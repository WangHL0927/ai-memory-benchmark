"""试跑工具共用的格式、路径、原子写入和失败记录；只用标准库。"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import os
import tempfile
import threading

from tools.paths import ROOT, TMP, inside_repo, is_public, private_path
from tools import score

_cache_progress = threading.local()


def set_cache_status(status):
    """每个工作线程独立记录本样本的缓存状态，供结束日志使用。"""
    _cache_progress.status = status


class ModelOutputError(ValueError):
    """输出解析/校验两次失败，不携带模型原文。"""
    attempts = 2


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value, overwrite=True):
    """同目录原子替换，失败时保留原文件。调用方先执行输出路径检查。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        if overwrite:
            os.replace(temporary, path)
        else:
            # 同目录硬链接只在目标不存在时成功，消除 exists→replace 的覆盖竞态。
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def output_path(path, data_dir=None):
    path = inside_repo(path)
    return path if data_dir is not None and is_public(data_dir) else private_path(path)


def identifier(value):
    if not re.fullmatch(r'[\w.-]+', value) or value in ('.', '..'):
        raise ValueError('系统标识只能含字母、数字、下划线、点和连字符')
    return value


def transcript(text):
    """轮次标题及正文去缩进规则，工具块仍附在助手轮内。"""
    meta, turns = {}, []
    for line in text.splitlines():
        match = re.fullmatch(r'## \[T(\d+)\]\s+(用户|助手)(?:\s+(.*))?', line)
        if match:
            turns.append(dict(turn=int(match[1]), role=match[2], lines=[]))
        elif turns:
            turns[-1]['lines'].append(line[1:] if line.startswith(' ') else line)
        else:
            match = re.fullmatch(r'- ([^：]+)：(.*)', line)
            if match:
                meta[match[1]] = match[2].strip()
    for turn in turns:
        turn['text'] = '\n'.join(turn.pop('lines')).strip()
    if not turns or len({t['turn'] for t in turns}) != len(turns):
        raise ValueError('transcript 无轮次或轮次重复')
    return meta, turns


def section(number):
    text = (ROOT / 'docs/framework.md').read_text(encoding='utf-8')
    marker = f'## {number}. '
    start = text.index(marker)
    following = re.search(r'\n## ', text[start:])
    return text[start:start + following.start()] if following else text[start:]


def policy_material(path=None, reference=None):
    policy = score.load_policy(path, reference)
    md = Path(path).with_suffix('.md') if path else ROOT / 'policies' / (policy['id'] + '-' + policy['version'] + '.md')
    return policy, md.read_text(encoding='utf-8')


def parse_json(text):
    """允许完整 JSON 代码围栏；不从夹杂说明的文本里偷偷截取 JSON。"""
    if not isinstance(text, str):
        raise ValueError('模型输出必须为 JSON 文本')
    text = text.strip()
    match = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, re.S)
    return json.loads(match[1] if match else text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('非有限 JSON 数值')))


def run_shell(gold, identity, values):
    return dict(standard_version=gold['standard_version'], policy=gold['policy'], system=identity,
                D=[dict(draft=d) for d in values])


def batch(tasks, worker, failures, concurrency=1, max_consecutive_failures=3):
    """并发任务互不共享会话；失败只记录位置、类型和尝试次数，不落模型原文或凭据。

    长任务以日志文件为准：tmp/logs/<失败记录名>-<时间>.log，每个样本开始、结束各一行，立即落盘。
    连续失败达到 max_consecutive_failures 就停止派发剩余样本，防止系统性错误空耗模型额度。
    """
    from concurrent.futures import ThreadPoolExecutor
    import time
    if concurrency < 1:
        raise ValueError('并发度必须为正')
    failure_path = output_path(failures)
    log_dir = TMP / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    # 日志名带上失败记录所在目录（通常就是数据集目录），避免不同数据集的同名任务混在一起
    log_path = log_dir / f"{Path(failure_path).parent.name}-{Path(failure_path).stem}-{now().replace(':', '')}.log"
    lock, state = threading.Lock(), dict(streak=0, stopped=False)
    tasks = list(tasks)

    def log(line):
        with lock, open(log_path, 'a', encoding='utf-8') as stream:
            stream.write(f"{now()} {line}\n")
            stream.flush()

    log(f"开始：{len(tasks)} 个样本，并发 {concurrency}")

    def execute(task):
        set_cache_status('miss')
        if state['stopped']:
            log(f"未执行（已熔断）：{task} cache=miss")
            return 'not-run', None
        started = time.monotonic()
        log(f"开始：{task}")
        try:
            status = worker(task)
            with lock:
                state['streak'] = 0
            log(f"完成：{task} 状态={status} cache={_cache_progress.status} 耗时={time.monotonic() - started:.1f}s")
            return status, None
        except Exception as exc:
            detail = str(exc)[:500] if isinstance(exc, (ValueError, KeyError, TypeError, RuntimeError)) else None
            with lock:
                state['streak'] += 1
                if state['streak'] >= max_consecutive_failures:
                    state['stopped'] = True
            log(f"失败：{task} 类型={type(exc).__name__} cache={_cache_progress.status} 耗时={time.monotonic() - started:.1f}s 原因={detail}")
            return 'failed', dict(sample=str(task), error_type=type(exc).__name__,
                                  message='处理失败，请核查输入或模型输出；未写入目标文件',
                                  # 只记我们自己校验代码的报错（不含模型原文和凭据）；原始返回见 tmp/model-raw/
                                  detail=detail, attempts=getattr(exc, 'attempts', 1), time=now())
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(execute, tasks))
    errors = [error for _, error in results if error]
    not_run = sum(status == 'not-run' for status, _ in results)
    write_json(failure_path, dict(failures=errors, failed=len(errors), not_run=not_run, circuit_broken=state['stopped'],
                                completed=sum(status == 'written' for status, _ in results),
                                skipped=sum(status == 'skipped' for status, _ in results), log=str(log_path)))
    summary = (f"已写入 {sum(s == 'written' for s, _ in results)}；跳过 {sum(s == 'skipped' for s, _ in results)}；"
               f"失败 {len(errors)}；未执行 {not_run}{'（连续失败熔断）' if state['stopped'] else ''}；日志：{log_path}")
    log('结束：' + summary)
    print(summary + f"；记录：{failure_path}")
    return 1 if errors else 0
