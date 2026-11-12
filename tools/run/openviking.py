"""OpenViking 0.5.4.1：每样本独立空库，只导出原生长期记忆 L2。

先安装：UV_CACHE_DIR=$PWD/tmp/envs/.uv-cache uv pip install \
    --python tmp/envs/openviking/bin/python openviking==0.5.4.1
运行：python3 -m tools.run.openviking --data-dir tmp/trial/e2e/trace/data
产品提示词不改写；HTTP 代理只校验模型名、统计调用并保存审计原文。
"""
import argparse
from inspect import getsource
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

from tools.config import sha256_file, environment, sha256_text
from tools.paths import ROOT, TMP
from tools.run.proxy import ModelProxy, settings
from tools.trial import batch, now, output_path, read_json, run_shell, transcript, write_json

VERSION = '0.5.4.1'
SYSTEM = 'openviking-0.5.4.1-ltm-defaults'
SOURCE_COMMIT = '86edb40a01733490b930266dcfcab7c2cebf94b3'


def transcript_to_messages(text):
    """工具块保留在助手正文里，既不解释也不执行；T 编号机械写入 turn_id。"""
    _, turns = transcript(text)
    return [dict(role={'用户': 'user', '助手': 'assistant'}[t['role']],
                 content=t['text'], turn_id=f"T{t['turn']}") for t in turns]


def native_to_draft(record):
    """一个原生 L2 文件就是一项；原生类型、所有者和 ranges 不推断成政策字段。"""
    content = record['content']
    if not isinstance(content, str) or not content.strip():
        raise ValueError('原生记忆正文为空或类型错误')
    return dict(statement=content, category='', scope='', attributes={}, conditions=[], evidence=[])


def select_records(records, initial_templates):
    """只排除与固定版初始化模板逐字相同的文件；修改过的模板完整提交。"""
    selected, excluded = [], []
    for record in records:
        filename = record['uri'].rsplit('/', 1)[-1]
        if record['content'].strip() == initial_templates.get(filename):
            excluded.append(dict(uri=record['uri'], sha256=sha256_text(record['content']),
                                 reason='与固定版初始化模板逐字一致'))
        else:
            selected.append(record)
    return selected, excluded


def api(base, path, body=None, timeout=30, **query):
    url = base + path + ('?' + urlencode(query) if query else '')
    payload = None if body is None else json.dumps(body).encode()
    with urlopen(Request(url, data=payload, headers={'Content-Type': 'application/json'}), timeout=timeout) as response:
        value = json.load(response)
    if value.get('status') not in ('ok', 'ready'):
        raise RuntimeError('产品 API 返回失败状态')
    return value


def result(base, path, body=None, timeout=30, **query):
    return api(base, path, body, timeout, **query)['result']


def list_files(base, uri, hidden=False):
    """逐层递归并分页：固定版 recursive=true 默认只下钻三层，会漏深层事件。"""
    rows, pending, visited = [], [uri], set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        offset = 0
        while True:
            page = api(base, '/api/v1/fs/ls', uri=current, recursive='false', output='original',
                       show_all_hidden=str(hidden).lower(), offset=offset, limit=200)
            entries = page['result']
            rows.extend(entries)
            pending.extend(r['uri'] for r in entries if r.get('isDir', r.get('is_dir', False)))
            if not page.get('has_more', len(entries) == 200):
                break
            if not entries:
                raise RuntimeError('目录分页未前进')
            offset += len(entries)
    return rows


def read_content(base, uri):
    return result(base, '/api/v1/content/read', uri=uri, offset=0, limit=-1)


def memory_files(base, user_root):
    # 遍历整个新身份，包括 peers/self，排除 sessions、资源和 L0/L1。
    return {r['uri']: read_content(base, r['uri']) for r in list_files(base, user_root)
            if not r.get('isDir', r.get('is_dir', False)) and '/memories/' in r['uri']
            and not r['uri'].rsplit('/', 1)[-1].startswith('.')}


def product_metadata(python):
    code = 'from pathlib import Path\nimport hashlib\n' + getsource(sha256_file) + '''
import importlib.metadata as m,json,pathlib,sys,platform
import yaml
from jinja2 import Template
p=m.distribution('openviking')
files={str(f):sha256_file(p.locate_file(f)) for f in p.files if str(f).startswith(('openviking/system_assets/', 'openviking/prompts/')) and pathlib.Path(p.locate_file(f)).is_file() and pathlib.Path(str(f)).suffix in ('.yaml','.yml','.md','.jinja2','.j2','.txt')}
initial={}
for f in p.files:
 if str(f).startswith('openviking/system_assets/memory/memory_templates/builtin/') and str(f).endswith('.yaml'):
  schema=yaml.safe_load(pathlib.Path(p.locate_file(f)).read_text())
  filename=schema.get('filename_template','')
  if schema.get('enabled') and filename and '{{' not in filename and schema.get('content_template'):
   fields={f['name']:f.get('init_value') or '' for f in schema.get('fields',[])}
   if fields:
    initial[filename]=Template(schema['content_template']).render(**fields).strip()
print(json.dumps({'version':p.version,'python':sys.version,'platform':platform.platform(),'initial_templates':initial,'dependencies':sorted(d.metadata['Name']+'=='+d.version for d in m.distributions()),'prompt_asset_sha256':files}))'''
    proc = subprocess.run([str(python), '-c', code], capture_output=True, text=True, timeout=30, check=True)
    meta = json.loads(proc.stdout)
    meta['published_wheel'] = dict(filename='openviking-0.5.4.1-cp310-abi3-macosx_14_0_arm64.whl',
                                  pypi_sha256='ad9dad29813ced92f764d9217e0366939961c557d05290d26e5e4bf4cd82edf9',
                                  verification='PyPI 发布哈希；本机安装文件哈希见 prompt_asset_sha256')
    if meta['version'] != VERSION:
        raise ValueError('必须安装 openviking==0.5.4.1')
    return meta


@contextmanager
def isolated_server(python, directory, config, proxy, startup_timeout):
    """不继承用户 HOME、模型凭据或产品配置；退出时终止服务并删除配置。"""
    store = directory / 'store'
    for name in ('home', 'temp'):
        (directory / name).mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    conf = dict(storage=dict(workspace=str(store), vectordb=dict(backend='local', name='context'),
                             agfs=dict(backend='local')),
                embedding=dict(dense=dict(provider='openai', api_base=proxy.url, api_key='local-proxy',
                                          model=config['embedding_model'], dimension=config['dimensions'])),
                vlm=dict(provider='openai', api_base=proxy.url, api_key='local-proxy', model=config['model'],
                         temperature=0.0, max_retries=2, timeout=proxy.timeout,
                         # 本地代理遮住百炼域名；复现产品对百炼原生发送的开关。
                         extra_request_body=dict(enable_thinking=False)),
                memory=dict(extraction_enabled=True, session_skill_extraction_enabled=False,
                            session_auto_commit=dict(enabled=False)),
                server=dict(agent_evolution=dict(enabled=False)))
    conf_path = directory / 'ov.conf'
    # mkstemp/原子写入创建的配置为 600，其中也没有真实凭据。
    write_json(conf_path, conf)
    os.chmod(conf_path, 0o600)
    env = dict(PATH=os.environ.get('PATH', ''), HOME=str(directory / 'home'),
               TMPDIR=str(directory / 'temp'), LANG='en_US.UTF-8', PYTHONUNBUFFERED='1',
               XDG_CONFIG_HOME=str(directory / 'home'), XDG_CACHE_HOME=str(directory / 'home'))
    command = [str(python.with_name('openviking-server')), '--config', str(conf_path),
               '--host', '127.0.0.1', '--port', str(port)]
    proc = None
    try:
        # 产品没有真实 key，日志无法回显该凭据。
        with (directory / 'server.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            base, deadline = f'http://127.0.0.1:{port}', time.monotonic() + startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('OpenViking 启动失败，见独立 server.log')
                try:
                    api(base, '/ready', timeout=2)
                    break
                except (URLError, TimeoutError, RuntimeError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('OpenViking 启动超时')
                    time.sleep(0.5)
            write_json(directory / 'process.json', dict(pid=proc.pid, port=port))
            yield base
    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            except ProcessLookupError:
                pass
        conf_path.unlink(missing_ok=True)


def wait_commit(base, commit, deadline, poll):
    if commit.get('status') != 'accepted' or not commit.get('task_id'):
        raise RuntimeError('commit 未启动抽取')
    while time.monotonic() < deadline:
        task = result(base, '/api/v1/tasks/' + commit['task_id'],
                      timeout=max(0.1, min(30, deadline - time.monotonic())))
        if task['status'] in ('failed', 'cancelled'):
            raise RuntimeError('异步抽取任务失败')
        if task['status'] == 'completed':
            return task
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
    raise TimeoutError('异步抽取超时')


def generate(sample, args, config, package):
    target = output_path(sample / 'runs' / (SYSTEM + '.json'), sample.parent)
    if target.exists() and not args.force:
        return 'skipped'
    text = (sample / 'transcript.md').read_text(encoding='utf-8')
    messages = transcript_to_messages(text)
    started = time.monotonic()
    directory = TMP / 'openviking' / (sample.name + '-' + uuid.uuid4().hex)
    directory.mkdir(parents=True, mode=0o700)
    manifest = dict(sample=sample.name, system=SYSTEM, time=now(), input_sha256=sha256_text(text),
                    raw_directory=str(directory.relative_to(ROOT)), timeout_seconds=args.timeout,
                    status='running', product=package)
    proxy = ModelProxy(config, directory, args.request_timeout)
    try:
        with proxy, isolated_server(args.python, directory, config, proxy, args.startup_timeout) as base:
            session_id = 'eval-' + uuid.uuid4().hex
            session = result(base, '/api/v1/sessions', dict(session_id=session_id, auto_commit_policy=None))
            if session.get('auto_commit_policy') is not None:
                raise RuntimeError('自动 commit 未关闭')
            user_root = session['uri'].split('/sessions/')[0]
            before = memory_files(base, user_root)
            manifest['initial_memory_sha256'] = {u: sha256_text(c) for u, c in before.items()}
            for offset in range(0, len(messages), 100):
                added = result(base, f'/api/v1/sessions/{session_id}/messages/batch', dict(messages=messages[offset:offset + 100]))
                if added.get('added') != len(messages[offset:offset + 100]) or added.get('message_count') != min(offset + 100, len(messages)):
                    raise RuntimeError('产品未完整接收消息或意外自动 commit')
            commit = result(base, f'/api/v1/sessions/{session_id}/commit',
                            dict(keep_recent_count=0, enable_working_memory=False))
            deadline = time.monotonic() + args.timeout
            task = wait_commit(base, commit, deadline, args.poll_interval)
            # 服务端 wait 的 timeout 字段被忽略，用客户端剩余时限约束。
            queues = result(base, '/api/v1/system/wait', {}, timeout=max(0.1, deadline - time.monotonic()))
            if any(q.get('error_count', 0) for q in queues.values()):
                raise RuntimeError('产品后续队列存在错误')
            archive = commit['archive_uri'].rstrip('/')
            archive_files = list_files(base, archive, hidden=True)
            write_json(directory / 'completion.json', dict(task=task, queues=queues, archive_files=archive_files))
            names = {r['uri'].rsplit('/', 1)[-1] for r in archive_files}
            if '.failed.json' in names or '.done' not in names or 'memory_diff.json' not in names:
                raise RuntimeError('归档未完整完成')
            diff = json.loads(read_content(base, archive + '/memory_diff.json'))
            final = memory_files(base, user_root)
            operations = diff['operations']
            touched = {r['uri'] for k in ('adds', 'updates') for r in operations.get(k, [])}
            deleted = {r['uri'] for r in operations.get('deletes', [])}
            for extra_root in {u.split('/memories/')[0] + '/memories' for u in touched
                               if '/memories/' in u and not u.startswith(user_root + '/')} :
                final.update(memory_files(base, extra_root))
            changed = {u: c for u, c in final.items() if before.get(u) != c}
            if (touched - deleted) - set(final):
                raise RuntimeError('memory_diff 与最终枚举不一致')
            all_records = [dict(uri=u, content=c) for u, c in sorted(changed.items())]
            records, excluded = select_records(all_records, package['initial_templates'])
            manifest['excluded_initial_templates'] = excluded
            values = [native_to_draft(r) for r in records]
            write_json(directory / 'export.json', dict(records=records, all_records=all_records, excluded_initial_templates=excluded, memory_diff=diff, task=task,
                                                       queues=queues, session=session, commit=commit))
            if any(c.get('status') != 200 for c in proxy.calls):
                raise RuntimeError('模型调用曾失败，见脱敏调用审计')
            if any('length' in c.get('finish_reasons', []) for c in proxy.calls):
                raise RuntimeError('模型输出达到上限被截断')
        # gold 仅在产品退出后读取外壳，不把标注内容交给产品。
        gold = read_json(sample / 'gold.json')
        meta = dict(name=SYSTEM, version=VERSION, deployment='venv / 每样本独立进程和 workspace',
                    source_commit=SOURCE_COMMIT, package=package, llm=config['model'],
                    excluded_initial_templates=excluded,
                    base_url=config['base_url'], parameters=proxy.parameters(),
                    embedding_model=config['embedding_model'], embedding=dict(dimensions=config['dimensions'],
                    normalization='未记录；适配器不额外归一化', chunking='产品原生默认；具体参数未记录'),
                    prompt_version='OpenViking v' + VERSION + ' 原生模板',
                    prompt_sha256=package['prompt_asset_sha256'], input_sha256=sha256_text(text),
                    usage=proxy.summary(), calls=proxy.calls, time=now(), environment=environment(),
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    context=dict(window='未记录', input_characters=len(text), message_characters=sum(len(m['content']) for m in messages),
                                 header_metadata_forwarded=False, adapter_input_truncated=False,
                                 source_truncation_markers=text.count('截断'), product_input_truncated='未记录',
                                 output_truncated=False), timeout=dict(seconds=args.timeout, occurred=False),
                    retries='见逐调用审计；产品内部重试次数未记录', raw_directory=str(directory.relative_to(ROOT)),
                    raw_export_sha256=sha256_file(directory / 'export.json'),
                    dataset_manifest='见样本来源；版本未记录', isolation=dict(auto_commit=False, working_memory=False,
                    skill_extraction=False, agent_evolution=False), output_sha256=sha256_text(json.dumps(values, ensure_ascii=False, sort_keys=True)))
        write_json(target, run_shell(gold, meta, values), overwrite=args.force)
        manifest.update(status='written', output=str(target.relative_to(ROOT)), exported=len(values),
                        run_sha256=sha256_file(target))
        print(f"{sample.name}：{meta['elapsed_seconds']:.1f}s；LLM={meta['usage']['llm']['calls']}；"
              f"embedding={meta['usage']['embedding']['calls']}；导出={len(values)}", flush=True)
        return 'written'
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, timed_out=isinstance(exc, TimeoutError),
                        detail=str(exc) if isinstance(exc, RuntimeError) else None)
        raise
    finally:
        manifest.update(elapsed_seconds=round(time.monotonic() - started, 3), usage=proxy.summary(), parameters=proxy.parameters(), calls=proxy.calls)
        write_json(directory / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=TMP / 'envs/openviking/bin/python')
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--startup-timeout', type=float, default=120)
    parser.add_argument('--poll-interval', type=float, default=1)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--failures', type=Path, default=TMP / 'trial/openviking-failures.json')
    parser.add_argument('--concurrency', type=int, default=1, help='同时处理的样本数；每个样本独立进程、端口和存储')
    args = parser.parse_args()
    try:
        if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.request_timeout, args.startup_timeout, args.poll_interval)):
            raise ValueError('所有超时和轮询间隔必须为有限正数')
        args.python = args.python.absolute()
        config, package = settings(), product_metadata(args.python)
        return batch(sorted(p.parent for p in args.data_dir.glob('*/gold.json')),
                     lambda p: generate(p, args, config, package), args.failures, concurrency=args.concurrency)
    except (ValueError, OSError, subprocess.SubprocessError):
        parser.exit(2, '错误：配置或隔离环境无效，请检查必要环境变量及固定版本安装\n')


if __name__ == '__main__':
    raise SystemExit(main())
