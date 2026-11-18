"""ai-memory single：每样本独立实例，默认 hook 摄取后手动单页整理。

python3 -m tools.run.ai_memory --data-dir tmp/trial/e2e/trace/data
只使用 brew 保留的固定版真实二进制，不调用默认链接、宿主 hooks 或现役服务。
原生配置依据固定源码 7bb659ceda9cad1c3867079313d6667e2abeb321；
https://github.com/akitaonrails/ai-memory/tree/v3.4.0
"""
import argparse
from collections import Counter
from contextlib import closing, contextmanager
import json
import math
import os
from pathlib import Path
import re
import socket
import sqlite3
from string import Template
import shutil
import subprocess
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

from tools.config import sha256_file, environment, sha256_text
from tools.paths import ROOT, TMP
from tools.run.proxy import ModelProxy, settings
from tools.run.project import declared_project
from tools.trial import batch, identifier, now, output_path, read_json, run_shell, transcript, write_json

VERSION = '3.4.0'
SYSTEM = 'ai-memory-3.4.0-single-defaults-qwen'
SOURCE_COMMIT = '7bb659ceda9cad1c3867079313d6667e2abeb321'
BINARY = Path(os.getenv('AI_MEMORY_340_BINARY', 'ai-memory'))
RELEASES = {
    VERSION: (SYSTEM, SOURCE_COMMIT, BINARY),
    '3.4.7': (
        'ai-memory-3.4.7-single-defaults-qwen',
        'fa0ec25dbe987b7574a7487ecf92f0873040c2f2',
        Path(os.getenv('AI_MEMORY_347_BINARY', 'ai-memory')),
    ),
}
ENVS = TMP / 'envs/ai-memory'
TEMPLATE = Path(__file__).with_suffix('') / 'single.toml'
# 冻结机械排除清单；会话总结是被测产物，必须保留。
STRUCTURAL_NAMES = {'_meta.md', '_index.md', 'index.md'}
TOOL = re.compile(r'^工具调用：([^\n]+)\n参数：', re.M)
TOOL_OUTPUT = re.compile(r'^工具输出：([^\n]*)\n?', re.M)


def tool_events(text):
    """只解析 transcript 的工具标记，不执行历史命令；不认识的参数保留原文。

    旧 transcript 没有工具结果结束标记，结果延续到下一工具块或本轮末尾。
    不从自然语言猜测工具名或输出边界。
    """
    matches = list(TOOL.finditer(text))
    rows = []
    for index, match in enumerate(matches):
        segment = text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        # JSON 参数可包含换行和工具标记字符串，先消耗完整 JSON 再查结果。
        try:
            arguments, end = json.JSONDecoder().raw_decode(segment.lstrip())
            tail = segment.lstrip()[end:]
        except json.JSONDecodeError:
            output = TOOL_OUTPUT.search(segment)
            arguments = segment[:output.start() if output else len(segment)].strip()
            tail = segment[output.start():] if output else ''
        output = TOOL_OUTPUT.search(tail)
        row = dict(tool_name=match[1].strip(), tool_input=arguments)
        if output:
            row.update(tool_use_id=output[1].strip(), tool_response=tail[output.end():].strip())
        rows.append(row)
    return rows


def replay_requests(text, base, session_id, project, cwd):
    """逐轮回放原生 hook 信封；助手 Stop 不带正文、marker 或采集 opt-in。"""
    _, turns = transcript(text)
    events, mapping = [], []

    def add(event, turn=None, **fields):
        index = len(events)
        query = dict(event=event, agent='claude-code', workspace='eval', project=project,
                     ingest_key=f'{session_id}-{index:06d}')
        events.append(dict(url=base + '/hook?' + urlencode(query),
                           body=dict(session_id=session_id, cwd=str(cwd), **fields)))
        mapping.append(dict(index=index, turn=turn, event=event))

    add('session-start')
    assistant_bytes, tool_count = 0, 0
    for index, turn in enumerate(turns):
        if turn['role'] == '用户':
            add('user-prompt-submit', turn['turn'], prompt=turn['text'])
        else:
            assistant_bytes += len(turn['text'].encode('utf-8'))
            for tool in tool_events(turn['text']):
                tool_count += 1
                before = {k: v for k, v in tool.items() if k != 'tool_response'}
                add('pre-tool-use', turn['turn'], **before)
                if 'tool_response' in tool:
                    add('post-tool-use', turn['turn'], **tool)
            # 连续的助手/工具消息属于同一响应；仅在响应末尾发一次 Stop。
            if index + 1 == len(turns) or turns[index + 1]['role'] == '用户':
                add('stop', turn['turn'])
    add('session-end')
    return events, dict(turns=len(turns), event_mapping=mapping, tools=tool_count,
                        capture_assistant=False, assistant_turns_not_captured=sum(t['role'] == '助手' for t in turns),
                        assistant_turn_bytes=assistant_bytes, assistant_body_forwarded=False, adapter_input_truncated=False,
                        source_truncation_markers=text.count('截断'), header_metadata_forwarded=False,
                        tool_output_boundary='下一工具块或本轮末尾；源格式无显式结束标记')


def validate_ack(ack, count):
    """2xx 不等于完整摄取：验证每个索引和终态，原生 policy drop 单独计数。"""
    expected = list(range(count))
    accepted = ack.get('accepted_indices', list(range(ack.get('accepted', 0))))
    results = ack.get('results', [])
    if (accepted != expected or ack.get('accepted') != count or 'failed_index' in ack
            or [r.get('index') for r in results] != expected):
        raise RuntimeError('hook batch 未完整确认，禁止当作空记忆')
    counts = Counter(r.get('outcome') for r in results)
    if set(counts) - {'stored', 'dropped_policy'}:
        raise RuntimeError('新实例发生重放、权限/身份冲突或无效事件，见 hook 审计')
    return counts


def native_to_draft(record):
    """single 整页原文作为一项；技术 namespace/tier 不转换为语义范围或类别。"""
    body = record['body']
    if not isinstance(body, str) or not body.strip():
        raise ValueError('原生记忆正文为空或类型错误')
    return dict(statement=body, category='', scope='', attributes={}, conditions=[], evidence=[])


def database(store):
    """只读实验数据库，绝不使用产品的默认目录发现或客户端路由。"""
    path = store / 'db/memory.sqlite'
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def session_snapshot(store, project):
    with closing(database(store)) as connection:
        sessions = connection.execute('''SELECT hex(s.id) AS id, s.ended_at, w.name AS workspace,
            p.name AS project FROM sessions s JOIN workspaces w ON s.workspace_id=w.id
            JOIN projects p ON s.project_id=p.id''').fetchall()
        if len(sessions) != 1 or sessions[0]['workspace'] != 'eval' or sessions[0]['project'] != project:
            raise RuntimeError('实验会话数量或作用域不符')
        if sessions[0]['ended_at'] is None:
            raise RuntimeError('实验会话尚未结束')
        rows = connection.execute('SELECT hex(id) AS id, kind, title, body FROM observations ORDER BY created_at, id').fetchall()
        if not rows:
            raise RuntimeError('回放后没有任何原生观察')
        return dict(session_id=str(uuid.UUID(hex=sessions[0]['id'])),
                    observations=[dict(row) for row in rows])


def export_pages(store):
    """停进程后全库读取最新页；schema 缺失报错，不将历史版或 ledger 重复提交。"""
    with closing(database(store)) as connection:
        columns = {r['name'] for r in connection.execute('PRAGMA table_info(pages)')}
        if not {'id', 'path', 'body', 'frontmatter_json', 'is_latest', 'workspace_id', 'project_id'} <= columns:
            raise RuntimeError('原生 pages schema 不兼容')
        rows = connection.execute('''SELECT hex(g.id) AS id, g.path, g.title, g.body,
            g.frontmatter_json, g.tier, w.name AS workspace, p.name AS project
            FROM pages g JOIN workspaces w ON g.workspace_id=w.id
            JOIN projects p ON g.project_id=p.id WHERE g.is_latest=1
            ORDER BY w.name, p.name, g.path''').fetchall()
        schema = {r['name']: r['sql'] for r in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('pages','sessions','observations')")}
    selected, excluded = [], []
    for row in rows:
        record = dict(row)
        name = record['path'].rsplit('/', 1)[-1]
        if name in STRUCTURAL_NAMES or re.fullmatch(r'log-\d{4}-\d{2}\.md', name):
            excluded.append(record)
        else:
            record['frontmatter'] = json.loads(record.pop('frontmatter_json'))
            selected.append(record)
    return selected, excluded, schema


def http(base, path, body=None, timeout=30):
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}
    with urlopen(Request(base + path, data=payload, headers=headers), timeout=timeout) as response:
        return json.load(response)


def consolidate(base, session_id, timeout, audit=None):
    """只调用一次 tools/call；传库内 UUID，不虚构 workspace/project 参数。"""
    value = http(base, '/mcp', dict(jsonrpc='2.0', id=1, method='tools/call', params=dict(
        name='memory_consolidate', arguments=dict(session_id=session_id, dry_run=False, multi_page=False))), timeout)
    if audit is not None:
        write_json(audit, value)
    if 'error' in value or value.get('result', {}).get('isError'):
        raise RuntimeError('受控 consolidation 失败，见 MCP 审计和模型审计')
    blocks = value.get('result', {}).get('content', [])
    if len(blocks) != 1 or blocks[0].get('type') != 'text':
        raise RuntimeError('consolidation 返回结构不符')
    outcome = json.loads(blocks[0]['text'])
    if outcome.get('dry_run') is not False or not outcome.get('page_id') or not outcome.get('new_body_markdown'):
        raise RuntimeError('consolidation 未生成有效记忆')
    return value, outcome


def child_environment(directory):
    """白名单环境：即使调用者配置了现役地址/凭据，也不会传入子进程。"""
    return dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', HOME=str(directory / 'home'),
                TMPDIR=str(directory / 'temp'), XDG_CONFIG_HOME=str(directory / 'home'),
                XDG_DATA_HOME=str(directory / 'home'), XDG_CACHE_HOME=str(directory / 'home'),
                AI_MEMORY_HOME=str(directory / 'home'), AI_MEMORY_IGNORE_MARKER='true',
                LANG='en_US.UTF-8', NO_COLOR='1', LLM_API_KEY='local-proxy', EMBEDDING_API_KEY='local-proxy')


def product_metadata(binary, version=VERSION):
    probe = ENVS / ('version-' + uuid.uuid4().hex)
    for name in ('home', 'temp'):
        (probe / name).mkdir(parents=True, mode=0o700)
    proc = subprocess.run([str(binary), '--version'], cwd=probe, env=child_environment(probe),
                          capture_output=True, text=True, check=True, timeout=10)
    if proc.stdout.strip() != 'ai-memory ' + version:
        raise ValueError(f'必须使用 ai-memory {version}，不能跟随 brew 默认链接')
    system, source_commit, _ = RELEASES[version]
    return dict(version=version, system=system, binary=str(binary), binary_sha256=sha256_file(binary),
                source_commit=source_commit, source_tag=f'v{version}',
                source_binary_correspondence='发布版本核对；未复现源码构建',
                template_sha256=sha256_file(TEMPLATE),
                adapter_sha256=sha256_file(Path(__file__)),
                proxy_sha256=sha256_file(Path(__file__).with_name('proxy.py')),
                dependencies='Rust 单二进制；动态依赖锁未记录')


@contextmanager
def isolated_server(binary, directory, config, proxy, startup_timeout, *, project=None, native_vectors=False):
    """全新数据目录、空 HOME、随机端口；仅在本进程成功绑定后连接该端口。"""
    for name in ('home', 'temp'):
        (directory / name).mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    variables = dict(model=json.dumps(config['model']), proxy_url=json.dumps(proxy.url),
                     embedding_model=json.dumps(config['embedding_model']), dimensions=config['dimensions'],
                     request_timeout=math.ceil(proxy.timeout))
    rendered = Template(TEMPLATE.read_text()).substitute(variables)
    if native_vectors:
        rendered = '\n'.join(line for line in rendered.split('\n') if not line.startswith('embedding_'))
    conf_path = directory / 'config.toml'
    conf_path.write_text(rendered)
    os.chmod(conf_path, 0o600)
    write_json(directory / 'configuration.json', dict(template_sha256=sha256_file(TEMPLATE),
                                                     rendered_sha256=sha256_text(rendered), config=rendered))
    project = project or 'sample-' + directory.parent.name
    command = [str(binary), '--data-dir', str(directory / 'store'), '--config', str(conf_path),
               'serve', '--transport', 'http', '--bind', f'127.0.0.1:{port}', '--no-watcher',
               '--workspace', 'eval', '--project', project]
    process = None
    try:
        with (directory / 'server.log').open('w') as log:
            process = subprocess.Popen(command, cwd=directory, env=child_environment(directory),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(directory / 'process.json', dict(pid=process.pid, port=port, command=command))
            base = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError('ai-memory 启动失败，见隔离 server.log')
                # 不靠探测一个候选端口判断就绪，避免误连恰好占用端口的其他服务。
                if 'MCP HTTP server ready' in (directory / 'server.log').read_text():
                    if http(base, '/healthz', timeout=2).get('status') != 'ok':
                        raise RuntimeError('实验服务健康检查失败')
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('ai-memory 启动超时')
                time.sleep(0.1)
            yield base, project
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise RuntimeError('实验服务未正常停止，不能导出可能未落稳的数据')
            write_json(directory / 'exit.json', dict(returncode=process.returncode))
            if process.returncode != 0:
                raise RuntimeError('实验服务非正常退出，见独立 server.log')


def generate(sample, args, config, package, *, max_input_tokens=None, stop_file=None):
    if stop_file is not None and stop_file.exists():
        raise SystemExit('评测输入 token 已达到停止条件')
    system, version = package.get('system', SYSTEM), package.get('version', VERSION)
    target = output_path(sample / 'runs' / (system + '.json'), sample.parent)
    if target.exists() and not args.force:
        return 'skipped'
    text = (sample / 'transcript.md').read_text(encoding='utf-8')
    identifier(sample.name)
    started = time.monotonic()
    directory = ENVS / sample.name / uuid.uuid4().hex
    directory.mkdir(parents=True, mode=0o700)
    manifest = dict(sample=sample.name, system=system, time=now(), status='running', product=package,
                    input_sha256=sha256_text(text), raw_directory=str(directory.relative_to(ROOT)),
                    environment=environment(), timeout_seconds=args.timeout)
    proxy = ModelProxy(config, directory, args.request_timeout,
                       max_input_tokens=max_input_tokens, stop_file=stop_file)
    try:
        with proxy:
            native = {}
            if getattr(args, 'v07', False):
                native = dict(project=declared_project(text), native_vectors=True)
            with isolated_server(args.binary, directory, config, proxy, args.startup_timeout, **native) as (base, project):
                session_id = str(uuid.uuid4())
                events, capture = replay_requests(text, base, session_id, project, directory)
                write_json(directory / 'replay.json', dict(events=events, **capture))
                deadline = time.monotonic() + args.timeout
                counts = Counter()
                for offset in range(0, len(events), 256):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('回放超时')
                    chunk = events[offset:offset + 256]
                    ack = http(base, '/hook/batch', chunk, timeout=remaining)
                    write_json(directory / f'ack-{offset:06d}.json', ack)
                    counts.update(validate_ack(ack, len(chunk)))
                snapshot = session_snapshot(directory / 'store', project)
                write_json(directory / 'observations.json', snapshot)
                # batch 按项等待写入完成；会话映射与落库另行校验，不靠任意 sleep。
                if len(snapshot['observations']) != counts['stored']:
                    raise RuntimeError('hook 确认数与库内观察数不一致')
                capture.update(acknowledgements=dict(counts), observations=len(snapshot['observations']),
                               product_input_truncated='原生消毒/投影有损；详见 observations.json 与线上请求，精确截断量未记录')
                write_json(directory / 'capture.json', capture)
                if any(c['kind'] == 'llm' for c in proxy.calls):
                    raise RuntimeError('手动 consolidation 前出现非预期 LLM 调用')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('consolidation 总时限已过')
                _, outcome = consolidate(base, snapshot['session_id'], remaining, directory / 'consolidation.json')
            # 必须停进程后导出最新原生页，防止背景写入改变停止点。
            records, excluded, schema = export_pages(directory / 'store')
            if not any(r['path'] == outcome['path'] and r['body'] == outcome['new_body_markdown'] for r in records):
                raise RuntimeError('MCP 结果与最新原生页不一致')
            write_json(directory / 'export.json', dict(records=records, excluded=excluded, schema=schema,
                                                       source_session=snapshot['session_id']))
            if proxy.errors or not any(c['kind'] == 'llm' for c in proxy.calls) or any(c.get('status') != 200 for c in proxy.calls):
                raise RuntimeError('模型调用缺失或失败，禁止提交空结果')
            if any('length' in c.get('finish_reasons', []) for c in proxy.calls):
                raise RuntimeError('模型输出被上限截断')
            if proxy.input_limit_exceeded:
                raise RuntimeError('样本超过输入 token 上限')
        values = [native_to_draft(r) for r in records]
        # 产品已经退出后才读取金标外壳，U/N 从不交给被测进程。
        gold = read_json(sample / 'gold.json')
        prompt_hashes = {str(c['index']): sha256_text(json.dumps(read_json(ROOT / c['raw_path'])['request']['messages'],
                         ensure_ascii=False, sort_keys=True)) for c in proxy.calls if c['kind'] == 'llm'}
        meta = dict(name=system, version=version, configuration='single', product=package,
                    source_commit=package.get('source_commit', SOURCE_COMMIT),
                    deployment='brew 固定版本真实二进制 / 每样本独立 HOME、进程、端口、数据库',
                    llm=config['model'], base_url=config['base_url'],
                    parameters=proxy.parameters(),
                    embedding_model=config['embedding_model'], embedding=dict(dimensions=config['dimensions'],
                        normalization='未记录；适配器不额外归一化', chunking='产品原生默认；见原始请求'),
                    prompt_version=f'ai-memory v{version} 原生 single', prompt_sha256=prompt_hashes,
                    input_sha256=sha256_text(text), output_sha256=sha256_text(json.dumps(values, ensure_ascii=False, sort_keys=True)),
                    raw_directory=str(directory.relative_to(ROOT)), raw_export_sha256=sha256_file(directory / 'export.json'),
                    usage=proxy.summary(), calls=proxy.calls, proxy_errors=proxy.errors,
                    time=now(), elapsed_seconds=round(time.monotonic() - started, 3),
                    context=dict(window='未记录', input_characters=len(text), capture=capture, output_truncated=False),
                    isolation=dict(backfill_on_start=False, capture_assistant=False, consolidate_on_session_end=False,
                        auto_improve=False, maintenance=False, profile=False, dream=False, watcher=False, manual_consolidations=1),
                    timeout=dict(seconds=args.timeout, occurred=False), retries='适配器不重试；产品重试见逐调用审计',
                    dataset_manifest='见样本来源；版本未记录', environment=environment())
        run = run_shell(gold, meta, values)
        if getattr(args, 'v07', False):
            meta['embedding_model'] = 'all-MiniLM-L6-v2'
            meta['embedding'] = dict(provider='native-local', dimensions=384)
            for item, record in zip(run['D'], records):
                item['reader_metadata'] = dict(scope='project:' + record['project'], source=record['path'])
        write_json(target, run, overwrite=args.force)
        manifest.update(status='written', output=str(target.relative_to(ROOT)), exported=len(values),
                        run_sha256=sha256_file(target))
        print(f"{sample.name}：{meta['elapsed_seconds']:.1f}s；LLM={meta['usage']['llm']['calls']}；"
              f"embedding={meta['usage']['embedding']['calls']}；导出={len(values)}", flush=True)
        return 'written'
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, timed_out=isinstance(exc, TimeoutError),
                        detail=str(exc) if isinstance(exc, RuntimeError) else None)
        if proxy.input_limit_exceeded:
            raise SystemExit('样本超过输入 token 上限，停止评测') from None
        raise
    finally:
        manifest.update(elapsed_seconds=round(time.monotonic() - started, 3), usage=proxy.summary(), parameters=proxy.parameters(),
                        calls=proxy.calls, proxy_errors=proxy.errors, input_limit_exceeded=proxy.input_limit_exceeded)
        write_json(directory / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--v07', action='store_true', help='使用 v0.7 项目头部和原生向量默认值')
    parser.add_argument('--product-version', choices=RELEASES, default=VERSION)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--startup-timeout', type=float, default=60)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--failures', type=Path, default=TMP / 'trial/ai-memory-failures.json')
    parser.add_argument('--concurrency', type=int, default=1, help='同时处理的样本数；每个样本独立进程、端口和存储')
    parser.add_argument('--max-input-tokens', type=int, help='停止评测的单样本输入 token 上限')
    parser.add_argument('--stop-file', type=Path, help='跨适配器共享评测停止状态（限 tmp/）')
    args = parser.parse_args()
    try:
        if (any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.request_timeout, args.startup_timeout))
                or (args.max_input_tokens is not None and args.max_input_tokens <= 0)):
            raise ValueError('超时必须是有限正数')
        binary = args.binary or RELEASES[args.product_version][2]
        if args.binary is None and binary == Path('ai-memory'):
            brew = shutil.which('brew')
            if brew is None:
                raise ValueError('需要固定版本的 --binary')
            cellar = Path(subprocess.check_output([brew, '--cellar', 'ai-memory'], text=True).strip())
            binary = cellar / args.product_version / 'libexec/ai-memory'
        args.binary = binary.resolve(strict=True)
        if args.stop_file is not None:
            args.stop_file = output_path(args.stop_file)
        config = settings()
        # 生成参数沿用产品默认；代理只校验模型并记录实际请求。
        package = product_metadata(args.binary, args.product_version)
        return batch(sorted(p.parent for p in args.data_dir.glob('*/gold.json')),
                     lambda p: generate(p, args, config, package, max_input_tokens=args.max_input_tokens,
                                        stop_file=args.stop_file), args.failures, concurrency=args.concurrency)
    except (ValueError, OSError, subprocess.SubprocessError):
        parser.exit(2, '错误：配置或固定版本二进制无效，请核查 .env 和隔离环境\n')


if __name__ == '__main__':
    raise SystemExit(main())
