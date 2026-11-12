"""claude-mem 15.17.0 本地观察者：每样本全新 worker，关闭向量。

安装（不运行任何安装脚本，不安装 hooks）：
npm --cache tmp/envs/.npm-cache install --prefix tmp/envs/claude-mem \
    --ignore-scripts --no-audit --no-fund claude-mem@15.17.0
运行：python3 -m tools.run.claude_mem --data-dir tmp/trial/e2e/trace/data
需要已安装 Bun；直接运行包内 worker 的前台入口，不调用会发现/接管服务的 start。
"""
import argparse
from collections import Counter
from contextlib import closing, contextmanager
import json
import math
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
import uuid

from tools.config import sha256_file, environment, sha256_text
from tools.paths import ROOT, TMP
from tools.run.proxy import ModelProxy, settings
from tools.trial import batch, identifier, now, output_path, read_json, run_shell, transcript, write_json

VERSION = '15.17.0'
SYSTEM = 'claude-mem-15.17.0-openai-compatible-novec'
SOURCE_COMMIT = '5681b2a3a9ec1be2c53264e6009ab82ad11ae601'
ENVS = TMP / 'envs/claude-mem'
PACKAGE = ENVS / 'node_modules/claude-mem'
GUARD = Path(__file__).with_suffix('') / 'network.cjs'
INTEGRITY = 'sha512-fudqrmuttL65Shg9PlN+GitsYW20iO0Y3lnbte+vIwI8omEU9DtjUx1+x+DlQc5OuZ8GgkY8i6R5ngzWNyT1qw=='
MARKER = re.compile(r'^工具调用：([^\n]+)\n参数：|^工具输出：([^\n]*)\n?', re.M)
FIELDS = dict(observation=('title', 'subtitle', 'facts', 'narrative'),
              summary=('request', 'investigated', 'learned', 'completed', 'next_steps', 'notes'))


def parse_tools(text):
    """仅解析框架工具标记，历史命令绝不执行，输出保留 transcript 已有截断。

    提取器丢失调用侧 id；同轮的调用/结果只能按出现顺序 FIFO 配对，不能靠
    工具名猜测。JSON 参数先完整解码，避免把参数字符串中的标记当作边界。
    源格式没有结果结束标记，结果到下一个标记或轮末；数量不匹配直接拒绝。
    助手可辨认的正文仅为首个工具块之前的文字。
    """
    calls, outputs, position, prefix = [], [], 0, None
    while match := MARKER.search(text, position):
        if prefix is None:
            prefix = text[:match.start()].strip()
        if match[1] is not None:
            tail = text[match.end():]
            leading = len(tail) - len(tail.lstrip())
            try:
                value, end = json.JSONDecoder().raw_decode(tail.lstrip())
                position = match.end() + leading + end
            except json.JSONDecodeError:
                following = MARKER.search(text, match.end())
                position = following.start() if following else len(text)
                value = text[match.end():position].strip()
            calls.append(dict(tool_name=match[1].strip(), tool_input=value))
        else:
            following = MARKER.search(text, match.end())
            position = following.start() if following else len(text)
            outputs.append(dict(tool_use_id=match[2].strip(),
                                tool_response=text[match.end():position].strip()))
    if len(calls) != len(outputs):
        raise ValueError('工具调用/结果数量不匹配，源 transcript 不足以可靠配对')
    for call, output in zip(calls, outputs):
        call.update(output)
    return (text if prefix is None else prefix), calls


def replay_requests(text, session_id, project, cwd):
    """用户轮→init；工具完成→observation；每个响应末尾→summarize。

    连续助手轮属于同一响应，只有最后可辨认正文经 last_assistant_message
    进入 Stop；中间正文不会伪装为工具输出。没有工具的聊天只经 summary
    输入，裸 prompt 独立抽取保持产品默认关闭。头部不转发，不读取 gold。
    Stop 已从 transcript 丢失，机械采用“下轮为用户或会话结束”的边界。
    """
    _, turns = transcript(text)
    events, omitted = [], []
    last_text = ''

    def add(kind, turn=None, **body):
        events.append(dict(path='/api/sessions/' + kind, turn=turn,
                           body=dict(contentSessionId=session_id, platformSource='claude', cwd=str(cwd), **body)))

    if turns[0]['role'] != '用户':
        raise ValueError('本地观察者回放必须从用户轮开始')
    for index, turn in enumerate(turns):
        if turn['role'] == '用户':
            add('init', turn['turn'], project=project, prompt=turn['text'],
                nativePromptId=f"{session_id}-T{turn['turn']}")
            last_text = ''
        else:
            prose, tools = parse_tools(turn['text'])
            if last_text:
                omitted.append(dict(turn=last_turn, characters=len(last_text)))
            last_text, last_turn = prose, turn['turn']
            for number, tool in enumerate(tools):
                if not tool['tool_use_id']:
                    tool['tool_use_id'] = f"{session_id}-T{turn['turn']}-{number}"
                add('observations', turn['turn'], **tool)
            if index + 1 == len(turns) or turns[index + 1]['role'] == '用户':
                add('summarize', turn['turn'], last_assistant_message=last_text)
                last_text = ''
    add('session-end', reason='eval-completed')
    return events, dict(turns=len(turns), event_counts=dict(Counter(e['path'].rsplit('/', 1)[1] for e in events)),
                        omitted_intermediate_assistant=omitted, header_metadata_forwarded=False,
                        adapter_input_truncated=False, source_truncation_markers=text.count('截断'),
                        tool_pairing='同轮 FIFO；调用侧 id 已被源格式丢弃；数量不符拒绝',
                        tool_output_boundary='下一工具标记或轮末；源格式无显式结束标记',
                        stop_boundary='响应末尾（下一用户轮之前或会话结尾）')


def native_to_draft(record):
    """一条原生记录一项，固定字段顺序机械拼接；不推断政策属性或证据。

    title/subtitle/facts/narrative 与六个 summary 字段完整保留。facts 的
    JSON 数组仅解包为逐行原文，不拆成多个 D；其他原生元信息保存在 export。
    prompt_number 只有用户轮归属，没有原文 quote，因此 evidence 必须留空。
    """
    parts = []
    for field in FIELDS[record['kind']]:
        value = record.get(field)
        if field == 'facts' and isinstance(value, str) and value.startswith('['):
            value = json.loads(value)
        if value is None or value == '' or value == []:
            continue
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            value = '\n'.join(value)
        if not isinstance(value, str):
            raise ValueError('原生正文字段类型不符')
        parts.append(field + ':\n' + value)
    statement = '\n\n'.join(parts)
    if not statement.strip():
        raise ValueError('原生记录正文为空，禁止静默丢弃')
    return dict(statement=statement, category='', scope='', attributes={}, conditions=[], evidence=[])


def native_run(gold, meta, records):
    """来源放在提交项元数据里，不借用政策 category/attributes 表达 summary。"""
    run = run_shell(gold, meta, [native_to_draft(r) for r in records])
    for item, record in zip(run['D'], records):
        item['native_source'] = dict(kind=record['kind'], native_id=record['id'])
    return run


def http(base, path, body=None, timeout=30):
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    with urlopen(Request(base + path, data=payload, headers={'Content-Type': 'application/json'}), timeout=timeout) as response:
        return json.load(response)


def database(directory):
    connection = sqlite3.connect((directory / 'store/claude-mem.db').resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def export_records(directory, session_id):
    """停稳后读取全库；不走默认用户目录、不分页漏项、不重复计来源表。"""
    with closing(database(directory)) as connection:
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('实验 SQLite 完整性检查失败')
        tables = ('sdk_sessions', 'observations', 'session_summaries', 'user_prompts', 'committed_observation_tool_uses')
        raw = {name: [dict(r) for r in connection.execute(f'SELECT * FROM {name} ORDER BY rowid')] for name in tables}
        schema = {r['name']: r['sql'] for r in connection.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")}
    sessions = raw['sdk_sessions']
    if len(sessions) != 1 or sessions[0]['content_session_id'] != session_id:
        raise RuntimeError('实验库会话身份不符')
    records = [dict(r, kind=kind) for kind, table in (('observation', 'observations'), ('summary', 'session_summaries'))
               for r in raw[table]]
    return dict(records=records, tables=raw, schema=schema)


def product_metadata(package, bun):
    if read_json(package / 'package.json')['version'] != VERSION:
        raise ValueError('必须本地安装 claude-mem@15.17.0')
    lock = package.parent.parent / 'package-lock.json'
    entry = read_json(lock)['packages']['node_modules/claude-mem']
    if entry['version'] != VERSION or entry['integrity'] != INTEGRITY:
        raise ValueError('固定包 integrity 不符')
    worker = package / 'plugin/scripts/worker-service.cjs'
    return dict(version=VERSION, source_commit=SOURCE_COMMIT, npm_integrity=INTEGRITY,
                source_binary_correspondence='npm 固定发布版本及完整性；未复现源码构建',
                package_lock_sha256=sha256_file(lock),
                worker_sha256=sha256_file(worker),
                bun=subprocess.check_output([str(bun), '--version'], text=True).strip(),
                prompt_asset_sha256={str(p.relative_to(package)): sha256_file(p)
                                     for p in sorted((package / 'plugin/modes').rglob('*.json'))},
                adapter_sha256=sha256_file(Path(__file__)),
                proxy_sha256=sha256_file(Path(__file__).with_name('proxy.py')),
                network_guard_sha256=sha256_file(GUARD))


def child_environment(directory, config, proxy, port):
    """白名单环境与空 HOME；产品只能得到本次代理占位 key。"""
    env = dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', HOME=str(directory / 'home'),
               TMPDIR=str(directory / 'temp'), LANG='en_US.UTF-8', NO_COLOR='1',
               XDG_CONFIG_HOME=str(directory / 'home'), XDG_DATA_HOME=str(directory / 'home'),
               XDG_CACHE_HOME=str(directory / 'home'), CLAUDE_CONFIG_DIR=str(directory / 'home/.claude'),
               DO_NOT_TRACK='1', DISABLE_TELEMETRY='1', CLAUDE_MEM_TELEMETRY='false',
               CLAUDE_MEM_DATA_DIR=str(directory / 'store'), CLAUDE_MEM_ENV_FILE=str(directory / 'empty.env'),
               CLAUDE_MEM_WORKER_PORT=str(port), CLAUDE_MEM_WORKER_HOST='127.0.0.1',
               CLAUDE_MEM_RUNTIME='worker', CLAUDE_MEM_PROVIDER='openai-compatible',
               CLAUDE_MEM_OPENAI_COMPAT_BASE_URL=proxy.url, CLAUDE_MEM_OPENAI_COMPAT_MODEL=config['model'],
               CLAUDE_MEM_OPENAI_COMPAT_API_KEY='local-proxy', CLAUDE_MEM_QUOTA_FALLBACK_PROVIDER='off',
               CLAUDE_MEM_TIER_ROUTING_ENABLED='false', CLAUDE_MEM_CHROMA_ENABLED='false',
               CLAUDE_MEM_SERVER_VECTOR_SEARCH='false', CLAUDE_MEM_OBSERVE_BARE_PROMPTS='false',
               CLAUDE_MEM_TRANSCRIPTS_ENABLED='false', CLAUDE_MEM_FOLDER_CLAUDEMD_ENABLED='false',
               CLAUDE_MEM_AUTO_UPDATE='false', CLAUDE_MEM_FORBID_REAL_AUTO_UPDATE='1',
               CLAUDE_MEM_TELEGRAM_ENABLED='false', CLAUDE_MEM_BACKUP_ENABLED='false',
               CLAUDE_MEM_DELIVERABLES_ENABLED='false', CLAUDE_MEM_MEDIA_CAPTURE_ENABLED='false',
               CLAUDE_MEM_LLM_TIMEOUT_MS=str(math.ceil(proxy.timeout * 1000)),
               CLAUDE_MEM_FIELD_OPTIMIZE_TIMEOUT_MS=str(math.ceil(proxy.timeout * 1000)))
    return env


@contextmanager
def isolated_worker(args, directory, config, proxy):
    for name in ('home', 'temp', 'store', 'work'):
        (directory / name).mkdir()
    (directory / 'empty.env').touch(mode=0o600)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = child_environment(directory, config, proxy, port)
    write_json(directory / 'configuration.json', env)
    # --daemon 是前台 worker 入口；不用 start 的查找、抢占、后台派生逻辑。
    command = [str(args.bun), '--preload', str(GUARD), str(args.package / 'plugin/scripts/worker-service.cjs'), '--daemon']
    process, verified = None, False
    base = f'http://127.0.0.1:{port}'
    try:
        with (directory / 'server.log').open('w') as log:
            process = subprocess.Popen(command, cwd=directory / 'work', env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            write_json(directory / 'process.json', dict(pid=process.pid, port=port, command=command))
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError('实验 worker 启动退出，见独立日志')
                # 先看本次进程的绑定日志，绝不把候选端口上的现役服务当探测目标。
                bound = any('HTTP server started' in p.read_text() and f'pid={process.pid}' in p.read_text()
                            for p in (directory / 'store/logs').glob('*.log'))
                if not bound:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('实验 worker 尚未确认绑定自己的端口')
                    time.sleep(0.1)
                    continue
                try:
                    health = http(base, '/api/health', timeout=1)
                    # 随机端口仍可能发生竞争：首个只读探测必须核对我们启动的 PID。
                    if health.get('pid') != process.pid or health.get('version') != VERSION:
                        raise RuntimeError('端口所有者或版本不符，拒绝连接')
                    verified = True
                    if health.get('initialized'):
                        if health['ai']['provider'] != 'openai-compatible':
                            raise RuntimeError(f"观察者 provider 不符：{health['ai']['provider']}")
                        stats = http(base, '/api/stats')
                        if Path(stats['database']['path']) != directory / 'store/claude-mem.db':
                            raise RuntimeError('数据库隔离失败')
                        if any(stats['database'][k] for k in ('observations', 'sessions', 'summaries')):
                            raise RuntimeError('新 worker 数据库非空')
                        break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('实验 worker 启动超时')
                time.sleep(0.1)
            yield base, process
    finally:
        if process is not None:
            if verified and process.poll() is None:
                try:
                    http(base, '/api/admin/shutdown', {}, timeout=5)
                except (URLError, TimeoutError):
                    process.terminate()
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise RuntimeError('实验 worker 未正常停止，禁止导出半成品')
            write_json(directory / 'exit.json', dict(returncode=process.returncode))
            if process.returncode != 0:
                raise RuntimeError('实验 worker 异常退出，见独立日志')


def check_model(proxy):
    if proxy.errors or any(c.get('status') not in (200, 'pending') for c in proxy.calls):
        raise RuntimeError('模型调用失败，禁止提交半成品')
    if any(c['kind'] == 'embedding' for c in proxy.calls):
        raise RuntimeError('关闭向量后仍发生 embedding 调用')
    # ponytail: 默认 4096 输出上限下被截断是产品默认行为，不判失败；截断次数写进运行记录。


def wait_idle(base, process, proxy, deadline):
    """RAM depth 包含已 claim 的请求，确认为零后才投递下一条，避免过期/溢出。"""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('观察者处理期间退出')
        check_model(proxy)
        status = http(base, '/api/processing-status', timeout=min(5, max(.1, deadline - time.monotonic())))
        health = http(base, '/api/health', timeout=5)
        if status.get('abandonedStallSessions') or status.get('parkedSessions'):
            raise RuntimeError('观察者请求停滞或被停放')
        interaction = health['ai'].get('lastInteraction')
        if interaction and interaction.get('success') is False:
            raise RuntimeError('观察者报告处理失败')
        if (status.get('queueDepth') == 0 and status.get('isProcessing') is False
                and all(c.get('status') != 'pending' for c in proxy.calls)):
            return status
        time.sleep(0.1)
    raise TimeoutError('观察者处理超时')


def generate(sample, args, config, package):
    target = output_path(sample / 'runs' / (SYSTEM + '.json'), sample.parent)
    if target.exists() and not args.force:
        return 'skipped'
    text = (sample / 'transcript.md').read_text(encoding='utf-8')
    identifier(sample.name)
    started = time.monotonic()
    directory = ENVS / 'runs' / (sample.name + '-' + uuid.uuid4().hex)
    directory.mkdir(parents=True, mode=0o700)
    manifest = dict(sample=sample.name, system=SYSTEM, time=now(), status='running', product=package,
                    input_sha256=sha256_text(text), raw_directory=str(directory.relative_to(ROOT)),
                    environment=environment(), timeout_seconds=args.timeout)
    proxy = ModelProxy(config, directory, args.request_timeout)
    try:
        with proxy:
            with isolated_worker(args, directory, config, proxy) as (base, process):
                session_id = 'eval-' + uuid.uuid4().hex
                events, capture = replay_requests(text, session_id, 'sample-' + sample.name, directory / 'work')
                write_json(directory / 'replay.json', dict(events=events, capture=capture))
                deadline, acknowledgements = time.monotonic() + args.timeout, []
                # 产品自身的跳过、丢弃、截断属于默认行为：只计数写进运行记录，不判失败。
                anomalies = Counter()
                for index, event in enumerate(events):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('会话回放总时限已过')
                    ack = http(base, event['path'], event['body'], timeout=min(30, remaining))
                    acknowledgements.append(dict(index=index, response=ack))
                    write_json(directory / 'acknowledgements.json', acknowledgements)
                    if event['path'].endswith('/init'):
                        if ack.get('skipped'):
                            anomalies['init_skipped'] += 1
                        elif not ack.get('sessionDbId'):
                            raise RuntimeError('init 未返回会话')
                    elif event['path'].endswith('/session-end'):
                        if ack.get('status') != 'accepted':
                            raise RuntimeError('session-end 未接收')
                    elif ack.get('status') == 'skipped':
                        # 原生过滤是产品行为，单独计量，不能假装有模型调用。
                        if ack.get('reason') not in {'tool_excluded', 'memory_retrieval', 'session_memory_meta', 'private'}:
                            anomalies['event_skipped:' + str(ack.get('reason'))] += 1
                    elif ack.get('status') != 'queued':
                        raise RuntimeError('事件未排队')
                    wait_idle(base, process, proxy, deadline)
                check_model(proxy)
            exported = export_records(directory, session_id)
            write_json(directory / 'export.json', exported)
            records = exported['records']
            # 每个 queued 工具必须有原生消费凭据，即使观察者决定不存记忆。
            committed = {r['tool_use_id'] for r in exported['tables']['committed_observation_tool_uses']}
            queued = {e['body']['tool_use_id'] for e, a in zip(events, acknowledgements)
                      if e['path'].endswith('/observations') and a['response'].get('status') == 'queued'}
            anomalies['tool_events_not_committed'] += len(queued - committed)
            logs = '\n'.join(p.read_text() for p in (directory / 'store/logs').glob('*.log'))
            if 'EVAL_NETWORK_DENIED' in logs:
                raise RuntimeError('产品尝试非预期联网')
            for marker in ('batch was dropped', 'reply was cut off', 'SKIPPED_DUPLICATE_OR_LIMIT', 'expired live'):
                anomalies['log:' + marker] += logs.count(marker)
            if 'EVAL_NETWORK_DENIED' in (directory / 'server.log').read_text():
                raise RuntimeError('产品尝试非回环联网')
            if not proxy.calls:
                raise RuntimeError('未出现观察者调用，禁止伪造成功')
            if any(c.get('status') != 200 for c in proxy.calls):
                raise RuntimeError('停机后仍有未完成模型请求')
        values = [native_to_draft(r) for r in records]
        capture['acknowledgements'] = dict(Counter(a['response'].get('reason') or a['response'].get('status', 'init') for a in acknowledgements))
        prompt_hashes = {str(c['index']): sha256_text(json.dumps(read_json(ROOT / c['raw_path'])['request']['messages'],
                         ensure_ascii=False, sort_keys=True)) for c in proxy.calls}
        meta = dict(name=SYSTEM, version=VERSION, configuration='openai-compatible-novec', product=package,
                    deployment='npm 固定版 / 每样本独立 HOME、数据目录、进程及端口', source_commit=SOURCE_COMMIT,
                    llm=config['model'], base_url=config['base_url'],
                    parameters=proxy.parameters(),
                    embedding_model=None, embedding=dict(enabled=False, reason='embedding 模型不可指定（chroma-mcp 固定 text-embedding-ada-002），无法统一为评测指定的 embedding，关闭向量'),
                    prompt_version='claude-mem v15.17.0 原生 code 模式', prompt_sha256=prompt_hashes,
                    input_sha256=sha256_text(text), output_sha256=sha256_text(json.dumps(values, ensure_ascii=False, sort_keys=True)),
                    raw_directory=str(directory.relative_to(ROOT)), raw_export_sha256=sha256_file(directory / 'export.json'),
                    usage=proxy.summary(), truncated_calls=sum('length' in c.get('finish_reasons', []) for c in proxy.calls),
                    product_anomalies={k: v for k, v in anomalies.items() if v},
                    calls=proxy.calls, proxy_errors=proxy.errors,
                    exported_sources=[dict(index=i, kind=r['kind'], native_id=r['id']) for i, r in enumerate(records)],
                    time=now(), elapsed_seconds=round(time.monotonic() - started, 3), environment=environment(),
                    context=dict(window='未记录', input_characters=len(text), capture=capture,
                                 product_input_truncated='原生隐私过滤/字段压缩/上下文裁剪；详见逐调用原文，精确截断量未记录', output_truncated=False),
                    isolation=dict(chroma=False, vector_search=False, bare_prompts=False, tier_routing=False,
                                   fallback=False, cloud_sync=False, transcript_watcher=False, auto_update=False),
                    timeout=dict(seconds=args.timeout, occurred=False), retries='适配器不重试；产品内部重试见调用审计',
                    dataset_manifest='见样本来源；版本未记录')
        # 进程退出后才读 gold 的运行外壳，金标内容不交给被测产品。
        write_json(target, native_run(read_json(sample / 'gold.json'), meta, records), overwrite=args.force)
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
        manifest.update(elapsed_seconds=round(time.monotonic() - started, 3), usage=proxy.summary(), parameters=proxy.parameters(),
                        calls=proxy.calls, proxy_errors=proxy.errors)
        write_json(directory / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    parser.add_argument('--bun', type=Path, default=Path(shutil.which('bun') or 'bun'))
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--request-timeout', type=float, default=180)
    parser.add_argument('--startup-timeout', type=float, default=60)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--failures', type=Path, default=TMP / 'trial/claude-mem-failures.json')
    parser.add_argument('--concurrency', type=int, default=1, help='同时处理的样本数；每个样本独立进程、端口和存储')
    args = parser.parse_args()
    try:
        if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.request_timeout, args.startup_timeout)):
            raise ValueError('超时必须为有限正数')
        args.package, args.bun = args.package.resolve(strict=True), args.bun.resolve(strict=True)
        config = settings()
        package = product_metadata(args.package, args.bun)
        return batch(sorted(p.parent for p in args.data_dir.glob('*/gold.json')),
                     lambda p: generate(p, args, config, package), args.failures, concurrency=args.concurrency,
                     max_consecutive_failures=10)  # ponytail: 配对失败的样本会连续秒失败，阈值 3 会误熔断
    except (ValueError, OSError, subprocess.SubprocessError):
        parser.exit(2, '错误：配置或固定版本隔离环境无效，请核查 .env、npm 安装和 Bun\n')


if __name__ == '__main__':
    raise SystemExit(main())
