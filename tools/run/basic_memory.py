"""Basic Memory 1.0.0 + 固定 agent + 官方 memory-capture 原文。

安装：UV_CACHE_DIR=$PWD/tmp/envs/.uv-cache uv venv tmp/envs/basic-memory --python 3.12
      UV_CACHE_DIR=$PWD/tmp/envs/.uv-cache uv pip install --python tmp/envs/basic-memory/bin/python basic-memory==1.0.0
运行：python3 -m tools.run.basic_memory --data-dir tmp/trial/e2e/trace/data
每样本一个 stdio MCP 进程；历史会话只作为数据，历史工具不执行。
"""
import argparse
from inspect import getsource
from contextlib import contextmanager
import json
import math
from pathlib import Path
import queue
import re
import sqlite3
import subprocess
import threading
import time
from urllib.request import Request, urlopen
import uuid

from tools.config import sha256_file, environment, sha256_text
from tools.paths import ROOT, TMP
from tools.run.proxy import ModelProxy, settings
from tools.trial import batch, now, output_path, read_json, run_shell, transcript, write_json

VERSION = '1.0.0'
SYSTEM = 'basic-memory-1.0.0-agent-capture-defaults-qwen'
SOURCE_COMMIT = 'c9bd187b83baba549d5cd16bacc180f8023d51c8'
ENVS = TMP / 'envs/basic-memory'
ASSETS = Path(__file__).with_suffix('')
# 官方 memory-capture 提示词（AGPL-3.0-or-later）不随本仓库分发，运行时从已安装的固定版本包中读取。
PROMPT = next(iter(sorted(ENVS.glob('lib/python*/site-packages/basic_memory/data/skills/memory-capture/SKILL.md'))),
              ENVS / 'lib/python3/site-packages/basic_memory/data/skills/memory-capture/SKILL.md')
PROMPT_SHA256 = 'db63802b9b6cbf42f02c14b5384d3096e2fc99ecb24e2f7697813611c6b60004'
# 官方提示要求查找同一 thread 的笔记，因此保留必要的 bm_search 读取。
ALLOWED = {'bm_write', 'bm_edit', 'bm_read', 'bm_search', 'bm_list'}


class BudgetExceeded(RuntimeError):
    """预算耗尽是未完成，不能把已写的半成品提交为成功。"""


def native_to_draft(record):
    """整篇正文是一项；不重复导出 observations，不从标题/技术 project 猜政策字段。"""
    body = record['body']
    if not isinstance(body, str) or not body.strip():
        raise ValueError('原生笔记正文为空或类型不符')
    return dict(statement=body, category='', scope='', attributes={}, conditions=[], evidence=[])


def note_body(text):
    """只机械移除开头的 YAML frontmatter，正文逐字保留（包括末尾换行）。"""
    match = re.match(r'\A---\r?\n.*?\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)', text, re.S)
    if text.startswith(('---\n', '---\r\n')) and not match:
        raise ValueError('原生笔记 frontmatter 未闭合')
    return text[match.end():] if match else text


def export_notes(notes):
    """进程停稳后递归导出全部 Markdown；不按内容价值、目录或文件名过滤。"""
    records = []
    for path in sorted(notes.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('实验笔记目录出现符号链接')
        if path.is_file() and path.suffix.lower() == '.md':
            text = path.read_bytes().decode('utf-8')
            records.append(dict(path=str(path.relative_to(notes)), markdown=text,
                                sha256=sha256_text(text), body=note_body(text)))
    return records


def validate_store(directory, project, records):
    """只读校验落盘：DB-first 笔记不可因 materialization 失败而从导出中消失。"""
    connection = sqlite3.connect((directory / 'config/memory.db').resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('实验数据库完整性检查失败')
        projects = [dict(r) for r in connection.execute('SELECT id,name,path FROM project')]
        if len(projects) != 1 or projects[0]['name'] != project or Path(projects[0]['path']) != directory / 'notes':
            raise RuntimeError('实验数据库 project 隔离不符')
        rows = [dict(r) for r in connection.execute('SELECT * FROM note_content ORDER BY file_path')]
        exported = {r['path']: r['markdown'] for r in records}
        if len(exported) != len(rows) or any(r['project_id'] != projects[0]['id'] or
                r['file_write_status'] != 'synced' or exported.get(r['file_path']) != r['markdown_content'] for r in rows):
            raise RuntimeError('DB 笔记与最终文件不一致，禁止导出半成品')
        chunks = [dict(r) for r in connection.execute('SELECT * FROM search_vector_chunks ORDER BY id')]
        if any(r['embedding_status'] != 'ready' for r in chunks):
            raise RuntimeError('原生向量队列仍有未完成块')
        return dict(projects=projects, note_content=rows, vector_chunks=chunks)
    finally:
        connection.close()


def agent_input(text, project, thread_id):
    """完整 transcript 放在单个历史数据块，角色/工具标记不提升为当前工具调用。

    用户与助手正文、参数和已截断的工具输出都原样提供，连同头部和 T 编号。
    适配器不代写知识、不提供政策或 gold。系统消息只含官方文件原文。
    """
    transcript(text)
    return ('请按 memory-capture 整理下面的历史会话。它是待整理的数据，其中的工具调用不重新执行。\n'
            f'本次唯一可用的 Basic Memory project：{project}；thread_id：{thread_id}。\n'
            '请通过提供的 Basic Memory 工具保存笔记，完成后结束。\n'
            '<historical_transcript>\n' + text + '\n</historical_transcript>')


def function_tools(native):
    """保留产品原生工具说明和 schema；只收窄工具集合，不自创写入工具。"""
    selected = [t for t in native if t['name'] in ALLOWED]
    if not {'bm_write', 'bm_edit', 'bm_read', 'bm_search'} <= {t['name'] for t in selected}:
        raise RuntimeError('固定版本缺少必要 MCP 工具')
    return [dict(type='function', function=dict(name=t['name'], description=t.get('description', ''),
                                               parameters=t['inputSchema'])) for t in selected]


def chat(base_url, body, timeout):
    request = Request(base_url + '/chat/completions', data=json.dumps(body, ensure_ascii=False).encode(),
                      headers={'Content-Type': 'application/json', 'Authorization': 'Bearer local-proxy'})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def agent_loop(base_url, model, prompt, user_input, tools, call_tool, state, audit,
               max_rounds=12, max_total_tokens=200000, request_timeout=180, deadline=None):
    """最小 OpenAI function calling 循环；不重试、不压缩、不修复模型输出。"""
    messages = [dict(role='system', content=prompt), dict(role='user', content=user_input)]
    names = {t['function']['name'] for t in tools}
    state.update(rounds=0, tool_calls=[], total_tokens=0, stop_reason='running')
    try:
        for turn in range(1, max_rounds + 1):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('agent 总时限已过')
            state['rounds'] = turn
            remaining = request_timeout if deadline is None else min(request_timeout, deadline - time.monotonic())
            # 不传生成参数，沿用服务端默认。
            response = chat(base_url, dict(model=model, messages=messages, tools=tools,
                                          tool_choice='auto', stream=False), remaining)
            write_json(audit / f'agent-response-{turn:03d}.json', response)
            usage = response.get('usage', {}).get('total_tokens')
            if not isinstance(usage, int) or usage < 0:
                raise RuntimeError('缺少 token 用量，无法执行累计预算')
            state['total_tokens'] += usage
            if state['total_tokens'] > max_total_tokens:
                state['stop_reason'] = 'max_total_tokens'
                raise BudgetExceeded('agent 达到累计 token 上限')
            choices = response.get('choices', [])
            if len(choices) != 1:
                raise RuntimeError('模型返回 choices 数量不符')
            choice = choices[0]
            if choice.get('finish_reason') == 'length':
                state['stop_reason'] = 'max_output_tokens'
                raise BudgetExceeded('模型输出被服务端上限截断')
            message = choice['message']
            # 只回送 OpenAI 对话字段；厂商辅助字段留在原始返回里。
            assistant = {k: message[k] for k in ('role', 'content', 'tool_calls') if k in message}
            if assistant.get('role') != 'assistant':
                raise RuntimeError('模型角色不符')
            messages.append(assistant)
            calls = message.get('tool_calls') or []
            if not calls:
                if choice.get('finish_reason') != 'stop':
                    raise RuntimeError('agent 未正常结束')
                state['stop_reason'] = 'completed'
                return state
            if choice.get('finish_reason') != 'tool_calls':
                raise RuntimeError('模型工具调用结束标记不符')
            for call in calls:
                name = call['function']['name']
                if name not in names or name not in ALLOWED:
                    raise RuntimeError('模型请求未开放工具')
                arguments = json.loads(call['function']['arguments'])
                if not isinstance(arguments, dict):
                    raise RuntimeError('工具参数不是对象')
                record = dict(round=turn, name=name, id=call['id'], status='pending')
                state['tool_calls'].append(record)
                result = call_tool(name, arguments)
                record['status'] = 'error' if result.get('isError') else 'ok'
                write_json(audit / f'tool-{len(state["tool_calls"]):03d}.json',
                           dict(**record, arguments=arguments, result=result))
                # 原生工具拒绝原样交还 agent，可在预算内读取并修正；不伪装成功。
                messages.append(dict(role='tool', tool_call_id=call['id'],
                                     content=json.dumps(result, ensure_ascii=False)))
        state['stop_reason'] = 'max_rounds'
        raise BudgetExceeded('agent 达到最大轮数')
    finally:
        if state['stop_reason'] == 'running':
            state['stop_reason'] = 'failed'
        write_json(audit / 'agent.json', dict(**state, tool_call_count=len(state['tool_calls']), messages=messages))


class StdioMCP:
    """单样本 stdio JSON-RPC；读取线程只负责排队，每次调用有明确截止时间。"""
    def __init__(self, process, directory, deadline):
        self.process, self.directory, self.deadline = process, directory, deadline
        self.sequence, self.responses = 0, queue.Queue()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(json.loads(line))
        except Exception:
            self.responses.put({'reader_error': True})
        finally:
            self.responses.put(None)

    def send(self, method, params=None, request_id=None):
        value = dict(jsonrpc='2.0', method=method)
        if params is not None:
            value['params'] = params
        if request_id is not None:
            value['id'] = request_id
        self.process.stdin.write(json.dumps(value, ensure_ascii=False) + '\n')
        self.process.stdin.flush()

    def request(self, method, params=None):
        self.sequence += 1
        sequence = self.sequence
        self.send(method, params, sequence)
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('MCP 总时限已过')
            try:
                value = self.responses.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError('MCP 响应超时') from None
            if value is None or value.get('reader_error'):
                raise RuntimeError('MCP 进程退出或 stdio 非 JSON')
            if value.get('id') == sequence:
                write_json(self.directory / f'mcp-{sequence:04d}.json', dict(method=method, params=params, response=value))
                if 'error' in value:
                    raise RuntimeError('MCP 协议返回错误')
                return value['result']


def child_environment(directory):
    """白名单；不继承用户 BM、MCP、云端地址、凭据或模型环境。"""
    return dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', HOME=str(directory / 'home'),
                TMPDIR=str(directory / 'temp'), LANG='en_US.UTF-8', PYTHONUNBUFFERED='1',
                XDG_CONFIG_HOME=str(directory / 'home'), XDG_DATA_HOME=str(directory / 'home'),
                XDG_CACHE_HOME=str(directory / 'home'), BASIC_MEMORY_CONFIG_DIR=str(directory / 'config'),
                BASIC_MEMORY_HOME=str(directory / 'notes'), PYTHONPATH=str(ASSETS / 'network'),
                OPENAI_API_KEY='local-proxy', HF_HUB_OFFLINE='1', DO_NOT_TRACK='1')


def product_metadata(python):
    code = 'from pathlib import Path\nimport hashlib\n' + getsource(sha256_file) + '''
import importlib.metadata as m,json,pathlib,sys,platform
p=m.distribution('basic-memory')
print(json.dumps(dict(version=p.version,python=sys.version,platform=platform.platform(),
 dependencies=sorted(d.metadata['Name']+'=='+d.version for d in m.distributions()),
 installed_sha256={str(f):sha256_file(p.locate_file(f))
 for f in p.files if str(f).startswith('basic_memory/') and str(f).endswith('.py')})))'''
    result = subprocess.run([str(python), '-I', '-c', code], capture_output=True, text=True, timeout=30, check=True)
    package = json.loads(result.stdout)
    if package['version'] != VERSION or sha256_file(PROMPT) != PROMPT_SHA256:
        raise ValueError('必须使用 Basic Memory 1.0.0 与固定版官方提示词原文')
    package.update(source_commit=SOURCE_COMMIT, source_correspondence='PyPI 固定版；未复现源码构建',
                   published_wheel_sha256='a2d2c1589616c9599aec9edd6597d508028c5af88632312fc0990dcaccf02867',
                   prompt_source=f'https://github.com/basicmachines-co/basic-memory/blob/{SOURCE_COMMIT}/skills/memory-capture/SKILL.md',
                   prompt_source_path='skills/memory-capture/SKILL.md', prompt_sha256=PROMPT_SHA256,
                   prompt_license='AGPL-3.0-or-later；运行时从已安装的固定版本包读取，不随本仓库分发',
                   adapter_sha256=sha256_file(Path(__file__)),
                   proxy_sha256=sha256_file(Path(__file__).with_name('proxy.py')),
                   network_guard_sha256=sha256_file(ASSETS / 'network/sitecustomize.py'))
    return package


@contextmanager
def isolated_mcp(args, directory, config, proxy, project):
    for name in ('home', 'temp', 'config', 'notes'):
        (directory / name).mkdir()
    # v1.0.0 写入会异步生成向量；不能靠关闭 watcher 假装停用了 embedding。
    conf = dict(projects={project: dict(path=str(directory / 'notes'), mode='local')}, default_project=project,
                auto_update=False, logfire_enabled=False, logfire_send_to_logfire=False,
                cloud_promo_opt_out=True, index_changes=False, reranker_enabled=False,
                semantic_embedding_provider='openai', semantic_embedding_model=config['embedding_model'],
                semantic_embedding_api_base=proxy.url, semantic_embedding_dimensions=config['dimensions'],
                semantic_embedding_batch_size=20, default_search_type='text')
    write_json(directory / 'config/config.json', conf)
    command = [str(args.python.with_name('basic-memory')), 'mcp', '--project', project]
    with (directory / 'server.log').open('w') as log:
        process = subprocess.Popen(command, cwd=directory, env=child_environment(directory), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, start_new_session=True)
        write_json(directory / 'process.json', dict(pid=process.pid, command=command))
        client = StdioMCP(process, directory, time.monotonic() + args.timeout)
        clean = False
        try:
            client.deadline = time.monotonic() + args.startup_timeout
            init = client.request('initialize', dict(protocolVersion='2025-03-26', capabilities={},
                                  clientInfo=dict(name=SYSTEM, version='1')))
            client.send('notifications/initialized')
            client.deadline = time.monotonic() + args.timeout
            if export_notes(directory / 'notes'):
                raise RuntimeError('新项目不是空库')
            native, cursor = [], None
            while True:
                page = client.request('tools/list', {'cursor': cursor} if cursor else {})
                native.extend(page['tools'])
                cursor = page.get('nextCursor')
                if not cursor:
                    break
            write_json(directory / 'mcp-tools.json', dict(initialize=init, tools=native))

            def call_tool(name, arguments):
                if name not in ALLOWED:
                    raise RuntimeError('禁止未开放工具')
                if arguments.get('project', project) != project or arguments.get('all_projects') or arguments.get('projects'):
                    raise RuntimeError('工具试图访问其他 project')
                arguments = dict(arguments, project=project)
                return client.request('tools/call', dict(name=name, arguments=arguments))

            yield function_tools(native), call_tool, client.deadline
            clean = True
        finally:
            # 固定版原生 lifespan 在 EOF 后先 drain materialization 和向量后台任务。
            process.stdin.close()
            try:
                process.wait(timeout=max(.1, min(args.request_timeout + 30, client.deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                if clean:
                    raise TimeoutError('MCP 原生后台队列未在时限内停稳')
            finally:
                process.stdout.close()
                client.thread.join(timeout=2)
                write_json(directory / 'exit.json', dict(returncode=process.returncode))
            if clean and process.returncode != 0:
                raise RuntimeError('Basic Memory 非正常退出')


def generate(sample, args, config, package):
    target = output_path(sample / 'runs' / (SYSTEM + '.json'), sample.parent)
    if target.exists() and not args.force:
        return 'skipped'
    text = (sample / 'transcript.md').read_text(encoding='utf-8')
    started = time.monotonic()
    directory = ENVS / 'runs' / (sample.name + '-' + uuid.uuid4().hex)
    directory.mkdir(parents=True, mode=0o700)
    project, state = 'eval-' + uuid.uuid4().hex, {}
    manifest = dict(sample=sample.name, system=SYSTEM, time=now(), status='running', product=package,
                    input_sha256=sha256_text(text), raw_directory=str(directory.relative_to(ROOT)), environment=environment())
    proxy = ModelProxy(config, directory, args.request_timeout)
    try:
        with proxy:
            with isolated_mcp(args, directory, config, proxy, project) as (tools, call_tool, deadline):
                agent_loop(proxy.url, config['model'], PROMPT.read_text(), agent_input(text, project, sample.name),
                           tools, call_tool, state, directory, args.max_rounds, args.max_total_tokens,
                           args.request_timeout, deadline)
            records = export_notes(directory / 'notes')
            store = validate_store(directory, project, records)
            write_json(directory / 'export.json', dict(records=records, store=store, granularity='整篇笔记正文'))
            if proxy.errors or any(c.get('status') != 200 for c in proxy.calls):
                raise RuntimeError('模型调用失败或未完成')
            if not any(c['kind'] == 'llm' for c in proxy.calls):
                raise RuntimeError('缺少 agent 模型调用')
            if not records and any(c['status'] == 'error' for c in state['tool_calls']):
                raise RuntimeError('工具失败后无原生笔记，不能当作有效空结果')
            logs = '\n'.join(p.read_text() for p in directory.rglob('*.log'))
            if any(s in logs for s in ('EVAL_NETWORK_DENIED', 'Background task failed', 'ERROR')):
                raise RuntimeError('产品日志报告后台失败或非预期联网')
        values = [native_to_draft(r) for r in records]
        meta = dict(name=SYSTEM, version=VERSION, configuration='固定 agent + 官方 memory-capture 原文',
                    source_commit=SOURCE_COMMIT, product=package, llm=config['model'], base_url=config['base_url'],
                    parameters=proxy.parameters(),
                    embedding_model=config['embedding_model'], embedding=dict(enabled=True, dimensions=config['dimensions'],
                        reason='v1.0.0 原生写入异步建立向量；使用指定模型', normalization='产品 OpenAI embedder 原生归一化',
                        chunking='产品原生默认；见请求审计', batch_size=20),
                    prompt_version='Basic Memory v1.0.0 skills/memory-capture/SKILL.md', prompt_sha256=PROMPT_SHA256,
                    deployment='venv / 每样本空 HOME、独立 config/SQLite/notes 与 stdio 进程',
                    input_sha256=sha256_text(text), output_sha256=sha256_text(json.dumps(values, ensure_ascii=False, sort_keys=True)),
                    raw_directory=str(directory.relative_to(ROOT)), raw_export_sha256=sha256_file(directory / 'export.json'),
                    usage=proxy.summary(), calls=proxy.calls, agent=dict(**state, tool_call_count=len(state['tool_calls']),
                        max_rounds=args.max_rounds, max_total_tokens=args.max_total_tokens),
                    context=dict(window='未记录', input_characters=len(text), adapter_input_truncated=False,
                        source_truncation_markers=text.count('截断'), product_input_truncated=False, output_truncated=False),
                    isolation=dict(project=project, cloud=False, auto_update=False, watcher=False, network='仅回环'),
                    timeout=dict(seconds=args.timeout, occurred=False), retries='agent 不重试；代理限流重试见调用审计',
                    dataset_manifest='见样本来源；版本未记录', time=now(), environment=environment(),
                    elapsed_seconds=round(time.monotonic() - started, 3))
        # 被测进程已停稳；gold 只用于输出外壳，U/N 从未进入 agent 或产品。
        run = run_shell(read_json(sample / 'gold.json'), meta, values)
        for item, record in zip(run['D'], records):
            item['native_source'] = dict(kind='note', path=record['path'], sha256=record['sha256'])
        write_json(target, run, overwrite=args.force)
        manifest.update(status='written', output=str(target.relative_to(ROOT)), exported=len(values),
                        run_sha256=sha256_file(target))
        print(f"{sample.name}：{meta['elapsed_seconds']:.1f}s；LLM={proxy.summary()['llm']['calls']}；"
              f"embedding={proxy.summary()['embedding']['calls']}；导出={len(values)}；"
              f"轮数={state['rounds']}；工具={len(state['tool_calls'])}", flush=True)
        return 'written'
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, timed_out=isinstance(exc, TimeoutError),
                        detail=str(exc) if isinstance(exc, RuntimeError) else None)
        raise
    finally:
        manifest.update(elapsed_seconds=round(time.monotonic() - started, 3), usage=proxy.summary(), parameters=proxy.parameters(),
                        calls=proxy.calls, proxy_errors=proxy.errors, agent=state)
        write_json(directory / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=ENVS / 'bin/python')
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--request-timeout', type=float, default=180)
    parser.add_argument('--startup-timeout', type=float, default=90)
    parser.add_argument('--max-rounds', type=int, default=12)
    parser.add_argument('--max-total-tokens', type=int, default=200000)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--failures', type=Path, default=TMP / 'trial/basic-memory-failures.json')
    parser.add_argument('--concurrency', type=int, default=1, help='同时处理的样本数；每个样本独立进程、端口和存储')
    args = parser.parse_args()
    try:
        if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.request_timeout, args.startup_timeout)) or min(args.max_rounds, args.max_total_tokens) <= 0:
            raise ValueError('时限、轮数和 token 预算必须为有限正数')
        # 保留 venv 路径；resolve() 会跟随 python 符号链接逃出隔离环境。
        args.python = args.python.absolute()
        config, package = settings(), product_metadata(args.python)
        return batch(sorted(p.parent for p in args.data_dir.glob('*/gold.json')),
                     lambda p: generate(p, args, config, package), args.failures, concurrency=args.concurrency)
    except (ValueError, OSError, subprocess.SubprocessError):
        parser.exit(2, '错误：配置或固定版隔离环境无效，请核查 .env、官方提示词和本地安装\n')


if __name__ == '__main__':
    raise SystemExit(main())
