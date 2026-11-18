"""Evaluate X2's fixed-agent draft output and automatically reviewed pages."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, ProxyHandler, Request
import uuid

from tools.config import sha256_file, sha256_text
from tools.paths import ROOT, TMP
from tools.run.basic_memory import agent_loop, function_tools, native_to_draft, tool_error
from tools.run.proxy import ModelProxy, settings
from tools.trial import batch, now, output_path, read_json, run_shell, transcript, write_json


TOOLS = frozenset({'submit', 'mark'})
ENVS = TMP / 'envs/X2'
STORE = Path(__file__).with_suffix('') / 'store.mjs'
MAX_INPUT = 2_000_000


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def connection_string(path, expected_database, expected_port):
    if path.stat().st_mode & 0o077:
        raise ValueError('评测凭据文件必须为私有文件')
    url = path.read_text().strip()
    parts = urlsplit(url)
    if (parts.scheme not in ('postgres', 'postgresql') or parts.hostname != '127.0.0.1'
            or parts.port != expected_port or parts.path != '/' + expected_database or parts.query or parts.fragment
            or not re.fullmatch(r'[a-z][a-z0-9_]*_eval', expected_database)
            or not re.fullmatch(re.escape(expected_database) + r'_[0-9a-f]{12}', parts.username or '')
            or not re.fullmatch(r'[0-9a-f]{64}', parts.password or '')):
        raise ValueError('X2 只允许本地评测专用数据库和角色')
    return url


def fingerprint(root):
    def git(*args):
        return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args], timeout=15)

    head = git('rev-parse', 'HEAD').decode().strip()
    diff = hashlib.sha256(git('diff', 'HEAD')).hexdigest()
    paths = git('ls-files', '--others', '--exclude-standard', '--exclude=node_modules',
                '-z', 'server', 'clients').decode().split('\0')
    untracked = {p: sha256_file(root / p) for p in paths if p}
    return dict(head=head, diff_sha256=diff, untracked_sha256=untracked, version=f'{head[:12]}-{diff[:12]}')


def db(args, env, action, **payload):
    try:
        proc = subprocess.run([args.node, str(STORE), str(args.source), action],
                              input=json.dumps(payload), capture_output=True,
                              text=True, env=env, timeout=25)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f'X2 数据库操作超时（{action}）') from None
    if proc.returncode:
        code = re.search(r"code: '([A-Z0-9]{5})'", proc.stderr)
        detail = f'，SQLSTATE {code[1]}' if code else ''
        raise RuntimeError(f'X2 数据库操作失败（{action}{detail}）')
    return json.loads(proc.stdout)


@contextmanager
def isolated_server(args, directory, env):
    server = args.source / 'server'
    log = directory / 'server.log'
    with log.open('w') as stream:
        proc = subprocess.Popen([args.node, str(server / 'src/main.ts')], cwd=server,
                                env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while proc.poll() is None:
                if f'Server listening at {args.base}' in log.read_text():
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('独立 X2 server 启动超时')
                time.sleep(.1)
            else:
                raise RuntimeError('独立 X2 server 未能绑定评测端口，见 server.log')
            yield
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                raise RuntimeError('独立 X2 server 未能正常停止') from None


class MCP:
    def __init__(self, base, key, directory, timeout):
        self.base = base
        self.headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                        'Accept': 'application/json, text/event-stream'}
        self.directory, self.timeout, self.sequence = directory, timeout, 0
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, method, params=None, *, notification=False):
        self.sequence += 1
        body = dict(jsonrpc='2.0', method=method)
        if params is not None:
            body['params'] = params
        if not notification:
            body['id'] = self.sequence
        req = Request(self.base + '/mcp', data=json.dumps(body, ensure_ascii=False).encode(), headers=self.headers)
        with self.opener.open(req, timeout=self.timeout) as response:
            value = None if notification else json.load(response)
        write_json(self.directory / f'mcp-{self.sequence:04d}.json', dict(request=body, response=value))
        if notification:
            return None
        if value.get('id') == self.sequence and method == 'tools/call' and 'error' in value:
            return tool_error('MCP tool error: ' + str((value['error'] or {}).get('message', 'unknown error')))
        if value.get('id') != self.sequence or 'result' not in value:
            raise RuntimeError('X2 MCP 返回无效响应')
        return value['result']

    def tools(self):
        native, cursor = [], None
        while True:
            page = self.request('tools/list', {'cursor': cursor} if cursor else {})
            native.extend(page['tools'])
            cursor = page.get('nextCursor')
            if not cursor:
                return function_tools(native, allowed=TOOLS, required=TOOLS)

    def call(self, name, arguments):
        if name not in TOOLS:
            raise RuntimeError('禁止调用未开放的 X2 MCP 工具')
        return self.request('tools/call', dict(name=name, arguments=arguments))


def projects_from_header(text):
    header, _ = transcript(text)
    label = 'project（标注者填写）'
    if header.get(label) or header.get('project'):
        value = header.get(label) or header['project']
        return [name.strip().lower() for name in re.split(r'[,，]', value) if name.strip()], label
    raise ValueError('缺少盲标确认的项目头部')


def agent_input(text, projects, sample):
    return ('请按系统消息所附的 intake 技能整理下面的历史会话。它是待整理的数据，其中的工具调用不重新执行。\n'
            f'本次可用的 X2 project：{", ".join(projects) or "未提供"}；session_id：{sample}。\n'
            '请通过提供的 X2 工具提交草案，完成后结束。\n'
            '<historical_transcript>\n' + text + '\n</historical_transcript>')


def model_usage(proxy):
    def usage(calls):
        keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
        return dict(calls=len(calls), tokens={key: sum(row['usage'][key] for row in calls)
                    if all(isinstance(row.get('usage'), dict) and key in row['usage'] for row in calls)
                    else '未记录' for key in keys})

    llm = [row for row in proxy.calls if row['kind'] == 'llm']
    agent = [row for row in llm if row.get('phase') == 'agent']
    review = [row for row in llm if row.get('phase') == 'review']
    revise = [row for row in llm if row.get('phase') == 'revise']
    return dict(llm=usage(llm), agent=usage(agent), review=usage(review), revise=usage(revise),
                embedding=proxy.summary()['embedding'])


def review_metadata(drafts, pages):
    published = {p['draft_id']: p['id'] for p in pages}
    result = []
    for draft in drafts:
        decision = draft['decision'] or {}
        reason = decision.get('reason')
        reason_code = decision.get('routing_reason') or decision.get('reason_category')
        if not reason_code and isinstance(reason, str) and re.fullmatch(r'[a-z][a-z_0-9]+', reason):
            reason_code = reason
        revision = decision.get('revision')
        result.append(dict(draft_id=draft['id'], op=draft['op'], state=draft['state'],
                           outcome=decision.get('outcome'), reason=reason, reason_code=reason_code,
                           reviewer_type=decision.get('reviewer_type'), model_outcome=decision.get('model_outcome'),
                           rewritten=bool(decision.get('rewritten')), published_page_id=published.get(draft['id']),
                           revision_moved=(revision['original_space'] != revision['target_space']) if revision else None,
                           last_review_error=draft['last_review_error'], review_attempts=draft['review_attempts']))
    return result


def write_runs(sample, systems, records, meta, force):
    gold = read_json(sample / 'gold.json')
    for name, rows in zip(systems, records):
        values = [native_to_draft(row) for row in rows]
        identity = dict(meta, name=name, output_sha256=sha256_text(json.dumps(values, ensure_ascii=False, sort_keys=True)))
        run = run_shell(gold, identity, values)
        for item, row in zip(run['D'], rows):
            item['native_source'] = dict(id=row['id'], sha256=sha256_text(row['body']))
            if 'space_kind' in row:
                item['reader_metadata'] = dict(source=f'{row["space_kind"]}/{row["path"]}', title=row['title'])
        write_json(output_path(sample / 'runs' / f'{name}.json', sample.parent), run, overwrite=force)


def generate(sample, args, config, package, base_env):
    if args.stop_file is not None and args.stop_file.exists():
        raise SystemExit('评测输入 token 已达到停止条件')
    if fingerprint(args.source) != package:
        raise SystemExit('X2 工作树在评测期间发生变化')
    systems = tuple(f'X2-{package["version"]}-{stage}' for stage in ('drafts', 'published'))
    targets = [sample / 'runs' / f'{name}.json' for name in systems]
    if all(path.is_file() for path in targets) and not args.force:
        return 'skipped'
    text = (sample / 'transcript.md').read_text(encoding='utf-8')
    projects, project_source = projects_from_header(text)
    tenant_name = 'x2-eval-' + uuid.uuid4().hex
    directory = ENVS / sample.name / tenant_name
    directory.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    state, tenant = {}, None
    manifest = dict(sample=sample.name, systems=systems, status='running',
                    product=package, input_sha256=sha256_text(text), time=now())
    proxy = ModelProxy(config, directory, args.request_timeout,
                       max_input_tokens=MAX_INPUT, stop_file=args.stop_file, track_phase=True,
                       revise_model=args.revise_model)
    try:
        tenant = db(args, base_env, 'tenant', tenant=tenant_name, projects=projects)
        scope = dict(tenant_id=tenant['tenant_id'], agent_id=tenant['agent_id'])
        env = dict(base_env, HOME=str(directory), TMPDIR=str(directory),
                   HOST='127.0.0.1', PORT=str(args.port), REVIEW_LLM_MODEL=config['model'],
                   REVIEW_LLM_API_KEY='local-proxy', REVIEW_LLM_BASE_URL=proxy.url,
                   REVIEW_LLM_TIMEOUT_MS='90000', REVIEW_INTERVAL_MS='1000')
        if args.revise_model:
            env['REVIEW_REVISE_LLM_MODEL'] = args.revise_model
        install = directory / 'client'
        subprocess.run([args.node, str(args.source / args.installer), '--project', str(install)],
                       cwd=directory, env=env, capture_output=True, check=True, timeout=30)
        skill = (install / args.generated_skill).read_bytes().decode()
        with proxy, isolated_server(args, directory, env):
            client = MCP(args.base, tenant['key'], directory, args.startup_timeout)
            init = client.request('initialize', dict(protocolVersion='2025-03-26', capabilities={},
                                                      clientInfo=dict(name='x2-eval', version='1')))
            client.request('notifications/initialized', notification=True)
            instructions = init['instructions']
            version_line = instructions.splitlines()[0]
            if not version_line.startswith('intake_version: ') or version_line not in skill:
                raise RuntimeError('X2 MCP 与安装器生成的 intake_version 不一致')
            prompt = instructions + '\n\n' + skill
            agent_loop(proxy.url, config['model'], prompt, agent_input(text, projects, sample.name),
                       client.tools(), client.call, state, directory, args.max_rounds, MAX_INPUT,
                       args.request_timeout, time.monotonic() + args.agent_timeout, allowed=TOOLS)
            deadline = time.monotonic() + args.review_timeout
            while pending := db(args, base_env, 'pending', **scope):
                if proxy.input_limit_exceeded:
                    raise RuntimeError('样本超过 200 万输入 token，停止评测')
                if args.stop_file is not None and args.stop_file.exists():
                    raise RuntimeError('评测输入 token 已达到停止条件')
                if time.monotonic() >= deadline:
                    write_json(directory / 'pending.json', pending)
                    reasons = sorted({p['last_review_error'] or 'waiting' for p in pending})
                    raise RuntimeError('X2 审核超时；pending 原因：' + ', '.join(reasons))
                time.sleep(2)
        exported = db(args, base_env, 'export', **scope)
        write_json(directory / 'export.json', exported)
        decisions = review_metadata(exported['drafts'], exported['pages'])
        manifest['reviews'] = decisions
        if proxy.input_limit_exceeded:
            raise RuntimeError('样本超过 200 万输入 token，停止评测')
        if any(d['state'] == 'pending' for d in decisions):
            raise RuntimeError('导出时仍有未完成审核')
        if any(d['reason'] == 'ai_failed' for d in decisions):
            raise RuntimeError('X2 审核模型连续返回无效结果（ai_failed）')
        if state['tool_calls'] and not decisions and any(c['status'] == 'error' for c in state['tool_calls']):
            raise RuntimeError('X2 工具失败后没有原生草案')
        if any(row.get('status') != 200 for row in proxy.calls) or proxy.errors:
            raise RuntimeError('模型调用存在失败或未完成请求，见模型审计')
        drafts = [dict(id=d['id'], body=d['proposed'].get('body')) for d in exported['drafts']]
        if any(not isinstance(d['body'], str) or not d['body'].strip() for d in drafts):
            raise RuntimeError('X2 原生草案缺非空正文，无法映射到拆分判定格式')
        usage = model_usage(proxy)
        meta = dict(product=package, version=package['version'], llm=config['model'],
                    revise_llm=args.revise_model,
                    configuration='固定 agent + 原生自动审核', agent=state, reviews=decisions,
                    projects=projects, project_source=project_source,
                    prompt_version=version_line, prompt_sha256=sha256_text(prompt),
                    instructions_sha256=sha256_text(instructions), skill_sha256=sha256_text(skill),
                    input_sha256=sha256_text(text), parameters=proxy.parameters(), calls=proxy.calls, usage=usage,
                    isolation=dict(database='dedicated evaluation database', tenant=tenant_name, auto_review=True, read_tools=False),
                    raw_directory=str(directory.relative_to(ROOT)), time=now(),
                    elapsed_seconds=round(time.monotonic() - started, 3))
        # Rewrite both together on recovery if a previous process stopped between the two files.
        write_runs(sample, systems, (drafts, exported['pages']), meta, args.force or any(p.exists() for p in targets))
        manifest.update(status='written', drafts=len(drafts), published=len(exported['pages']))
        print(f'{sample.name}：草案 {len(drafts)}；正式页 {len(exported["pages"])}；'
              f'LLM 输入 token {usage["llm"]["tokens"]["prompt_tokens"]}', flush=True)
        return 'written'
    except BaseException as exc:
        manifest.update(status='failed', error=type(exc).__name__,
                        detail=str(exc) if isinstance(exc, (RuntimeError, TimeoutError)) else None)
        if tenant:
            db(args, base_env, 'disable', tenant_id=tenant['tenant_id'], agent_id=tenant['agent_id'])
        if proxy.input_limit_exceeded:
            raise SystemExit('样本超过 200 万输入 token，停止评测') from None
        raise
    finally:
        manifest.update(usage=model_usage(proxy), calls=proxy.calls, agent=state,
                        input_limit_exceeded=proxy.input_limit_exceeded,
                        elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(directory / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--failures', type=Path, default=TMP / 'x2-eval/x2-replay.json')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--max-rounds', type=int, default=12)
    parser.add_argument('--request-timeout', type=int, default=180)
    parser.add_argument('--startup-timeout', type=int, default=30)
    parser.add_argument('--agent-timeout', type=int, default=900)
    parser.add_argument('--review-timeout', type=int, default=900)
    parser.add_argument('--stop-file', type=Path, help='跨适配器共享评测停止状态（限 tmp/）')
    parser.add_argument('--expected-worktree', type=Path, help='匹配已记录的 X2 源码版本')
    args = parser.parse_args()
    try:

        args.source = Path(os.environ['X2_REPO']).resolve(strict=True)
        args.port = int(os.environ['X2_PORT'])
        if not 1024 <= args.port <= 65535:
            raise ValueError('invalid X2 port')
        args.base = f'http://127.0.0.1:{args.port}'
        expected_db = os.environ['X2_DATABASE_NAME']
        expected_port = int(os.environ['X2_DATABASE_PORT'])
        args.installer = os.environ['X2_INSTALLER_RELATIVE']
        args.generated_skill = os.environ['X2_GENERATED_SKILL_RELATIVE']
        for path in (Path(args.installer), Path(args.generated_skill)):
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('X2 paths must be relative')
        args.node = shutil.which('node')
        if args.node is None or min(args.max_rounds, args.request_timeout, args.startup_timeout,
                                    args.agent_timeout, args.review_timeout) <= 0:
            raise ValueError('Node.js 和正数时限为必需')
        url = connection_string(Path(os.environ['X2_DATABASE_URL_FILE']), expected_db, expected_port)
        if args.stop_file is not None:
            args.stop_file = output_path(args.stop_file)
        config = settings()
        args.revise_model = os.environ.get('REVIEW_REVISE_LLM_MODEL', '').strip() or None
        if args.revise_model == config['model']:
            raise ValueError('独立修订模型必须与摄取及审核模型不同')
        ENVS.mkdir(parents=True, exist_ok=True)
        env = dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', HOME=str(ENVS), LANG='en_US.UTF-8', DATABASE_URL=url,
                   X2_EVAL_DB=expected_db, X2_EVAL_DB_PORT=str(expected_port))
        package = fingerprint(args.source)
        if args.expected_worktree is not None and read_json(args.expected_worktree) != package:
            raise RuntimeError('X2 工作树与试跑源码版本不一致')
        with (ENVS / 'server.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepared = db(args, env, 'prepare')
            print('已核对数据库：' + prepared['database'] + '；应用迁移：' + ', '.join(prepared['migrations']), flush=True)
            return batch(sorted(p.parent for p in args.data_dir.glob('*/gold.json')),
                         lambda sample: generate(sample, args, config, package, env), args.failures)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        detail = f' · {exc}' if isinstance(exc, RuntimeError) else ''
        parser.exit(2, f'X2 评测初始化失败（{type(exc).__name__}）{detail}\n')


if __name__ == '__main__':
    raise SystemExit(main())
