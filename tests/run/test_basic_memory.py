"""合成数据、假 OpenAI 服务和假 stdio MCP；默认不访问真实产品/模型。"""
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import threading
import time
import unittest

from tools.paths import ROOT, TMP
from tools.run.basic_memory import (ALLOWED, ASSETS, BudgetExceeded, PROMPT, PROMPT_SHA256,
    SYSTEM, StdioMCP, agent_input, agent_loop, child_environment, export_notes, function_tools,
    native_to_draft, note_body, validate_store)
from tools.config import sha256_text


@contextmanager
def fake_openai(responses):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            response = responses[min(len(seen) - 1, len(responses) - 1)]
            raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(raw)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def response(calls=None, finish=None, tokens=10):
    message = dict(role='assistant', content=None if calls else '整理完成')
    if calls:
        message['tool_calls'] = [dict(id=f'call-{i}', type='function', function=dict(
            name=name, arguments=json.dumps(arguments))) for i, (name, arguments) in enumerate(calls)]
    return dict(choices=[dict(message=message, finish_reason=finish or ('tool_calls' if calls else 'stop'))],
                usage=dict(total_tokens=tokens))


def tools():
    native = [dict(name=name, description='合成原生说明', inputSchema=dict(type='object', properties={}))
              for name in sorted(ALLOWED | {'bm_delete', 'bm_project', 'shell'})]
    return function_tools(native)


class ConversionTests(unittest.TestCase):
    def test_official_prompt_hash_and_input_is_only_history(self):
        if PROMPT.exists():  # 提示词来自本机安装的 basic-memory 包；未安装时跳过此项核对
            self.assertEqual(sha256_text(PROMPT.read_text()), PROMPT_SHA256)
        text = (ROOT / 'tests/fixtures/openviking/transcript.md').read_text()
        value = agent_input(text, 'synthetic-project', 'synthetic-thread')
        self.assertIn(text, value)
        self.assertIn('历史会话', value)
        self.assertIn('synthetic-thread', value)
        with self.assertRaises(ValueError):
            agent_input('没有轮次', 'p', 't')

    def test_whole_body_preserved_no_double_count_no_inferred_fields(self):
        content = '---\ntitle: 合成标题\ncategory: decision\n---\n# 合成笔记\n\n- [decision] 仅冬季用蓝色。\n\n尾注\n'
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            root = Path(directory)
            (root / '.hidden/deep').mkdir(parents=True)
            (root / '.hidden/deep/A.MD').write_text(content)
            (root / 'other.md').write_text('第二篇\n')
            records = export_notes(root)
        self.assertEqual(len(records), 2)
        row = records[0]
        self.assertEqual(row['markdown'], content)
        self.assertEqual(row['body'], '# 合成笔记\n\n- [decision] 仅冬季用蓝色。\n\n尾注\n')
        draft = native_to_draft(row)
        self.assertEqual(draft, dict(statement=row['body'], category='', scope='', attributes={}, conditions=[], evidence=[]))
        self.assertEqual(note_body('正文\n---\n分隔线'), '正文\n---\n分隔线')
        with self.assertRaises(ValueError):
            note_body('---\n未闭合')
        with self.assertRaises(ValueError):
            native_to_draft(dict(body='  '))

    def test_export_rejects_symlink(self):
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            root = Path(directory)
            (root / 'bad.md').symlink_to(ROOT / 'README.md')
            with self.assertRaises(RuntimeError):
                export_notes(root)

    def test_environment_does_not_inherit_credentials_or_home(self):
        env = child_environment(Path('/synthetic'))
        self.assertEqual(env['HOME'], '/synthetic/home')
        self.assertEqual(env['BASIC_MEMORY_CONFIG_DIR'], '/synthetic/config')
        self.assertEqual(env['OPENAI_API_KEY'], 'local-proxy')
        self.assertNotIn('SUT_API_KEY', env)
        self.assertNotIn('AI_MEMORY_HOME', env)
        self.assertEqual({t['function']['name'] for t in tools()}, ALLOWED)

    def test_database_first_notes_must_all_be_materialized_and_vectors_finished(self):
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            root = Path(directory)
            (root / 'config').mkdir()
            with closing(sqlite3.connect(root / 'config/memory.db')) as db, db:
                db.executescript('''CREATE TABLE project(id INTEGER,name TEXT,path TEXT);
                    CREATE TABLE note_content(project_id INTEGER,file_path TEXT,file_write_status TEXT,markdown_content TEXT);
                    CREATE TABLE search_vector_chunks(id INTEGER,embedding_status TEXT);''')
                db.execute('INSERT INTO project VALUES(1,?,?)', ('p', str(root / 'notes')))
                db.execute('INSERT INTO note_content VALUES(1,?,?,?)', ('a.md', 'synced', '合成正文\n'))
                db.execute('INSERT INTO search_vector_chunks VALUES(1,?)', ('ready',))
            records = [dict(path='a.md', markdown='合成正文\n')]
            self.assertEqual(len(validate_store(root, 'p', records)['note_content']), 1)
            for invalid in ([], [dict(path='a.md', markdown='过期正文')]):
                with self.assertRaises(RuntimeError):
                    validate_store(root, 'p', invalid)
            with closing(sqlite3.connect(root / 'config/memory.db')) as db, db:
                db.execute("UPDATE search_vector_chunks SET embedding_status='pending'")
            with self.assertRaises(RuntimeError):
                validate_store(root, 'p', records)
            with self.assertRaises(RuntimeError):
                validate_store(root, '其他项目', records)

    def test_child_guard_blocks_remote_connections_and_dns(self):
        code = '''import socket
for action in (lambda: socket.getaddrinfo('synthetic.invalid', 443),
               lambda: socket.socket().connect_ex(('203.0.113.1', 443)),
               lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'x', ('203.0.113.1', 53))):
 try:
  action()
 except OSError as e:
  assert str(e)=='EVAL_NETWORK_DENIED'
 else:
  raise AssertionError('远程访问未被阻止')
assert socket.getaddrinfo('127.0.0.1', 80)
'''
        result = subprocess.run([sys.executable, '-c', code], env=dict(os.environ,
                                PYTHONPATH=str(ASSETS / 'network')), capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)


class AgentTests(unittest.TestCase):
    def run_agent(self, responses, callback, **kwargs):
        state = {}
        with tempfile.TemporaryDirectory(dir=TMP) as directory, fake_openai(responses) as (url, seen):
            agent_loop(url, 'fake-llm', '官方原文替身', '合成历史', tools(), callback, state, Path(directory), **kwargs)
            audit = json.loads((Path(directory) / 'agent.json').read_text())
        return state, seen, audit

    def test_fake_openai_forwards_write_edit_and_read_to_stdio_mcp(self):
        # 假产品回显 MCP tools/call，证明 function calling 经过真实 JSON-RPC 管道。
        script = '''import sys,json
for line in sys.stdin:
 v=json.loads(line)
 if 'id' in v:
  print(json.dumps(dict(jsonrpc='2.0',id=v['id'],result=dict(content=[dict(type='text',text=json.dumps(v['params']))]))),flush=True)
'''
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            process = subprocess.Popen([sys.executable, '-u', '-c', script], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, text=True)
            client = StdioMCP(process, Path(directory), time.monotonic() + 10)
            forwarded = []
            def call(name, arguments):
                result = client.request('tools/call', dict(name=name, arguments=arguments))
                forwarded.append(json.loads(result['content'][0]['text']))
                return result
            try:
                rows = [response([('bm_write', dict(content='合成笔记', project='p'))]),
                        response([('bm_read', dict(identifier='note', project='p')),
                                  ('bm_edit', dict(identifier='note', command='append', content='补充', project='p'))]), response()]
                state, seen, audit = self.run_agent(rows, call)
                self.assertEqual([c['name'] for c in forwarded], ['bm_write', 'bm_read', 'bm_edit'])
                self.assertEqual(forwarded[0]['arguments']['content'], '合成笔记')
                self.assertEqual(state['rounds'], 3)
                self.assertEqual(state['stop_reason'], 'completed')
                self.assertEqual(audit['tool_call_count'], 3)
                self.assertEqual(seen[0]['messages'][0]['content'], '官方原文替身')
                self.assertEqual([m['role'] for m in seen[-1]['messages']], ['system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'tool'])
                self.assertNotIn('temperature', seen[0])
                self.assertNotIn('max_tokens', seen[0])
            finally:
                process.stdin.close()
                process.wait(timeout=5)
                process.stdout.close()
                client.thread.join(timeout=1)

    def test_round_limit_stops_and_persists_trace(self):
        calls, state = [], {}
        with tempfile.TemporaryDirectory(dir=TMP) as directory, fake_openai([response([('bm_read', {})])]) as (url, seen):
            with self.assertRaises(BudgetExceeded):
                agent_loop(url, 'fake', 'prompt', 'history', tools(),
                           lambda n, a: calls.append(n) or {}, state, Path(directory), max_rounds=2)
            self.assertEqual(state['stop_reason'], 'max_rounds')
            self.assertEqual(len(seen), 2)
            self.assertEqual(calls, ['bm_read', 'bm_read'])
            self.assertEqual(json.loads((Path(directory) / 'agent.json').read_text())['rounds'], 2)

    def test_token_limit_and_length_refuse_tool_execution(self):
        for value, budget, reason in [(response([('bm_write', {})], tokens=11), 10, 'max_total_tokens'),
                                      (response(finish='length'), 100, 'max_output_tokens')]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory(dir=TMP) as directory, fake_openai([value]) as (url, seen):
                state, calls = {}, []
                with self.assertRaises(BudgetExceeded):
                    agent_loop(url, 'fake', 'p', 'h', tools(), lambda n, a: calls.append(n),
                               state, Path(directory), max_total_tokens=budget)
                self.assertEqual(state['stop_reason'], reason)
                self.assertEqual(calls, [])

    def test_tool_refusal_returned_and_unknown_tool_rejected(self):
        state, seen, audit = self.run_agent([response([('bm_write', {})]), response()],
            lambda n, a: dict(isError=True, content=[dict(type='text', text='合成冲突')]))
        self.assertEqual(state['tool_calls'][0]['status'], 'error')
        self.assertIn('合成冲突', seen[1]['messages'][-1]['content'])
        with self.assertRaises(RuntimeError):
            self.run_agent([response([('shell', dict(command='不执行'))])], lambda n, a: self.fail('不应执行'))

    def test_mcp_timeout_is_bounded(self):
        process = subprocess.Popen([sys.executable, '-u', '-c', 'import time; time.sleep(10)'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            with tempfile.TemporaryDirectory(dir=TMP) as directory:
                client = StdioMCP(process, Path(directory), time.monotonic() + .05)
                with self.assertRaises(TimeoutError):
                    client.request('initialize')
        finally:
            process.terminate()
            process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
            client.thread.join(timeout=1)


@unittest.skipUnless(os.environ.get('RUN_BASIC_MEMORY_LIVE') == '1', '真实产品和模型默认跳过')
class LiveTests(unittest.TestCase):
    def test_live_synthetic_ingestion(self):
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            sample = Path(directory) / 'synthetic'
            sample.mkdir()
            (sample / 'transcript.md').write_text((ROOT / 'tests/fixtures/openviking/transcript.md').read_text())
            (sample / 'gold.json').write_text(json.dumps(dict(standard_version='v0.4.4', policy=dict(id='default', version='v1'))))
            result = subprocess.run([sys.executable, '-m', 'tools.run.basic_memory', '--data-dir', directory,
                                     '--failures', str(Path(directory) / 'failures.json')],
                                    cwd=ROOT, capture_output=True, text=True, timeout=1200)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = json.loads((sample / 'runs' / (SYSTEM + '.json')).read_text())
            self.assertGreater(len(run['D']), 0)
            self.assertEqual(run['system']['agent']['stop_reason'], 'completed')
            self.assertTrue(all('judgment' not in d for d in run['D']))


if __name__ == '__main__':
    unittest.main()
