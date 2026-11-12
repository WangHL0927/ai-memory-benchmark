"""全部默认测试使用合成数据和假 HTTP，不访问真实产品或模型。"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tools.run.openviking import (ModelProxy, memory_files, native_to_draft,
                                 transcript_to_messages, wait_commit, select_records, list_files)
from tools.paths import ROOT, TMP


class ConversionTests(unittest.TestCase):
    def test_transcript_roles_order_tools_and_turns(self):
        text = '''# 合成材料
- project：不传给被测模型的头部
## [T1] 用户
 以后用 Python。
## [T3] 助手
 好的。
 <details><summary>工具调用</summary>
 rm -rf /fictional 不执行
 </details>
## [T9] 用户
 谢谢。
'''
        messages = transcript_to_messages(text)
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'user'])
        self.assertEqual([m['turn_id'] for m in messages], ['T1', 'T3', 'T9'])
        self.assertEqual(messages[0]['content'], '以后用 Python。')
        self.assertIn('rm -rf /fictional 不执行', messages[1]['content'])
        self.assertNotIn('头部', json.dumps(messages, ensure_ascii=False))

    def test_invalid_transcript_rejected(self):
        for text in ('没有轮次', '## [T1] 用户\n 甲\n## [T1] 助手\n 乙'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                transcript_to_messages(text)

    def test_native_page_preserved_without_inferred_fields(self):
        content = '# 偏好\n\n- 仅冬季使用蓝色水管。\n- 曾用红色。\n'
        value = native_to_draft(dict(uri='viking://user/eval/memories/preferences/a.md', content=content,
                                    memory_type='preferences', ranges=[dict(start=1, end=3)]))
        self.assertEqual(value, dict(statement=content, category='', scope='', attributes={},
                                     conditions=[], evidence=[]))
        with self.assertRaises(ValueError):
            native_to_draft(dict(content='  '))

    def test_only_exact_initial_templates_are_excluded(self):
        rows = [dict(uri='viking://user/e/memories/identity.md', content='空模板'),
                dict(uri='viking://user/e/memories/soul.md', content='空模板加上真实知识'),
                dict(uri='viking://user/e/memories/profile.md', content='原生知识')]
        selected, excluded = select_records(rows, {'identity.md': '空模板', 'soul.md': '空模板'})
        self.assertEqual(selected, rows[1:])
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]['uri'], rows[0]['uri'])

    def test_walk_has_no_depth_limit_and_paginates(self):
        root = 'viking://user/e'
        def fake_api(base, path, **query):
            self.assertEqual(query['recursive'], 'false')
            if query['uri'] == root and query['offset'] == 0:
                return dict(result=[dict(uri=root + '/a', isDir=True)], has_more=True)
            if query['uri'] == root and query['offset'] == 1:
                return dict(result=[dict(uri=root + '/other.md', isDir=False)], has_more=False)
            depth = query['uri'].count('/') - root.count('/')
            return dict(result=[dict(uri=query['uri'] + '/a', isDir=depth < 6)], has_more=False)
        with patch('tools.run.openviking.api', side_effect=fake_api):
            rows = list_files('fake', root)
        self.assertEqual(len(rows), 8)
        self.assertTrue(any(r['uri'].count('/') - root.count('/') == 7 for r in rows))

    def test_fixture_is_synthetic_and_has_four_turns(self):
        path = ROOT / 'tests/fixtures/openviking/transcript.md'
        self.assertEqual(len(transcript_to_messages(path.read_text())), 4)

    def test_export_excludes_summaries_sessions_and_covers_peer(self):
        rows = [dict(uri=u, isDir=d) for u, d in [
            ('viking://user/eval/memories/preferences', True),
            ('viking://user/eval/memories/preferences/a.md', False),
            ('viking://user/eval/memories/.overview.md', False),
            ('viking://user/eval/sessions/a/messages.jsonl', False),
            ('viking://user/eval/peers/b/memories/profile.md', False)]]
        with patch('tools.run.openviking.list_files', return_value=rows), patch(
                'tools.run.openviking.read_content', side_effect=lambda b, u: u):
            files = memory_files('fake', 'viking://user/eval')
        self.assertEqual(len(files), 2)
        self.assertIn('viking://user/eval/peers/b/memories/profile.md', files)

    def test_commit_polls_until_completed_and_rejects_failure_timeout_noop(self):
        commit = dict(status='accepted', task_id='fake')
        with patch('tools.run.openviking.result', side_effect=[dict(status='running'), dict(status='completed')]) as fetch:
            wait_commit('fake', commit, time.monotonic() + 1, 0.001)
            self.assertEqual(fetch.call_count, 2)
        with patch('tools.run.openviking.result', return_value=dict(status='failed')):
            with self.assertRaises(RuntimeError):
                wait_commit('fake', commit, time.monotonic() + 1, 0.001)
        with self.assertRaises(TimeoutError):
            wait_commit('fake', commit, time.monotonic() - 1, 0.001)
        with self.assertRaises(RuntimeError):
            wait_commit('fake', dict(status='skipped'), time.monotonic() + 1, 0.001)


class ProxyTests(unittest.TestCase):
    def test_unset_parameters_usage_raw_hash_and_key_not_persisted(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append((self.path, body, self.headers['Authorization']))
                value = (dict(data=[dict(embedding=[0.0, 1.0])], usage=dict(prompt_tokens=3, total_tokens=3))
                         if self.path.endswith('embeddings') else
                         dict(choices=[dict(finish_reason='stop')], usage=dict(prompt_tokens=5, completion_tokens=2, total_tokens=7)))
                raw = json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(raw)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=TMP) as directory:
                config = dict(base_url=f'http://127.0.0.1:{server.server_port}/v1', key='SYNTHETIC-SECRET',
                              model='fake-llm', temperature=0.2, max_tokens=100, enable_thinking=False,
                              embedding_model='fake-embedding', dimensions=2)
                from urllib.request import Request, urlopen
                with ModelProxy(config, Path(directory), 5) as proxy:
                    for endpoint, body in [('chat/completions', dict(model='fake-llm', messages=[])),
                                           ('embeddings', dict(model='fake-embedding', input=['合成输入']))]:
                        with urlopen(Request(proxy.url + '/' + endpoint, data=json.dumps(body).encode())) as response:
                            self.assertEqual(response.status, 200)
                    usage = proxy.summary()
                self.assertEqual(usage['llm']['calls'], 1)
                self.assertEqual(usage['llm']['tokens']['total_tokens'], 7)
                self.assertEqual(usage['embedding']['calls'], 1)
                for field in ('enable_thinking', 'max_tokens', 'temperature'):
                    self.assertNotIn(field, seen[0][1])
                self.assertNotIn('dimensions', seen[1][1])
                self.assertEqual(seen[0][2], 'Bearer SYNTHETIC-SECRET')
                for path in Path(directory).glob('*.json'):
                    self.assertNotIn('SYNTHETIC-SECRET', path.read_text())
                    self.assertIn('response_sha256', json.loads(path.read_text()))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


@unittest.skipUnless(os.environ.get('RUN_OPENVIKING_LIVE') == '1', '真实模型冒烟默认跳过；手动运行适配器完成验收')
class LiveTests(unittest.TestCase):
    def test_live_synthetic_ingestion(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            sample = Path(directory) / 'synthetic'
            sample.mkdir()
            (sample / 'transcript.md').write_text((ROOT / 'tests/fixtures/openviking/transcript.md').read_text())
            (sample / 'gold.json').write_text(json.dumps(dict(standard_version='v0.4.3',
                policy=dict(id='default', version='v1'), project=['garden'], U=[], N=[])))
            completed = subprocess.run(['python3', '-m', 'tools.run.openviking', '--data-dir', directory,
                '--timeout', '120', '--failures', str(Path(directory) / 'failures.json')],
                cwd=ROOT, capture_output=True, text=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            run = json.loads((sample / 'runs/openviking-0.5.4.1-ltm-defaults.json').read_text())
            self.assertGreater(len(run['D']), 0)
