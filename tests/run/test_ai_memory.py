"""仅使用合成记录；默认不启动真实服务、不访问线上模型。"""
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
import uuid

from tools.paths import ROOT, TMP
from tools.run import ai_memory as adapter


class ReplayTests(unittest.TestCase):
    def test_order_body_default_capture_and_tool_pair(self):
        text = '''# 合成材料
- project：garden
## [T1] 用户
 记住我偏好 Python。
## [T2] 助手
 私密助手正文不采集。
 工具调用：Bash
 参数：{"command":"echo 合成命令，不执行"}

 工具输出：synthetic-id
 合成结果
## [T3] 助手
 已完成。
## [T9] 用户
 下一项。
'''
        sid = str(uuid.uuid4())
        requests, capture = adapter.replay_requests(text, 'http://127.0.0.1:12345', sid, 's', '/eval')
        queries = [parse_qs(urlsplit(r['url']).query) for r in requests]
        self.assertEqual([q['event'][0] for q in queries], [
            'session-start', 'user-prompt-submit', 'pre-tool-use', 'post-tool-use', 'stop',
            'user-prompt-submit', 'session-end'])
        self.assertEqual(requests[2]['body']['tool_input'], {'command': 'echo 合成命令，不执行'})
        self.assertEqual(requests[3]['body']['tool_response'], '合成结果')
        self.assertEqual(requests[3]['body']['tool_use_id'], 'synthetic-id')
        self.assertEqual(len({q['ingest_key'][0] for q in queries}), len(queries))
        for request, query in zip(requests, queries):
            self.assertEqual(request['body']['session_id'], sid)
            self.assertEqual(query['workspace'], ['eval'])
            self.assertEqual(query['agent'], ['claude-code'])
            self.assertNotIn('capture_assistant', query)
            self.assertNotIn('_ai_memory_assistant', request['body'])
        self.assertNotIn('私密助手正文', json.dumps(requests, ensure_ascii=False))
        self.assertNotIn('已完成', json.dumps(requests, ensure_ascii=False))
        self.assertFalse(capture['capture_assistant'])
        self.assertEqual(capture['assistant_turns_not_captured'], 2)
        self.assertEqual(capture['event_mapping'][4]['turn'], 3)

    def test_user_markers_are_never_executed_or_reinterpreted(self):
        text = '## [T1] 用户\n 工具调用：Bash\n参数：{"command":"rm -rf /fictional"}'
        requests, _ = adapter.replay_requests(text, 'http://unused', str(uuid.uuid4()), 's', '/eval')
        self.assertEqual(len(requests), 3)
        self.assertIn('rm -rf /fictional', requests[1]['body']['prompt'])

    def test_multiple_tools_json_and_missing_result(self):
        text = '''工具调用：Write
参数：{"content": "甲\\n工具输出：仅为参数数据", "path": "fictional.txt"}

工具输出：id-1
完成

工具调用：Read
参数：{"path":"fictional.txt"}
'''
        tools = adapter.tool_events(text)
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]['tool_input']['content'], '甲\n工具输出：仅为参数数据')
        self.assertEqual(tools[0]['tool_response'], '完成')
        self.assertNotIn('tool_response', tools[1])
        self.assertEqual(adapter.tool_events('普通正文，没有工具'), [])
        self.assertEqual(adapter.tool_events('工具调用：custom\n参数：原样脚本')[0]['tool_input'], '原样脚本')

    def test_ack_requires_all_terminal_results_and_counts_policy_drop(self):
        ack = dict(accepted=2, results=[dict(index=0, outcome='stored'), dict(index=1, outcome='dropped_policy')])
        self.assertEqual(adapter.validate_ack(ack, 2), {'stored': 1, 'dropped_policy': 1})
        bad = [dict(accepted=2, results=[]), dict(ack, accepted_indices=[1]), dict(ack, accepted=1),
               dict(ack, failed_index=1)]
        for outcome in ('replayed', 'resumed', 'ignored_end', 'dropped_invalid', 'dropped_unauthorized', 'dropped_collision'):
            bad.append(dict(accepted=2, results=[dict(index=0, outcome='stored'), dict(index=1, outcome=outcome)]))
        for value in bad:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                adapter.validate_ack(value, 2)


class ExportTests(unittest.TestCase):
    def test_native_page_is_preserved_and_technical_scope_not_inferred(self):
        body = '# 合成总结\n\n- 偏好 Python。\n- garden 选择 SQLite。\n'
        record = dict(body=body, tier='semantic', workspace='eval', project='sample-s001',
                      frontmatter={'kind': 'session-summary', 'sources': ['session://abc']})
        draft = adapter.native_to_draft(record)
        self.assertEqual(draft, dict(statement=body, category='', scope='', attributes={}, conditions=[], evidence=[]))
        for body in ('', '   ', None):
            with self.assertRaises(ValueError):
                adapter.native_to_draft(dict(body=body))

    def test_export_only_latest_nonstructural_pages_and_keep_session_summary(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            store = Path(temporary)
            (store / 'db').mkdir()
            with closing(sqlite3.connect(store / 'db/memory.sqlite')) as db:
                db.executescript('''CREATE TABLE workspaces(id BLOB, name TEXT);
                    CREATE TABLE projects(id BLOB, name TEXT);
                    CREATE TABLE pages(id BLOB, workspace_id BLOB, project_id BLOB, path TEXT,
                    title TEXT, body TEXT, frontmatter_json TEXT, tier TEXT, is_latest INT);
                    INSERT INTO workspaces VALUES (1, 'eval'); INSERT INTO projects VALUES (2, 's');''')
                for i, (path, body, latest) in enumerate([
                        ('sessions/a.md', '旧版', 0), ('sessions/a.md', '新版\n原文', 1),
                        ('_meta.md', '目录', 1), ('index.md', '索引', 1), ('log-2026-11.md', '事件', 1),
                        ('facts/other.md', '另一条原生知识', 1)]):
                    db.execute('INSERT INTO pages VALUES (?,1,2,?,?,?,?,?,?)',
                               (i, path, path, body, '{"kind":"fact"}', 'episodic', latest))
                db.commit()
            records, excluded, schema = adapter.export_pages(store)
            self.assertEqual([r['body'] for r in records], ['另一条原生知识', '新版\n原文'])
            self.assertEqual(len(excluded), 3)
            self.assertIn('pages', schema)
            self.assertEqual(records[0]['frontmatter'], {'kind': 'fact'})

    def test_mcp_arguments_error_audit_and_no_fallback_to_empty_result(self):
        outcome = dict(page_id='id', path='sessions/a.md', dry_run=False, new_body_markdown='# 原文')
        response = dict(result=dict(content=[dict(type='text', text=json.dumps(outcome))]))
        with patch.object(adapter, 'http', return_value=response) as http:
            _, value = adapter.consolidate('http://unused', 'fake-uuid', 10)
        self.assertEqual(value, outcome)
        arguments = http.call_args.args[2]['params']['arguments']
        self.assertEqual(arguments, dict(session_id='fake-uuid', dry_run=False, multi_page=False))
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            path = Path(temporary) / 'error.json'
            error = dict(error=dict(code=-32603, message='合成失败'))
            with patch.object(adapter, 'http', return_value=error), self.assertRaises(RuntimeError):
                adapter.consolidate('http://unused', 'id', 10, path)
            self.assertEqual(json.loads(path.read_text()), error)


class IsolationTests(unittest.TestCase):
    def test_child_environment_does_not_inherit_host_routes_or_secrets(self):
        with patch.dict(os.environ, {'AI_MEMORY_SERVER_URL': 'http://existing', 'SUT_API_KEY': 'synthetic-secret',
                                    'AI_MEMORY_DATA_DIR': '/existing', 'GIT_CONFIG_GLOBAL': '/existing'}):
            env = adapter.child_environment(Path('/fictional/experiment'))
        self.assertEqual(env['HOME'], '/fictional/experiment/home')
        self.assertEqual(env['LLM_API_KEY'], 'local-proxy')
        for name in ('AI_MEMORY_SERVER_URL', 'AI_MEMORY_DATA_DIR', 'SUT_API_KEY', 'GIT_CONFIG_GLOBAL'):
            self.assertNotIn(name, env)
        self.assertNotIn('synthetic-secret', json.dumps(env))

    def test_failure_preserves_existing_run_and_records_manifest(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            directory = Path(temporary)
            sample = directory / 's'
            (sample / 'runs').mkdir(parents=True)
            (sample / 'transcript.md').write_text('## [T1] 用户\n 合成消息')
            target = sample / 'runs' / (adapter.SYSTEM + '.json')
            target.write_text('原有结果不可丢失')
            args = argparse_args(force=True)
            config = dict(model='fake', embedding_model='fake-embedding', key='synthetic-secret', dimensions=2)

            @contextmanager
            def fail(*args):
                raise RuntimeError('合成启动失败')
                yield

            with patch.object(adapter, 'ENVS', directory / 'envs'), patch.object(adapter, 'isolated_server', fail):
                with self.assertRaises(RuntimeError):
                    adapter.generate(sample, args, config, {})
            self.assertEqual(target.read_text(), '原有结果不可丢失')
            manifest = json.loads(next((directory / 'envs/s').glob('*/manifest.json')).read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertNotIn('synthetic-secret', json.dumps(manifest))
            args.force = False
            with patch.object(adapter, 'isolated_server') as server:
                self.assertEqual(adapter.generate(sample, args, config, {}), 'skipped')
                server.assert_not_called()


def argparse_args(force=False):
    from argparse import Namespace
    return Namespace(force=force, binary=adapter.BINARY, timeout=10, request_timeout=5, startup_timeout=5)


@unittest.skipUnless(os.environ.get('RUN_AI_MEMORY_LIVE') == '1', '真实模型和服务默认跳过')
class LiveTests(unittest.TestCase):
    def test_synthetic_real_instance(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            directory = Path(temporary)
            sample = directory / 'synthetic-live'
            sample.mkdir()
            (sample / 'transcript.md').write_text((ROOT / 'tests/fixtures/openviking/transcript.md').read_text())
            (sample / 'gold.json').write_text(json.dumps(dict(standard_version='v0.4.3',
                policy=dict(id='default', version='v1'), project=['garden'], U=[], N=[])))
            env = dict(os.environ)
            env.pop('EVAL_DOTENV', None)
            result = subprocess.run(['python3', '-m', 'tools.run.ai_memory', '--data-dir', str(directory),
                '--failures', str(directory / 'failures.json')], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = json.loads((sample / 'runs' / (adapter.SYSTEM + '.json')).read_text())
            self.assertEqual(len(run['D']), 1)
            self.assertNotIn('judgment', run['D'][0])


if __name__ == '__main__':
    unittest.main()
