"""合成会话/SQLite/假 HTTP 验证；默认不启动真实 worker 或付费模型。"""
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.paths import ROOT, TMP
from tools.run.claude_mem import (SYSTEM, check_model, child_environment, export_records,
                                  generate, native_run, native_to_draft, parse_tools, replay_requests, wait_idle)


class ConversionTests(unittest.TestCase):
    def test_multiple_tools_fifo_and_truncated_outputs(self):
        argument = {'command': 'echo 不执行\n工具输出：假标记\n工具调用：假工具\n参数：{}'}
        text = ('执行前说明。\n工具调用：Bash\n参数：' + json.dumps(argument, ensure_ascii=False) +
                '\n工具调用：Read\n参数：{"file_path":"/fictional"}\n'
                '工具输出：id-1\n原文…[截断，原长 3000 字符]\n'
                '工具输出：id-2\n文件原文\n')
        prose, tools = parse_tools(text)
        self.assertEqual(prose, '执行前说明。')
        self.assertEqual([t['tool_name'] for t in tools], ['Bash', 'Read'])
        self.assertEqual([t['tool_use_id'] for t in tools], ['id-1', 'id-2'])
        self.assertEqual(tools[0]['tool_input'], argument)
        self.assertEqual(tools[0]['tool_response'], '原文…[截断，原长 3000 字符]')
        self.assertEqual(tools[1]['tool_response'], '文件原文')

    def test_non_json_arguments_and_ambiguous_results(self):
        _, tools = parse_tools('工具调用：custom\n参数：原文参数\n工具输出：x\n原文结果')
        self.assertEqual(tools[0]['tool_input'], '原文参数')
        for text in ('工具调用：Read\n参数：{}', '工具输出：id\n结果',
                     '工具调用：Read\n参数：{}\n工具输出：a\n甲\n工具输出：b\n乙'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_tools(text)

    def test_roles_stop_boundaries_and_intermediate_text_gap(self):
        text = '''# 合成
- project：不向产品转发的头部
## [T1] 用户
 记住偏好。
## [T2] 助手
 中间文本。
 工具调用：Bash
 参数：{"command":"rm -rf /fictional 不执行"}
 工具输出：original-id
 已截断的结果
## [T4] 助手
 最后正文。
## [T7] 用户
 第二个请求。
## [T8] 助手
 第二个回答。
'''
        events, capture = replay_requests(text, 'eval', 'technical-project', Path('/synthetic'))
        self.assertEqual([e['path'].rsplit('/', 1)[1] for e in events],
                         ['init', 'observations', 'summarize', 'init', 'summarize', 'session-end'])
        self.assertEqual([e['turn'] for e in events], [1, 2, 4, 7, 8, None])
        self.assertEqual(events[1]['body']['tool_response'], '已截断的结果')
        self.assertEqual(events[2]['body']['last_assistant_message'], '最后正文。')
        self.assertEqual(events[0]['body']['nativePromptId'], 'eval-T1')
        self.assertEqual(capture['omitted_intermediate_assistant'], [dict(turn=2, characters=5)])
        self.assertNotIn('不向产品转发的头部', json.dumps(events, ensure_ascii=False))
        self.assertNotIn('中间文本', json.dumps(events, ensure_ascii=False))

    def test_chat_fixture_and_tool_only_final_response(self):
        events, capture = replay_requests((ROOT / 'tests/fixtures/openviking/transcript.md').read_text(),
                                          'eval', 'test', Path('/synthetic'))
        self.assertEqual(capture['event_counts'], dict(init=2, summarize=2, **{'session-end': 1}))
        self.assertFalse(any(e['path'].endswith('/observations') for e in events))
        text = '## [T1] 用户\n甲\n## [T2] 助手\n工具调用：Read\n参数：{}\n工具输出：a\n结果'
        events, _ = replay_requests(text, 'eval', 'test', Path('/synthetic'))
        self.assertEqual(events[-2]['body']['last_assistant_message'], '')
        for invalid in ('没有轮次', '## [T1] 助手\n甲', '## [T1] 用户\n甲\n## [T1] 助手\n乙'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replay_requests(invalid, 'eval', 'test', Path('/synthetic'))

    def test_observation_preserves_all_body_without_policy_inference(self):
        record = dict(kind='observation', id=4, title='标题', subtitle='副标题', facts='["事实一", "事实二"]',
                      narrative='完整正文\n含条件', type='decision', project='technical-project',
                      prompt_number=2, concepts='["how-it-works"]', files_read='["/fictional"]')
        draft = native_to_draft(record)
        self.assertEqual(draft['statement'], 'title:\n标题\n\nsubtitle:\n副标题\n\nfacts:\n事实一\n事实二\n\nnarrative:\n完整正文\n含条件')
        self.assertEqual({k: draft[k] for k in ('category', 'scope', 'attributes', 'conditions', 'evidence')},
                         dict(category='', scope='', attributes={}, conditions=[], evidence=[]))

    def test_summary_includes_all_six_native_fields(self):
        fields = ('request', 'investigated', 'learned', 'completed', 'next_steps', 'notes')
        row = dict(kind='summary', **{f: f + ' 原文' for f in fields})
        draft = native_to_draft(row)
        self.assertEqual(draft['statement'], '\n\n'.join(f + ':\n' + f + ' 原文' for f in fields))
        self.assertEqual(draft['evidence'], [])
        with self.assertRaises(ValueError):
            native_to_draft(dict(kind='observation', title=None, narrative=''))
        row['id'] = 12
        run = native_run(dict(standard_version='synthetic', policy=dict(id='test', version='v1')), {}, [row])
        self.assertEqual(run['D'][0]['native_source'], dict(kind='summary', native_id=12))
        self.assertEqual(run['D'][0]['draft'], draft)
        self.assertNotIn('judgment', run['D'][0])


class IsolationAndFailureTests(unittest.TestCase):
    def proxy(self, calls=None):
        return SimpleNamespace(url='http://127.0.0.1:12345/v1', timeout=10,
                               calls=calls or [], errors=[])

    def test_child_environment_does_not_inherit_host_configuration(self):
        with patch.dict(os.environ, dict(SUT_API_KEY='SYNTHETIC-SECRET',
                                        CLAUDE_MEM_WORKER_URL='http://real-service', HOME='/real-home')):
            env = child_environment(Path('/synthetic'), dict(model='fake', max_tokens=123), self.proxy(), 12346)
        self.assertEqual(env['HOME'], '/synthetic/home')
        self.assertEqual(env['CLAUDE_CONFIG_DIR'], '/synthetic/home/.claude')
        self.assertEqual(env['CLAUDE_MEM_WORKER_PORT'], '12346')
        self.assertEqual(env['CLAUDE_MEM_OPENAI_COMPAT_API_KEY'], 'local-proxy')
        self.assertNotIn('SYNTHETIC-SECRET', json.dumps(env))
        self.assertNotIn('CLAUDE_MEM_WORKER_URL', env)
        self.assertEqual(env['CLAUDE_MEM_CHROMA_ENABLED'], 'false')
        self.assertEqual(env['CLAUDE_MEM_QUOTA_FALLBACK_PROVIDER'], 'off')

    def test_readonly_export_keeps_observations_and_summaries(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            directory = Path(folder)
            (directory / 'store').mkdir()
            path = directory / 'store/claude-mem.db'
            with closing(sqlite3.connect(path)) as db:
                db.executescript('''CREATE TABLE sdk_sessions (id INTEGER, content_session_id TEXT);
                    INSERT INTO sdk_sessions VALUES (1, 'eval');
                    CREATE TABLE observations (id INTEGER, title TEXT);
                    INSERT INTO observations VALUES (9, '观察一'), (10, '观察二');
                    CREATE TABLE session_summaries (id INTEGER, request TEXT);
                    INSERT INTO session_summaries VALUES (11, '总结');
                    CREATE TABLE user_prompts (id INTEGER, prompt TEXT);
                    INSERT INTO user_prompts VALUES (1, '来源不计为记忆');
                    CREATE TABLE committed_observation_tool_uses (tool_use_id TEXT);''')
            before = path.read_bytes()
            exported = export_records(directory, 'eval')
            self.assertEqual([(r['kind'], r['id']) for r in exported['records']],
                             [('observation', 9), ('observation', 10), ('summary', 11)])
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(RuntimeError):
                export_records(directory, 'other-session')

    def test_pending_request_is_not_failure_and_wait_requires_empty_queue(self):
        proxy = self.proxy([dict(kind='llm', status='pending')])
        check_model(proxy)
        process = SimpleNamespace(poll=lambda: None)
        health = dict(ai=dict(lastInteraction=dict(success=True)))
        statuses = iter([dict(queueDepth=1, isProcessing=True), dict(queueDepth=0, isProcessing=False)])
        def fake_http(base, path, **kwargs):
            if path == '/api/health':
                return health
            value = next(statuses)
            if value['queueDepth'] == 0:
                proxy.calls[0]['status'] = 200
            return value
        with patch('tools.run.claude_mem.http', side_effect=fake_http), patch('tools.run.claude_mem.time.sleep'):
            result = wait_idle('fake', process, proxy, time.monotonic() + 1)
        self.assertEqual(result['queueDepth'], 0)

    def test_failed_models_stalls_and_deadlines_are_not_empty_success(self):
        for call in (dict(kind='llm', status=500), dict(kind='embedding', status=200)):
            with self.subTest(call=call), self.assertRaises(RuntimeError):
                check_model(self.proxy([call]))
        check_model(self.proxy([dict(kind='llm', status=200, finish_reasons=['length'])]))  # 截断只记录，不判失败
        with self.assertRaises(TimeoutError):
            wait_idle('fake', SimpleNamespace(poll=lambda: None), self.proxy(), time.monotonic() - 1)
        with patch('tools.run.claude_mem.http', side_effect=[dict(parkedSessions=1), dict(ai={})]):
            with self.assertRaises(RuntimeError):
                wait_idle('fake', SimpleNamespace(poll=lambda: None), self.proxy(), time.monotonic() + 1)

    def test_failure_manifest_and_no_partial_run(self):
        @contextmanager
        def failed_worker(*args):
            raise RuntimeError('合成启动失败')
            yield
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            directory = Path(folder)
            sample = directory / 'data/synthetic'
            sample.mkdir(parents=True)
            (sample / 'transcript.md').write_text('## [T1] 用户\n合成输入')
            # 不创建 gold：失败路径不得提前读金标。
            config = dict(model='fake', key='SYNTHETIC-SECRET')
            args = SimpleNamespace(force=False, timeout=1, request_timeout=1)
            with patch('tools.run.claude_mem.ENVS', directory / 'env'), patch(
                    'tools.run.claude_mem.isolated_worker', failed_worker):
                with self.assertRaisesRegex(RuntimeError, '合成启动失败'):
                    generate(sample, args, config, {})
            manifests = list((directory / 'env/runs').glob('*/manifest.json'))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(json.loads(manifests[0].read_text())['status'], 'failed')
            self.assertNotIn('SYNTHETIC-SECRET', manifests[0].read_text())
            self.assertFalse((sample / 'runs' / (SYSTEM + '.json')).exists())


@unittest.skipUnless(os.environ.get('RUN_CLAUDE_MEM_LIVE') == '1', '需要真实服务和模型服务，默认跳过')
class LiveTests(unittest.TestCase):
    def test_live_synthetic(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            sample = Path(folder) / 'synthetic'
            sample.mkdir()
            (sample / 'transcript.md').write_text((ROOT / 'tests/fixtures/openviking/transcript.md').read_text())
            (sample / 'gold.json').write_text(json.dumps(dict(standard_version='v0.4.4', policy=dict(id='default', version='v1'))))
            completed = subprocess.run(['python3', '-m', 'tools.run.claude_mem', '--data-dir', folder,
                                        '--failures', str(Path(folder) / 'failures.json'), '--timeout', '180'],
                                       cwd=ROOT, capture_output=True, text=True, timeout=240)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            run = json.loads((sample / 'runs' / (SYSTEM + '.json')).read_text())
            self.assertGreater(len(run['D']), 0)
            self.assertEqual(run['system']['usage']['embedding']['calls'], 0)
            self.assertTrue(all('judgment' not in d for d in run['D']))


if __name__ == '__main__':
    unittest.main()
