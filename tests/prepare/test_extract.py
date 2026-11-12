import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.paths import inside_repo
from tools.prepare import extract
from tests.helpers import ROOT, TMP, SESSIONS


class ExtractTests(unittest.TestCase):
    def test_both_sources(self):
        for source, project in [('claude','garden'),('codex','library')]:
            meta, turns = extract.read_session(SESSIONS / (source+'.jsonl'))
            self.assertEqual(meta['source'],source)
            self.assertEqual(meta['project'],project)
            self.assertEqual(meta['turns'],5)
            self.assertEqual(meta['user_messages'],3)
            text=extract.render(meta,turns)
            self.assertIn('工具调用：read_config',text)
            self.assertIn('工具输出：',text)
            self.assertIn('.conf',text)
            self.assertNotIn('不应保留',text)
            self.assertIn('## [T5]',text)
            self.assertIn('- cwd 推断项目：'+project,text)
            self.assertIn('- project（标注者填写）：\n',text)
            self.assertNotIn('- 项目名：',text)

    def test_truncate_and_suggest_cuts(self):
        self.assertEqual(extract.tool_output('字'*2000),'字'*2000)
        self.assertEqual(extract.tool_output('字'*2001),'字'*2000+'…[截断，原长 2001 字符]')
        turns=[dict(role='user',text='合成文字',timestamp='2026-11-11T08:00:00Z') for _ in range(301)]
        turns[10]['timestamp']='2026-11-11T11:00:01Z'
        meta=dict(session_id='synthetic',source='codex',date='2026-11-11',cwd='/synthetic',project='synthetic',turns=301)
        text=extract.render(meta,turns)
        self.assertIn('T11 前（时间间隔超过 2 小时）',text)
        self.assertIn('T301 前（每 300 轮）',text)
        self.assertEqual(text.count('## [T'),301)

    def test_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            result=subprocess.run(['python3','-m','tools.prepare.extract',str(SESSIONS/'codex.jsonl'),'synthetic','--data-dir',d],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            path=Path(d)/'synthetic/transcript.md'
            self.assertTrue(path.exists())
            with self.assertRaisesRegex(ValueError,'拒绝覆盖'):
                extract.extract(SESSIONS/'codex.jsonl','synthetic',data_dir=d)
        with self.assertRaises(ValueError):
            inside_repo('/not-in-repo/output')
        with self.assertRaisesRegex(ValueError,'不能写进仓库的受跟踪目录'):
            extract.extract(SESSIONS/'codex.jsonl','synthetic',data_dir=ROOT/'data')

    def test_bad_json_reports_line(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            p=Path(d)/'bad.jsonl';p.write_text('{}\n无效\n')
            with self.assertRaisesRegex(ValueError,':2:'):
                extract.read_session(p)
