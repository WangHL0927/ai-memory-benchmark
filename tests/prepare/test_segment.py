import tempfile
import unittest
from pathlib import Path

from tools.prepare.segment import segment_dir
from tools.trial import transcript
from tests.helpers import TMP

TEXT = '''# 会话记录

- 会话 id：synthetic
- 总轮数：4
- project（标注者填写）：

## [T1] 用户 2026-01-01T00:00:00Z

第一轮

## [T2] 助手 2026-01-01T00:00:01Z

第二轮

> 建议切点：T3 前（每 300 轮）；未自动拆分。

## [T3] 用户 2026-01-01T00:00:02Z

第三轮

## [T4] 助手 2026-01-01T00:00:03Z

第四轮
'''


class SegmentTests(unittest.TestCase):
    def test_split_renumbers_and_keeps_session(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            sample = Path(d) / 'tc01'
            sample.mkdir()
            (sample / 'transcript.md').write_text(TEXT, encoding='utf-8')
            self.assertEqual(segment_dir(sample), ['tc01-p1', 'tc01-p2'])
            self.assertFalse(sample.exists())
            second = (Path(d) / 'tc01-p2' / 'transcript.md').read_text(encoding='utf-8')
            meta, turns = transcript(second)
            self.assertEqual([t['turn'] for t in turns], [1, 2])
            self.assertEqual(turns[0]['text'], '第三轮')
            self.assertIn('第 2/2 段（原 T3~T4）', second)
            self.assertEqual(meta['会话 id'], 'synthetic')
            self.assertEqual(meta['总轮数'], '2')


class SegmentAtUserTurnTests(unittest.TestCase):
    def test_cut_before_assistant_moves_to_next_user_turn(self):
        text = TEXT.replace('> 建议切点：T3 前（每 300 轮）；未自动拆分。\n\n## [T3] 用户', '## [T3] 用户')
        text = text.replace('## [T2] 助手', '> 建议切点：T2 前（每 300 轮）；未自动拆分。\n\n## [T2] 助手')
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            sample = Path(d) / 'tc01'
            sample.mkdir()
            (sample / 'transcript.md').write_text(text, encoding='utf-8')
            self.assertEqual(segment_dir(sample), ['tc01-p1', 'tc01-p2'])
            second = (Path(d) / 'tc01-p2' / 'transcript.md').read_text(encoding='utf-8')
            self.assertIn('（原 T3~T4）', second)   # T2 前的切点顺延到 T3（用户轮）前
            self.assertEqual(transcript(second)[1][0]['role'], '用户')
