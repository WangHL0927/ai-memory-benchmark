from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tests.helpers import TMP, unit
from tests.helpers import sample, cli, fake_cli
from tools import score
from tools.model.annotate import annotate
from tools.trial import read_json
from types import SimpleNamespace


class ModelTests(unittest.TestCase):
    def test_annotate_frozen_valid_and_no_runs_read(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); s = sample(root / 'data', with_gold=False)
            (s / 'runs').mkdir(); (s / 'runs/forbidden.json').write_text('SECRET_RUN_SENTINEL')
            program, state = fake_cli(root, [dict(project=['garden'], U=[unit()], N=[])])
            args = SimpleNamespace(data_dir=s.parent, policy=None, layer='random', annotator='codex',
                                   model='fake-snapshot', cli_path=str(program), reasoning_effort='high', timeout=10)
            real_read = Path.read_text
            def guarded(path, *a, **kw):
                if 'runs' in path.parts:
                    raise AssertionError('盲标不得读取 runs')
                return real_read(path, *a, **kw)
            with patch.object(Path, 'read_text', guarded):
                self.assertEqual(annotate(s, args), 'written')
            gold = read_json(s / 'gold.json'); score.validate_gold(gold)
            self.assertEqual(gold['blind_snapshot'], dict(U=gold['U'], N=gold['N']))
            for field in ('frozen_at', 'frozen_policy', 'frozen_project'):
                self.assertIn(field, gold)
            self.assertEqual(gold['annotator']['model'], 'fake-snapshot')
            invocation = read_json(state)[0]
            self.assertNotIn('SECRET_RUN_SENTINEL', invocation['prompt'])
            self.assertNotIn(str(s), invocation['prompt'])
            self.assertIn('model_reasoning_effort=high', invocation['args'])
            self.assertIn('read-only', invocation['args'])
            self.assertNotEqual(invocation['cwd'], str(s))
            before = (s / 'gold.json').read_bytes()
            self.assertEqual(annotate(s, args), 'skipped')
            self.assertEqual(before, (s / 'gold.json').read_bytes())

    def test_annotation_retry_validation_failure_and_continue(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); a = sample(root / 'data', 'a', False); b = sample(root / 'data', 'b', False)
            invalid = unit(); invalid['evidence'][0]['quote'] = '原文中不存在'
            program, state = fake_cli(root, ['非 JSON', dict(project=['garden'], U=[invalid], N=[]),
                                              dict(project=['garden'], U=[unit()], N=[])])
            result = cli('tools.model.annotate', '--data-dir', a.parent, '--annotator', 'claude', '--model', 'fake',
                         '--cli-path', program, '--failures', root / 'failures.json')
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse((a / 'gold.json').exists())
            score.validate_gold(read_json(b / 'gold.json'))
            self.assertEqual(len(read_json(state)), 3)
            self.assertIn('--tools', read_json(state)[0]['args'])
            self.assertEqual(read_json(root / 'failures.json')['failed'], 1)


class RepairTests(unittest.TestCase):
    def test_unescape_turn_fix_and_drop(self):
        from tools.model.annotate import repair_units
        turns = [dict(turn=1, text='工具调用：Write {"content": "x = \\"a\\"\\nprint(x)"}'), dict(turn=2, text='用户确认：就用 A 方案')]
        units = [dict(unit_id='u1', evidence=[dict(turn=1, quote='x = "a"\nprint(x)')]),     # JSON 转义还原后能对上
                 dict(unit_id='u2', evidence=[dict(turn=1, quote='就用 A 方案')]),              # 轮次错了，唯一出现在 T2
                 dict(unit_id='u3', evidence=[dict(turn=2, quote='原文里没有的话')])]           # 对不上，删除，单元也删
        log = dict(evidence_total=0, turn_fixed=0, evidence_dropped=0, units_dropped=[])
        kept = repair_units(units, turns, log)
        self.assertEqual([u['unit_id'] for u in kept], ['u1', 'u2'])
        self.assertEqual(kept[1]['evidence'][0]['turn'], 2)
        self.assertEqual((log['turn_fixed'], log['evidence_dropped'], log['units_dropped']), (1, 1, ['u3']))
