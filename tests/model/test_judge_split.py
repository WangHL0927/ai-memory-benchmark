"""合成判定批次与假 CLI；不调用真实模型。"""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json

from tests.helpers import recall, precision, TMP, entry, run, unit
from tests.helpers import sample, fake_cli, cli
from tools.model import judge_split as split
from tools.trial import read_json, write_json


class SplitJudgeTests(unittest.TestCase):
    def args(self, directory, **kwargs):
        return SimpleNamespace(data_dir=directory, force=False, annotator='claude', model='fake-split',
                               reasoning_effort='high', policy=None, timeout=10, cli_path=None, no_cache=True, **kwargs)

    def test_batches_all_memories_and_number_bijection(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            s = sample(Path(d) / 'data')
            g = read_json(s / 'gold.json'); g['U'] = [unit(f'u{i}') for i in range(65)]; write_json(s / 'gold.json', g)
            r = run(*[entry() for _ in range(61)])
            path = s / 'runs/a.json'; write_json(path, r)
            originals = {p: p.read_bytes() for p in (s / 'gold.json', path, s / 'transcript.md')}
            calls = []
            def fake(args, prompt, version, validate, **kw):
                material = json.loads(prompt.split('输入材料 JSON：\n')[1])
                kind = 'recall' if 'units' in material else 'precision'
                calls.append((kind, material))
                if kind == 'recall':
                    self.assertEqual(len(material['memories']), 61)
                    self.assertNotIn('transcript', material)
                    self.assertEqual(set(material['units'][0]), {'unit_id', 'statement', 'conditions', 'category'})
                    output = [recall(u['unit_id']) for u in reversed(material['units'])]
                else:
                    self.assertIn('transcript', material)
                    self.assertNotIn('units', material)
                    self.assertEqual(set(material['memories'][0]), {'memory_id', 'statement'})
                    output = [precision(m['memory_id']) for m in reversed(material['memories'])]
                return validate(output), dict(model='fake-split', attempts=1)
            with patch.object(split, 'cli_json', fake):
                self.assertEqual(split.judge(path, self.args(s.parent)), 'written')
                self.assertEqual(split.judge(path, self.args(s.parent)), 'skipped')
            self.assertEqual([len(m['units']) for k, m in calls if k == 'recall'], [30, 30, 5])
            self.assertEqual([len(m['memories']) for k, m in calls if k == 'precision'], [30, 30, 1])
            result = read_json(s / 'judgments/a.json')
            self.assertEqual([v['unit_id'] for v in result['recall']], [f'u{i}' for i in range(65)])
            self.assertEqual([v['memory_id'] for v in result['precision']], [f'm{i}' for i in range(1, 62)])
            self.assertEqual(len(result['judge']['batches']), 6)
            for p, value in originals.items():
                self.assertEqual(p.read_bytes(), value)
            # 任一输入发生变化时，旧哈希不能导致跳过。
            for changed in originals:
                changed.write_bytes(originals[changed] + b'\n')
                with patch.object(split, 'cli_json', fake):
                    self.assertEqual(split.judge(path, self.args(s.parent)), 'written')
                changed.write_bytes(originals[changed])
            args = self.args(s.parent); args.force = True
            with patch.object(split, 'cli_json', fake):
                self.assertEqual(split.judge(path, args), 'written')

    def test_validate_rejects_missing_duplicate_extra_ids_and_bad_types(self):
        bad = [[], [recall('other')], [dict(recall(), extra=1)], [recall(refs=['m2'])],
               [recall(refs=['m1', 'm1'])], [recall(verdict='unknown')], [recall(refs=[])],
               [recall(verdict='none', refs=['m1'])]]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                split.validate_batch(value, ['u1'], 'recall', ['m1'])
        with self.assertRaises(ValueError):
            split.validate_batch([recall(), recall()], ['u1', 'u2'], 'recall', ['m1'])
        for value in ([precision('extra')], [dict(precision(), distorted='false')], [dict(precision(), grounded='unclear')],
                      [dict(precision(), worth=True)], [dict(precision(), note=[])], [precision(), precision()]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                split.validate_batch(value, ['m1'], 'precision')

    def test_claim_counts_errors_and_verbatim_validation(self):
        valid = precision(claims=5, unsupported=1, partial=1, distorted=True, distorted_claims=1, verbatim=True,
                          errors=[dict(text='错误陈述', type='distorted', reason='改变原意')])
        self.assertEqual(split.validate_batch([valid], ['m1'], 'precision'), [valid])
        for field in split.CLAIM_FIELDS:
            for invalid in (-1, True, 1.5, '1', None):
                value = dict(valid, **{field: invalid})
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    split.validate_batch([value], ['m1'], 'precision')
        bad = [dict(valid, claims=2), dict(valid, verbatim=1),
               precision(claims=0, verbatim=True), dict(valid, errors={}),
               dict(valid, errors=[dict(text='x', type='partial', reason='不应列为错误')]),
               dict(valid, errors=[dict(text='', type='unsupported', reason='缺少依据')]),
               dict(valid, errors=[dict(text='x', type='unsupported', reason='')]),
               dict(valid, errors=[dict(text='x', type='unsupported', reason='a', extra=1)]),
               dict(valid, errors=[dict(text='x', type='unsupported', reason='a')]*2),
               dict(valid, errors=[dict(text='x', type='distorted', reason='a')]*2)]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                split.validate_batch([value], ['m1'], 'precision')
        truncated = precision(claims=20, unsupported=20, errors=[dict(text=str(i), type='unsupported', reason='无依据') for i in range(10)])
        split.validate_batch([truncated], ['m1'], 'precision')
        long = precision(claims=20, unsupported=20, errors=[dict(text=str(i), type='unsupported', reason='无依据') for i in range(12)])
        self.assertEqual(len(split.validate_batch([long], ['m1'], 'precision')[0]['errors']), 10)  # 超过 10 项截取，不判失败
        split.validate_batch([precision(claims=0)], ['m1'], 'precision')
        legacy = {k: v for k, v in precision().items() if k not in (*split.CLAIM_FIELDS, 'verbatim', 'errors')}
        with self.assertRaisesRegex(ValueError, '字段不符'):
            split.validate_batch([legacy], ['m1'], 'precision')

    def test_precision_template_change_reuses_recall_cache_only(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); s = sample(root / 'data'); path = s / 'runs/a.json'; write_json(path, run(entry()))
            program, state = fake_cli(root, [[recall()], [precision()], [precision(claims=2)]])
            args = self.args(s.parent); args.cli_path = str(program); args.no_cache = False
            actual = split.prompt_file
            with patch('tools.cache.CACHE_ROOT', root / 'cache'):
                split.judge(path, args)
                result = read_json(s / 'judgments/a.json')
                # Simulate an existing legacy judgment lacking v4 fields; it must not be skipped.
                result['precision'] = [{k: v for k, v in item.items() if k not in (*split.CLAIM_FIELDS, 'verbatim', 'errors')} for item in result['precision']]
                write_json(s / 'judgments/a.json', result)
                with patch.object(split, 'prompt_file', side_effect=lambda name: actual(name) + ('\n新版精确提示词' if name == 'judge-precision' else '')):
                    split.judge(path, args)
            result = read_json(s / 'judgments/a.json')
            self.assertEqual(len(read_json(state)), 3)
            self.assertEqual([b['identity']['cache'] for b in result['judge']['batches']], ['hit', 'miss'])
            self.assertEqual(result['precision'][0]['claims'], 2)
            self.assertEqual(result['judge']['prompt_versions'], dict(recall='model-judge-recall-v1', precision='model-judge-precision-v4'))

    def test_cli_retry_raw_identity_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); s = sample(root / 'data')
            path = s / 'runs/a.json'; write_json(path, run(entry()))
            program, state = fake_cli(root, [[recall('wrong')], [recall()], [precision('bad')], [precision()]])
            args = self.args(s.parent); args.cli_path = str(program)
            self.assertEqual(split.judge(path, args), 'written')
            result = read_json(s / 'judgments/a.json')
            self.assertEqual([b['identity']['attempts'] for b in result['judge']['batches']], [2, 2])
            self.assertEqual(len(read_json(state)), 4)
            for b in result['judge']['batches']:
                identity = b['identity']
                self.assertEqual(identity['model'], 'fake-split')
                self.assertEqual(identity['cli_version'], 'fake-cli 0.0')
                self.assertEqual(len(identity['raw_outputs']), 2)
            self.assertNotIn('transcript', json.loads(read_json(state)[0]['prompt'].split('输入材料 JSON：\n')[1]))

    def test_failure_atomic_and_main_filter(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); s = sample(root / 'data')
            for name in ('a', 'b'):
                write_json(s / f'runs/{name}.json', run(entry()))
            old = dict(unchanged=True); target = s / 'judgments/a.json'; write_json(target, old)
            program, state = fake_cli(root, [[recall()], []])
            result = cli('tools.model.judge_split', '--data-dir', s.parent, '--systems', 'a', '--annotator', 'claude',
                         '--model', 'fake', '--cli-path', program, '--no-cache', '--failures', root / 'failures.json')
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(read_json(target), old)
            self.assertFalse((s / 'judgments/b.json').exists())
            self.assertEqual(read_json(root / 'failures.json')['failed'], 1)
            self.assertEqual(read_json(root / 'failures.json')['failures'][0]['attempts'], 2)
            self.assertEqual(len(read_json(state)), 3)

    def test_split_batches_reuse_content_cache_and_no_cache_bypasses_it(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); s = sample(root / 'data')
            path = s / 'runs/a.json'; write_json(path, run(entry()))
            program, state = fake_cli(root, [[recall()], [precision()], [recall()], [precision()]])
            args = self.args(s.parent); args.cli_path = str(program); args.no_cache = False
            split.judge(path, args)
            self.assertEqual(len(read_json(state)), 2)
            args.force = True
            split.judge(path, args)
            result = read_json(s / 'judgments/a.json')
            self.assertEqual([b['identity']['cache'] for b in result['judge']['batches']], ['hit', 'hit'])
            self.assertEqual(len(read_json(state)), 2)
            args.no_cache = True
            split.judge(path, args)
            self.assertEqual(len(read_json(state)), 4)

    def test_mutated_input_during_call_and_empty_input(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            s = sample(Path(d) / 'data'); path = s / 'runs/a.json'; write_json(path, run(entry()))
            def mutate(args, prompt, version, validate, **kw):
                (s / 'transcript.md').write_text('等待期间改变的会话', encoding='utf-8')
                return validate([recall()] if 'recall' in version else [precision()]), {}
            with patch.object(split, 'cli_json', mutate), self.assertRaisesRegex(ValueError, '等待期间'):
                split.judge(path, self.args(s.parent))
            self.assertFalse((s / 'judgments/a.json').exists())
            g = read_json(s / 'gold.json'); g['U'] = []; write_json(s / 'gold.json', g); write_json(path, run())
            with patch.object(split, 'cli_json', side_effect=AssertionError('空输入不调用模型')):
                self.assertEqual(split.judge(path, self.args(s.parent)), 'written')
            result = read_json(s / 'judgments/a.json'); self.assertEqual(result['recall'], []); self.assertEqual(result['precision'], [])
