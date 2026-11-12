"""以合成数据验证选型比较按 id 配对、共用样本计算指标。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.helpers import TMP, fixture, recall, precision
from tools import judge_select as select
from tools.trial import read_json, write_json


class JudgeSelectTests(unittest.TestCase):
    def test_compare_pairs_and_agreement(self):
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            root = Path(directory); source = root / 'source'; target = root / 'selection'
            fixture(source, 'a', precision_values=[precision(), precision('m2', 'no')])
            fixture(source, 'missing')
            for sample in source.iterdir():
                (sample / 'judgments').rename(sample / 'judgments-opus')
            with patch.object(select, 'SELECT_ROOT', target):
                self.assertEqual(select.setup(source, 'alpha', 'opus', ['candidate']), 2)
                sample = target / 'candidate/data/a'
                reference = read_json(source / 'a/judgments-opus/alpha.json')
                other = dict(reference, recall=[recall(verdict='none')], precision=[precision('m2', 'yes'), precision('m1')])
                write_json(sample / 'judgments/alpha.json', other)
                rows = select.compare(source, 'alpha', 'opus')
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual((row['samples'], row['of']), (1, 2))
            self.assertEqual(row['recall']['raw_agreement'], 0)
            self.assertEqual(row['grounded']['raw_agreement'], .5)
            self.assertEqual(row['error']['raw_agreement'], .5)
            self.assertEqual(row['miss_rate'], (1, 0))
            self.assertEqual(row['false_memory_rate'], (0, .5))
            self.assertIn('candidate', select.markdown(rows, 'opus'))
            self.assertEqual(select.pair([{'id': 'b', 'v': 2}, {'id': 'a', 'v': 1}],
                                         [{'id': 'a', 'v': 9}, {'id': 'c', 'v': 3}], 'id', 'v'), ([1], [9]))

    def test_legacy_reference_keeps_memory_rate_and_has_no_claim_rate(self):
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            root = Path(directory); source = root / 'source'; target = root / 'selection'
            fixture(source, precision_values=[precision(grounded='no')])
            sample = source / 's001'; (sample / 'judgments').rename(sample / 'judgments-opus')
            path = sample / 'judgments-opus/alpha.json'; reference = read_json(path)
            reference['precision'] = [{k: v for k, v in p.items() if k in ('memory_id', 'grounded', 'distorted', 'worth', 'note')} for p in reference['precision']]
            write_json(path, reference)
            with patch.object(select, 'SELECT_ROOT', target):
                select.setup(source, 'alpha', 'opus', ['v4'])
                other = dict(reference, precision=[precision(claims=10, unsupported=1, grounded='partial')])
                write_json(target / 'v4/data/s001/judgments/alpha.json', other)
                row = select.compare(source, 'alpha', 'opus')[0]
            self.assertEqual(row['false_memory_rate_by_memory'], (0, 1))
            self.assertEqual(row['false_memory_rate'], (.1, None))
            self.assertEqual(row['error']['raw_agreement'], 0)
            self.assertIn('按陈述', select.markdown([row], 'opus'))
