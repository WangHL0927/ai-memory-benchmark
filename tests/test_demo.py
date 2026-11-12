"""README 合成样例可直接评分和渲染；不调用任何模型。"""
from pathlib import Path
import tempfile
import unittest

from tests.helpers import FIXTURES, TMP, cli
from tools.trial import read_json


class DemoTests(unittest.TestCase):
    def test_demo_split_score_and_html(self):
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            output = Path(directory) / 'report'
            result = cli('tools.score_split', '--data-dir', FIXTURES / 'evaluation', '--systems', 'alpha', 'beta',
                         '--bootstrap', 20, '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = read_json(output.with_suffix('.json'))
            self.assertEqual(report['sample_count'], 2)
            self.assertEqual(report['failure_count'], 0)
            systems = report['datasets']['synthetic']['systems']
            self.assertEqual(set(systems), {'alpha', 'beta'})
            self.assertGreater(systems['beta']['overall']['metrics']['miss_rate']['value'],
                               systems['alpha']['overall']['metrics']['miss_rate']['value'])
            result = cli('tools.report_split_html', output.with_suffix('.json'), '--output', output.with_suffix('.html'))
            self.assertEqual(result.returncode, 0, result.stderr)
            html = output.with_suffix('.html').read_text()
            self.assertIn('未经人工校准', html)
            self.assertIn('漏记率', html)
            self.assertIn('误记率', html)
