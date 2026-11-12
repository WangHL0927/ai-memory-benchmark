"""拆分指标、缺失样本、会话聚类及新报告的合成测试。"""
from pathlib import Path
import tempfile
import unittest

from tests.helpers import TMP, unit, entry
from tests.helpers import recall, precision, fixture
from tests.helpers import sample, cli
from tools.trial import read_json, write_json
from tools import score_split as scoring
from tools.report_split_html import render


class SplitScoreTests(unittest.TestCase):
    def test_metrics_uncertain_added_and_independent_precision(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            units = [unit('full'), unit('partial'), unit('none'), unit('distorted'), unit('maybe', uncertain=True), unit('added', origin='added_by_run:a')]
            verdicts = [recall(u['unit_id'], v) for u, v in zip(units, ['full', 'partial', 'none', 'distorted', 'full', 'none'])]
            ps = [precision('m1', 'yes', False, 'no'), precision('m2', 'no', True), precision('m3', 'partial'), precision('m4', 'yes', True)]
            e = fixture(Path(d), units=units, verdicts=verdicts, precision_values=ps)
            c = scoring.counts(e); m = scoring.summarize([e], 20)['metrics']
            self.assertEqual(c['units'], 4); self.assertEqual(c['units_inclusive'], 5)
            self.assertEqual(m['miss_rate']['value'], .75)
            self.assertEqual(m['miss_rate_inclusive']['value'], .6)
            self.assertEqual(m['uncovered_rate']['value'], .5)
            self.assertEqual(m['partial_rate']['value'], .25)
            self.assertEqual(m['false_memory_rate']['value'], .5)  # 无依据且写歪只算一次；不值得记不算误记。
            self.assertEqual(m['insufficient_rate']['value'], .25)
            self.assertEqual(m['not_worth_rate']['value'], .25)
            self.assertEqual(m['memories_per_sample']['value'], 4)
            self.assertEqual(m['chars_per_memory']['value'], len(entry()['draft']['statement']))
            self.assertEqual(m['chars_per_sample']['value'], 4 * len(entry()['draft']['statement']))
            self.assertEqual(m['miss_rate']['ci95'], [.75, .75])
            self.assertNotIn('f1', m)

    def test_claim_weighting_priority_verbatim_and_memory_auxiliary_metric(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            # One large memory: 10 claims incl. one verbatim block; two erroneous claims.
            # A distorted unsupported claim belongs only to distorted_claims, not unsupported as well.
            large = precision(grounded='partial', distorted=True, claims=10, unsupported=1, partial=2,
                              distorted_claims=1, verbatim=True, errors=[
                                  dict(text='无依据的计划', type='unsupported', reason='会话中未提出'),
                                  dict(text='把局部许可扩大', type='distorted', reason='无依据且改变原意，优先归写歪')])
            small = precision('m2', grounded='partial', claims=2, partial=1)
            e = fixture(Path(d), precision_values=[large, small])
            c = scoring.counts(e); m = scoring.summarize([e], 10)['metrics']
            self.assertEqual(c['claims'], 12)
            self.assertEqual(c['claim_errors'], 2)
            self.assertEqual(c['partial_claims'], 3)
            self.assertEqual(m['false_memory_rate']['numerator'], 2)
            self.assertEqual(m['false_memory_rate']['denominator'], 12)
            self.assertAlmostEqual(m['false_memory_rate']['value'], 1/6)
            self.assertEqual(m['false_memory_rate_by_memory']['value'], .5)
            self.assertEqual(m['claims_per_sample']['value'], 12)
            self.assertEqual(m['verbatim_memory_share']['value'], .5)
            self.assertEqual(m['false_memory_rate']['ci95'], [1/6, 1/6])
            # Structural-only memory has no claims: claim rate undefined, old denominator remains one memory.
            empty = fixture(Path(d), precision_values=[precision(claims=0)])
            m = scoring.summarize([empty], 5)['metrics']
            self.assertIsNone(m['false_memory_rate']['value'])
            self.assertEqual(m['false_memory_rate_by_memory']['value'], 0)
            self.assertEqual(m['claims_per_sample']['value'], 0)
            excerpt = fixture(Path(d), precision_values=[precision(claims=1, verbatim=True)])
            excerpt['memories'][0]['statement'] = '# ChatLog\n用户：用蓝色水管。\n助手：好的。\n用户：只在冬季使用。'
            m = scoring.summarize([excerpt], 5)['metrics']
            self.assertEqual(m['claims_per_sample']['value'], 1)  # Whole excerpt is one claim, never split by the scorer.
            self.assertEqual(m['verbatim_memory_share']['value'], 1)

    def test_claim_rate_bootstrap_clusters_segments_with_unequal_denominators(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d)
            es = [fixture(root, 'a', session='same', precision_values=[precision(claims=10, unsupported=2)]),
                  fixture(root, 'b', session='same', precision_values=[precision(claims=2)]),
                  fixture(root, 'c', session='other', precision_values=[precision(grounded='no')])]
            summary = scoring.summarize(es, 200, 123)
            self.assertEqual(summary['clusters'], 2)
            metric = summary['metrics']['false_memory_rate']
            self.assertEqual((metric['numerator'], metric['denominator']), (3, 13))
            self.assertAlmostEqual(metric['value'], 3/13)
            self.assertEqual(metric['ci95'], [1/6, 1])
            self.assertEqual(summary, scoring.summarize(list(reversed(es)), 200, 123))

    def test_report_header_aggregates_runs_without_exposing_hashes(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d)
            metadata = dict(name='alpha', llm='fixed-model', prompt_version='产品原生模板',
                            prompt_sha256={'sample-template': 'PROMPT_HASH_SENTINEL'},
                            usage=dict(llm=dict(calls=3), embedding=dict(calls=2)))
            fixture(root, 'a', metadata=metadata); fixture(root, 'b', metadata=metadata)
            report = scoring.build_report(root, iterations=5)
            self.assertEqual(len(report['systems']['alpha']), 1)  # dedup is okay for raw metadata, not totals.
            self.assertEqual(report['system_summary']['alpha']['llm_calls'], '6')
            self.assertEqual(report['system_summary']['alpha']['embedding_calls'], '4')
            self.assertIn('PROMPT_HASH_SENTINEL', str(report['systems']))
            for rendered in (render(report), scoring.markdown(report)):
                self.assertNotIn('PROMPT_HASH_SENTINEL', rendered)
                self.assertIn('提示词来源 产品原生模板', rendered)
                self.assertEqual(rendered.count('调用合计 LLM 6 / embedding 4'), 1)
                self.assertIn('误记率（按陈述）', rendered)
                self.assertIn('每段陈述数', rendered)
            self.assertEqual(report['false_memory_unit'], 'claim')
            self.assertEqual(report['precision_version'], 'model-judge-precision-v4')

    def test_legacy_precision_is_excluded_not_inferred_as_one_claim(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); fixture(root)
            path = root / 's001/judgments/alpha.json'; judgment = read_json(path)
            judgment['precision'] = [{k: v for k, v in p.items() if k in ('memory_id', 'grounded', 'distorted', 'worth', 'note')} for p in judgment['precision']]
            write_json(path, judgment)
            report = scoring.build_report(root, iterations=5)
            self.assertEqual(report['failure_count'], 1)
            self.assertEqual(report['failures'][0]['code'], 'invalid_judgment')
            self.assertIsNone(report['datasets']['synthetic']['systems']['alpha']['overall']['metrics']['false_memory_rate']['value'])

    def test_failures_missing_stale_and_no_denominator(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d)
            fixture(root, 'valid')
            fixture(root, 'stale'); (root / 'stale/transcript.md').write_text('改变过的会话')
            fixture(root, 'unjudged'); (root / 'unjudged/judgments/alpha.json').unlink()
            fixture(root, 'replay'); r = read_json(root / 'replay/runs/alpha.json'); r['status'] = 'failed'; r['reason'] = '无法回放'; write_json(root / 'replay/runs/alpha.json', r)
            sample(root, 'missing')
            report = scoring.build_report(root, ['alpha', 'absent'], iterations=20)
            system = report['datasets']['synthetic']['systems']['alpha']
            self.assertEqual(system['overall']['counts']['samples'], 1)
            self.assertEqual(system['overall']['metrics']['miss_rate']['denominator'], 1)
            self.assertEqual(system['failed_count'], 4)
            self.assertEqual(report['failure_count'], 9)
            self.assertEqual(report['failure_reason_counts'], dict(invalid_judgment=1, missing_judgment=1, replay_failed=1, missing_run=6))
            self.assertIn('无法回放', str(report['failures']))
            empty = report['datasets']['synthetic']['systems']['absent']['overall']
            self.assertIsNone(empty['metrics']['miss_rate']['value']); self.assertIsNone(empty['metrics']['false_memory_rate']['value'])
            self.assertEqual(empty['metrics']['miss_rate']['ci95'], [None, None])

    def test_empty_u_and_empty_d(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d)
            e = fixture(root, units=[], verdicts=[], precision_values=[precision()])
            m = scoring.summarize([e], 10)['metrics']
            self.assertIsNone(m['miss_rate']['value']); self.assertEqual(m['false_memory_rate']['value'], 0)
            e = fixture(root, precision_values=[], verdicts=[recall(verdict='none')])
            m = scoring.summarize([e], 10)['metrics']
            self.assertEqual(m['miss_rate']['value'], 1); self.assertIsNone(m['false_memory_rate']['value'])
            self.assertIsNone(m['chars_per_memory']['value']); self.assertEqual(m['chars_per_sample']['value'], 0)

    def test_cluster_bootstrap_is_reproducible_and_uses_session_not_segment(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d)
            es = [fixture(root, 'a', session='same', verdicts=[recall(verdict='none')]),
                  fixture(root, 'b', session='same'), fixture(root, 'c', session='other')]
            a = scoring.summarize(es, 200, 123)
            self.assertEqual(a, scoring.summarize(list(reversed(es)), 200, 123))
            self.assertEqual(a['clusters'], 2)
            self.assertEqual(a['metrics']['miss_rate']['ci95'], [0, .5])
            self.assertEqual(a['metrics']['miss_rate']['bootstrap_valid'], 200)
            self.assertEqual(a['metrics']['miss_rate']['numerator'], 1)

    def test_report_cli_html_identities_differences_and_basic_memory_separation(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); data = root / 'data'
            for system, metadata in [('alpha', dict(name='claude-mem', llm='fixed-model', parameters=dict(temperature=1), isolation=dict(chroma=False, vector_search=False))),
                                     ('basic-memory', dict(name='Basic Memory', llm='fixed-model', configuration='固定 agent'))]:
                fixture(data, system=system, metadata=metadata)
            # fixture 会写同一份 gold，内容保持相同，不影响输入哈希。
            result = cli('tools.score_split', '--data-dir', data, '--bootstrap', 10, '--output', root / 'report')
            self.assertEqual(result.returncode, 0, result.stderr)
            report = read_json(root / 'report.json')
            self.assertEqual(report['datasets']['synthetic']['systems']['basic-memory']['comparison_group'], 'agent_combination')
            self.assertIn('chroma', str(report['native_differences']))
            html = render(report); md = (root / 'report.md').read_text()
            for content in (html, md):
                for text in ('未经人工校准', 'Basic Memory + 固定 agent 的组合', '产品与原生流程的差异', 'fixed-model', '每条平均字符数', scoring.WORTH_NOTICE):
                    self.assertIn(text, content)
                self.assertNotIn('单元级 F1', content)
            self.assertNotIn('<script src', html); self.assertNotIn('<link ', html)
            html_result = cli('tools.report_split_html', root / 'report.json', '--output', root / 'report.html')
            self.assertEqual(html_result.returncode, 0, html_result.stderr)
            self.assertEqual((root / 'report.html').read_text(), html)
            attached = scoring.build_report(data, iterations=5, agreement=dict(raw_agreement=.9))
            self.assertEqual(attached['agreement']['raw_agreement'], .9)
            with self.assertRaises(ValueError):
                scoring.build_report(data, iterations=0)

    def test_mixed_policy_versions_rejected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); fixture(root, 'a'); fixture(root, 'b')
            path = root / 'b/gold.json'; g = read_json(path); g['standard_version'] = 'v-another'; write_json(path, g)
            with self.assertRaisesRegex(ValueError, '混用'):
                scoring.build_report(root, iterations=5)
