"""发布工具测试：全部输入均为合成数据，公开目录位于隔离临时仓库。"""
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.helpers import ROOT, TMP, gold, unit, entry, run, recall, precision
from tools.trial import read_json
from tools.model.judge_split import input_hashes, validate_judgment, memories
from tools import score_split
from tools import publish, score


class ScanRuleTests(unittest.TestCase):
    def kinds(self, text):
        return {kind for kind, _, _ in publish.detections(text)}

    def test_each_rule_has_positive_and_negative(self):
        # 每条规则分别验证正常格式和相似但不应命中的文本。
        cases = [
            ('email', 'person@example.org', 'person at example dot org'),
            ('phone', '+86 13800138000', '版本 2026-11-12'),
            ('ipv4', '192.168.1.20', '999.999.999.999'),
            ('ipv6', '2001:db8::7', '2001:db8:xyz::7'),
            ('url_credentials', 'https://user:pass@example.org/path?view=one', 'https://example.org/docs?lang=zh'),
            ('secret_token', 'sk-' + 'a' * 30, 'sk-short'),
            ('secret_assignment', 'api_key=Abcd1234Sensitive', 'api_key=[PLACEHOLDER]'),
            ('jwt', 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl', 'eyJub3QiOiJhbGciIn0.data.signature'),
            ('private_key', '-----BEGIN RSA PRIVATE KEY-----\nYWJj\n-----END RSA PRIVATE KEY-----', '-----BEGIN PUBLIC KEY-----\nYWJj\n-----END PUBLIC KEY-----'),
            ('wallet', '0x' + 'ab' * 20, '0x' + 'ab' * 19),
            ('cloud_id', 'prj_ABCdef123456', 'prj_'),
            ('user_path', '/Users/example/work', '/Users/<用户名>/work'),
        ]
        self.assertEqual({k for k, _, _ in cases}, {k for k, _, _ in publish.RULES})
        for kind, positive, negative in cases:
            with self.subTest(kind=kind, positive=True):
                self.assertIn(kind, self.kinds(positive))
            with self.subTest(kind=kind, positive=False):
                self.assertNotIn(kind, self.kinds(negative))

    def test_platform_and_address_variants(self):
        cases = {
            'phone': ['13800138000', '+1 (415) 555-0134', '415-555-0134'],
            'ipv6': ['::1', 'fe80::1%en0', '2001:db8:0:0:0:0:0:1', '::ffff:192.0.2.1'],
            'url_credentials': ['https://example.org/?token=opaque123', 'https://example.org/?api_key=opaque123', 'https://example.org/?%74oken=opaque123', 'ssh://username@example.org/repo', 'mongodb://user:pass@example.org/db', 'https://example.org/?token_id=opaque123'],
            'secret_token': ['ghp_' + 'a'*36, 'github_pat_' + 'a'*50, 'AKIA' + 'A'*16, 'ASIA' + 'A'*16, 'xoxb-1234567890-abcdefghijkl', 'AIza' + 'a'*35],
            'private_key': ['-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----', '-----BEGIN ENCRYPTED PRIVATE KEY-----\nabc\n-----END ENCRYPTED PRIVATE KEY-----'],
            'wallet': ['1BoatSLRHtKNngkdXEeobR76b53LETtpyT', 'bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kygt080', 'T' + 'a'*33],
            'cloud_id': ['team_abcdef12345', 'org_abcdefgh123', 'org-Abcd123456'],
            'user_path': ['/home/alice/project', r'C:\Users\alice\work'],
        }
        for kind, texts in cases.items():
            for text in texts:
                with self.subTest(kind=kind, text=text):
                    self.assertIn(kind, self.kinds(text))

    def test_ordinary_text_has_no_hits(self):
        self.assertEqual(self.kinds('花园项目采用蓝色水管。版本 v1.2.3，日期 2026-11-12；https://example.org/docs。'), set())

    def test_sentence_punctuation_and_json_credentials(self):
        for kind, text in [('email', 'person@example.org.'), ('ipv4', '192.168.1.20.'), ('ipv6', '2001:db8::7.')]:
            with self.subTest(kind=kind):
                self.assertIn(kind, self.kinds(text))
        content = publish.json_bytes({'parameters': {'api_key': 'OpaqueSecret0123', 'password': '[PASSWORD_1]'}})
        hits = publish.scan_files({'s001/runs/test.json': content})
        self.assertEqual([(h['type'], h['value']) for h in hits], [('secret_assignment', 'OpaqueSecret0123')])

    def test_json_escapes_context_and_position(self):
        text = '甲' * 40 + r'C:\Users\alice' + '/' + '乙' * 40
        raw = json.dumps({'note': text}, ensure_ascii=True).encode()
        hits = publish.scan_files({'s001/gold.json': raw})
        hit = next(h for h in hits if h['type'] == 'user_path')
        self.assertEqual(hit['value'], r'C:\Users\alice')
        self.assertEqual(hit['before'], '甲' * 30)
        self.assertEqual(hit['after'], '/' + '乙' * 29)
        self.assertEqual(hit['location']['pointer'], '/note')
        self.assertEqual(hit['location']['start'], 40)
        self.assertEqual(hit['location']['column'], 41)
        private = {'key': '-----BEGIN PRIVATE KEY-----\nYWJj\n-----END PRIVATE KEY-----'}
        self.assertIn('private_key', {h['type'] for h in publish.scan_files({'s001/runs/test.json': publish.json_bytes(private)})})


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='publish-test-', dir=TMP)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.patch = patch.object(publish, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        (self.root / 'datasets').mkdir()
        for name in ('trace-commons', 'halumem'):
            (self.root / 'datasets' / (name + '.json')).write_bytes((ROOT / 'datasets' / (name + '.json')).read_bytes())
        self.data = self.root / 'tmp/input'
        self.sample = self.data / 's001'
        (self.sample / 'runs').mkdir(parents=True)
        self.report_path = self.root / 'tmp/report.json'
        self.folder = self.root / 'tmp/publish/trace-commons'
        self.create_sample('花园项目联系 person@example.org，今后用蓝色水管。')

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(publish.json_bytes(value))

    def create_sample(self, quote):
        u = unit()
        u['statement'] = quote
        u['evidence'] = [dict(turn=1, quote=quote)]
        g = gold([u])
        g['turns'] = 1
        g['dataset'] = 'trace-commons'
        g['blind_snapshot'] = copy.deepcopy({'U': g['U'], 'N': g['N']})
        e = entry()
        e['draft']['statement'] = quote
        e['draft']['evidence'] = [dict(turn=1, quote=quote)]
        self.write_json(self.sample / 'gold.json', g)
        self.write_json(self.sample / 'runs/test.json', run(e))
        (self.sample / 'transcript.md').write_text('# 合成会话\n\n## [T1] 用户\n\n' + quote + '\n', encoding='utf-8')
        self.refresh_report()

    def refresh_report(self):
        for path in self.data.glob('*/runs/*.json'):
            sample = path.parent.parent
            g = read_json(sample / 'gold.json')
            self.write_json(sample / 'judgments' / path.name, dict(
                recall=[dict(recall(u['unit_id']), note=u['statement']) for u in g['U']],
                precision=[dict(precision(), note=g['U'][0]['statement'])],
                judge=dict(tool='synthetic', model='fake-judge'), **input_hashes(sample, path)))
        report = score_split.build_report(self.data, iterations=3)
        self.write_json(self.report_path, report)
        return report

    def scan(self):
        return publish.scan(self.data, 'trace-commons')

    def redact(self):
        result = self.scan()
        entries = []
        values = {}
        for hit in result['hits']:
            value = values.setdefault((hit['sample'], hit['value']), f'[REDACTED_{len(values)+1}]')
            entries.append({'id': hit['id'], 'replacement': value})
        self.write_json(self.folder / 'redactions.json', {'entries': entries})
        return result

    def export(self, force=False):
        return publish.export(self.data, 'trace-commons', 'eval-v1', self.report_path, force)

    def test_scan_is_read_only_and_lists_all_artifacts(self):
        before = publish.inventory(self.data)
        result = self.scan()
        self.assertEqual(publish.inventory(self.data), before)
        self.assertTrue((self.folder / 'scan.md').exists())
        self.assertEqual(result['files'], {p: publish.digest(b) for p, b in before.items()})
        self.assertEqual({h['file'] for h in result['hits']}, set(before))
        self.assertTrue(all(h['sample'] == 's001' for h in result['hits']))

    def test_scan_report_symlink_cannot_modify_source(self):
        self.folder.mkdir(parents=True)
        source = self.sample / 'transcript.md'
        before = source.read_bytes()
        (self.folder / 'scan.json').symlink_to(source)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            self.scan()
        self.assertEqual(source.read_bytes(), before)

    def test_missing_scan_refused(self):
        with self.assertRaisesRegex(ValueError, '先运行 scan'):
            self.export()
        self.assertFalse((self.root / 'reports').exists())

    def test_unhandled_hits_refused_without_echoing_secrets(self):
        result = self.scan()
        with self.assertRaisesRegex(ValueError, '未处理') as error:
            self.export()
        self.assertIn(result['hits'][0]['id'], str(error.exception))
        self.assertNotIn('person@example.org', str(error.exception))
        self.assertFalse((self.root / 'reports').exists())
        self.assertFalse((self.root / 'data').exists())

    def test_redaction_preserves_evidence_and_snapshots_and_source(self):
        self.redact()
        before = publish.inventory(self.data)
        manifest = self.export()
        target = self.root / 'data/trace-commons/default-v1/s001'
        g = read_json(target / 'gold.json')
        r = read_json(target / 'runs/test.json')
        text = (target / 'transcript.md').read_text()
        self.assertIn('[REDACTED_1]', text)
        self.assertNotIn('person@example.org', text)
        self.assertIn(g['U'][0]['evidence'][0]['quote'], text)
        self.assertEqual(g['U'][0]['evidence'], r['D'][0]['draft']['evidence'])
        score.validate_gold(g)
        judgment = read_json(target / 'judgments/test.json')
        validate_judgment(judgment, g, memories(r), input_hashes(target, target / 'runs/test.json'))
        self.assertEqual(publish.inventory(self.data), before)
        report_dir = self.root / 'reports/eval-v1'
        prov = (report_dir / 'PROVENANCE.md').read_text()
        for required in ('112ebd4d', 'CC-BY-4.0', 'Trace Commons contributors', '政策版本', '框架版本', '标注者', '判定者', '模型参数', '样本数', '脱敏处理', '未经人工校准'):
            self.assertIn(required, prov)
        self.assertIn('未经人工校准', (report_dir / 'report.html').read_text())
        public = read_json(report_dir / 'report.json')
        self.assertNotIn('data_dir', public)
        self.assertNotIn('files', public)
        self.assertNotIn('errors', public['datasets']['trace-commons']['systems']['test'])
        self.assertEqual(public['datasets']['trace-commons']['systems']['test']['overall'], read_json(self.report_path)['datasets']['trace-commons']['systems']['test']['overall'])
        expected = {p.relative_to(self.root).as_posix() for folder in (self.root / 'data', report_dir) for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        self.assertEqual(set(manifest['files']), expected)
        for filename, sha in manifest['files'].items():
            self.assertEqual(publish.digest((self.root / filename).read_bytes()), sha)
        self.assertEqual(manifest['artifacts'], {p: publish.digest(b) for p, b in before.items()})

    def test_sample_value_mapping(self):
        self.scan()
        self.write_json(self.folder / 'redactions.json', {'entries': [dict(sample='s001', value='person@example.org', replacement='[EMAIL_1]')]})
        self.export()
        self.assertIn('[EMAIL_1]', (self.root / 'data/trace-commons/default-v1/s001/transcript.md').read_text())

    def test_distinct_samples_can_use_distinct_placeholders(self):
        second = self.data / 's002'
        for relative, content in publish.inventory(self.data).items():
            destination = second / Path(relative).relative_to('s001')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        g = read_json(second / 'gold.json')
        g['sample_id'], g['session_id'] = 's002', 'synthetic-second'
        self.write_json(second / 'gold.json', g)
        self.refresh_report()
        self.scan()
        self.write_json(self.folder / 'redactions.json', {'entries': [
            dict(sample=sample, value='person@example.org', replacement=placeholder)
            for sample, placeholder in [('s001', '[EMAIL_1]'), ('s002', '[EMAIL_2]')]]})
        self.export()
        target = self.root / 'data/trace-commons/default-v1'
        self.assertIn('[EMAIL_1]', (target / 's001/transcript.md').read_text())
        self.assertIn('[EMAIL_2]', (target / 's002/transcript.md').read_text())

    def test_url_component_and_short_quote_are_redacted_consistently(self):
        token = 'sk-' + 'a' * 30
        self.create_sample('配置地址 https://example.org/?access_token=' + token + ' 。')
        g = read_json(self.sample / 'gold.json')
        g['U'][0]['evidence'][0]['quote'] = token
        g['blind_snapshot'] = copy.deepcopy({'U': g['U'], 'N': g['N']})
        self.write_json(self.sample / 'gold.json', g)
        self.refresh_report()
        self.redact()
        self.export()
        target = self.root / 'data/trace-commons/default-v1/s001'
        public = read_json(target / 'gold.json')
        self.assertIn(public['U'][0]['evidence'][0]['quote'], (target / 'transcript.md').read_text())
        self.assertNotIn(token, (target / 'transcript.md').read_text())

    def test_allowlist_exact_hits_with_reasons(self):
        result = self.scan()
        # 只豁免一处不能放行其他文件里同值的命中。
        self.write_json(self.folder / 'allowlist.json', {'entries': [dict(id=result['hits'][0]['id'], reason='合成邮箱')]})
        with self.assertRaisesRegex(ValueError, '未处理'):
            self.export()
        self.write_json(self.folder / 'allowlist.json', {'entries': [dict(id=h['id'], reason='合成邮箱') for h in result['hits']]})
        self.export()
        self.assertIn('person@example.org', (self.root / 'data/trace-commons/default-v1/s001/transcript.md').read_text())

    def test_allowlist_requires_reason(self):
        result = self.scan()
        self.write_json(self.folder / 'allowlist.json', {'entries': [dict(id=result['hits'][0]['id'])]})
        with self.assertRaisesRegex(ValueError, 'reason'):
            self.export()

    def test_redaction_conflicts_refused(self):
        result = self.scan()
        hit = result['hits'][0]
        self.write_json(self.folder / 'redactions.json', {'entries': [dict(id=hit['id'], replacement='[EMAIL_1]'), dict(id=hit['id'], replacement='[EMAIL_2]')]})
        with self.assertRaisesRegex(ValueError, '冲突'):
            self.export()

    def test_invalid_evidence_refused(self):
        g = read_json(self.sample / 'gold.json')
        g['U'][0]['evidence'][0]['quote'] = '不存在于原文的引文'
        g['blind_snapshot'] = copy.deepcopy({'U': g['U'], 'N': g['N']})
        self.write_json(self.sample / 'gold.json', g)
        self.refresh_report()
        self.redact()
        with self.assertRaisesRegex(ValueError, '证据无法'):
            self.export()
        self.assertFalse((self.root / 'reports').exists())

    def test_wrong_turn_evidence_refused(self):
        r = read_json(self.sample / 'runs/test.json')
        r['D'][0]['draft']['evidence'][0]['turn'] = 2
        self.write_json(self.sample / 'runs/test.json', r)
        self.refresh_report()
        self.redact()
        with self.assertRaisesRegex(ValueError, '证据无法'):
            self.export()

    def test_post_redaction_gold_validation_required(self):
        self.redact()
        # 输入快照先校验一次；第二次必须检查脱敏后的金标。
        with patch.object(score, 'validate_gold', side_effect=[None, ValueError('金标校验失败')]) as validate:
            with self.assertRaisesRegex(ValueError, '金标校验失败'):
                self.export()
            self.assertEqual(validate.call_count, 2)

    def test_stale_scan_and_removed_hits_refused(self):
        self.redact()
        (self.sample / 'transcript.md').write_text('## [T1] 用户\n新的 person2@example.org')
        self.refresh_report()
        with self.assertRaisesRegex(ValueError, 'scan.json 已过期'):
            self.export()
        self.redact()
        snapshot = read_json(self.folder / 'scan.json')
        snapshot['hits'] = []
        self.write_json(self.folder / 'scan.json', snapshot)
        with self.assertRaisesRegex(ValueError, 'scan.json 已过期'):
            self.export()

    def test_stale_report_refused(self):
        self.redact()
        g = read_json(self.sample / 'gold.json')
        g['annotator'] = {'name': '合成标注者'}
        self.write_json(self.sample / 'gold.json', g)
        with self.assertRaisesRegex(ValueError, '报告输入哈希'):
            self.export()

    def test_split_protocol_versions_judgment_hash_and_symlinks_checked(self):
        original = read_json(self.report_path)
        for mutation in ('protocol', 'dataset', 'version', 'missing_judgment'):
            report = copy.deepcopy(original)
            if mutation == 'protocol': report['protocol'] = 'another'
            if mutation == 'dataset': report['datasets']['other'] = report['datasets'].pop('trace-commons')
            if mutation == 'version': report['datasets']['trace-commons']['protocol_versions'][0][0] = 'another'
            if mutation == 'missing_judgment': report['files'] = [f for f in report['files'] if '/judgments/' not in f['path']]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                publish.verify_report(report, publish.inventory(self.data))
        self.redact()
        path = self.sample / 'judgments/test.json'
        value = read_json(path); value['precision'][0]['note'] = '变更判定'
        self.write_json(path, value)
        with self.assertRaisesRegex(ValueError, '报告输入哈希'):
            self.export()
        target = self.root / 'tmp/other-judgment.json'; target.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            self.scan()

    def test_v4_error_details_are_scanned_redacted_and_exported(self):
        path = self.sample / 'judgments/test.json'
        judgment = read_json(path)
        judgment['precision'][0] = precision(grounded='no', unsupported=1, errors=[
            dict(text='person@example.org', type='unsupported', reason='会话没有确认 person@example.org')])
        self.write_json(path, judgment)
        report = score_split.build_report(self.data, iterations=3)
        self.write_json(self.report_path, report)
        snapshot = self.redact()
        self.assertTrue(any('/errors/' in h['location']['pointer'] for h in snapshot['hits']))
        self.export()
        exported = read_json(self.root / 'data/trace-commons/default-v1/s001/judgments/test.json')
        self.assertEqual(exported['precision'][0]['claims'], 1)
        self.assertEqual(exported['precision'][0]['unsupported'], 1)
        self.assertNotIn('person@example.org', json.dumps(exported, ensure_ascii=False))
        public = read_json(self.root / 'reports/eval-v1/report.json')
        self.assertEqual(public['false_memory_unit'], 'claim')
        self.assertEqual(public['precision_version'], 'model-judge-precision-v4')
        self.assertEqual(public['datasets']['trace-commons']['systems']['test']['overall']['metrics']['false_memory_rate']['value'], 1)
        legacy = copy.deepcopy(report); legacy.pop('precision_version')
        with self.assertRaises(ValueError):
            publish.verify_report(legacy, publish.inventory(self.data))

    def test_scores_only_has_no_data_or_raw_report_content(self):
        report = read_json(self.report_path)
        report['untrusted_extra'] = '不能公开的原始会话内容'
        report['failures'].append(dict(dataset='trace-commons', system='test', sample_id='s001', code='synthetic', reason='person@example.org 不能公开'))
        report['agreement'] = dict(labels={'scope': dict(n=3, agreed=2, raw=2/3, kappa=.5)}, samples=[{'statement': '不能公开'}])
        self.write_json(self.report_path, report)
        manifest = publish.export(self.data, 'halumem', 'scores-v1', self.report_path)
        self.assertFalse((self.root / 'data').exists())
        folder = self.root / 'reports/scores-v1'
        self.assertEqual({p.name for p in folder.iterdir()}, {'report.json', 'report.html', 'PROVENANCE.md', 'manifest.json'})
        for path in folder.iterdir():
            self.assertNotIn('person@example.org', path.read_text())
            self.assertNotIn('不能公开', path.read_text())
        public = read_json(folder / 'report.json')
        self.assertEqual(public['agreement']['labels'], report['agreement']['labels'])
        self.assertEqual(manifest['artifacts'], {p: publish.digest(b) for p, b in publish.inventory(self.data).items()})
        self.assertTrue(all(p.startswith('reports/') for p in manifest['files']))

    def test_existing_target_and_force(self):
        self.redact()
        self.export()
        existing = self.root / 'reports/eval-v1'
        marker = existing / 'old.txt'
        marker.write_text('旧导出')
        with self.assertRaisesRegex(ValueError, '已存在'):
            self.export()
        self.assertEqual(marker.read_text(), '旧导出')
        self.export(force=True)
        self.assertFalse(marker.exists())
        # 数据目标已存在时，即使报告版本不同也必须拒绝，且不留半份报告。
        with self.assertRaisesRegex(ValueError, '已存在'):
            publish.export(self.data, 'trace-commons', 'eval-v2', self.report_path)
        self.assertFalse((self.root / 'reports/eval-v2').exists())

    def test_force_install_failure_restores_previous_trees(self):
        self.redact()
        self.export()
        before = {p.relative_to(self.root): p.read_bytes() for parent in ('data', 'reports') for p in (self.root / parent).rglob('*') if p.is_file()}
        real_rename = publish.os.rename
        def fail_on_data(source, target):
            if str(source).endswith('stage-1'):
                raise OSError('合成的磁盘写入失败')
            return real_rename(source, target)
        with patch.object(publish.os, 'rename', side_effect=fail_on_data):
            with self.assertRaisesRegex(OSError, '磁盘写入失败'):
                self.export(force=True)
        after = {p.relative_to(self.root): p.read_bytes() for parent in ('data', 'reports') for p in (self.root / parent).rglob('*') if p.is_file()}
        self.assertEqual(after, before)

    def test_symlink_input_and_output_refused(self):
        self.redact()
        (self.root / 'reports').symlink_to(self.root / 'tmp', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            self.export()
        source = self.sample / 'transcript.md'
        copy_path = self.root / 'tmp/source.md'
        copy_path.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(copy_path)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            self.scan()

    def test_private_source_is_never_publishable(self):
        g = read_json(self.sample / 'gold.json')
        g['source'] = 'codex'
        self.write_json(self.sample / 'gold.json', g)
        self.refresh_report()
        self.redact()
        with self.assertRaisesRegex(ValueError, '自有或私有'):
            self.export()

    def test_unsafe_identifiers_and_missing_redistribution(self):
        for value in ('../halumem', '/halumem', '..', 'a/b'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                publish.dataset_manifest(value)
        meta = read_json(self.root / 'datasets/trace-commons.json')
        meta.pop('redistribution')
        self.write_json(self.root / 'datasets/trace-commons.json', meta)
        with self.assertRaisesRegex(ValueError, 'redistribution'):
            self.scan()

    def test_cli_reports_counts_only(self):
        output = io.StringIO()
        with redirect_stdout(output):
            publish.main(['scan', '--data-dir', str(self.data), '--dataset', 'trace-commons.json'])
        self.assertIn('email', output.getvalue())
        self.assertNotIn('person@example.org', output.getvalue())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            publish.main(['export', '--data-dir', str(self.data), '--dataset', 'trace-commons', '--eval-version', 'v1', '--report', str(self.report_path)])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
