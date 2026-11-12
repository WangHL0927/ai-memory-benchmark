"""固定种子人工抽检、手算 κ、离线页面交互的合成测试。"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.helpers import TMP, unit
from tests.helpers import recall, precision
from tests.helpers import fixture
from tests.helpers import cli
from tools import calibration
from tools.trial import read_json, write_json


def exported(sheet):
    return dict(format='calibration-answers-v1', dataset=sheet['dataset'], sheet_id=sheet['sheet_id'], annotator='合成人工',
                results=[dict(id=q['id'], kind=q['kind'], system=q['system'], model=q['model'],
                              human={k: q['model'][k] for k in calibration.ANSWER_FIELDS[q['kind']]}) for q in sheet['questions']])


class CalibrationTests(unittest.TestCase):
    def test_sampling_reproducible_balanced_system_then_sample(self):
        candidates = [dict(id=f'{system}-{sample}-{i}', system=system, sample_id=sample)
                      for system in ('a', 'b', 'c') for sample in ('x', 'y', 'z') for i in range(12)]
        first = calibration.stratified_sample(candidates, 29, 7)
        self.assertEqual(first, calibration.stratified_sample(list(reversed(candidates)), 29, 7))
        self.assertNotEqual(first, calibration.stratified_sample(candidates, 29, 8))
        self.assertEqual(len({q['id'] for q in first}), 29)
        counts = [sum(q['system'] == s for q in first) for s in ('a', 'b', 'c')]
        self.assertLessEqual(max(counts) - min(counts), 1)
        for system in ('a', 'b', 'c'):
            counts = [sum(q['system'] == system and q['sample_id'] == s for q in first) for s in ('x', 'y', 'z')]
            self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(len(calibration.stratified_sample(candidates, 1000, 7)), len(candidates))
        self.assertEqual(calibration.stratified_sample(candidates, 0, 7), [])
        with self.assertRaises(ValueError):
            calibration.stratified_sample(candidates, -1, 7)

    def test_sheet_materials_hidden_comparison_and_safe_embedded_data(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d)
            for system in ('a', 'b'):
                for name in ('s1', 's2'):
                    fixture(root, name, system=system, units=[unit('u1'), unit('u2')], verdicts=[recall(), recall('u2', 'none')],
                            precision_values=[precision(), precision('m2')])
            sheet = calibration.make_sheet(root, n=5, seed=42)
            self.assertEqual(sheet, calibration.make_sheet(root, n=5, seed=42))
            self.assertEqual(sheet['sampled'], dict(recall=5, precision=5))
            for q in sheet['questions']:
                if q['kind'] == 'recall':
                    self.assertEqual(len(q['memories']), 2)  # 包括模型未引用的记忆，供人工独立判断。
                else:
                    self.assertIn('以后用蓝色水管', q['transcript'])
                    self.assertTrue(q['excerpts'])
            sheet['questions'][0]['model']['note'] = '</script><img src=x onerror=alert(1)>'
            html = calibration.render_sheet(sheet)
            self.assertIn('class="comparison" hidden', html)
            self.assertIn('class="reveal" disabled', html)
            self.assertNotIn('</script><img', html)
            self.assertIn('\\u003c/script\\u003e', html)
            self.assertNotIn('<script src', html)
            self.assertIn('先根据材料独立选择', html)

    def test_relevant_full_turns_and_fallback(self):
        text = '## [T1] 用户\n苹果方案\n\n## [T2] 助手\n已知\n\n## [T3] 用户\nwater pipe blue\n\n## [T4] 用户\n水管蓝色需要确认\n'
        turns = calibration.relevant_turns('water pipe 蓝色水管', text)
        self.assertTrue(any(t['turn'] == 3 for t in turns))
        self.assertTrue(any(t['turn'] == 4 for t in turns))
        self.assertEqual(calibration.relevant_turns('未知', '无轮次原文')[0]['text'], '无轮次原文')

    def test_kappa_hand_calculated_and_degenerate(self):
        # 4 对：观察一致率 3/4；两侧边际 A=3/4、A=2/4，期望 1/2，κ=1/2。
        result = calibration.agreement(['A', 'A', 'A', 'B'], ['A', 'A', 'B', 'B'])
        self.assertEqual(result['raw_agreement'], .75)
        self.assertEqual(result['expected_agreement'], .5)
        self.assertEqual(result['kappa'], .5)
        self.assertIsNone(calibration.agreement([], [])['kappa'])
        self.assertIsNone(calibration.agreement(['A'], ['A'])['kappa'])
        self.assertEqual(calibration.agreement(['A', 'B'], ['B', 'A'])['kappa'], -1)
        with self.assertRaises(ValueError):
            calibration.agreement(['A'], [])

    def test_score_dimensions_unanswered_provenance_and_rejections(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d)
            fixture(root, precision_values=[precision(), precision('m2', 'partial', True, 'no')],
                    units=[unit('u1'), unit('u2')], verdicts=[recall(), recall('u2', 'none')])
            sheet = calibration.make_sheet(root, 50)
            answers = exported(sheet)
            result = calibration.score_answers(answers, sheet)
            self.assertEqual(result['recall']['verdict']['raw_agreement'], 1)
            self.assertEqual(result['precision']['grounded']['kappa'], 1)
            self.assertEqual(result['precision']['distorted']['kappa'], 1)
            self.assertEqual(result['precision']['worth']['kappa'], 1)
            self.assertEqual(result['precision']['joint']['raw_agreement'], 1)
            answers['results'][0]['human'] = None
            result = calibration.score_answers(answers)
            self.assertEqual(result['unanswered']['recall'], 1)
            self.assertEqual(result['source_validation'], 'export_only')
            for change in ('duplicate', 'missing', 'model', 'label', 'empty_model'):
                bad = copy.deepcopy(answers)
                if change == 'duplicate': bad['results'].append(bad['results'][0])
                if change == 'missing': bad['results'].pop()
                if change == 'model': bad['results'][0]['model']['verdict'] = 'distorted'
                if change == 'label': bad['results'][-1]['human']['distorted'] = 0
                if change == 'empty_model': bad['results'][0]['model'] = None
                with self.subTest(change=change), self.assertRaises(ValueError):
                    calibration.score_answers(bad, sheet)

    def test_independent_claim_counts_and_verbatim_are_validated(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d)
            fixture(root, precision_values=[precision(grounded='partial', distorted=True, claims=5,
                    unsupported=1, partial=1, distorted_claims=1, verbatim=True)])
            sheet = calibration.make_sheet(root)
            answers = exported(sheet)
            row = next(r for r in answers['results'] if r['kind'] == 'precision')
            row['human'].update(claims=4, unsupported=1, partial=0, distorted_claims=0, verbatim=False)
            result = calibration.score_answers(answers, sheet)
            self.assertEqual(result['precision']['claims']['raw_agreement'], 0)
            self.assertEqual(result['precision']['verbatim']['raw_agreement'], 0)
            self.assertEqual(result['precision']['false_memory_rate']['human']['value'], .25)
            self.assertEqual(result['precision']['false_memory_rate']['model']['value'], .4)
            for changes in (dict(claims=True), dict(partial=-1), dict(claims=1.5),
                            dict(claims=0), dict(verbatim=1)):
                bad = copy.deepcopy(answers)
                target = next(r for r in bad['results'] if r['kind'] == 'precision')
                target['human'].update(changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    calibration.score_answers(bad, sheet)
            del row['human']['claims']
            with self.assertRaises(ValueError):
                calibration.score_answers(answers, sheet)

    def test_cli_sample_score_attach_and_stale_skip(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); data = root / 'data'; fixture(data)
            result = cli('tools.calibration', 'sample', '--data-dir', data, '-n', 50, '--output', root / 'sheet.html')
            self.assertEqual(result.returncode, 0, result.stderr)
            sheet = read_json(root / 'sheet.json'); self.assertEqual(sheet['sampled'], dict(recall=1, precision=1))
            write_json(root / 'answers.json', exported(sheet))
            result = cli('tools.calibration', 'score', root / 'answers.json', '--sheet', root / 'sheet.json', '--output', root / 'agreement.json')
            self.assertEqual(result.returncode, 0, result.stderr)
            result = cli('tools.score_split', '--data-dir', data, '--bootstrap', 5, '--agreement', root / 'agreement.json', '--output', root / 'report')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_json(root / 'report.json')['agreement']['recall']['verdict']['raw_agreement'], 1)
            (data / 's001/transcript.md').write_text('被改过的会话')
            with self.assertRaisesRegex(ValueError, '没有可抽检'):
                calibration.make_sheet(data)

    @unittest.skipUnless(shutil.which('node'), '未安装 Node，跳过离线页面脚本替身测试')
    def test_page_independent_answer_gate_lock_and_download(self):
        # 用小型 DOM 替身执行真实内嵌脚本；不需要网络、浏览器或真实数据。
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d); fixture(root, precision_values=[precision(grounded='partial', claims=2, partial=1, verbatim=True)])
            sheet = calibration.make_sheet(root, 1)
            harness = r'''
const vm=require('vm'); const assert=require('assert');
class Element {
 constructor(){this.children=[];this.handlers={};this.dataset={};this.value='';this.disabled=false;this.hidden=true;this.textContent='';}
 appendChild(x){this.children.push(x);}
 addEventListener(k,fn){this.handlers[k]=fn;}
 click(){if(!this.disabled&&this.handlers.click)this.handlers.click();}
 querySelector(s){if(s.startsWith('[data-field='))return this.selects.find(x=>x.dataset.field===s.split('"')[1]);return this.parts[s];}
 querySelectorAll(){return this.selects;}
}
const sheet=JSON.parse(process.argv[1]);const script=JSON.parse(process.argv[2]);const els={};
for(const id of ['sheet-data','progress','annotator','export'])els[id]=new Element();
els['sheet-data'].textContent=JSON.stringify(sheet);
for(const q of sheet.questions){const card=new Element();card.selects=[];
 const form=new Element();form.appendChild=label=>card.selects.push(label.children[0]);
 card.parts={'.answers':form,'.reveal':new Element(),'.comparison':new Element()};card.parts['.reveal'].disabled=true;
 els['q-'+q.id]=card;}
let blob=null,link=null;
const ctx={document:{getElementById:id=>els[id],createElement:tag=>{const e=new Element();if(tag==='a')link=e;return e;}},
 Blob,URL:{createObjectURL:b=>{blob=b;return 'local:test';},revokeObjectURL:()=>{}},setTimeout:()=>{}};
vm.runInNewContext(script,ctx);
for(const q of sheet.questions){const card=els['q-'+q.id];assert(card.parts['.reveal'].disabled);assert(card.parts['.comparison'].hidden);
 for(const select of card.selects){select.value=JSON.stringify(q.model[select.dataset.field]);select.handlers.change();}
 if(q.kind==='precision'){
  const claims=card.selects.find(s=>s.dataset.field==='claims');
  const original=claims.value;
  for(const invalid of ['-1','1.5','0']){claims.value=invalid;claims.handlers.input();assert(card.parts['.reveal'].disabled);assert(card.parts['.comparison'].hidden);}
  claims.value=original;claims.handlers.input();
 }
 assert(!card.parts['.reveal'].disabled);card.parts['.reveal'].click();
 assert(!card.parts['.comparison'].hidden);assert(card.selects.every(s=>s.disabled));}
els.export.click();
(async()=>{const result=JSON.parse(await blob.text());assert.equal(result.results.length,2);assert(result.results.every(r=>r.human&&r.compared));assert.equal(result.sheet_id,sheet.sheet_id);assert.equal(link.download,'calibration-synthetic.json');})().catch(e=>{console.error(e);process.exitCode=1;});
'''
            result = subprocess.run(['node', '-e', harness, json.dumps(sheet), json.dumps(calibration.SCRIPT)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
