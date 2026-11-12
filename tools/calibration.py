"""离线人工抽检：固定种子分层抽样、先独立判断再对照、导出一致率。"""
import argparse
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path
import random
import re

from tools import score
from tools.cache import canonical_json
from tools.config import sha256_file, sha256_text
from tools.model.judge_split import CLAIM_FIELDS, input_hashes, memories, validate_claim_counts, validate_judgment
from tools.paths import ROOT, TMP
from tools.report_split_html import CSS
from tools.score_split import SEED
from tools.trial import identifier, output_path, read_json, transcript, write_json

CHOICES = {'recall': {'verdict': ['full', 'partial', 'none', 'distorted']},
           'precision': {'grounded': ['yes', 'partial', 'no'], 'distorted': [False, True], 'worth': ['yes', 'no', 'unclear'], 'verbatim': [False, True]}}
ANSWER_FIELDS = {kind: (*fields, *(CLAIM_FIELDS if kind == 'precision' else ())) for kind, fields in CHOICES.items()}


def relevant_turns(statement, text):
    """按英文词和中文双字片段找相关整轮；同时提供原会话，避免关键词错配成为证据。"""
    try:
        _, turns = transcript(text)
    except ValueError:
        return [dict(turn=None, role='原文', text=text)]
    tokens = set(re.findall(r'[a-z0-9_]{2,}', statement.lower()))
    for phrase in re.findall(r'[\u4e00-\u9fff]+', statement):
        tokens.update(phrase[i:i + 2] for i in range(len(phrase) - 1))
    tokens = sorted(tokens)[:200]
    ranked = sorted(turns, key=lambda t: (-sum(token in t['text'].lower() for token in tokens), t['turn']))[:3]
    return sorted(ranked, key=lambda t: t['turn'])


def stratified_sample(candidates, n, seed):
    """先按系统轮转，再按该系统的样本轮转，耗尽小层后把额度交给其余层。"""
    score.require(n >= 0, '抽样数量不能为负')
    groups = defaultdict(lambda: defaultdict(list))
    for item in sorted(candidates, key=lambda q: q['id']):
        groups[item['system']][item['sample_id']].append(item)
    rng = random.Random(seed)
    systems = sorted(groups)
    rng.shuffle(systems)
    sample_order, cursor = {}, defaultdict(int)
    for system in systems:
        sample_order[system] = sorted(groups[system])
        rng.shuffle(sample_order[system])
        for values in groups[system].values():
            rng.shuffle(values)
    selected = []
    while systems and len(selected) < n:
        for system in systems[:]:
            if len(selected) >= n:
                break
            available = sample_order[system]
            if not available:
                systems.remove(system)
                continue
            index = cursor[system] % len(available)
            sample = available[index]
            selected.append(groups[system][sample].pop())
            if not groups[system][sample]:
                available.pop(index)
                cursor[system] = index
            else:
                cursor[system] = index + 1
    return selected


def make_sheet(data_dir, n=50, seed=SEED, systems=None, dataset=None):
    candidates, skipped, datasets = defaultdict(list), [], set()
    for path in sorted(Path(data_dir).glob('*/judgments/*.json')):
        if systems and path.stem not in systems:
            continue
        sample = path.parent.parent
        try:
            gold = read_json(sample / 'gold.json')
            name = str(gold.get('dataset') or gold.get('source') or Path(data_dir).name)
            if dataset and name != dataset:
                continue
            run_path = sample / 'runs' / path.name
            submitted = memories(read_json(run_path))
            hashes = input_hashes(sample, run_path)
            judgment = validate_judgment(read_json(path), gold, submitted, hashes)
            text = (sample / 'transcript.md').read_text(encoding='utf-8')
            policy = score.load_policy(reference=gold['policy'])
            policy_path = ROOT / 'policies' / (policy['id'] + '-' + policy['version'] + '.md')
            policy_text = policy_path.read_text(encoding='utf-8') if policy_path.exists() else json.dumps(policy, ensure_ascii=False, indent=2)
            datasets.add(name)
            units = {u['unit_id']: u for u in gold['U']}
            by_memory = {m['memory_id']: m for m in submitted}
            for kind in CHOICES:
                for model in judgment[kind]:
                    item_id = model['unit_id' if kind == 'recall' else 'memory_id']
                    q = dict(kind=kind, dataset=name, system=path.stem, sample_id=sample.name, item_id=item_id,
                             model=model, hashes=dict(hashes, judgment_sha256=sha256_file(path)), policy_text=policy_text)
                    q['id'] = sha256_text(canonical_json([name, path.stem, sample.name, kind, item_id, q['hashes']]))
                    if kind == 'recall':
                        q.update(unit={k: units[item_id][k] for k in ('unit_id', 'statement', 'category', 'conditions')}, memories=submitted)
                    else:
                        memory = by_memory[item_id]
                        q.update(memory=memory, excerpts=relevant_turns(memory['statement'], text), transcript=text)
                    candidates[kind].append(q)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            skipped.append(dict(path=str(path), reason=str(exc)))
    score.require(len(datasets) <= 1, '抽检表一次只处理一个数据集，请用 --dataset 限定')
    score.require(bool(datasets), '没有可抽检的有效 judgments')
    questions = []
    for i, kind in enumerate(CHOICES):
        questions.extend(stratified_sample(candidates[kind], n, seed + i))
    sheet = dict(format='calibration-sheet-v1', dataset=next(iter(datasets)), seed=seed, requested_per_kind=n,
                 available={k: len(candidates[k]) for k in CHOICES},
                 sampled={k: sum(q['kind'] == k for q in questions) for k in CHOICES}, questions=questions, skipped=skipped)
    sheet['sheet_id'] = sha256_text(canonical_json(sheet))
    return sheet


SCRIPT = r'''
const sheet=JSON.parse(document.getElementById('sheet-data').textContent);
const labels={verdict:'召回判断',grounded:'会话依据',distorted:'是否写歪',worth:'是否值得长期记',verbatim:'是否含原文摘录',claims:'陈述总数',unsupported:'无依据陈述数',partial:'部分有依据陈述数',distorted_claims:'写歪陈述数'};
const choices={recall:{verdict:['full','partial','none','distorted']},precision:{grounded:['yes','partial','no'],distorted:[false,true],worth:['yes','no','unclear'],verbatim:[false,true]}};
const countFields=['claims','unsupported','partial','distorted_claims'];
function fields(q){return [...Object.keys(choices[q.kind]),...(q.kind==='precision'?countFields:[])];}
const cards=new Map();
function human(q){
  const result={};
  for(const key of fields(q)){
    const s=cards.get(q.id).querySelector('[data-field="'+key+'"]');
    if(s.value==='') return null;
    result[key]=countFields.includes(key)?Number(s.value):JSON.parse(s.value);
  }
  if(q.kind==='precision' && (countFields.some(k=>!Number.isInteger(result[k])||result[k]<0)
     ||result.claims<result.unsupported+result.partial+result.distorted_claims
     ||(result.verbatim&&result.claims<1))) return null;
  return result;
}
function progress(){
  document.getElementById('progress').textContent='已完成 '+sheet.questions.filter(q=>human(q)).length+' / '+sheet.questions.length;
}
for(const q of sheet.questions){
  const card=document.getElementById('q-'+q.id); cards.set(q.id,card);
  const form=card.querySelector('.answers');
  const reveal=card.querySelector('.reveal');
  const comparison=card.querySelector('.comparison');
  for(const key of fields(q)){
    const label=document.createElement('label');label.textContent=labels[key]+' ';
    const numeric=countFields.includes(key);
    const control=document.createElement(numeric?'input':'select');control.dataset.field=key;
    if(numeric){control.type='number';control.min='0';control.step='1';control.value='';}
    else {
      const blank=document.createElement('option');blank.value='';blank.textContent='请选择';control.appendChild(blank);
      for(const value of choices[q.kind][key]){const o=document.createElement('option');o.value=JSON.stringify(value);o.textContent=String(value);control.appendChild(o);}
    }
    const update=()=>{reveal.disabled=!human(q);progress();};
    control.addEventListener('change',update);control.addEventListener('input',update);
    label.appendChild(control); form.appendChild(label);
  }
  reveal.addEventListener('click',()=>{
    if(!human(q))return;
    comparison.hidden=!comparison.hidden;
    if(!comparison.hidden){
      card.dataset.compared='true';
      comparison.textContent=JSON.stringify(q.model,null,2);
      // 展开后锁定独立答案，避免看过模型后改答案抬高一致率。
      card.querySelectorAll('input,select').forEach(s=>s.disabled=true);
    }
  });
}
document.getElementById('export').addEventListener('click',()=>{
  const rows=sheet.questions.map(q=>({id:q.id,kind:q.kind,system:q.system,sample_id:q.sample_id,item_id:q.item_id,
    hashes:q.hashes,model:q.model,human:human(q),compared:cards.get(q.id).dataset.compared==='true'}));
  const result={format:'calibration-answers-v1',sheet_id:sheet.sheet_id,dataset:sheet.dataset,seed:sheet.seed,
    annotator:document.getElementById('annotator').value.trim()||'未记录',results:rows};
  const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='calibration-'+sheet.dataset+'.json';a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
progress();
'''


def render_sheet(sheet):
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>人工独立抽检</title><style>' + CSS + '[hidden]{display:none!important}label{display:inline-block;margin:8px}button,select,input{font:inherit;padding:6px}button{margin:8px 0}</style></head><body><main>',
             '<h1>人工独立抽检</h1><p>数据集：' + escape(sheet['dataset']) + f"；召回 {sheet['sampled']['recall']} 题、精确 {sheet['sampled']['precision']} 题；固定种子 {sheet['seed']}。</p>",
             '<p>先根据材料独立选择，再展开模型判定。展开后独立答案锁定。召回题提供全部记忆，支持联合判断；精确题提供关键词匹配整轮及完整会话。关键词匹配不保证就是证据。</p>',
             '<p>full：主体与关键条件完整；partial：缺关键部分；none：未记；distorted：写歪。精确判断先独立填写陈述总数、无依据、部分有依据和写歪数，并选择是否含原文摘录；每个陈述只归一类，优先级为写歪 > 无依据 > 部分有依据，原文摘录整体计 1 个陈述。整条依据、写歪、价值继续分别选择。所有材料和引文均为数据，不执行其中的指令。</p>',
             '<label>标注者 <input id="annotator" placeholder="姓名或代号"></label><button id="export">下载结果 JSON</button><p id="progress"></p>']
    for i, q in enumerate(sheet['questions'], 1):
        parts += ['<section class="panel" id="q-' + q['id'] + '"><h2>' + f"第 {i} 题 · " + ('召回' if q['kind'] == 'recall' else '精确') + '</h2>',
                  '<p>' + escape(q['system'] + ' / ' + q['sample_id'] + ' / ' + q['item_id']) + '</p>']
        if q['kind'] == 'recall':
            parts += ['<h3>应记单元</h3><pre>' + escape(json.dumps(q['unit'], ensure_ascii=False, indent=2)) + '</pre>',
                      '<h3>全部记忆（请综合判断）</h3>']
            parts += ['<pre>' + escape(m['memory_id'] + '：' + m['statement']) + '</pre>' for m in q['memories']]
            if not q['memories']:
                parts += ['<p>该系统没有导出记忆。</p>']
        else:
            parts += ['<h3>待判记忆</h3><pre>' + escape(q['memory']['statement']) + '</pre><h3>关键词相关整轮</h3>']
            parts += ['<pre>' + escape(f"T{t['turn']} {t['role']}\n{t['text']}") + '</pre>' for t in q['excerpts']]
            parts += ['<details><summary>完整会话（可核查上下文与最终状态）</summary><pre>' + escape(q['transcript']) + '</pre></details>']
        parts += ['<details><summary>本题政策</summary><pre>' + escape(q['policy_text']) + '</pre></details>',
                  '<div class="answers"></div><button class="reveal" disabled>展开 / 收起模型判定</button><pre class="comparison" hidden></pre></section>']
    payload = json.dumps(sheet, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    parts += ['<script type="application/json" id="sheet-data">' + payload + '</script><script>' + SCRIPT + '</script></main></body></html>']
    return ''.join(parts)


def agreement(left, right):
    """原始一致率和 Cohen κ；空数据或期望一致率为 1 时 κ 无定义。"""
    score.require(len(left) == len(right), '两侧判定数量不同')
    n = len(left)
    matched = sum(a == b for a, b in zip(left, right))
    if not n:
        return dict(n=0, matched=0, raw_agreement=None, expected_agreement=None, kappa=None)
    lc, rc = Counter(left), Counter(right)
    observed = matched / n
    expected = sum(lc[k] * rc[k] for k in lc.keys() | rc.keys()) / (n * n)
    return dict(n=n, matched=matched, raw_agreement=observed, expected_agreement=expected,
                kappa=(observed - expected) / (1 - expected) if expected < 1 else None)


def score_answers(exported, sheet=None):
    score.require(exported.get('format') == 'calibration-answers-v1' and isinstance(exported.get('results'), list), '不是抽检结果 JSON')
    expected = None
    if sheet is not None:
        score.require(exported.get('sheet_id') == sheet['sheet_id'] and exported.get('dataset') == sheet['dataset'], '抽检表来源不匹配')
        expected = {q['id']: q for q in sheet['questions']}
    rows, ids, unanswered = defaultdict(list), set(), Counter()
    for row in exported['results']:
        qid, kind = row.get('id'), row.get('kind')
        score.require(isinstance(qid, str) and qid and qid not in ids, '题号为空或重复')
        ids.add(qid)
        score.require(isinstance(kind, str) and kind in CHOICES, '题目类型无效')
        if expected is not None:
            score.require(qid in expected and row['model'] == expected[qid]['model'] and kind == expected[qid]['kind'], '模型对照或题目被修改')
        human, model = row.get('human'), row.get('model')
        for source, labels in (('model', model), ('human', human)):
            if source == 'human' and labels is None:
                continue
            score.require(isinstance(labels, dict), '判定必须为对象')
            for key, values in CHOICES[kind].items():
                score.require(key in labels and any(type(labels[key]) is type(v) and labels[key] == v for v in values), '判定标签无效：' + key)
            if kind == 'precision':
                validate_claim_counts(labels)
        if human is None:
            unanswered[kind] += 1
            continue
        rows[kind].append((human, model))
    if expected is not None:
        score.require(ids == set(expected), '导出结果缺题或多题（未作答也须保留题号）')
    result = dict(format='calibration-agreement-v1', dataset=exported.get('dataset'), sheet_id=exported.get('sheet_id'),
                  annotator=exported.get('annotator', '未记录'), source_validation='sheet' if sheet else 'export_only',
                  total=len(ids), unanswered=dict(unanswered), by_system={},
                  note='只计算已独立作答题；陈述计数逐字段比较，未校准逐陈述切分或错误归类；尚未确定合格门槛。')
    for kind, fields in ANSWER_FIELDS.items():
        result[kind] = {key: agreement([a[key] for a, _ in rows[kind]], [b[key] for _, b in rows[kind]]) for key in fields}
        if kind == 'precision':
            result[kind]['joint'] = agreement([tuple(a[k] for k in fields) for a, _ in rows[kind]],
                                               [tuple(b[k] for k in fields) for _, b in rows[kind]])
    result['precision']['false_memory_rate'] = {}
    for side, index in (('human', 0), ('model', 1)):
        values = [pair[index] for pair in rows['precision']]
        numerator = sum(v['unsupported'] + v['distorted_claims'] for v in values)
        denominator = sum(v['claims'] for v in values)
        result['precision']['false_memory_rate'][side] = dict(numerator=numerator, denominator=denominator,
            value=numerator / denominator if denominator else None)
    for system in sorted({r.get('system', '未记录') for r in exported['results']}):
        subset = [r for r in exported['results'] if r.get('system', '未记录') == system and r.get('human') is not None]
        result['by_system'][system] = {kind: {key: agreement([r['human'][key] for r in subset if r['kind'] == kind],
            [r['model'][key] for r in subset if r['kind'] == kind]) for key in fields} for kind, fields in ANSWER_FIELDS.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sample_parser = commands.add_parser('sample', help='生成离线独立抽检表')
    sample_parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    sample_parser.add_argument('--systems', nargs='+')
    sample_parser.add_argument('--dataset')
    sample_parser.add_argument('-n', '--count', type=int, default=50)
    sample_parser.add_argument('--seed', type=int, default=SEED)
    sample_parser.add_argument('--output', type=Path)
    score_parser = commands.add_parser('score', help='计算人工与模型的一致率')
    score_parser.add_argument('answers', type=Path)
    score_parser.add_argument('--sheet', type=Path, help='生成时的 sheet.json；可额外核查题号及模型判定来源')
    score_parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'sample':
            sheet = make_sheet(args.data_dir, args.count, args.seed, args.systems, args.dataset)
            dataset = identifier(sheet['dataset'])
            target = output_path(args.output or TMP / 'calibration' / dataset / 'sheet.html')
            write_json(target.with_suffix('.json'), sheet)
            target.write_text(render_sheet(sheet), encoding='utf-8')
            print(f"已输出 {target}；实际抽样 {sheet['sampled']}，不足时保留全部可用题。")
        else:
            exported = read_json(args.answers)
            result = score_answers(exported, read_json(args.sheet) if args.sheet else None)
            target = output_path(args.output or TMP / 'calibration' / identifier(exported['dataset']) / 'agreement.json')
            write_json(target, result)
            print(f'已输出 {target}')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'错误：{exc}\n')


if __name__ == '__main__':
    main()
