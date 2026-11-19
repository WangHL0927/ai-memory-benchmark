"""Aggregate model receipts and render v0.7 reports without source text or paths."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from tools.model.v07.cli import JUDGE_MODEL, READER_MODEL, READER_WINDOW
from tools.model.v07.evaluate import BASELINES
from tools.model.v07.protocol import SCENARIOS
from tools.model.v07.scoring import (diagnostic_rates, diagnostics, flatten, grade_record, log_penalty, paired_values,
                                    ratio, resampling, sensitivity, value_and_cost)
from tools.paths import private_path
from tools.trial import read_json, write_json

PRODUCTS = ('openviking', 'ai-memory', 'claude-mem', 'basic-memory', 'X2')
LABELS = {'openviking': 'OpenViking 0.5.4.1', 'ai-memory': 'ai-memory 3.4.7',
          'claude-mem': 'claude-mem 15.17.0', 'basic-memory': 'Basic Memory 1.0.0 + agent',
          'X2': 'X2', 'no-memory': '无记忆基线', 'full-transcript': '完整 transcript 基线'}


def product_name(name):
    return next((key for key in PRODUCTS if name.startswith(key + '-')), name)


def partition(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['session_id']].append(row)
    return [grouped[key] for key in sorted(grouped)]


def collect(root):
    golds, records = {}, defaultdict(dict)
    for p in sorted(root.glob('*/gold.json')):
        golds[p.parent.name] = read_json(p)
        finished = {f.name: f for f in p.parent.glob('v07-task-results/*.json')}
        finished.update({f.name: f for f in p.parent.glob('v07-results/*.json')})
        for path in sorted(finished.values()):
            r = read_json(path)
            key = product_name(r['system'])
            if p.parent.name in records[key]:
                raise ValueError('multiple versions of the same system on one sample')
            records[key][p.parent.name] = r
    if not golds:
        raise ValueError('expected frozen samples')
    return golds, records


def values(golds, records, excluded):
    rows = {}
    diag_rows = defaultdict(list)
    empty, counts, failures = 0, Counter(), Counter()
    overflows = defaultdict(list)
    for sid, r in records.items():
        gold = golds[sid]
        empty += r['empty']
        counts[gold['dataset']] += 1
        if gold['tasks']:
            row = grade_record(gold, r, excluded.get(sid, ()))
            rows[sid] = row
            failures.update(row['failures'])
            for task, answer in r['answers'].items():
                if task not in excluded.get(sid, ()) and answer['status'] == 'reader_overflow':
                    overflows[sid].append(answer['required_tokens'])
        if r['system'] not in BASELINES and r.get('diagnosis_complete', True):
            diag_rows[gold['dataset']].append((gold, r))
    by_dataset = {}
    for dataset, parts in diag_rows.items():
        diag = Counter()
        for gold, r in parts:
            diag.update(diagnostics(gold, r))
        grouped = defaultdict(list)
        for gold, r in parts:
            grouped[gold['session_id']].append(diagnostics(gold, r))

        def metric(groups):
            total = Counter()
            for row in flatten(groups):
                total.update(row)
            return {key: entry['value'] for key, entry in diagnostic_rates(total).items()}
        by_dataset[dataset] = dict(counts=dict(diag), rates=diagnostic_rates(diag),
                                   intervals=resampling([grouped[k] for k in sorted(grouped)], metric),
                                   samples=len(parts), clusters=len(grouped))
    numbers = value_and_cost(rows.values())
    return dict(rows=rows, valid=dict(counts), empty=empty, task_failures=dict(failures), diagnosis=by_dataset,
                reader_overflows=dict(overflows),
                summaries=numbers, sensitivity=sensitivity(rows.values()))


def token_stats(tokens):
    return ' / '.join(f'{measure(tokens):,.0f}' for measure in (median, mean, max)) if tokens else 'NA'


def writer_ledger(root, golds):
    ledger = defaultdict(lambda: defaultdict(list))
    for sample, gold in golds.items():
        for path in (root / sample / 'runs').glob('*.json'):
            if path.stem.endswith('-drafts'):
                continue
            native = read_json(path)['system']
            name = product_name(path.stem)
            if name in PRODUCTS:
                ledger[name][gold['dataset']].append(native)
    collected = {}
    for name, entries in ledger.items():
        collected[name] = {}
        for dataset, rows in entries.items():
            stats = dict(samples=len(rows), llm_calls=0, llm_input=0, llm_output=0, external_embedding_calls=0,
                         write_input_missing=0, write_output_missing=0, cached_input=0, cached_known=0)
            for row in rows:
                usage = row['usage']
                stats['llm_calls'] += usage['llm']['calls']
                stats['external_embedding_calls'] += usage.get('embedding', {}).get('calls', 0)
                for field, token in [('input', 'prompt_tokens'), ('output', 'completion_tokens')]:
                    measured = usage['llm']['tokens'].get(token)
                    if type(measured) is int:
                        stats['llm_' + field] += measured
                    else:
                        stats['write_' + field + '_missing'] += 1
                for call in row.get('calls', []):
                    if call.get('kind') != 'llm':
                        continue
                    call_usage = call.get('usage')
                    if not isinstance(call_usage, dict):
                        continue
                    cached = (call_usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                    if type(cached) is int:
                        stats['cached_known'] += 1
                        stats['cached_input'] += cached
            durations = [r['elapsed_seconds'] for r in rows if type(r.get('elapsed_seconds')) in (int, float)]
            stats['median_seconds'] = median(durations) if durations else None
            collected[name][dataset] = stats
    return collected


def x2_review_summary(records, golds):
    counts = Counter()
    for sid, row in records.items():
        for review in row.get('x2_reviews') or []:
            counts['submitted'] += 1
            if review['state'] != 'rejected':
                continue
            counts['rejected'] += 1
            counts['unsupported'] += review.get('reason_code') in ('missing_evidence', 'unconfirmed')
            if golds[sid]['dataset'] == 'halumem':
                reason = (review.get('reason') or '').lower()
                counts['halumem_future'] += 'after the current date' in reason or 'event is future-dated' in reason
    return {key: counts[key] for key in ('submitted', 'rejected', 'unsupported', 'halumem_future')}


def summarize(root, exclusions=()):
    golds, records = collect(root)
    excluded = defaultdict(set)
    for task in exclusions:
        excluded[task['sample_id']].add(task['task_id'])
    scenario_sources = {s: set() for s in SCENARIOS}
    for sid, gold in golds.items():
        for task in gold['tasks']:
            if task['task_id'] not in excluded.get(sid, ()):
                scenario_sources[task['scenario']].add(gold['session_id'])
    by_system = {label: values(golds, records[label], excluded) for label in LABELS}
    trace = {s for s, g in golds.items() if g['tasks']}
    matched = trace.copy()
    for system in PRODUCTS:
        rows = by_system[system]['rows']
        matched &= {s for s, row in rows.items() if row['num_tasks'] == row['valid_tasks'] and row['valid_tasks']}
    row_map = {n: {s: r for s, r in block['rows'].items() if s in matched} for n, block in by_system.items()}
    available = {n: b for n, b in by_system.items() if b['rows']}
    for name, block in by_system.items():
        rows = block['rows']
        block['read_tokens_per_question'] = [
            answer['read_tokens'] for sid, result in records[name].items()
            for task, answer in result['answers'].items()
            if task not in excluded.get(sid, ()) and answer['status'] == 'answered'
        ]
        planned = {s: sum(task['task_id'] not in excluded.get(s, ()) for task in golds[s]['tasks']) for s in trace}
        correct = sum(row['outcomes'].get('correct', 0) for row in rows.values())
        adapter = {'ai-memory': 'ai_memory', 'basic-memory': 'basic_memory', 'claude-mem': 'claude_mem',
                   'openviking': 'openviking', 'X2': 'x2'}.get(name)
        status_path = root / 'status' / f'{adapter}.json'
        unreplayed = set()
        if adapter is not None and status_path.exists():
            unreplayed = {Path(f['sample']).name for f in read_json(status_path).get('failures', [])}
        denominator = sum(row['valid_tasks'] + row['failures'].get('reader_overflow', 0) for row in rows.values())
        denominator += sum(planned[sid] for sid in (trace & unreplayed) - set(rows))
        block['conservative'] = {'correct': correct, 'planned_tasks': sum(planned.values()),
                                 'measured_tasks': denominator, 'unmeasured_tasks': sum(planned.values()) - denominator}
        block['common'] = value_and_cost(row_map[name].values())
        block['common_sensitivity'] = sensitivity(row_map[name].values())
        block['cost_per_segment'] = {s: {k: row[k] for k in
                                     ('transcript_tokens', 'read_tokens', 'write_tokens', 'write_multiple')}
                                     for s, row in rows.items()}
        scored = [r for r in rows.values() if r['valid_tasks']]
        block['coding_scored'] = len(scored)
        block['coding_clusters'] = len(partition(scored))
        block['strata_success'] = {}
        for n, (low, high) in enumerate(((0, 50), (51, 150), (151, float('inf')))):
            stratum = [r for s, r in rows.items() if low <= golds[s]['turns'] <= high]
            block['strata_success'][n] = dict(samples=len(stratum), clusters=len(partition(stratum)),
                                              success=value_and_cost(stratum)['observed_success'])
        block['intervals'] = resampling(partition(scored),
                                       lambda g: {k: v for k, v in value_and_cost(flatten(g)).items()
                                                  if k in ('composite', 'value', 'read_penalty', 'write_penalty')})
    pairwise = {}
    if matched:
        for left, right in combinations(available, 2):
            pairwise[f'{left} - {right}'] = paired_values(row_map[left], row_map[right])
    assessment_usage = {}
    for name, systems in records.items():
        forms = defaultdict(lambda: dict(calls=0, input=0, output=0, input_measured=0, output_measured=0))
        for result in systems.values():
            reader = [entry['receipt'] for entry in result['answers'].values() if 'receipt' in entry]
            for stage, calls in (('reader', reader), ('judge', result['calls'])):
                for call in calls:
                    totals = forms[stage]
                    totals['calls'] += 1
                    usage = call.get('usage') or {}
                    for key, aliases in (('input', ('input_tokens', 'prompt_tokens')),
                                         ('output', ('output_tokens', 'completion_tokens'))):
                        amount = next((usage[a] for a in aliases if type(usage.get(a)) is int), None)
                        if amount is not None:
                            totals[key] += amount
                            totals[key + '_measured'] += 1
        assessment_usage[name] = dict(forms)
    return dict(samples={s: dict(dataset=g['dataset'], turns=g['turns'], tasks=len(g['tasks']),
                                 transcript_sha256=g['transcript_sha256'])
                         for s, g in golds.items()}, common_trace=sorted(matched), systems=by_system,
                paired=pairwise, exclusions=list(exclusions), tokenizer=next(iter(golds.values()))['tokenizer'],
                writer_ledger=writer_ledger(root, golds), assessment_usage=assessment_usage,
                x2_review=x2_review_summary(records['X2'], golds),
                scenario_sources={s: len(sources) for s, sources in scenario_sources.items()})


def fmt(value, percentage=False):
    if value is None:
        return 'NA'
    return f'{value:.1%}' if percentage else f'{value:.2f}'


def compact_metric(entry):
    result = fmt(entry['value'], True)
    return f"{result} ({entry['numerator']}/{entry['denominator']})"


def scenario_metric(entry):
    return f"{fmt(entry['value'], True)}（{entry['denominator']} 题）"


def status_summaries(root, report):
    details = defaultdict(list)
    names = {'ai_memory': 'ai-memory', 'basic_memory': 'basic-memory', 'openviking': 'openviking',
             'claude_mem': 'claude-mem', 'x2': 'X2'}
    for code, name in names.items():
        status = root / 'status' / f'{code}.json'
        if status.exists():
            data = read_json(status)
            for row in data.get('failures', []):
                sample = Path(row['sample']).name
                reason = row.get('detail') or row.get('error') or ''
                kind = '审核未完成' if '审核超时' in reason or '未完成审核' in reason else '摄取失败'
                error = '模型工具参数不是有效 JSON' if name == 'X2' and row['error_type'] == 'JSONDecodeError' else row['error_type']
                details[kind].append(f'{LABELS[name]} / {sample}（{error}）')
            if data.get('not_run'):
                details['未执行'].append(f'{LABELS[name]} {data["not_run"]} 段')
    path = root / 'evaluation-failures.json'
    if path.exists():
        for row in read_json(path):
            details['判定失败'].append(f"{LABELS[product_name(row['system'])]} / {row['sample']} ({row['error_type']})")
    for name, row in report['systems'].items():
        if row['task_failures']:
            detail = ', '.join(f'{k} {v}' for k, v in sorted(row['task_failures'].items()))
            details['作答状态'].append(f'{LABELS[name]} ({detail})')
        for sample, tokens in row['reader_overflows'].items():
            interval = f'{min(tokens):,}–{max(tokens):,}' if min(tokens) != max(tokens) else f'{tokens[0]:,}'
            details['超窗输入'].append(f'{LABELS[name]} / {sample}（{len(tokens)} 题，{interval} token）')
    return details


def render(report, date, statuses):
    lines = [f'# 记忆摄取评测 v0.7.1 缩减版 · {date}', '',
             'X2 是未公开的内部系统。以下为模型判定，未做人类校准。', '',
             f'盲标和出题使用 `{JUDGE_MODEL}` high，判分使用同型号 medium。'
             f'作答使用 `{READER_MODEL}`，端点默认生成参数，按实际端点上限采用 {READER_WINDOW:,} token 窗口。'
             f'T 与 R 采用 tiktoken {report["tokenizer"]["version"]} / {report["tokenizer"]["encoding"]}。'
             '写入使用 `deepseek-v4.1-flash-1017`。'
             'Coding 使用 [default-v2](../../policies/default-v2.md)，'
             'HaluMem 使用 [halumem-v1](../../policies/halumem-v1.json)。',
             f'全五产品共同有效的 coding 段共 {len(report["common_trace"])} 段。'
             '缺必要场景或写入/读取计量时，综合分显示 NA。', '',
             '## 1　综合分', '',
             '| 系统 | 摄取完成 | Trace 有任务记录 | HaluMem 判定段 | 成功空库 | 已判定的隐私命中 | 综合分 | 截底前分数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for system, d in report['systems'].items():
        secrets = sum(v['counts'].get('precision_contains_secret', 0) for v in d['diagnosis'].values())
        privacy = '**🔴 ' + str(secrets) + '**' if secrets else '0'
        captures = report['writer_ledger'].get(system, {})
        completed = str(sum(c['samples'] for c in captures.values())) if system in PRODUCTS else '—'
        empty = str(d['empty']) if system in PRODUCTS else '—'
        privacy = privacy if system in PRODUCTS else '—'
        lines.append(f"| {LABELS[system]} | {completed} | {d['valid'].get('trace-commons',0)} | "
                     f"{d['valid'].get('halumem',0)} | {empty} | {privacy} | "
                     f"{fmt(d['summaries']['composite'])} | {fmt(d['summaries']['margin'])} |")
    lines += ['', '## 2　价值分', '',
              '| 系统 | 有效 coding 段数 / 来源数 | 价值分 | S1 理解用户 | S2 解释决策 | S3 避免重复研究 | S4 延续工作 | S5 可归因误导率 | 保守全对率 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for system, d in report['systems'].items():
        row = d['summaries']
        c = d['conservative']
        conservative = ratio(c['correct'], c['measured_tasks'])
        options = [scenario_metric(row['scenarios'][s]) for s in SCENARIOS]
        lines.append(f"| {LABELS[system]} | {d['coding_scored']} / {d['coding_clusters']} | {fmt(row['value'])} | " + ' | '.join(options)
                     + f" | {compact_metric(row['misleading'])} | {compact_metric(conservative)} |")
    lines += ['', 'S1–S4 对每题的要点完成比例取等权均值。S5 只扣有证据可归因于所给记忆的行动错误。'
              '保守全对率将已知摄取失败及超窗计为失败，缺判定或评测服务故障计为未测。']
    lines += ['', '出题所覆盖的独立来源数为 ' +
              '、'.join(f"{s} {report['scenario_sources'][s]}" for s in SCENARIOS) + '。']
    if report['exclusions']:
        reasons = {'answer_leak': '题面泄露', 'general_knowledge': '通识可答',
                   'insufficient_source_evidence': '会话证据不足'}
        rows = [f"`{i['sample_id']}/{i['task_id']}`（{reasons[i['code']]}）" for i in report['exclusions']]
        lines += ['', '全组撤销题目 ' + '、'.join(rows) + '。']
    lines += ['', '| 系统 | 价值分 [95% 区间] | 价值有效重抽 | 综合分 [95% 区间] | 综合有效重抽 |',
              '|---|---|---:|---|---:|']
    for system, d in report['systems'].items():
        v, c = (d['intervals'][name] for name in ('value','composite'))
        def bounds(row):
            lo,hi = row['ci95']
            return f"{fmt(row['value'])} [{fmt(lo)}, {fmt(hi)}]"
        lines.append(f"| {LABELS[system]} | {bounds(v)} | {v['valid_resamples']} | "
                     f"{bounds(c)} | {c['valid_resamples']} |")
    lines += ['', '| 系统 | 已判定全对率 | 要点完成率 | 行动错误率（诊断） | 部分 / 具体弃答 / 答错 | 错误与材料一致 / 可归因误导 / 归因不清 / 高风险 |',
              '|---|---:|---:|---:|---:|---:|']
    for system, d in report['systems'].items():
        r = d['summaries']
        tally = r['results']
        points = ratio(tally.get('accepted_requirements', 0), tally.get('requirements', 0))
        outcomes = ' / '.join(str(tally.get(k, 0)) for k in ('partial', 'specific_abstention', 'wrong'))
        risks = ' / '.join(str(tally.get(k, 0))
                           for k in ('error_matches_memory', 'misleading', 'unclear_attribution', 'high_risk'))
        lines.append(f"| {LABELS[system]} | {compact_metric(r['observed_success'])} | {compact_metric(points)} | "
                     f"{compact_metric(r['action_error_rate'])} | {outcomes} | {risks} |")
    for title, status in statuses.items():
        if status:
            lines += ['', f'- {title} ' + ' · '.join(status)]
    lines += ['', '## 3　效率', '',
              '| 系统 | 每题读取 token 中位数 / 均值 / 最大值 | 每段读取 token 中位数 / 均值 / 最大值 | 读扣分 | 写扣分 |',
              '|---|---:|---:|---:|---:|']
    for system, d in report['systems'].items():
        tokens = [r['read_tokens'] for r in d['rows'].values() if r['read_tokens'] is not None]
        lines.append(f"| {LABELS[system]} | {token_stats(d['read_tokens_per_question'])} | {token_stats(tokens)} | "
                     f"{fmt(d['summaries']['read_penalty'])} | {fmt(d['summaries']['write_penalty'])} |")
    lines += ['', '<details>', '<summary>Coding 各段读写 token 及扣分</summary>', '',
              '| 系统 | 样本 | T | R | W | 读扣分 | 写扣分 |', '|---|---|---:|---:|---:|---:|---:|']
    for system, d in report['systems'].items():
        for sid, r in d['rows'].items():
            cost = value_and_cost([r])
            write_penalty = log_penalty(r['write_multiple'], 2.5) if r['write_multiple'] is not None else None
            lines.append(f"| {LABELS[system]} | {sid} | {r['transcript_tokens']:,} | {fmt(r['read_tokens'])} | "
                         f"{fmt(r['write_tokens'])} | {fmt(cost['read_penalty'])} | {fmt(write_penalty)} |")
    lines += ['', '</details>']
    lines += ['', '## 4　摄取诊断', '',
              '| 数据集 | 系统 | 段数 / 来源数 | 漏记率 | 完全未覆盖 | 错记率 | 噪声 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for system, d in report['systems'].items():
        for ds, info in d['diagnosis'].items():
            values = [compact_metric(info['rates'][k]) for k in ('miss', 'uncovered', 'claim_errors', 'worth_no')]
            lines.append(f"| {ds} | {LABELS[system]} | {info['samples']} / {info['clusters']} | " + ' | '.join(values) + ' |')
    lines += ['', '| 数据集 | 系统 | full / partial / none / distorted | 含争议漏记 | 逐字完整 | 范围可判 / 准确 | 追溯率 | 定位成功率 | HaluMem 父点完整率 |',
              '|---|---|---|---:|---:|---|---:|---:|---:|']
    for system, d in report['systems'].items():
        for ds, info in d['diagnosis'].items():
            counts, rates = info['counts'], info['rates']
            distribution = ' / '.join(str(counts.get(k,0)) for k in ('full','partial','none','distorted'))
            columns = [compact_metric(rates[k]) for k in ('inclusive_miss','exact')]
            columns.append(compact_metric(rates['project_judged']) + ' / ' + compact_metric(rates['project_accuracy']))
            columns.extend(compact_metric(rates[k]) for k in ('traceable','pointer_success','parent_full'))
            lines.append(f"| {ds} | {LABELS[system]} | {distribution} | " + ' | '.join(columns) + ' |')
    lines += ['', '| 数据集 | 系统 | 必要元素缺失 | 误收 N | 部分有依据 | 含部分的依据不足或错误 | 去摘录错记 | 范围扩大 / 陈旧 / 验证边界错误 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for system, d in report['systems'].items():
        for ds, info in d['diagnosis'].items():
            cols = [compact_metric(info['rates'][k]) for k in
                    ('necessary_missing', 'negatives', 'partial', 'insufficient', 'errors_without_excerpts')]
            cols.append(' / '.join(str(info['counts'].get('precision_' + k, 0))
                                   for k in ('scope_expansion', 'outdated_as_current', 'validation_boundary_error')))
            lines.append(f"| {ds} | {LABELS[system]} | " + ' | '.join(cols) + ' |')
    lines += ['', '| 数据集 | 系统 | 条目 / 陈述 | 摘录条目 / 字符 | 整段对话照搬 | 外部上下文 / 重复 | 召回冲突 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for system, d in report['systems'].items():
        for ds, info in d['diagnosis'].items():
            c = Counter(info['counts'])
            lines.append(f"| {ds} | {LABELS[system]} | {c['items']} / {c['precision_claims']} | "
                         f"{c['excerpt_claims']} / {c['precision_excerpt_chars']:,} | {c['precision_verbatim_chat']} | "
                         f"{c['precision_external_context']} / {c['precision_duplicate']} | {c['conflict']} |")
    lines += ['', '<details>', '<summary>诊断区间及有效重抽次数</summary>', '',
              '| 数据集 | 系统 | 漏记率 [95% 区间] / 有效重抽 | 错记率 [95% 区间] / 有效重抽 |',
              '|---|---|---:|---:|']
    for system, d in report['systems'].items():
        for ds, info in d['diagnosis'].items():
            bounds = []
            for key in ('miss', 'claim_errors'):
                entry = info['intervals'][key]
                lo, hi = entry['ci95']
                bounds.append(f"[{fmt(lo, True)}, {fmt(hi, True)}] / {entry['valid_resamples']}")
            lines.append(f"| {ds} | {LABELS[system]} | " + ' | '.join(bounds) + ' |')
    lines += ['', '</details>', '', '### 轮长分层', '',
              '每格列 coding 段数、独立来源数及已判定任务全对率。', '',
              '| 系统 | ≤50 | 51–150 | >150 |', '|---|---:|---:|---:|']
    for system, d in report['systems'].items():
        groups = [f"{x['samples']} 段 / {x['clusters']} 源 · {compact_metric(x['success'])}"
                  for x in d['strata_success'].values()]
        lines.append(f"| {LABELS[system]} | " + ' | '.join(groups) + ' |')
    lines += ['', '## 5　写入成本', '', '| 系统 | 已计量写入输入 token（coding） | 写入用量缺失段数 |', '|---|---:|---:|']
    for system, d in report['systems'].items():
        rows = list(d['rows'].values())
        tokens = [r['write_tokens'] for r in rows if r['write_tokens'] is not None]
        lines.append(f"| {LABELS[system]} | {format(sum(tokens), ',') if tokens else 'NA'} | "
                     f"{sum(r['write_multiple'] is None for r in rows)} |")
    lines += ['', '| 数据集 | 系统 | 回放完成 | 写入 LLM 调用 | 已计量输入 / 输出 token | 未计量输入段 | 缓存 token / 有计量的调用 | 外部向量调用 | 回放中位秒数 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for system, rows in report['writer_ledger'].items():
        for ds, r in rows.items():
            cache = f"{r['cached_input']:,} / {r['cached_known']}" if r['cached_known'] else 'NA'
            lines.append(f"| {ds} | {LABELS[system]} | {r['samples']} | {r['llm_calls']} | "
                         f"{r['llm_input']:,} / {r['llm_output']:,} | {r['write_input_missing']} | {cache} | "
                         f"{r['external_embedding_calls']} | {fmt(r['median_seconds'])} |")
    lines += ['', 'W 使用写入供应商在 usage.prompt_tokens 中返回的输入用量。'
              'T 和 R 由 o200k_base 计量。W/T 是不同 tokenizer 下的扣分口径，缓存用量另列。'
              '综合分在参与价值分的 coding 段上取成本均值。整段超窗仍保留实际写入账。', '',
              '<details>', '<summary>已形成判定记录的评测调用用量</summary>', '',
              '| 系统 | 阶段 | 调用 | 有输入 / 输出计量的调用 | 已计量输入 / 输出 token |',
              '|---|---|---:|---:|---:|']
    for system in LABELS:
        for stage, row in report['assessment_usage'].get(system, {}).items():
            lines.append(f"| {LABELS[system]} | {stage} | {row['calls']} | "
                         f"{row['input_measured']} / {row['output_measured']} | "
                         f"{row['input']:,} / {row['output']:,} |")
    lines += ['', '</details>', '', '## 6　九格敏感性', '']
    for system, d in report['systems'].items():
        lines += [f'### {LABELS[system]}', '', '| a / b | 1 | 2.5 | 4 |', '|---|---:|---:|---:|']
        for a in (2, 4, 6):
            cells = [fmt(d['sensitivity'][f'{a}/{b}']) for b in (1, 2.5, 4)]
            if a == 4:
                cells[1] = '**' + cells[1] + '**'
            lines.append(f"| {a} | " + ' | '.join(cells) + ' |')
        lines.append('')
    reversals, ties = [], []
    for left, right in combinations(PRODUCTS, 2):
        a, b = (report['systems'][name]['common_sensitivity'] for name in (left, right))
        signs = {(av > b[k]) - (av < b[k]) for k, av in a.items() if av is not None and b[k] is not None}
        if {-1, 1} <= signs:
            reversals.append(f'{LABELS[left]} / {LABELS[right]}')
        if 0 in signs:
            ties.append(f'{LABELS[left]} / {LABELS[right]}')
    lines += ['九格产品间排序在共同有效样本上核对。'
              '差值变号 ' + ('、'.join(reversals) or '无可报告项') + '。'
              '并列 ' + ('、'.join(ties) or '无可报告项') + '。', '']
    if report['paired']:
        cohort = '、'.join(f'`{sample}`' for sample in report['common_trace'])
        lines += ['## 共同样本的配对差', '', f'五个产品共同有效的样本为 {cohort}。'
                  '缺场景的重抽无定义。区间收为一点时，表示有定义的重抽给出了相同结果。', '',
                  '| 产品差值 | 综合分差 [95% 区间] / 有效重抽 | 价值分差 [95% 区间] / 有效重抽 |',
                  '|---|---:|---:|']
        def ci(row):
            return (f'{fmt(row["value"])} [{fmt(row["ci95"][0])}, {fmt(row["ci95"][1])}]'
                    f" / {row['valid_resamples']}")
        for pair, values in report['paired'].items():
            left, right = pair.split(' - ')
            lines.append(f"| {LABELS[left]} − {LABELS[right]} | {ci(values['composite'])} | {ci(values['value'])} |")
        lines.append('')
    lines += ['## 样本与局限', '',
              '本轮为框架 §8 的缩减版，共 4 段 coding 和 4 段日常对话，覆盖范围为表 1 的子集。'
              '每段 coding 冻结 3 题。每个配置和样本只评一次，小样本区间及缺项限制总体比较。'
              'HaluMem 仅作摄取诊断，两条作答基线用于 coding。'
              '区间来自按来源聚类的 10,000 次重抽（seed 20261112）。', '']
    review = report['x2_review']
    if review['submitted']:
        lines += ['X2 审核的落盘终态只有通过和驳回，没有修订档。'
                  f"主轮 {review['submitted']} 条草案驳回 {review['rejected']} 条，"
                  f"其中 {review['unsupported']} 条主因是细节无据或未确认。"]
        if review.get('halumem_future'):
            lines[-1] += ('部分 HaluMem 会话日期晚于审核运行日，审核将这些日期视为未来，'
                          f"并在 {review['halumem_future']} 条驳回中引用这一点。")
        lines[-1] += '这些观测限于本轮 X2 的审核执行方式，标准本身的效果超出这一观察范围。'
        lines.append('')
    lines += [
              '五个产品使用同一批冻结 transcript。按本版修订重放适用写入配置，其余写入复用冻结导出。'
              '初次修订补了题面截点并重答重判。最终 S5 复算沿用已有答卷和判分。', '',
              '| 配置 | 实际摄取及导出 | 写入请求主要参数 |',
              '|---|---|---|',
              '| OpenViking 0.5.4.1 | 原生长期记忆提交，包含归档的完整会话附件。使用端点所需的外部向量配置。 | `temperature=0`，`max_tokens=32768`，`enable_thinking=false` |',
              '| ai-memory 3.4.7 | 原生 hook 事件与一次单页整理。使用本地向量。 | `temperature=0.2`，`max_tokens=32000`，JSON Schema |',
              '| claude-mem 15.17.0 | 原生观察与阶段总结。使用本地 Chroma。 | `temperature=0.3`，`max_tokens=4096`，`reasoning.enabled=false`，`stream=true` |',
              '| Basic Memory 1.0.0 + agent | 固定 agent 采用官方 `memory-capture` 提示，经原生工具写入笔记。使用本地 FastEmbed。 | `stream=false`，温度与输出上限沿用端点默认 |',
              '| X2 | 固定摄取 agent 与原生自动审核。读者只收到审核后的正式页。 | 摄取 agent 用 `stream=false`，审核用 JSON 对象，温度与输出上限沿用端点默认 |', '']
    for sid, r in report['samples'].items():
        lines.append(f"- `{sid}` · {r['turns']} 轮，{r['tasks']} 题，冻结 SHA-256 前 16 位 `{r['transcript_sha256'][:16]}`")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--date', required=True)
    parser.add_argument('--task-exclusions', type=Path)
    args = parser.parse_args()
    output = private_path(args.output_dir)
    exclusions = read_json(args.task_exclusions) if args.task_exclusions else ()
    root = private_path(args.data_dir)
    report = summarize(root, exclusions)
    write_json(output / 'metrics.json', report)
    (output / 'README.md').write_text(render(report, args.date, status_summaries(root, report)))


if __name__ == '__main__':
    main()
