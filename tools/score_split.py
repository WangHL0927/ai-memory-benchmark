"""拆分判定评分：漏记、误记、政策参考项和粒度；不混入旧路线指标。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from tools import score
from tools.config import sha256_file, environment
from tools.model.judge_split import CLAIM_FIELDS, VERSIONS, input_hashes, memories, run_failure, validate_claim_counts, validate_judgment
from tools.paths import ROOT
from tools.trial import identifier, output_path, read_json, write_json

METRIC_KEYS = ('miss_rate', 'false_memory_rate', 'claims_per_sample', 'memories_per_sample', 'chars_per_memory')
SEED = 20261112
WORTH_NOTICE = '与政策相关，不同产品未拿到同一份政策，仅供参考'
LABELS = {
    'miss_rate': '漏记率', 'uncovered_rate': '完全没记住率', 'partial_rate': '部分记住率',
    'miss_rate_inclusive': '漏记率（含 uncertain）', 'uncovered_rate_inclusive': '完全没记住率（含 uncertain）',
    'partial_rate_inclusive': '部分记住率（含 uncertain）', 'false_memory_rate': '误记率（按陈述）',
    'false_memory_rate_by_memory': '误记率（按条，辅助）', 'claims_per_sample': '每段陈述数',
    'verbatim_memory_share': '含原文摘录的记忆占比',
    'insufficient_rate': '依据不足率', 'not_worth_rate': '不值得记占比',
    'memories_per_sample': '每段平均记忆条数', 'chars_per_memory': '每条平均字符数',
    'chars_per_sample': '每段平均总字符数'}


def unique(values):
    return [json.loads(v) for v in sorted({json.dumps(v, sort_keys=True, ensure_ascii=False) for v in values})]


def basic_memory(name, metadata=()):
    names = [name] + [str(v.get('name', '')) for v in metadata if isinstance(v, dict)]
    return any('basicmemory' in ''.join(c for c in n.lower() if c.isalnum()) for n in names)


def display_name(name, metadata=()):
    return 'Basic Memory + 固定 agent 的组合（' + name + '）' if basic_memory(name, metadata) else name


def counts(evaluation, allow_legacy=False):
    gold, judgment = evaluation['gold'], evaluation['judgment']
    states = {v['unit_id']: v['verdict'] for v in judgment['recall']}
    units = [u for u in gold['U'] if u['origin'] == 'blind']
    c = dict(samples=1, memories=len(evaluation['memories']),
             chars=sum(len(m['statement']) for m in evaluation['memories']))
    for suffix, inclusive in (('', False), ('_inclusive', True)):
        selected = [u for u in units if inclusive or not u['uncertain']]
        c['units' + suffix] = len(selected)
        c['miss' + suffix] = sum(states[u['unit_id']] != 'full' for u in selected)
        c['uncovered' + suffix] = sum(states[u['unit_id']] in ('none', 'distorted') for u in selected)
        c['partial' + suffix] = sum(states[u['unit_id']] == 'partial' for u in selected)
    p = judgment['precision']
    has_claims = all(all(k in v for k in (*CLAIM_FIELDS, 'verbatim')) for v in p)
    score.require(has_claims or allow_legacy, '精确判定缺少 v4 陈述计数，需要重新判定')
    for v in p:
        if any(k in v for k in CLAIM_FIELDS):
            validate_claim_counts(v)
    c['claims_missing_samples'] = int(not has_claims)
    if has_claims:
        c.update(claims=sum(v['claims'] for v in p), unsupported_claims=sum(v['unsupported'] for v in p),
                 partial_claims=sum(v['partial'] for v in p), distorted_claims=sum(v['distorted_claims'] for v in p),
                 claim_errors=sum(v['unsupported'] + v['distorted_claims'] for v in p),
                 verbatim_memories=sum(v['verbatim'] for v in p))
    c.update(errors=sum(v['grounded'] == 'no' or v['distorted'] for v in p),
             insufficient=sum(v['grounded'] == 'partial' for v in p), not_worth=sum(v['worth'] == 'no' for v in p))
    return c


def metrics(c):
    def ratio(n, d):
        numerator, denominator = c.get(n, 0), c.get(d, 0)
        return dict(numerator=numerator, denominator=denominator,
                    value=numerator / denominator if denominator else None)
    result = {}
    for suffix in ('', '_inclusive'):
        for name, count in (('miss_rate', 'miss'), ('uncovered_rate', 'uncovered'), ('partial_rate', 'partial')):
            result[name + suffix] = ratio(count + suffix, 'units' + suffix)
    for name, n, d in (('false_memory_rate', 'claim_errors', 'claims'),
                       ('false_memory_rate_by_memory', 'errors', 'memories'),
                       ('claims_per_sample', 'claims', 'samples'),
                       ('verbatim_memory_share', 'verbatim_memories', 'memories'), ('insufficient_rate', 'insufficient', 'memories'),
                       ('not_worth_rate', 'not_worth', 'memories'), ('memories_per_sample', 'memories', 'samples'),
                       ('chars_per_memory', 'chars', 'memories'), ('chars_per_sample', 'chars', 'samples')):
        result[name] = (dict(numerator=None, denominator=None, value=None)
                        if c.get('claims_missing_samples') and name in ('false_memory_rate', 'claims_per_sample', 'verbatim_memory_share')
                        else ratio(n, d))
    return result


def summarize(evaluations, iterations=10000, seed=SEED):
    """同一 session_id 的所有切段整组抽取；先求和再算比率（微平均）。"""
    score.require(iterations > 0, 'bootstrap 次数必须为正')
    clusters = defaultdict(list)
    for e in evaluations:
        clusters[e['gold']['session_id']].append(counts(e))
    vectors = [score.add_counts(clusters[k]) for k in sorted(clusters)]
    total = score.add_counts(vectors)
    out, draws, rng = metrics(total), defaultdict(list), random.Random(seed)
    for _ in range(iterations if vectors else 0):
        sampled = metrics(score.add_counts(vectors[rng.randrange(len(vectors))] for _ in vectors))
        for key, value in sampled.items():
            if value['value'] is not None:
                draws[key].append(value['value'])
    for key, value in out.items():
        value.update(ci95=[score.percentile(draws[key], .025), score.percentile(draws[key], .975)],
                     bootstrap_valid=len(draws[key]))
    return dict(counts=total, clusters=len(clusters), metrics=out)


def native_difference(metadata):
    """只摘流程差异，不把逐次工具调用或运行轨迹塞进精简报告。"""
    result = {k: metadata.get(k, '未记录') for k in
              ('configuration', 'deployment', 'isolation', 'embedding')}
    context = metadata.get('context')
    context_fields = ('window', 'header_metadata_forwarded', 'adapter_input_truncated',
                      'product_input_truncated', 'output_truncated', 'input_truncated')
    result['context'] = {k: context[k] for k in context_fields if k in context} if isinstance(context, dict) else '未记录'
    agent = metadata.get('agent')
    result['agent'] = {k: agent.get(k, '未记录') for k in ('max_rounds', 'max_total_tokens', 'stop_reason')} if isinstance(agent, dict) else '未记录'
    return result


def system_summary(records):
    """每系统一份简表；调用次数按运行累加，不能先对同配置去重。"""
    models, sources = set(), set()
    total, missing = dict(llm=0, embedding=0), dict(llm=0, embedding=0)
    for record in records:
        record = record if isinstance(record, dict) else {}
        model = record.get('llm')
        if isinstance(model, dict):
            model = model.get('model') or model.get('name')
        models.add(str(model or '未记录'))
        product = record.get('product') or record.get('package') or {}
        sources.add(str(record.get('prompt_version') or product.get('prompt_source') or '未记录'))
        for kind in total:
            value = record.get('usage', {}).get(kind, {}).get('calls')
            if type(value) is not int or value < 0:
                calls = record.get('calls')
                value = sum(c.get('kind') == kind for c in calls) if isinstance(calls, list) else None
            if value is None:
                missing[kind] += 1
            else:
                total[kind] += value
    def calls(kind):
        if not records or missing[kind] == len(records):
            return '未记录'
        return str(total[kind]) + (f"（另 {missing[kind]} 次运行未记录）" if missing[kind] else '')
    return dict(model='、'.join(sorted(models)) or '未记录', prompt_source='；'.join(sorted(sources)) or '未记录',
                llm_calls=calls('llm'), embedding_calls=calls('embedding'))


def identity_summary(records):
    return unique({k: item.get(k, '未记录') for k in
                   ('tool', 'model', 'reasoning_effort', 'cli_version', 'prompt_version', 'prompt_versions')}
                  if isinstance(item, dict) else '未记录' for item in records)


def system_identity_lines(report):
    for name, summary in report['system_summary'].items():
        yield (f"{name}：模型 {summary['model']}；提示词来源 {summary['prompt_source']}；"
               f"调用合计 LLM {summary['llm_calls']} / embedding {summary['embedding_calls']}")


def build_report(data_dir, systems=None, iterations=10000, seed=SEED, agreement=None):
    data_dir = Path(data_dir)
    score.require(iterations > 0, 'bootstrap 次数必须为正')
    samples = sorted(p for p in data_dir.iterdir() if p.is_dir() and
                     any((p / name).exists() for name in ('gold.json', 'transcript.md', 'runs', 'judgments')))
    score.require(bool(samples), '未找到样本')
    names = sorted(set(systems or [p.stem for s in samples for folder in ('runs', 'judgments') for p in (s / folder).glob('*.json')]))
    score.require(bool(names), '没有系统标识；全无 run 时请用 --systems 指定被测对象')
    for name in names:
        identifier(name)
    groups, failures, sources = defaultdict(lambda: defaultdict(list)), [], []
    metadata, annotators, judges = defaultdict(list), [], []
    dataset_samples, protocols = defaultdict(list), defaultdict(set)
    for sample in samples:
        gold = None
        dataset = data_dir.name
        try:
            gold = read_json(sample / 'gold.json')
            dataset = str(gold.get('dataset') or gold.get('source') or data_dir.name)
            score.validate_gold(gold)
            score.require(isinstance(gold['session_id'], str) and bool(gold['session_id']), 'session_id 无效')
            annotators.append(gold.get('annotator'))
            protocols[dataset].add(json.dumps([gold['standard_version'], gold['policy']], sort_keys=True))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            gold_error = str(exc)
        else:
            gold_error = None
        dataset_samples[dataset].append(sample.name)
        for name in names:
            run_path, judgment_path = sample / 'runs' / (name + '.json'), sample / 'judgments' / (name + '.json')
            stage, code = 'gold', 'invalid_gold'
            try:
                if gold_error:
                    raise ValueError(gold_error)
                stage, code = 'run', 'missing_run' if not run_path.exists() else 'invalid_run'
                run = read_json(run_path)
                score.require(isinstance(run, dict), 'run 必须为对象')
                metadata[name].append(run.get('system'))
                reason = run_failure(run)
                if reason:
                    code = 'replay_failed'
                    raise ValueError(reason)
                submitted = memories(run)
                score.require(run.get('standard_version') == gold['standard_version'] and run.get('policy') == gold['policy'], 'run 的框架或政策不匹配')
                stage, code = 'judgment', 'missing_judgment' if not judgment_path.exists() else 'invalid_judgment'
                hashes = input_hashes(sample, run_path)
                judgment = validate_judgment(read_json(judgment_path), gold, submitted, hashes)
                judges.append(judgment['judge'])
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                reason = str(exc)
                failures.append(dict(dataset=dataset, sample_id=sample.name, system=name, stage=stage, code=code, reason=reason))
                continue
            groups[dataset][name].append(dict(gold=gold, judgment=judgment, memories=submitted, sample_id=sample.name))
            for path in (sample / 'gold.json', sample / 'transcript.md', run_path, judgment_path):
                sources.append(dict(path=str(path), sha256=sha256_file(path)))
    # 不同政策或框架不能悄悄合并；分别运行即可产出各自的报告。
    for dataset, versions in protocols.items():
        score.require(len(versions) == 1, f'{dataset} 内框架或政策版本混用，请分开评分')
    report = dict(protocol='split-v1', precision_version=VERSIONS['precision'], false_memory_unit='claim', sample_count=len(samples), data_dir=str(data_dir), environment=environment(),
                  bootstrap=dict(iterations=iterations, seed=seed, confidence=.95, cluster='gold.session_id'),
                  annotators=unique(annotators), judges=unique(judges), systems={n: unique(metadata[n]) for n in names},
                  system_summary={n: system_summary(metadata[n]) for n in names},
                  agreement=agreement, calibration_notice='已附人工一致率；未设合格门槛' if agreement else '未经人工校准',
                  warnings=[WORTH_NOTICE, '失败或缺失样本不计入分母，也不当作 0 分；不同系统的有效样本可能不同。',
                            '区间按会话聚类；零错误或单会话区间可能退化，不能证明真实错误率为零。',
                            '精确判断 v4 按陈述计误记率；原文摘录整体计 1 个陈述，仍应结合陈述数、记忆条数和摘录占比理解结果。',
                            '漏记主口径只含 blind 且排除 uncertain；揭盲补写单元不进主指标。'],
                  files=unique(sources), datasets={}, failures=failures, failure_count=len(failures),
                  failure_reason_counts=dict(Counter(f['code'] for f in failures)))
    for dataset, sample_ids in sorted(dataset_samples.items()):
        table = dict(sample_count=len(sample_ids), samples=sample_ids, protocol_versions=[json.loads(p) for p in sorted(protocols[dataset])], systems={})
        for name in names:
            es = groups[dataset][name]
            excluded = [f for f in failures if f['dataset'] == dataset and f['system'] == name]
            table['systems'][name] = dict(display_name=display_name(name, metadata[name]),
                comparison_group='agent_combination' if basic_memory(name, metadata[name]) else 'product',
                valid_samples=[e['sample_id'] for e in es], failed_count=len(excluded), failures=excluded,
                overall=summarize(es, iterations, seed),
                by_layer={layer: summarize([e for e in es if e['gold']['layer'] == layer], iterations, seed)
                          for layer in sorted({e['gold']['layer'] for e in es})},
                samples=[dict(sample_id=e['sample_id'], session_id=e['gold']['session_id'], **counts(e)) for e in es])
        report['datasets'][dataset] = table
    report['native_differences'] = {n: unique(native_difference(v) for v in metadata[n] if isinstance(v, dict))
                                  or ['未记录'] for n in names}
    return report


def number(value):
    return '未定义' if value is None else f'{value:.4f}'


def cell(metric):
    return f"{number(metric['value'])}（{metric['numerator']:g}/{metric['denominator']:g}；95% CI {number(metric['ci95'][0])}～{number(metric['ci95'][1])}）"


def main_tables(table):
    """混有定向样本时，主表按来源层拆开；组合对象始终单列。"""
    layers = sorted({layer for result in table['systems'].values() for layer in result['by_layer']})
    for layer in layers if len(layers) > 1 else ['overall']:
        for group, title in (('product', '被测产品（按标识排列）'), ('agent_combination', '固定 agent 组合（单独展示）')):
            rows = [(n, r, r['overall'] if layer == 'overall' else r['by_layer'].get(layer))
                    for n, r in table['systems'].items() if r['comparison_group'] == group]
            if rows:
                yield layer + ' / ' + title, rows


def markdown(report):
    lines = ['# 拆分判定报告', '', f"协议 split-v1；样本 {report['sample_count']}；{report['calibration_notice']}。", '',
             f"会话级 bootstrap {report['bootstrap']['iterations']} 次，种子 {report['bootstrap']['seed']}，95% 区间。", '']
    from tools.config import summarize_identities
    lines += ['被测对象与模型参数：'] + ['- ' + x for x in system_identity_lines(report)] + ['']
    for label, key in (('标注者', 'annotators'), ('判定者', 'judges')):
        identities = identity_summary(report[key])
        lines += [label + '：'] + ['- ' + x for x in summarize_identities(identities)] + ['']
    lines += ['（完整的模型参数、提示词哈希和运行环境见 report.json）', '']
    if report['agreement']:
        lines += ['人工一致率：' + json.dumps(report['agreement'], ensure_ascii=False), '']
    lines += ['> ' + w for w in report['warnings']]
    for dataset, table in report['datasets'].items():
        lines += ['', f"## {dataset}（{table['sample_count']} 个样本）", '', '框架 / 政策：' + json.dumps(table['protocol_versions'], ensure_ascii=False)]
        for title, rows in main_tables(table):
            keys = METRIC_KEYS
            lines += ['', '### ' + title + ' / 主口径', '',
                      '| 系统 | 有效样本 | 排除样本 | ' + ' | '.join(LABELS[k] for k in keys) + ' |',
                      '|---|---:|---:|' + '---:|' * len(keys)]
            for name, result, summary in rows:
                if summary is None:
                    continue
                lines.append('| ' + result['display_name'] + f" | {summary['counts'].get('samples', 0):g} | {result['failed_count']} | " +
                             ' | '.join(cell(summary['metrics'][k]) if not k.startswith(('claims_', 'memories_', 'chars_')) else
                                        (f"{summary['metrics'][k]['value']:.1f}" if summary['metrics'][k].get('value') is not None else '—')
                                        for k in keys) + ' |')
    lines += ['', '## 失败或缺失', '', f"共 {report['failure_count']} 个系统 × 样本对（不计分母）。", '']
    lines += [f"- {f['dataset']} / {f['system']} / {f['sample_id']}：{f['code']}；{f['reason']}" for f in report['failures']]
    lines += ['', '## 产品与原生流程的差异', '', 'Basic Memory 按“Basic Memory + 固定 agent 的组合”单独展示。以下取自 run.system；未记录项不推断。', '']
    lines += ['- ' + n + '：' + json.dumps(v, ensure_ascii=False) for n, v in report['native_differences'].items()]
    lines += ['', '完整身份、样本粒度与分层结果见 report.json。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--systems', nargs='+')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--agreement', type=Path)
    parser.add_argument('--output', type=Path, help='路径前缀；默认 <data-dir>/report')
    args = parser.parse_args()
    try:
        target = output_path(args.output or args.data_dir / 'report', args.data_dir)
        report = build_report(args.data_dir, args.systems, args.bootstrap, args.seed,
                              read_json(args.agreement) if args.agreement else None)
        write_json(target.with_suffix('.json'), report)
        target.with_suffix('.md').write_text(markdown(report), encoding='utf-8')
        print(f"已输出 {target.with_suffix('.json')} 和 {target.with_suffix('.md')}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'错误：{exc}\n')


if __name__ == '__main__':
    main()
