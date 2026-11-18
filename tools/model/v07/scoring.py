"""v0.7.1 task credit, measured costs, and session-cluster intervals."""
from collections import Counter, defaultdict
from itertools import product
from math import log2
import random
from statistics import mean

from tools.model.v07.protocol import SCENARIOS
from tools.score import percentile, require

SEED = 20261112
ITERATIONS = 10_000
READ_COEFFICIENTS = (2, 4, 6)
WRITE_COEFFICIENTS = (1, 2.5, 4)


def ratio(n, d):
    return dict(numerator=n, denominator=d, value=n / d if d else None)


def log_penalty(x, multiplier):
    require(x >= 0, 'cost ratio must be nonnegative')
    return multiplier * log2(x) if x > 1 else 0


def write_input(record):
    """Return the provider's input usage, including writing calls that needed correction."""
    usage = record['usage']['llm']
    if usage['calls'] == 0:
        return 0
    tokens = usage['tokens'].get('prompt_tokens')
    return tokens if type(tokens) is int and tokens >= 0 else None


def grade_record(gold, result, excluded=()):
    outcomes = Counter()
    rows = [task for task in gold['tasks'] if task['task_id'] not in excluded]
    failed = Counter()
    for task in rows:
        tid, scope = task['task_id'], task['scenario']
        if tid not in result['tasks']:
            status = result['answers'].get(tid, {}).get('status', 'missing')
            failed[status] += 1
            continue
        verdict = result['tasks'][tid]
        outcomes[f'{scope}_valid'] += 1
        outcomes[f'{scope}_correct'] += verdict['verdict'] == 'correct'
        outcomes[verdict['verdict']] += 1
        for name in ('action_error', 'misleading', 'error_matches_memory', 'unclear_attribution', 'high_risk'):
            outcomes[name] += verdict[name]
        supported = sum(r['status'] == 'supported' for r in verdict['requirements'])
        required = len(verdict['requirements'])
        require(required > 0, 'scoring requires at least one mandatory point')
        outcomes[f'{scope}_credit'] += supported / required
        outcomes['accepted_requirements'] += supported
        outcomes['requirements'] += required
    valid = sum(outcomes[f'{scope}_valid'] for scope in SCENARIOS)
    read = [result['answers'][r['task_id']]['read_tokens'] for r in rows
            if r['task_id'] in result['tasks'] and r['task_id'] in result['answers']]
    read_avg = mean(read) if read and len(read) == valid else None
    input_tokens = write_input(result)
    require(gold['transcript_tokens'] > 0, 'invalid transcript tokens')
    write_multiple = input_tokens / gold['transcript_tokens'] if input_tokens is not None else None
    return dict(session_id=gold['session_id'], sample=gold['sample_id'], outcomes=dict(outcomes), failures=dict(failed),
                num_tasks=len(rows), valid_tasks=valid, read_tokens=read_avg, write_tokens=input_tokens,
                transcript_tokens=gold['transcript_tokens'], write_multiple=write_multiple)


def value_and_cost(rows, a=4, b=2.5):
    stats = Counter()
    for row in rows:
        stats.update(row['outcomes'])
    by_scenario = {s: ratio(stats[f'{s}_credit'], stats[f'{s}_valid']) for s in SCENARIOS}
    scenario_full = {s: ratio(stats[f'{s}_correct'], stats[f'{s}_valid']) for s in SCENARIOS}
    valid = sum(i['valid_tasks'] for i in rows)
    trials = sum(i['num_tasks'] for i in rows)
    misled = ratio(stats['misleading'], valid)
    complete = all(v['value'] is not None for v in by_scenario.values())
    value = max(0, 25 * sum(v['value'] for v in by_scenario.values()) - 10 * misled['value']) if complete else None
    scored = [i for i in rows if i['valid_tasks']]
    read_ratios = [i['read_tokens'] / i['transcript_tokens'] for i in scored if i['read_tokens'] is not None]
    write_ratios = [i['write_multiple'] for i in scored if i['write_multiple'] is not None]
    penalty_r = mean(log_penalty(r / .05, a) for r in read_ratios) if scored and len(scored) == len(read_ratios) else None
    penalty_w = mean(log_penalty(w, b) for w in write_ratios) if scored and len(scored) == len(write_ratios) else None
    margin = None if None in (value, penalty_r, penalty_w) else value - penalty_r - penalty_w
    return dict(composite=max(0, margin) if margin is not None else None, margin=margin, value=value,
                scenarios=by_scenario, scenario_full=scenario_full, misleading=misled,
                action_error_rate=ratio(stats['action_error'], valid), results=dict(stats),
                conservative_success=ratio(stats['correct'], trials), observed_success=ratio(stats['correct'], valid),
                read_penalty=penalty_r, write_penalty=penalty_w, valid_tasks=valid, total_tasks=trials)


def sensitivity(rows):
    return {f'{a}/{b}': value_and_cost(rows, a, b)['composite']
            for a, b in product(READ_COEFFICIENTS, WRITE_COEFFICIENTS)}


def diagnostics(gold, result):
    status = {r['unit_id']: r for r in result['recall']}
    certain = [u for u in gold['U'] if u['origin'] == 'blind' and not u['uncertain']]
    all_blind = [u for u in gold['U'] if u['origin'] == 'blind']
    counts = Counter()
    counts['units'] = len(certain)
    counts['units_inclusive'] = len(all_blind)
    for unit in all_blind:
        row = status[unit['unit_id']]
        counts['miss_inclusive'] += row['verdict'] != 'full'
        if unit['uncertain']:
            continue
        counts[row['verdict']] += 1
        counts['verbatim_full'] += row['verbatim_full']
        counts['conflict'] += row['conflict']
        necessary = {e['element_id']: e for e in unit['elements'] if e['required']}
        for elem in row['elements']:
            if elem['element_id'] not in necessary:
                continue
            counts['necessary'] += 1
            counts['necessary_missing'] += elem['status'] == 'missing'
            if necessary[elem['element_id']]['representation'] == 'pointer':
                counts['pointers'] += 1
                counts['located'] += elem['status'] == 'supported'
    if gold['dataset'] == 'halumem':
        parents = defaultdict(list)
        for unit in certain:
            parents[unit['parent_point']].append(status[unit['unit_id']]['verdict'] == 'full')
        counts['parents'] = len(parents)
        counts['parents_full'] = sum(all(v) for v in parents.values())
    names = ('claims', 'distorted', 'unsupported', 'partial', 'excerpt_chars', 'excerpt_errors')
    flags = ('contains_secret', 'verbatim_chat', 'traceable', 'external_context', 'duplicate',
             'scope_expansion', 'outdated_as_current', 'validation_boundary_error')
    counts['items'] = len(result['precision'])
    counts['negatives'] = len(gold['N'])
    hit = set()
    for row in result['precision']:
        for name in (*names, *flags):
            counts['precision_' + name] += row[name]
        counts['excerpt_claims'] += bool(row['excerpt_chars'])
        counts['worth_no'] += row['worth'] == 'no'
        counts['scope_applicable'] += row['scope'] != 'inapplicable'
        counts['scope_judged'] += row['scope'] in ('correct', 'incorrect')
        counts['scope_correct'] += row['scope'] == 'correct'
        hit.update(row['matched_negatives'])
    counts['negatives_hit'] = len(hit)
    return counts


def diagnostic_rates(c):
    claims, errors = c['precision_claims'], c['precision_distorted'] + c['precision_unsupported']
    return dict(miss=ratio(c['partial'] + c['none'] + c['distorted'], c['units']),
                uncovered=ratio(c['none'] + c['distorted'], c['units']),
                inclusive_miss=ratio(c['miss_inclusive'], c['units_inclusive']),
                exact=ratio(c['verbatim_full'], c['units']), parent_full=ratio(c['parents_full'], c['parents']),
                claim_errors=ratio(errors, claims), partial=ratio(c['precision_partial'], claims),
                insufficient=ratio(errors + c['precision_partial'], claims),
                errors_without_excerpts=ratio(errors - c['precision_excerpt_errors'], claims - c['excerpt_claims']),
                negatives=ratio(c['negatives_hit'], c['negatives']), worth_no=ratio(c['worth_no'], c['items']),
                project_judged=ratio(c['scope_judged'], c['scope_applicable']),
                project_accuracy=ratio(c['scope_correct'], c['scope_judged']),
                traceable=ratio(c['precision_traceable'], c['items']), pointer_success=ratio(c['located'], c['pointers']),
                necessary_missing=ratio(c['necessary_missing'], c['necessary']))


def resampling(groups, metric, iterations=ITERATIONS, seed=SEED):
    """metric takes an iterable of sampled source groups and returns named scalars."""
    groups = list(groups)
    observed = metric(groups)
    draws = defaultdict(list)
    if len(groups) > 1:
        rng = random.Random(seed)
        for _ in range(iterations):
            row = metric([groups[rng.randrange(len(groups))] for _ in groups])
            for key, value in row.items():
                if value is not None:
                    draws[key].append(value)
    return {k: dict(value=v, valid_resamples=len(draws[k]),
                    ci95=[percentile(draws[k], .025), percentile(draws[k], .975)])
            for k, v in observed.items()}


def flatten(groups):
    return [value for group in groups for value in group]


def paired_values(left, right, iterations=ITERATIONS, seed=SEED):
    matched = defaultdict(list)
    for sample in sorted(left.keys() & right.keys()):
        require(left[sample]['session_id'] == right[sample]['session_id'], 'paired cluster mismatch')
        matched[left[sample]['session_id']].append((left[sample], right[sample]))

    def metric(groups):
        pairs = flatten(groups)
        a = value_and_cost([l for l, _ in pairs])
        b = value_and_cost([r for _, r in pairs])
        return {key: a[key] - b[key] if a[key] is not None and b[key] is not None else None
                for key in ('composite', 'value', 'read_penalty', 'write_penalty')}
    return resampling([matched[k] for k in sorted(matched)], metric, iterations, seed)
