"""Evaluate one frozen sample and one anonymous memory view."""
from concurrent.futures import ThreadPoolExecutor
import json

from tools.config import sha256_file, sha256_text
from tools.model.judge_split import run_failure
from tools.model.v07.cli import codex, read_answer, tokenizer
from tools.model.v07.protocol import (COVERAGE, ELEMENTS, POLICIES, bool_field,
                                     contract, material_prompt)
from tools.paths import ROOT
from tools.score import require
from tools.trial import read_json, write_json

BATCH_SIZE = 30
BASELINES = frozenset({'no-memory', 'full-transcript'})


def memory_view(run):
    """Product IDs and infrastructure paths are omitted. Native scope and source stay visible."""
    result = []
    for i, entry in enumerate(run['D'], 1):
        draft = entry['draft']
        require(isinstance(draft['statement'], str) and draft['statement'].strip(), 'missing exported memory')
        value = dict(memory_id=f'm{i}', statement=draft['statement'])
        for key in ('scope', 'source', 'title'):
            if draft.get(key):
                value[key] = draft[key]
        value.update({k: v for k, v in entry.get('reader_metadata', {}).items() if k in ('scope', 'source', 'title') and v})
        result.append(value)
    return result


def serialize(rows):
    # JSON keeps the native order, delimiting metadata and text without rewriting them.
    return json.dumps(rows, ensure_ascii=False) if rows else ''


def validate_refs(refs, ids):
    require(type(refs) is list and set(refs) <= set(ids) and len(refs) == len(set(refs)), 'invalid memory references')


def validate_recall(value, units, memory):
    found = contract(value, 'unit_id', [row['unit_id'] for row in units])
    by_id = {m['memory_id']: m for m in memory}
    for unit in units:
        record = found[unit['unit_id']]
        require(record['verdict'] in COVERAGE, 'invalid recall verdict')
        bool_field(record['conflict'])
        bool_field(record['verbatim_full'])
        expected = {e['element_id']: e for e in unit['elements']}
        rows = contract(record['elements'], 'element_id', expected)
        for row in rows.values():
            require(row['status'] in ELEMENTS, 'invalid element status')
            validate_refs(row['memory_ids'], by_id)
            quote = row['quote']
            require(isinstance(quote, str) and (not quote or any(
                quote in by_id[mid]['statement'] or any(quote in body for key, body in by_id[mid].items()
                                                       if key in ('scope', 'source', 'title') and isinstance(body, str))
                for mid in row['memory_ids'])), 'invalid recall quote')
            if row['status'] in ('supported', 'contradicted'):
                require(bool(row['memory_ids']), 'element support requires a reference')
        required = [rows[k]['status'] for k, e in expected.items() if e['required']]
        inferred = ('full' if all(s == 'supported' for s in required) else 'distorted' if 'contradicted' in required
                    else 'partial' if 'supported' in required else 'none')
        # Apply the frozen rule over validated element outcomes, even when the judge's
        # redundant unit label disagrees. The unmodified judge response stays in the receipt.
        record['verdict'] = inferred
        record['verbatim_full'] = record['verbatim_full'] and inferred == 'full'
    return value


def validate_precision(value, items, negatives):
    contract(value, 'memory_id', [row['memory_id'] for row in items])
    lengths = {r['memory_id']: len(r['statement']) for r in items}
    for row in value:
        for key in ('claims', 'distorted', 'unsupported', 'partial', 'excerpt_chars', 'excerpt_errors'):
            require(type(row[key]) is int and row[key] >= 0, f'invalid {key}')
        require(row['distorted'] + row['unsupported'] + row['partial'] <= row['claims'], 'claim counts overlap')
        require(row['excerpt_chars'] <= lengths[row['memory_id']], 'invalid excerpt length')
        require(row['excerpt_errors'] <= min(bool(row['excerpt_chars']), row['distorted'] + row['unsupported'])
                and (not row['excerpt_chars'] or row['claims'] >= 1), 'invalid excerpt errors')
        require(row['worth'] in ('yes', 'no', 'unclear'), 'invalid worth')
        require(row['scope'] in ('correct', 'incorrect', 'unknown', 'inapplicable'), 'invalid scope')
        validate_refs(row['matched_negatives'], [item['unit_id'] for item in negatives])
        for key in ('contains_secret', 'verbatim_chat', 'traceable', 'external_context', 'duplicate',
                    'scope_expansion', 'outdated_as_current', 'validation_boundary_error'):
            bool_field(row[key])
        require(isinstance(row['errors'], list) and len(row['errors']) <= 10, 'invalid error sample')
    return value


def validate_tasks(value, tasks, memory):
    records = contract(value, 'task_id', tasks)
    for key, row in records.items():
        require(row['verdict'] in ('correct', 'partial', 'specific_abstention', 'wrong'), 'invalid task verdict')
        requirements = contract(row['requirements'], 'requirement_id',
                                [item['requirement_id'] for item in tasks[key]['requirements']])
        for point in requirements.values():
            require(point['status'] in ('supported', 'missing', 'contradicted'), 'invalid task requirement verdict')
            validate_refs(point['memory_ids'], [r['memory_id'] for r in memory])
        for name in ('action_error', 'error_matches_memory', 'actual_reuse', 'misleading',
                     'unclear_attribution', 'high_risk'):
            bool_field(row[name])
        require(isinstance(row['invalid_reason'], str) and isinstance(row['note'], str), 'expected notes')
        require(not row['actual_reuse'] or bool(memory), 'empty memory cannot be reused')
        has_signals = row['action_error'] and row['error_matches_memory'] and row['actual_reuse']
        require(not row['misleading'] or has_signals, 'attributed misleading answer requires all evidence signals')
        if has_signals and not row['misleading']:
            # Retain an explicitly non-attributed result as unclear when the signals disagree.
            row['unclear_attribution'] = True
        require(row['verdict'] != 'correct' or
                (not row['action_error'] and all(r['status'] == 'supported' for r in requirements.values())),
                'correct result requires all points')
    return value


def batch(iterable, size=BATCH_SIZE):
    for pos in range(0, len(iterable), size):
        yield iterable[pos:pos+size]


def evaluate(sample, system, *, diagnosis=True, precision_workers=1, precision_batch_size=BATCH_SIZE):
    goldfile = sample / 'gold.json'
    gold = read_json(goldfile)
    inputfile = sample / 'transcript.md'
    source = inputfile.read_text()
    require(sha256_text(source) == gold['transcript_sha256'], 'frozen input drift')
    if diagnosis or system in BASELINES:
        target = sample / 'v07-results' / f'{system}.json'
    else:
        target = sample / 'v07-task-results' / f'{system}.json'
    raw = sample / 'runs' / f'{system}.json'
    if system in BASELINES:
        require(bool(gold['tasks']), 'reader baseline requires tasks')
        mem = [] if system == 'no-memory' else [dict(memory_id='m1', statement=source)]
        usage = dict(llm=dict(calls=0, tokens=dict(prompt_tokens=0, completion_tokens=0)))
    else:
        run = read_json(raw)
        require(not run_failure(run) and run.get('standard_version') == 'v0.7', 'replay is not valid for this round')
        require(run['system']['input_sha256'] == gold['transcript_sha256'], 'replay input differs from frozen transcript')
        mem = memory_view(run)
        usage = run['system']['usage']
    write_model = None if system in BASELINES else run['system'].get('llm')
    inputs = dict(gold=sha256_file(goldfile), transcript=sha256_file(inputfile),
                  run=sha256_file(raw) if system not in BASELINES else None)
    if target.exists():
        previous = read_json(target)
        require(previous['inputs'] == inputs, 'result inputs changed')
        return previous
    policy = (ROOT / 'policies' / f'{POLICIES[gold["dataset"]]}.md').read_text()
    result = dict(sample_id=gold['sample_id'], session_id=gold['session_id'], dataset=gold['dataset'], system=system,
                  turns=gold['turns'], inputs=inputs, transcript_tokens=gold['transcript_tokens'],
                  usage=usage, write_model=write_model, evaluation_protocol='v0.7.1',
                  empty=(not mem), calls=[], answers={}, tasks={}, recall=[], precision=[],
                  diagnosis_complete=diagnosis and system not in BASELINES,
                  x2_reviews=run['system'].get('reviews') if system not in BASELINES else None)
    count, meta = tokenizer()
    require(meta == gold['tokenizer'], 'tokenizer must match frozen transcript')
    result['tokenizer'] = meta

    def judge(stage, payload, validator, sequence=0):
        prompt = material_prompt(stage, payload)
        receipt_path = sample / 'v07-judge-stages' / system / f'{stage}-{sequence}.json'
        prompt_sha = sha256_text(prompt)
        if receipt_path.exists():
            old = read_json(receipt_path)
            if old['inputs'] == inputs and old['prompt_sha256'] == prompt_sha:
                return validator(old['result']), old['calls']
        rows, receipts = codex(stage, prompt, validator)
        write_json(receipt_path, dict(inputs=inputs, prompt_sha256=prompt_sha, result=rows, calls=receipts))
        return rows, receipts
    for task in gold['tasks']:
        aid = task['task_id']
        answer_file = sample / 'v07-answers' / system / f'{aid}.json'
        if answer_file.exists():
            answer = read_json(answer_file)
            require(answer['inputs'] == inputs, 'answer inputs changed')
        else:
            answer = read_answer(serialize(mem), task['question'], count)
            answer['inputs'] = inputs
            write_json(answer_file, answer)
        result['answers'][aid] = answer
    completed = {t['task_id']: t for t in gold['tasks'] if result['answers'][t['task_id']]['status'] == 'answered'}
    if completed:
        payload = dict(tasks=list(completed.values()), memory=mem,
                       answers={k: result['answers'][k]['answer'] for k in completed})
        records, calls = judge('tasks', payload, lambda v: validate_tasks(v, completed, mem))
        result['calls'].extend(calls)
        result['tasks'] = {row['task_id']: row for row in records}
    if diagnosis and system not in BASELINES:
        unit_fields = ('unit_id', 'statement', 'expected_use', 'scope', 'elements')
        element_fields = ('element_id', 'statement', 'role', 'required', 'representation', 'equivalents', 'locator', 'type')
        for index, group in enumerate(batch(gold['U'])):
            units = [{k: u[k] if k != 'elements' else [{j: e[j] for j in element_fields if j in e}
                                                      for e in u[k]] for k in unit_fields} for u in group]
            rows, calls = judge('recall', dict(units=units, memory=mem),
                                lambda v: validate_recall(v, group, mem), index)
            result['calls'].extend(calls)
            result['recall'].extend(rows)
        nrows = [{k: row[k] for k in ('unit_id', 'statement', 'reject_reason')} for row in gold['N']]

        def precision(job):
            index, group = job
            payload = dict(transcript=source, policy=policy, negatives=nrows, memory=group)
            if precision_batch_size != BATCH_SIZE:
                payload['memory_character_limits'] = {m['memory_id']: len(m['statement']) for m in group}
            return judge('precision', payload, lambda v: validate_precision(v, group, gold['N']), index)

        require(1 <= precision_batch_size <= BATCH_SIZE, 'invalid precision batch size')
        jobs = list(enumerate(batch(mem, precision_batch_size)))
        if precision_workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=precision_workers) as executor:
                judgments = list(executor.map(precision, jobs))
        else:
            judgments = map(precision, jobs)
        for rows, calls in judgments:
            result['calls'].extend(calls)
            result['precision'].extend(rows)
    write_json(target, result)
    return result
