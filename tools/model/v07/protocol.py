"""Contracts shared by annotation, scoring, and validation."""
import json
import re

from tools.config import sha256_text
from tools.paths import ROOT
from tools.score import require
from tools.trial import now, read_json, transcript, write_json
from tools.model.v07.cli import codex, tokenizer

POLICIES = {'trace-commons': 'default-v2', 'halumem': 'halumem-v1'}
SCENARIOS = ('S1', 'S2', 'S3', 'S4')
COVERAGE = ('full', 'partial', 'none', 'distorted')
ELEMENTS = ('supported', 'missing', 'contradicted', 'unverifiable')


def prompt_file(stage):
    return (ROOT / 'tools/model/prompts' / f'v07-{stage}.md').read_text()


def material_prompt(stage, payload):
    return prompt_file(stage) + '\n\n' + json.dumps(payload, ensure_ascii=False)


def contract(items, key, expected=None):
    require(type(items) is list and all(type(row) is dict for row in items), f'{key} rows expected')
    ids = [row[key] for row in items]
    require(all(isinstance(i, str) and bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', i)) for i in ids),
            f'invalid {key}')
    require(len(ids) == len(set(ids)), f'duplicate {key}')
    if expected is not None:
        require(set(ids) == set(expected), f'{key} set does not match')
    return {row[key]: row for row in items}


def text(value):
    require(isinstance(value, str) and value.strip(), 'expected nonempty text')


def bool_field(value):
    require(type(value) is bool, 'expected bool')


def evidence(rows, turns):
    require(type(rows) is list and rows, 'missing evidence')
    verified = []
    for row in rows:
        if isinstance(row['turn'], str) and re.fullmatch(r'T?[0-9]+', row['turn']):
            row['turn'] = int(row['turn'].removeprefix('T'))
        require(type(row['turn']) is int and row['turn'] in turns, 'evidence turn unavailable')
        if 'quote' in row:
            text(row['quote'])
            if row['quote'] not in turns[row['turn']]:
                hits = [turn for turn, body in turns.items() if row['quote'] in body]
                if len(hits) != 1:
                    continue
                row['turn'] = hits[0]
        else:
            text(row['redacted_locator'])
        verified.append(row)
    require(bool(verified), 'evidence has no continuous source quotation or redacted locator')
    rows[:] = verified


def validate_gold(value, input_text, dataset, raw_points=None):
    header, turns_raw = transcript(input_text)
    turns = {turn['turn']: turn['text'] for turn in turns_raw}
    policy = read_json(ROOT / 'policies' / f'{POLICIES[dataset]}.json')
    projects = value['project']
    require(type(projects) is list and projects, 'expected nonempty project list')
    for name in projects:
        text(name)
        require(len(name) <= 255 and not any(c in name for c in '\r\n,，'), 'invalid project')
    units, negatives = contract(value['U'], 'unit_id'), contract(value['N'], 'unit_id')
    require(not set(units) & set(negatives), 'unit IDs overlap')
    element_ids = set()
    for row in units.values():
        require(row['category'] in policy['categories'], 'unexpected category')
        for field in ('statement', 'scope', 'expected_use', 'note'):
            text(row[field])
        bool_field(row['uncertain'])
        if dataset == 'trace-commons':
            basis = row['attributes']['basis']
            require(basis in policy['attributes']['basis']['allowed_by_category'][row['category']], 'invalid basis')
        evidence(row['evidence'], turns)
        elems = contract(row['elements'], 'element_id')
        require(bool(elems) and any(e['required'] and e['role'] == 'subject' for e in elems.values()), 'missing required subject')
        for e in elems.values():
            require(e['element_id'] not in element_ids, 'duplicate element ID across units')
            element_ids.add(e['element_id'])
            require(e['role'] in ('subject', 'condition', 'detail') and e['representation'] in ('gist', 'pointer', 'exact'),
                    'invalid element role or representation')
            bool_field(e['required'])
            text(e['statement'])
            text(e['omission_effect'])
            if e['role'] == 'condition':
                require(e['type'] in policy['condition_types'], 'invalid condition type')
            require(type(e['equivalents']) is list and all(isinstance(t, str) for t in e['equivalents']), 'invalid equivalents')
            if e['representation'] == 'pointer':
                text(e['locator'])
            evidence(e['evidence'], turns)
    for row in negatives.values():
        text(row['statement'])
        require(row['reject_reason'] in policy['reject_reasons'] and row['layer'] in ('random', 'targeted'), 'invalid negative')
        evidence(row['evidence'], turns)
    for row in value['prior_context']:
        evidence(row['evidence'], turns)
    tasks = contract(value['tasks'], 'task_id')
    if dataset == 'trace-commons':
        require(len(tasks) == 3, 'expected exactly three Trace tasks for the reduced round')
    else:
        require(not tasks, 'unexpected HaluMem tasks')
        parent = {p['index']: p for p in raw_points}
        mapped = set()
        for row in (*units.values(), *negatives.values()):
            point = row['parent_point']
            require(type(point) is int and point in parent, 'unmapped source point')
            if row in units.values():
                require(parent[point]['memory_source'] != 'interference', 'interference belongs to N')
            mapped.add(point)
        require(mapped == set(parent), 'unmapped source points')
    for task in tasks.values():
        require(task['scenario'] in SCENARIOS, 'invalid scenario')
        if type(task['as_of']) is int and task['as_of'] in turns:
            task['as_of'] = f"T{task['as_of']}"
        for key in ('question', 'as_of', 'expected_action'):
            text(task[key])
        as_of = re.fullmatch(r'T([1-9][0-9]*)', task['as_of'])
        require(as_of and int(as_of[1]) in turns, 'expected an existing task cutoff turn')
        require(f"以 {task['as_of']} 时的状态为准" in task['question'], 'task cutoff must appear in the question')
        require(bool(task['unit_ids']) and set(task['unit_ids']) <= set(units), 'unknown task unit')
        require(bool(task['element_ids']) and set(task['element_ids']) <= element_ids, 'unknown task element')
        for name in ('prohibited', 'equivalents', 'severe_errors'):
            require(type(task[name]) is list and all(isinstance(t, str) for t in task[name]), 'invalid task qualifiers')
        require(bool(contract(task['requirements'], 'requirement_id')), 'empty task requirements')
        for req in task['requirements']:
            text(req['statement'])
        evidence(task['evidence'], turns)
    return value


def finalize(folder, source_text, value, source, raw_points, receipts, count, token_meta):
    # Products all receive the same frozen bytes. Other annotator-only material stays behind.
    source_lines = source_text.splitlines(keepends=True)
    marker = '- project（标注者填写）：'
    boundary = next(i for i, line in enumerate(source_lines) if line.startswith('## [T'))
    slots = [i for i, line in enumerate(source_lines[:boundary]) if line.startswith(marker)]
    require(len(slots) == 1, 'ambiguous project header')
    source_lines[slots[0]] = marker + ', '.join(value['project']) + '\n'
    frozen = ''.join(source_lines)
    policy_id, policy_version = POLICIES[source['dataset']].rsplit('-', 1)
    for row in value['U']:
        row['origin'] = 'blind'
    for row in value['N']:
        row['origin'] = 'blind'
    gold = dict(value, **source, policy=dict(id=policy_id, version=policy_version), standard_version='v0.7',
                frozen_at=now(), gold_revision=0, annotator=receipts, raw_points=raw_points,
                transcript_sha256=sha256_text(frozen), transcript_tokens=count(frozen), tokenizer=token_meta)
    (folder / 'transcript.md').write_text(frozen)
    write_json(folder / 'gold.json', gold)
    return gold


def annotate(folder):
    source = read_json(folder / 'source.json')
    source_text = (folder / 'source-transcript.md').read_text()
    require(sha256_text(source_text) == source['source_sha256'], 'registered input has changed')
    gold_path = folder / 'gold.json'
    if gold_path.exists():
        existing = read_json(gold_path)
        require(existing['source_sha256'] == source['source_sha256'] and
                existing['transcript_sha256'] == sha256_text((folder / 'transcript.md').read_text()), 'frozen gold or input drift')
        return existing
    dataset = source['dataset']
    raw_points = read_json(folder / 'source-points.json') if dataset == 'halumem' else None
    framework = (ROOT / 'docs/framework.md').read_text()
    # Budget, legacy outcomes, and the cohort are not part of annotation input.
    selected_sections = framework[framework.index('## 3　'):framework.index('## 7　')]
    policy = (ROOT / 'policies' / f'{POLICIES[dataset]}.md').read_text()
    count, token_meta = tokenizer()
    value, receipts = codex('annotate', material_prompt('annotate', dict(
        sample=source['sample_id'], dataset=dataset, transcript=source_text,
        memory_points=raw_points, framework=selected_sections, policy=policy)),
        lambda value: validate_gold(value, source_text, dataset, raw_points))
    return finalize(folder, source_text, value, source, raw_points, receipts, count, token_meta)
