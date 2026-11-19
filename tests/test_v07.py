from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.model.v07.cli import READER_WINDOW, read_answer
from tools.model.v07.cohort import clip
from tools.model.v07.evaluate import memory_view, validate_tasks
from tools.model.v07.protocol import contract, evidence
from tools.model.v07.scoring import grade_record, paired_values, sensitivity, value_and_cost, write_input
from tools.run.openviking import transcript_to_messages
from tools.run.x2 import connection_string


class V07ProtocolTests(unittest.TestCase):
    def test_rejects_path_ids_before_use(self):
        with self.assertRaises(ValueError):
            contract([{'task_id': '../system'}], 'task_id')

    def test_normalizes_turn_id_and_requires_actual_quote(self):
        rows = [{'turn': 'T5', 'quote': 'actual words'}, {'turn': 5, 'quote': 'invented'}]
        evidence(rows, {5: 'The actual words follow.'})
        self.assertEqual(rows, [{'turn': 5, 'quote': 'actual words'}])
        with self.assertRaises(ValueError):
            evidence([{'turn': 5, 'quote': 'invented'}], {5: 'actual words'})

    def test_prefix_requires_next_user_boundary(self):
        parent = '# Sample\n- 总轮数：3\n## [T1] 用户\n a\n## [T2] 助手\n b\n## [T3] 用户\n c\n'
        self.assertIn('- 总轮数：2', clip(parent, 2))
        self.assertNotIn('[T3]', clip(parent, 2))
        with self.assertRaises(ValueError):
            clip(parent, 1)

    def test_native_message_replay_preserves_frozen_project_header(self):
        source = '# Source\n- project（标注者填写）：alpha\n\n## [T1] 用户\nRead the summary.\n'
        legacy = transcript_to_messages(source)
        replay = transcript_to_messages(source, include_header=True)
        self.assertEqual(legacy[0]['content'], 'Read the summary.')
        self.assertEqual(replay[0]['turn_id'], 'T1')
        self.assertIn('- project（标注者填写）：alpha\n', replay[0]['content'])
        self.assertTrue(replay[0]['content'].endswith(legacy[0]['content']))

    def test_reader_limit_refuses_before_using_provider(self):
        with patch('tools.model.v07.cli.settings') as provider:
            result = read_answer('x' * READER_WINDOW, 'task', len)
        self.assertEqual(result['status'], 'reader_overflow')
        self.assertGreater(result['required_tokens'], READER_WINDOW)
        provider.assert_not_called()

    def test_native_order_and_visibility(self):
        view = memory_view(dict(D=[dict(draft=dict(statement='first')),
                                    dict(draft=dict(statement='second'),
                                         reader_metadata=dict(scope='project:site', internal_path='/here'))]))
        self.assertEqual(view, [dict(memory_id='m1', statement='first'),
                                dict(memory_id='m2', statement='second', scope='project:site')])

    def test_empty_memory_cannot_be_attributed(self):
        task = {'q': {'requirements': [{'requirement_id': 'r'}]}}
        response = [dict(task_id='q', verdict='wrong', requirements=[dict(requirement_id='r',
                            status='missing', memory_ids=[])], action_error=True, error_matches_memory=True,
                         actual_reuse=True, misleading=True, unclear_attribution=False, high_risk=False,
                         invalid_reason='', note='')]
        with self.assertRaises(ValueError):
            validate_tasks(response, task, [])


class V07ScoreTests(unittest.TestCase):
    def example(self, reader=1000, writing=1, *, correct=1):
        outcomes = {'correct': 4 * correct}
        for name in ('S1', 'S2', 'S3', 'S4'):
            outcomes[name + '_valid'] = 1
            outcomes[name + '_correct'] = correct
            outcomes[name + '_credit'] = correct
        return dict(outcomes=outcomes, valid_tasks=4, num_tasks=4, transcript_tokens=1000,
                    read_tokens=reader, write_multiple=writing, session_id='sample')

    def test_fixture_scores_and_sensitivity(self):
        example = self.example()
        result = value_and_cost([example])
        self.assertAlmostEqual(result['read_penalty'], 17.2877123795)
        self.assertEqual(result['write_penalty'], 0)
        self.assertAlmostEqual(result['composite'], 82.7122876205)
        matrix = sensitivity([example])
        self.assertEqual(len(matrix), 9)
        self.assertEqual(matrix['4/2.5'], result['composite'])

    def test_missing_provider_cost_is_not_assumed_zero(self):
        self.assertIsNone(write_input({'usage': {'llm': {'calls': 1, 'tokens': {}}}}))
        self.assertEqual(write_input({'usage': {'llm': {'calls': 0, 'tokens': {}}}}), 0)
        self.assertIsNone(value_and_cost([self.example(writing=None)])['composite'])

    def test_points_average_per_task_and_only_attributed_error_penalizes_s5(self):
        scenario = {'q1': 'S1', 'q2': 'S1', 'q3': 'S2', 'q4': 'S3', 'q5': 'S4'}
        gold = dict(sample_id='s', session_id='s', transcript_tokens=1000,
                    tasks=[dict(task_id=key, scenario=value) for key, value in scenario.items()])
        tasks = {key: dict(verdict='correct', requirements=[dict(status='supported')], action_error=False,
                           misleading=False, error_matches_memory=False, unclear_attribution=False, high_risk=False)
                 for key in scenario}
        tasks['q1'].update(verdict='partial', requirements=[dict(status='supported'), dict(status='contradicted')],
                           action_error=True)
        result = dict(tasks=tasks, answers={key: dict(read_tokens=0) for key in scenario},
                      usage=dict(llm=dict(calls=1, tokens=dict(prompt_tokens=4000))))
        score = value_and_cost([grade_record(gold, result)])
        self.assertEqual(score['scenarios']['S1']['value'], .75)
        self.assertEqual(score['scenario_full']['S1']['value'], .5)
        self.assertEqual(score['action_error_rate']['value'], .2)
        self.assertEqual(score['misleading']['value'], 0)
        self.assertEqual(score['value'], 93.75)
        self.assertEqual(score['write_penalty'], 5)
        self.assertEqual(score['composite'], 88.75)

        tasks['q1'].update(misleading=True, error_matches_memory=True, actual_reuse=True)
        attributed = value_and_cost([grade_record(gold, result)])
        self.assertEqual(attributed['action_error_rate']['value'], .2)
        self.assertEqual(attributed['misleading']['value'], .2)
        self.assertEqual(attributed['value'], 91.75)
        self.assertEqual(attributed['composite'], 86.75)

    def test_all_overflow_keeps_usage_out_of_comparable_value_and_cost(self):
        overflow = self.example(writing=100)
        overflow.update(outcomes={}, valid_tasks=0, read_tokens=None)
        score = value_and_cost([self.example(), overflow])
        self.assertAlmostEqual(score['composite'], 82.7122876205)
        self.assertEqual(score['conservative_success']['value'], .5)

    def test_pairing_intersects_sample_ids_and_checks_source(self):
        left = {'s': self.example(), 'only-left': self.example()}
        right = {'s': self.example(correct=0)}
        result = paired_values(left, right, iterations=10)
        self.assertEqual(result['value']['value'], 100)
        self.assertEqual(result['value']['valid_resamples'], 0)
        right['s']['session_id'] = 'different'
        with self.assertRaises(ValueError):
            paired_values(left, right, iterations=10)


class X2GuardTests(unittest.TestCase):
    def test_refuses_non_eval_database_and_overrides(self):
        url = 'postgresql://test_eval_000000000000:' + '0' * 64 + '@127.0.0.1:12345/test_eval'
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'url'
            path.touch(mode=0o600)
            for bad in (url.replace('/test_eval', '/user_data'), url + '?dbname=user_data', url.replace('12345','12346')):
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    connection_string(path, 'test_eval', 12345)
            path.write_text(url)
            self.assertEqual(connection_string(path, 'test_eval', 12345), url)
