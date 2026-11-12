"""政策驱动的金标校验；全部使用合成数据。"""
import copy
import unittest

from tests.helpers import ROOT, gold
from tools import score


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = score.load_policy(ROOT / 'policies/default-v1.json')

    def test_policy_identity_scope_prefix_and_membership_rejected(self):
        score.validate_gold(gold(), self.policy)
        for field in ('id', 'version'):
            g = gold(); g['policy'][field] = 'different'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, '政策 id/version'):
                score.validate_gold(g, self.policy)
        for scope in ('project:outside', 'project:', 'projectish:garden', 'project:*'):
            g = gold(); g['U'][0]['scope'] = scope
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, 'scope'):
                score.validate_gold(g, self.policy)
        for scope in ('global', 'project:garden'):
            g = gold(); g['U'][0]['scope'] = scope
            score.validate_gold(g, self.policy)

    def test_attribute_constraints_and_negative_reasons(self):
        g = gold(); g['U'][0]['attributes']['basis'] = 'verified'
        with self.assertRaisesRegex(ValueError, 'attributes.basis'):
            score.validate_gold(g, self.policy)
        p = copy.deepcopy(self.policy)
        p['attributes']['basis']['allowed_by_category']['decision'].append('verified')
        score.validate_gold(g, p)
        n = copy.deepcopy(g['U'][0]); n.update(unit_id='n1', reject_reason='prior_context', layer='random')
        g['N'] = [n]
        with self.assertRaisesRegex(ValueError, '负例原因'):
            score.validate_gold(g, p)

    def test_invalid_policy_and_frozen_snapshot(self):
        for field, value in (('categories', []), ('scope_values', ['global', 'global']), ('superseded', 'unknown'),
                             ('attributes', {'basis': {'values': ['yes'], 'allowed_by_category': {'unknown': ['yes']}}})):
            p = copy.deepcopy(self.policy); p[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                score.validate_policy(p)
        g = gold(); g['blind_snapshot'] = dict(U=[], N=[])
        with self.assertRaisesRegex(ValueError, '快照'):
            score.validate_gold(g, self.policy)
