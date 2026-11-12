"""合成标注/判定配合假 CLI，验证缓存不会再次调用模型。"""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.helpers import TMP, unit
from tests.helpers import cli, fake_cli, sample
from tools.model.annotate import annotate
from tools.model.common import prompt_file
from tools.trial import ModelOutputError, read_json, write_json


class CliCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        patcher = patch('tools.cache.CACHE_ROOT', self.root / 'cache')
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, program):
        return SimpleNamespace(data_dir=self.root / 'data', policy=None, layer='random', annotator='codex',
                               model='合成快照', cli_path=str(program), reasoning_effort='high',
                               timeout=10, no_cache=False, force=True)

    def test_annotation_hit_template_input_parameters_and_no_cache(self):
        s = sample(self.root / 'data', with_gold=False)
        program, state = fake_cli(self.root, [dict(project=['garden'], U=[unit()], N=[])])
        args = self.args(program)
        target = s / 'gold.json'
        annotate(s, args)
        first = read_json(target)
        self.assertEqual(first['annotator']['cache'], 'miss')
        target.unlink()
        annotate(s, args)
        second = read_json(target)
        self.assertEqual(len(read_json(state)), 1)
        self.assertEqual(second['annotator']['cache'], 'hit')
        self.assertEqual(first['annotator']['cache_key'], second['annotator']['cache_key'])
        # 同一模型输入可复用，但样本标识和分层必须按当前调用重建。
        other = sample(self.root / 'data', 'other', with_gold=False)
        (other / 'transcript.md').write_bytes((s / 'transcript.md').read_bytes())
        args.layer = 'targeted'
        annotate(other, args)
        self.assertEqual(read_json(other / 'gold.json')['sample_id'], 'other')
        self.assertEqual(read_json(other / 'gold.json')['layer'], 'targeted')
        self.assertEqual(len(read_json(state)), 1)
        for change in ('template', 'input', 'reasoning', 'no-cache'):
            target.unlink()
            if change == 'input':
                with (s / 'transcript.md').open('a') as stream:
                    stream.write('\n合成输入已变更\n')
            if change == 'reasoning':
                args.reasoning_effort = 'low'
            args.no_cache = change == 'no-cache'
            with patch('tools.model.annotate.prompt_file', return_value=prompt_file('annotate') + ('\n新模板' if change == 'template' else '')):
                annotate(s, args)
            self.assertEqual(read_json(target)['annotator']['cache'], 'miss')
        self.assertEqual(len(read_json(state)), 5)
        self.assertEqual(len(list((self.root / 'cache/annotate').glob('*/*.json'))), 4)


        s = sample(self.root / 'cli-data', with_gold=False)
        program, state = fake_cli(self.root / 'cli-data', [dict(project=['garden'], U=[unit()], N=[])])
        arguments = ['--data-dir', s.parent, '--annotator', 'claude', '--model', '合成模型',
                     '--cli-path', program, '--failures', self.root / 'failures.json']
        for flags, count, status in (([], 1, 'miss'), ([], 1, 'hit'), (['--no-cache'], 2, 'miss')):
            result = cli('tools.model.annotate', *arguments, *flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(read_json(state)), count)
            self.assertEqual(read_json(s / 'gold.json')['annotator']['cache'], status)
            record = read_json(self.root / 'failures.json')
            log = Path(record['log']).read_text()
            self.assertIn(f'完成：{s} 状态=written cache={status}', log)
            (s / 'gold.json').unlink()

    def test_failed_cli_output_is_not_cached(self):
        s = sample(self.root / 'data', with_gold=False)
        invalid = unit()
        invalid['evidence'][0]['quote'] = '不存在的证据'
        program, state = fake_cli(self.root, ['不是 JSON', dict(project=['garden'], U=[invalid], N=[])])
        args = self.args(program)
        for _ in range(2):
            with self.assertRaises(ModelOutputError):
                annotate(s, args)
        self.assertEqual(len(read_json(state)), 4)
        self.assertFalse((s / 'gold.json').exists())
        self.assertFalse(list((self.root / 'cache').glob('**/*.json')))
        write_json(self.root / 'responses.json', [dict(project=['garden'], U=[unit()], N=[])])
        annotate(s, args)
        self.assertEqual(len(read_json(state)), 5)
