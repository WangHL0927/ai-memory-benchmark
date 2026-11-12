"""缓存键和留存格式测试；全部为合成内容。"""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.helpers import ROOT, TMP
from tools.cache import ModelCache, canonical_json, save_raw
from tools.trial import read_json


class CacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        patcher = patch('tools.cache.CACHE_ROOT', self.root / 'cache')
        patcher.start()
        self.addCleanup(patcher.stop)

    def cache(self, **changes):
        options = dict(stage='annotate', tool='假工具', model='假快照',
                       parameters=dict(reasoning_effort='high', temperature=0, enable_thinking=False, max_tokens=1024),
                       template='合成模板', request=dict(messages=['合成输入'], model='假快照'))
        options.update(changes)
        return ModelCache(**options)

    def test_key_uses_all_inputs_and_canonical_json(self):
        original = self.cache()
        expected = hashlib.sha256(canonical_json(original.inputs).encode('utf-8')).hexdigest()
        self.assertEqual(original.key, expected)
        self.assertEqual(original.path, self.root / 'cache/annotate' / expected[:2] / (expected + '.json'))
        self.assertEqual(original.key, self.cache(request=dict(model='假快照', messages=['合成输入'])).key)
        for field, value in dict(stage='judge', tool='另一工具', model='另一快照',
                                 template='另一模板', request='另一输入').items():
            with self.subTest(field=field):
                self.assertNotEqual(original.key, self.cache(**{field: value}).key)
        for field, value in dict(reasoning_effort='low', temperature=0.5, enable_thinking=True, max_tokens=2048).items():
            parameters = dict(original.inputs['parameters'], **{field: value})
            with self.subTest(parameter=field):
                self.assertNotEqual(original.key, self.cache(parameters=parameters).key)
        with self.assertRaises(ValueError):
            self.cache(parameters=dict(temperature=float('nan')))

    def test_success_record_raw_integrity_and_atomic_write(self):
        cache = self.cache()
        raw = save_raw('[{"statement":"合成结果"}]')
        self.addCleanup((ROOT / raw['path']).unlink, missing_ok=True)
        identity = dict(tool='假工具', raw_outputs=[raw])
        result = [dict(statement='合成结果')]
        cache.save(result, identity)
        record = read_json(cache.path)
        self.assertEqual(record['result'], result)
        self.assertEqual(record['identity']['cache'], 'miss')
        self.assertEqual(record['raw_outputs'], [raw])
        self.assertIn('written_at', record)
        self.assertEqual(list(cache.path.parent.iterdir()), [cache.path])
        self.assertEqual(cache.load(lambda x: x)[1]['cache'], 'hit')
        before = cache.path.read_bytes()
        with self.assertRaises(TypeError):
            cache.save(result, dict(raw_outputs=[raw], invalid=object()))
        self.assertEqual(cache.path.read_bytes(), before)
        self.assertEqual(list(cache.path.parent.iterdir()), [cache.path])
        (ROOT / raw['path']).write_text('损坏的返回', encoding='utf-8')
        self.assertIsNone(cache.load(lambda x: x))
        cache.path.write_text('损坏的 JSON', encoding='utf-8')
        self.assertIsNone(cache.load(lambda x: x))

    def test_disabled_cache_neither_reads_nor_writes(self):
        cache = self.cache(enabled=False)
        cache.save([], dict(raw_outputs=[]))
        self.assertFalse(cache.path.exists())
        self.assertIsNone(cache.load(lambda x: self.fail('禁用后不得读取缓存')))
