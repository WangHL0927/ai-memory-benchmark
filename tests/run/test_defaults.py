"""只截获配置与进程启动请求，验证默认生成参数；不启动产品。"""
import json
import os
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.helpers import TMP
from tools.config import sha256_file, sha256_text
from tools.run import ai_memory, basic_memory, claude_mem, openviking, proxy


class DefaultConfigurationTests(unittest.TestCase):
    def config(self):
        return dict(model='synthetic-llm', embedding_model='synthetic-embedding', dimensions=2)

    def test_settings_require_model_and_embedding_only(self):
        env = dict(SUT_BASE_URL='http://synthetic/v1', SUT_API_KEY='synthetic-key', SUT_LLM_MODEL='synthetic-llm',
                   SUT_EMBEDDING_MODEL='synthetic-embedding', SUT_EMBEDDING_DIMENSIONS='2')
        with patch.dict(os.environ, env, clear=True), patch.object(proxy, 'load_env'):
            config = proxy.settings()
        self.assertEqual(config, dict(self.config(), base_url=env['SUT_BASE_URL'], key=env['SUT_API_KEY']))

    def test_openviking_wizard_config_and_bailian_switch(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            directory = Path(d); captured = {}
            def stop(command, **kwargs):
                captured.update(json.loads((kwargs['cwd'] / 'ov.conf').read_text()))
                raise RuntimeError('synthetic stop before product launch')
            with patch.object(openviking.subprocess, 'Popen', side_effect=stop), self.assertRaisesRegex(RuntimeError, 'synthetic stop'):
                with openviking.isolated_server(Path('/synthetic/python'), directory, self.config(),
                                                SimpleNamespace(url='http://127.0.0.1:1/v1', timeout=3), 1):
                    self.fail('must not start')
            vlm = captured['vlm']
            self.assertEqual(vlm['temperature'], 0.0)
            self.assertEqual(vlm['max_retries'], 2)
            self.assertEqual(vlm['extra_request_body'], dict(enable_thinking=False))
            self.assertNotIn('thinking', vlm); self.assertNotIn('max_tokens', vlm)
            self.assertEqual(captured['embedding']['dense']['dimension'], 2)

    def test_ai_memory_preserves_compat_defaults_and_qwen_fallback(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            directory = Path(d); captured = {}
            def stop(command, **kwargs):
                captured.update(tomllib.loads((kwargs['cwd'] / 'config.toml').read_text()))
                raise RuntimeError('synthetic stop before product launch')
            with patch.object(ai_memory.subprocess, 'Popen', side_effect=stop), self.assertRaisesRegex(RuntimeError, 'synthetic stop'):
                with ai_memory.isolated_server(Path('/synthetic/ai-memory'), directory, self.config(),
                                               SimpleNamespace(url='http://127.0.0.1:1/v1', timeout=3), 1):
                    self.fail('must not start')
            for field in ('llm_compat_disable_thinking', 'llm_compat_strict', 'llm_compat_stream', 'llm_max_tokens', 'temperature'):
                self.assertNotIn(field, captured)
            self.assertEqual(captured['embedding_model'], 'synthetic-embedding')
            self.assertEqual(captured['embedding_dim'], 2)

    def test_claude_mem_official_provider_without_generation_overrides(self):
        env = claude_mem.child_environment(Path('/synthetic'), self.config(),
                                          SimpleNamespace(url='http://127.0.0.1:1/v1', timeout=3), 2)
        self.assertEqual(env['CLAUDE_MEM_PROVIDER'], 'openai-compatible')
        self.assertEqual(env['CLAUDE_MEM_OPENAI_COMPAT_MODEL'], 'synthetic-llm')
        self.assertEqual(env['CLAUDE_MEM_CHROMA_ENABLED'], 'false')
        self.assertFalse(any('OPENROUTER' in field or 'MAX_OUTPUT' in field for field in env))

    def test_shared_file_hash_and_isolated_metadata_scripts(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            path = Path(d) / 'fixture'; path.write_text('合成文件\n')
            self.assertEqual(sha256_file(path), sha256_text('合成文件\n'))
        # 只编译传给隔离解释器的脚本，不执行产品 import 或安装操作。
        for adapter in (openviking, basic_memory):
            def stop(command, **kwargs):
                compile(command[-1], '<isolated-metadata>', 'exec')
                raise RuntimeError('synthetic script compiled')
            with self.subTest(adapter=adapter.__name__), patch.object(adapter.subprocess, 'run', side_effect=stop):
                with self.assertRaisesRegex(RuntimeError, 'script compiled'):
                    adapter.product_metadata(Path('/synthetic/python'))
