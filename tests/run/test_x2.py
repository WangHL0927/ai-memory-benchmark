import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError

from tools.paths import TMP
from tools.run.x2 import MCP


class X2TranscriptTests(unittest.TestCase):
    def test_upload_uses_draft_session_and_audits_without_text_or_credentials(self):
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            audit = Path(directory)
            client = MCP('http://127.0.0.1:1', 'synthetic-key', audit, 5, session_id='test-sample')
            text = '合成转录\n[T1] 仅用于本地测试'
            reply = io.BytesIO(b'{"code":0,"result":{"uploaded":true}}')
            reply.status = 200
            client.opener.open = Mock(side_effect=[
                reply, HTTPError(client.base, 413, 'too large', {}, io.BytesIO()),
            ])
            client.request = Mock(return_value={})

            before = client.upload_transcript(text, 'before')
            client.call('submit', {'session_id': 'uncorrelated-id', 'op': 'create'})
            after = client.upload_transcript(text, 'after')

            client.request.assert_called_once_with('tools/call', {
                'name': 'submit', 'arguments': {'session_id': 'test-sample', 'op': 'create'},
            })
            for sent in client.opener.open.call_args_list:
                (request,) = sent.args
                self.assertEqual(request.full_url, client.base + '/v1/session/transcript/upload')
                self.assertEqual(request.get_method(), 'POST')
                self.assertEqual(request.get_header('Authorization'), client.headers['Authorization'])
                self.assertEqual(json.loads(request.data), {
                    'session_id': 'test-sample', 'transcript': text, 'truncated': False,
                })
                self.assertEqual(sent.kwargs['timeout'], 8)

            self.assertTrue(before['uploaded'])
            self.assertFalse(after['uploaded'])
            self.assertEqual(after['http_status'], 413)
            self.assertEqual(before['sha256'], after['sha256'])
            self.assertEqual(before['bytes'], len(text.encode()))
            for result in (before, after):
                saved = (audit / f"transcript-upload-{result['stage']}.json").read_text()
                self.assertEqual(json.loads(saved), result)
                self.assertNotIn('合成转录', saved)
                self.assertNotIn('synthetic-key', saved)
