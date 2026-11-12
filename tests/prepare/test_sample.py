import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.prepare import sample
from tests.helpers import ROOT, TMP, SESSIONS


class SampleTests(unittest.TestCase):
    def test_filter_and_stratified_reproducible(self):
        paths=[('claude',SESSIONS/'claude.jsonl'),('codex',SESSIONS/'codex.jsonl')]
        records, errors=sample.inventory(paths)
        self.assertFalse(errors)
        self.assertTrue(all(r['eligible'] for r in records))
        strict,_=sample.inventory(paths,min_users=4)
        self.assertTrue(all(not r['eligible'] for r in strict))
        fake=[dict(eligible=True,length_layer=layer,session_id=f'{layer}-{i}') for layer in ['≤50','51~150','>150'] for i in range(10)]
        a=sample.choose(fake,8,7);b=sample.choose(fake,8,7)
        self.assertEqual(a,b)
        self.assertEqual(len({r['session_id'] for r in a}),8)
        self.assertEqual([sum(r['length_layer']==l for r in a) for l in ['≤50','51~150','>150']],[3,3,2])
        self.assertEqual(len(sample.choose(records,25)),2)

    def test_cli_only_synthetic_roots(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'claude/project').mkdir(parents=True);(root/'codex/day').mkdir(parents=True)
            (root/'claude/project/synthetic.jsonl').write_bytes((SESSIONS/'claude.jsonl').read_bytes())
            (root/'codex/day/rollout-synthetic.jsonl').write_bytes((SESSIONS/'codex.jsonl').read_bytes())
            result=subprocess.run(['python3','-m','tools.prepare.sample','-n','2','--claude-root',str(root/'claude'),'--codex-root',str(root/'codex'),'--data-dir',str(root/'out')],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((root/'out/samples.json').read_text())
            self.assertEqual(len(manifest['samples']),2)
            self.assertEqual(len(list((root/'out').glob('*/transcript.md'))),2)
