"""HaluMem 自编数据的转换及评分集成测试；不联网、不读取真实 data。"""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.prepare import halumem as converter
from tools import score


from tests.helpers import ROOT, TMP, FIXTURES as ALL_FIXTURES
FIXTURES = ALL_FIXTURES / "halumem"


class HaluMemTests(unittest.TestCase):
    def setUp(self):
        # 所有临时产物都位于本任务获准的 fixture 子目录，并在测试后移除。
        temporary = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.output = self.work / "converted"
        self.input = FIXTURES / "synthetic.jsonl"
        self.raw = json.loads(self.input.read_text(encoding="utf-8"))

    def convert(self, raw=None):
        source = self.input
        if raw is not None:
            source = self.work / "input.jsonl"
            source.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return converter.convert(source, self.output, prefix="synthetic",
                                 frozen_at="2026-11-11T12:00:00+00:00")

    def test_conversion_preserves_labels_and_real_session_evidence(self):
        samples = self.convert()
        self.assertEqual(len(samples), 2)
        for sid, sample in enumerate(samples):
            gold = json.loads((sample / "gold.json").read_text(encoding="utf-8"))
            score.validate_gold(gold)  # 从 gold 声明自动定位 halumem-v1.json。
            self.assertEqual(gold["policy"], dict(id="halumem", version="v1"))
            self.assertEqual(gold["project"], ["halumem"])
            self.assertEqual(gold["turns"], 4)  # 两个原始问答轮、四条框架消息。
            self.assertEqual(gold["halumem"]["source_dialogue_turns"], [0, 0, 1, 1])
            self.assertEqual(gold["frozen_at"], "2026-11-11T12:00:00+00:00")
            self.assertEqual(gold["session_id"], self.raw["uuid"])
            self.assertEqual(gold["halumem"]["session_index"], sid)
            self.assertEqual(gold["blind_snapshot"], {k:gold[k] for k in ("U", "N")})
            transcript = (sample / "transcript.md").read_text(encoding="utf-8")
            for forbidden in ("不应泄露给被测系统的背景标记", "隐藏的问答标记", "隐藏答案",
                              '"memory_points"', '"original_memories"'):
                self.assertNotIn(forbidden, transcript)
            for message in self.raw["sessions"][sid]["dialogue"]:
                self.assertIn(message["content"], transcript)
            for unit in gold["U"] + gold["N"]:
                self.assertEqual(unit["origin"], "blind")
                self.assertFalse(unit["uncertain"])
                self.assertEqual(unit["halumem"], next(p for p in self.raw["sessions"][sid]["memory_points"]
                                                      if p["index"] == unit["halumem"]["index"]))
                for evidence in unit["evidence"]:
                    original = self.raw["sessions"][sid]["dialogue"][evidence["turn"] - 1]
                    self.assertEqual(evidence["quote"], original["content"])
                    self.assertEqual(evidence["role"], original["role"])
        first, second = [json.loads((p / "gold.json").read_text()) for p in samples]
        self.assertEqual([len(first["U"]), len(first["N"]), len(second["U"]), len(second["N"])], [2, 1, 2, 0])
        self.assertEqual(first["N"][0]["reject_reason"], "distractor")
        self.assertEqual(second["U"][0]["attributes"]["is_update"], "true")
        self.assertEqual(second["U"][0]["halumem"]["original_memories"], [first["U"][0]["statement"]])
        self.assertNotIn("importance", second["U"][1]["attributes"])
        self.assertEqual(second["U"][1]["halumem"]["importance"], .3333)


    def test_converter_cli_and_refuses_overwrite(self):
        cmd = [sys.executable, "-m", "tools.prepare.halumem", str(self.input),
               "--data-dir", str(self.output)]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        golds = sorted(self.output.glob("*/gold.json"))
        before = [p.read_bytes() for p in golds]
        for path in golds:
            gold = json.loads(path.read_text())
            self.assertIsNotNone(datetime.fromisoformat(gold["frozen_at"]).tzinfo)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("拒绝覆盖", proc.stderr)
        self.assertEqual([p.read_bytes() for p in golds], before)

    def test_same_session_old_state_is_negative_without_changing_prior_session(self):
        raw = deepcopy(self.raw)
        old = deepcopy(raw["sessions"][0]["memory_points"][0])
        old["index"] = 3
        raw["sessions"][1]["memory_points"].insert(0, old)
        # 同段对话显式出现旧状态，之后更新；只把该段旧点转为 N。
        raw["sessions"][1]["dialogue"][0]["content"] = "我以前是建筑师，现在改行当教师了。"
        first, second = [json.loads((p / "gold.json").read_text()) for p in self.convert(raw)]
        self.assertEqual(len(first["U"]), 2)
        self.assertEqual(len(second["U"]), 2)
        self.assertEqual(second["N"][0]["reject_reason"], "superseded")
        score.validate_gold(second)

    def test_invalid_fields_do_not_leave_partial_output(self):
        mutations = (
            ("类别", lambda u: u["sessions"][1]["memory_points"][0].update(memory_type="unknown")),
            ("重要性", lambda u: u["sessions"][1]["memory_points"][0].update(importance=1.1)),
            ("布尔", lambda u: u["sessions"][1]["memory_points"][0].update(is_update="no")),
            ("更新链", lambda u: u["sessions"][1]["memory_points"][0].update(original_memories=[])),
            ("重复点", lambda u: u["sessions"][1]["memory_points"][1].update(index=1)),
            ("缺对话", lambda u: u["sessions"][1].update(dialogue=[])),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                raw = deepcopy(self.raw)
                mutate(raw)
                with self.assertRaises(ValueError):
                    self.convert(raw)
                self.assertFalse(self.output.exists())
                self.assertFalse(list(self.work.glob(".halumem-*")))

    def test_jsonl_multiple_users_blank_lines_and_unsafe_uuid(self):
        second = deepcopy(self.raw)
        second["uuid"] = "../../越界"
        source = self.work / "many.jsonl"
        source.write_text("\n" + json.dumps(self.raw) + "\n\n" + json.dumps(second) + "\n")
        samples = converter.convert(source, self.output)
        self.assertEqual(len(samples), 4)
        self.assertEqual(len({p.name for p in samples}), 4)
        self.assertTrue(all(p.parent == self.output for p in samples))

    def test_empty_gold_session_is_retained_and_bool_is_accepted(self):
        raw = deepcopy(self.raw)
        raw["sessions"][0]["memory_points"] = []
        raw["sessions"][0]["is_generated_qa_session"] = True
        raw["sessions"][1]["memory_points"][0]["is_update"] = True
        first, second = [json.loads((p / "gold.json").read_text()) for p in self.convert(raw)]
        self.assertEqual(first["U"], [])
        self.assertEqual(first["N"], [])
        self.assertTrue(first["halumem"]["is_generated_qa_session"])
        self.assertEqual(second["U"][0]["attributes"]["is_update"], "true")

    def test_broken_json_and_repeated_users_fail_without_partial_output(self):
        for tail in ("{broken json}", json.dumps(self.raw)):
            with self.subTest(tail=tail[:20]):
                source = self.work / "bad.jsonl"
                source.write_text(json.dumps(self.raw) + "\n" + tail + "\n")
                with self.assertRaisesRegex(ValueError, r"bad.jsonl:2:"):
                    converter.convert(source, self.output)
                self.assertFalse(self.output.exists())

    def test_oversized_session_requires_manual_segmentation(self):
        raw = deepcopy(self.raw)
        dialogue = raw["sessions"][1]["dialogue"]
        raw["sessions"][1]["dialogue"] = [dict(deepcopy(dialogue[i % 4]), dialogue_turn=i // 2)
                                           for i in range(302)]
        with self.assertRaisesRegex(ValueError, "超过 300"):
            self.convert(raw)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
