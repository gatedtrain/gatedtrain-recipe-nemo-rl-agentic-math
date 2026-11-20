"""Data split invariants of make_data.py's outputs (host CPU; reads data/, items-test/, raw/RAW.json)."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from collections import Counter

from _skips import HAVE_TRAIN, NO_TRAIN, skip_unless
from _support import AREA, read_jsonl

import make_data


def sha(p):
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


@skip_unless(HAVE_TRAIN, NO_TRAIN)
class DataSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_jsonl(AREA / "data" / "train.jsonl")
        cls.items = read_jsonl(AREA / "items-test" / "items.jsonl")
        cls.meta = json.loads((AREA / "data" / "data_meta.json").read_text())
        cls.imeta = json.loads((AREA / "items-test" / "item_set_meta.json").read_text())

    def test_counts_and_shards(self):
        self.assertEqual(len(self.train), 2000)
        self.assertEqual(len(self.items), 500)
        shards = sorted((AREA / "data" / "shards").glob("shard-*.jsonl"))
        self.assertEqual([s.name for s in shards], [f"shard-0000{k}.jsonl" for k in range(4)])
        self.assertEqual([len(read_jsonl(s)) for s in shards], [500] * 4)
        self.assertEqual(b"".join(s.read_bytes() for s in shards), (AREA / "data" / "train.jsonl").read_bytes())

    def test_ids_unique_and_ordered(self):
        self.assertEqual([r["row_id"] for r in self.train], [f"math-train-{k:05d}" for k in range(2000)])
        self.assertEqual([i["item"] for i in self.items], [f"math-test-{k:04d}" for k in range(500)])

    def test_overlap_zero(self):
        tr_exact = {r["input"] for r in self.train}
        tr_norm = {make_data.norm_text(r["input"]) for r in self.train}
        self.assertEqual(sum(i["text"] in tr_exact for i in self.items), 0)
        self.assertEqual(sum(make_data.norm_text(i["text"]) in tr_norm for i in self.items), 0)
        self.assertEqual(len({i["text"] for i in self.items}), 500)
        checks = {k: v for k, v in self.meta["checks"].items() if k != "train_candidates_dropped_as_test_duplicates"}
        self.assertEqual(checks,
                         {"train_vs_items_exact_overlap": 0, "train_vs_items_normalized_overlap": 0,
                          "duplicate_item_texts": 0})
        sources_tr = {(r["source"]["file"], r["source"]["index"]) for r in self.train}
        self.assertTrue(all("/train-" in f for f, _ in sources_tr))
        self.assertTrue(all("/test-" in i["source"]["file"] for i in self.items))

    def test_answers_are_integers_and_rows_consistent(self):
        for r in self.train:
            self.assertRegex(r["answer"], r"^-?\d+$")
            self.assertEqual(r["messages"], [{"role": "user", "content": r["input"]},
                                             {"role": "assistant", "content": r["answer"]}])
        for i in self.items:
            self.assertRegex(i["answer"], r"^-?\d+$")
            self.assertEqual(i["text"], i["problem"])

    def test_items_stratified(self):
        got = Counter(f"{i['subject']}|{i['level']}" for i in self.items)
        self.assertEqual(dict(got), {k: v for k, v in self.imeta["strata"].items() if v})
        self.assertEqual(sum(self.imeta["strata"].values()), 500)
        self.assertEqual(len({i["subject"] for i in self.items}), 7)

    def test_hashes_recorded(self):
        for rel, h in self.meta["files"].items():
            self.assertEqual(sha(AREA / rel), h, rel)
        self.assertEqual(self.imeta["items_sha256"], sha(AREA / "items-test" / "items.jsonl"))
        self.assertEqual(self.meta["prompt_file_sha256"], sha(AREA / "prompt_math_code.txt"))

    def test_integer_rule(self):
        cases = {"5": "5", "-3": "-3", "1{,}000": "1000", "10,000": "10000", "\\$5": None, "$12$": "12", "3.0": "3",
                 "\\frac{1}{2}": None, "5\\text{ cm}": None, "2\\sqrt{3}": None, "007": "7", "1,2": None,
                 "\\!42": "42", "90^\\circ": None, "": None, None: None}
        for raw, want in cases.items():
            self.assertEqual(make_data.integer_answer(raw), want, raw)
        self.assertEqual(make_data.last_boxed("a $\\boxed{3}$ then $\\boxed{\\frac{1}{2}}$"), "\\frac{1}{2}")
        self.assertEqual(make_data.last_boxed("so \\boxed 7$."), "7")

    def test_prompt_template(self):
        tmpl = (AREA / "prompt_math_code.txt").read_text()
        self.assertEqual(len(re.findall(r"(?<!\{)\{\}(?!\})", tmpl)), 1)
        text = tmpl.format("What is {x}?")
        self.assertIn("\\boxed{}", text)
        self.assertIn("<code>", text)
        self.assertTrue(text.endswith("Problem: What is {x}?"))


if __name__ == "__main__":
    unittest.main()
