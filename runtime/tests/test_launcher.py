"""Launcher overrides and the LoRA cut writer (host CPU; no NeMo RL import).

`OverridesAgainstConfig` needs hydra-core and omegaconf (not in the area-4 venv; the area-8 venv has hydra-core
1.3.8 / omegaconf 2.3.1, the NeMo RL image 1.3.7 / 2.3.1): it applies the launcher's overrides to the pinned clone's
`examples/configs/grpo_math_1B.yaml` exactly as NeMo RL's `parse_hydra_overrides` does (utils/config.py:174-194:
struct mode, OverridesParser, ConfigLoaderImpl._apply_overrides_to_config) and checks the resulting keys.
  <area-8 venv>/bin/python -m unittest discover -s tests -t tests -p test_launcher.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from _skips import HAVE_BASE_TOKENIZER, HAVE_NEMO_RL, NO_BASE, NO_NEMO_RL, skip_unless
from _support import AREA, NEMO_RL

import gen_agentic
import ptc_agentic_hooks as hooks
import run_agentic_nemo as launcher

try:
    from hydra._internal.config_loader_impl import ConfigLoaderImpl
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None


def args(memory="a"):
    return Namespace(base=AREA / "models" / "qwen3-1.7b", seed=10100, memory=memory, socket="/spool/mathexec.sock")


class Overrides(unittest.TestCase):
    def test_keys_unique_and_rungs(self):
        for rung in ("a", "b", "c"):
            ov = launcher.overrides(dict(launcher.PROFILES["probe"]), args(rung), Path("/out"))
            keys = [o.split("=", 1)[0].lstrip("+") for o in ov]
            self.assertEqual(len(keys), len(set(keys)), [k for k in keys if keys.count(k) > 1])
            d = {o.split("=", 1)[0].lstrip("+"): o.split("=", 1)[1] for o in ov}
            self.assertEqual(d["grpo.max_rollout_turns"], "4")
            self.assertEqual(d["policy.generation.vllm_cfg.cap_max_tokens_to_context"], "true")
            self.assertEqual(d["policy.tokenizer.chat_template_kwargs.enable_thinking"], "false")
            self.assertEqual(d["data.default.env_name"], "math_code")
            self.assertEqual(d["env.math_code.socket"], "/spool/mathexec.sock")
            self.assertEqual(d["data_plane.enabled"], "false")
            self.assertEqual(d["policy.generation.stop_strings"], '["</code>"]')
            self.assertEqual(d["loss_fn.reference_policy_kl_penalty"], "0.0")
            self.assertEqual(d.get("policy.automodel_cfg.lora_cfg.enabled"), None if rung == "c" else "true")
            if rung != "c":
                self.assertEqual((d["policy.automodel_cfg.lora_cfg.dim"], d["policy.automodel_cfg.lora_cfg.alpha"]),
                                 ("32", "64"))
                self.assertEqual(d["policy.max_total_sequence_length"], "4096")
                self.assertEqual(d["policy.logprob_chunk_size"], "512")
            self.assertEqual(d["policy.generation.max_new_tokens"], "1024")
            self.assertEqual(d["data.default.prompt_file"], str(AREA / "prompt_math_code_oneshot.txt"))
        self.assertNotIn("expandable_segments", " ".join(ov))

    def test_rung_c_refused(self):
        with self.assertRaises(SystemExit) as cm:
            launcher.main(["--profile", "probe", "--out", "/nonexistent/out", "--seed", "1", "--base", "/x",
                           "--memory", "c", "--socket", "/nonexistent/s"])
        self.assertIn("rung c", str(cm.exception))
        self.assertFalse(Path("/nonexistent/out").exists())


@unittest.skipIf(OmegaConf is None, "hydra-core / omegaconf not installed in this venv")
@skip_unless(HAVE_NEMO_RL, NO_NEMO_RL)
class OverridesAgainstConfig(unittest.TestCase):
    def test_apply_to_pinned_config(self):
        for rung in ("a", "b", "c"):
            cfg = OmegaConf.load(NEMO_RL / "examples/configs/grpo_math_1B.yaml")
            OmegaConf.set_struct(cfg, True)
            ov = launcher.overrides(dict(launcher.PROFILES["smoke"]), args(rung), Path("/out"))
            parsed = OverridesParser.create().parse_overrides(overrides=ov)
            ConfigLoaderImpl._apply_overrides_to_config(overrides=parsed, cfg=cfg)
            c = OmegaConf.to_container(cfg, resolve=False)
            self.assertEqual(c["grpo"]["max_rollout_turns"], 4)
            self.assertEqual(c["policy"]["tokenizer"]["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(c["env"]["math_code"]["max_turns"], 4)
            self.assertEqual(c["data"]["default"]["processor"], "math_code_rowid")
            self.assertEqual(c["policy"]["scheduler"][2]["milestones"], [10])
            self.assertFalse(c["policy"]["sequence_packing"]["enabled"])
            self.assertEqual(c["policy"]["automodel_cfg"]["lora_cfg"]["enabled"], rung != "c")
            if rung != "c":
                self.assertEqual(c["policy"]["automodel_cfg"]["lora_cfg"]["target_modules"], ["*_proj"])
                self.assertFalse(c["policy"]["automodel_cfg"]["lora_cfg"]["match_all_linear"])
            self.assertEqual(c["policy"]["generation"]["stop_strings"], ["</code>"])
            self.assertNotIn("skip_tokenizer_init", c["policy"]["generation"]["vllm_cfg"])   # NeMo RL derives it
            self.assertEqual(c["policy"]["generation"]["vllm_cfg"]["max_model_len"],
                             "${policy.max_total_sequence_length}")
            self.assertTrue(c["policy"]["generation"]["vllm_cfg"]["cap_max_tokens_to_context"])
            self.assertEqual(c["data"]["default"]["prompt_file"], str(gen_agentic.PROMPT_FILE))
            # training and evaluation share prompt, stop string and budget (gen_agentic.SAMPLER)
            self.assertEqual(c["policy"]["generation"]["stop_strings"], gen_agentic.SAMPLER["stop"])
            self.assertEqual(c["policy"]["generation"]["max_new_tokens"], 1024)
            self.assertEqual(c["policy"]["generation"]["max_new_tokens"],
                             gen_agentic.SAMPLER["max_new_tokens_per_turn"])
            self.assertIsNone(c["policy"]["generation"]["stop_token_ids"])           # EOS only, as the evaluator
            self.assertEqual((c["policy"]["generation"]["temperature"], c["policy"]["generation"]["top_p"],
                              c["policy"]["generation"]["top_k"]), (1.0, 1.0, None))
            if rung != "c":
                self.assertEqual(c["policy"]["max_total_sequence_length"], 4096)
                self.assertEqual(c["policy"]["max_total_sequence_length"], gen_agentic.SAMPLER["max_model_len"])
                self.assertEqual(c["policy"]["logprob_chunk_size"], 512)
                self.assertEqual((c["policy"]["train_micro_batch_size"], c["policy"]["logprob_batch_size"]), (1, 1))


@skip_unless(HAVE_BASE_TOKENIZER, NO_BASE)
class CutWriter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="a10-cut-"))
        self.step = self.tmp / "ckpt" / "step_5"
        w = self.step / "policy" / "weights" / "model"
        w.mkdir(parents=True)
        (w / "adapter_model.safetensors").write_bytes(b"\0" * 64)
        (w / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 32, "lora_alpha": 64,
                                                           "target_modules": ["q_proj"]}))
        (w / "automodel_peft_config.json").write_text("{}")
        t = self.step / "policy" / "tokenizer"
        t.mkdir(parents=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            shutil.copyfile(AREA / "models" / "qwen3-1.7b" / name, t / name)
        self.meta = {"seed": 1, "profile": "smoke", "stage": "agentic_grpo", "env_id": "math_code",
                     "memory_rung": "a", "lora": {"dim": 32, "alpha": 64}, "script_sha256": "sha256:x",
                     "examples_consumed": lambda s: 16 * s}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_lora_cut(self):
        out = self.tmp / "out"
        out.mkdir()
        cut = hooks.write_cut(self.step, 5, out, AREA / "models" / "qwen3-1.7b", self.meta)
        doc = json.loads((cut / "cut.json").read_text())
        self.assertEqual(doc["kind"], "lora_adapter")
        self.assertEqual(sorted(doc["files"]), sorted(f.name for f in cut.iterdir() if f.name != "cut.json"))
        for f, h in doc["files"].items():
            self.assertEqual(hooks.sha256_file(cut / f), h)
        self.assertNotIn("base_ref.json", doc["deployable"])
        self.assertIn("adapter_model.safetensors", doc["deployable"])
        self.assertEqual(os.stat(cut / "adapter_model.safetensors").st_nlink, 2)        # hard link
        self.assertEqual(json.loads((cut / "base_ref.json").read_text())["revision"],
                         "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
        sampler = json.loads((cut / "sampler.json").read_text())
        self.assertEqual(sampler["max_turns"], 4)
        self.assertEqual((sampler["max_new_tokens_per_turn"], sampler["max_model_len"]), (1024, 4096))
        self.assertEqual(sampler["prompt_file"], "prompt_math_code_oneshot.txt")
        self.assertEqual(sampler["prompt_file_sha256"], hooks.sha256_file(AREA / "prompt_math_code_oneshot.txt"))
        gen_agentic.check_sampler(cut)                                                   # the evaluator accepts it
        old = dict(sampler, max_new_tokens_per_turn=512, max_model_len=2048,
                   prompt_file_sha256=hooks.sha256_file(AREA / "prompt_math_code.txt"))
        (cut / "sampler.json").write_text(json.dumps(old))
        with self.assertRaises(SystemExit) as cm:                                       # a p1-a2-style cut
            gen_agentic.check_sampler(cut)
        self.assertIn("prompt_file_sha256", str(cm.exception))
        self.assertEqual(doc["examples_consumed"], 80)
        self.assertFalse(any(p.name.startswith(".") for p in (out / "cuts").iterdir()))

    def test_wrong_rank_refused(self):
        (self.step / "policy/weights/model/adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 8,
                                                                                       "lora_alpha": 64}))
        out = self.tmp / "out"
        out.mkdir()
        with self.assertRaises(RuntimeError):
            hooks.write_cut(self.step, 5, out, AREA / "models" / "qwen3-1.7b", self.meta)


if __name__ == "__main__":
    unittest.main()
