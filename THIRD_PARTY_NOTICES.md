# Third-party notices

This repository redistributes data from the MATH dataset and small configuration files of the Qwen3-1.7B model. Its
own code is licensed under Apache-2.0 (`LICENSE`).

## MATH dataset

- Source: the MATH dataset of Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset",
  NeurIPS 2021, https://github.com/hendrycks/math, as packaged on the Hugging Face Hub in `EleutherAI/hendrycks_math`
  at revision `21a5633873b6a120296cce3e2df9d5550074f4a3` (dataset card: `license: mit`).
- Redistributed here: `runtime/items-test/items.jsonl` (500 test problems with their solutions' integer answers,
  selected by `runtime/make_data.py`); test fixtures under `tests/fixtures/agentic/` (`real-arm/items-first20.jsonl`
  and the model outputs on those 20 test problems; six training problems in `p1-a3/data/shards/` and the probe's logs
  that contain them) and `runtime/tests/fixtures/p1-a2/` (training problems in the logged prompts).
- License: MIT. The text below is the `LICENSE` file of https://github.com/hendrycks/math at commit
  `985bdc1696e88e8643f081a0ff4719da39f2ae2a` (unchanged since the repository's first commit), fetched from
  https://raw.githubusercontent.com/hendrycks/math/985bdc1696e88e8643f081a0ff4719da39f2ae2a/LICENSE
  (SHA-256 `8f6401fb0b1fbfcac17aac8d61e4e1b0d2de9aa190d8aa655fedc5fd783b2ace`).

```
MIT License

Copyright (c) 2021 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen3-1.7B

- Source: `Qwen/Qwen3-1.7B` on the Hugging Face Hub at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
  (Qwen team, Alibaba Cloud).
- Redistributed here: `tests/fixtures/agentic/p1-a3/cuts/checkpoint-3/tokenizer_config.json` and
  `chat_template.jinja` (the tokenizer configuration and chat template NeMo RL saved with a LoRA cut). The model
  weights, tokenizer and vocabulary files are not redistributed (the fixture holds stand-ins).
- License: Apache License 2.0, the same text as this repository's `LICENSE` (the model repository's `LICENSE` file and
  model card `license: apache-2.0`).

## NeMo RL

- Source: https://github.com/NVIDIA-NeMo/RL at commit `4f39fd86289a2ffd5aaf5d202702f7fd7185c924`, Apache License 2.0.
- Not redistributed: the recipe runs NeMo RL inside its container image. `tests/fixtures/agentic/p1-a3/` and
  `runtime/tests/fixtures/p1-a2/` contain excerpts of logs NeMo RL wrote during training runs of this recipe.
