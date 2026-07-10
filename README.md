# RLHF Prototype

A working scaffold for the classic three-stage RLHF pipeline — supervised
fine-tuning (SFT) → reward modeling → PPO — built on Hugging Face's `trl`.
It runs end-to-end on tiny data/models as a wiring check, and is structured
so you can swap in your real dataset and a bigger model for an actual GPU run.

## Pipeline

```
data/sft_data.jsonl          data/preference_data.jsonl      data/ppo_prompts.jsonl
  (prompt, response)           (prompt, chosen, rejected)       (prompt)
        │                              │                              │
        ▼                              ▼                              ▼
  src/01_sft.py  ──────────►  src/02_reward_model.py  ──────►  src/03_ppo_train.py
  (SFTTrainer)      SFT ckpt     (RewardTrainer)      reward ckpt   (PPOTrainer)
        │                              │                              │
        ▼                              ▼                              ▼
  outputs/sft_model            outputs/reward_model          outputs/ppo_model
```

All three stages read shared settings from `configs/config.yaml`.

## Setup

```bash
pip install -r requirements.txt
```

GPU strongly recommended for anything beyond the tiny smoke test — PPO holds
4 models in memory at once (policy, reference, reward, value).

## Quickstart

```bash
cd src
python smoke_test.py       # tiny model, ~seconds, just checks the wiring
python 01_sft.py           # real run, stage 1
python 02_reward_model.py  # real run, stage 2
python 03_ppo_train.py     # real run, stage 3
```

`smoke_test.py` swaps in `hf-internal-testing/tiny-random-gpt2` and runs one
short pass of each stage — use it to catch config/data bugs before spending
GPU time on a real run. With a config that defines the optional `dpo` section
and `gen_eval_path` (like `configs/ckan_config.yaml`), it covers all five
stages (SFT, reward model, PPO, DPO, eval comparison); with the default
config it covers the classic three:

```bash
python smoke_test.py --config ../configs/ckan_config.yaml
```

## Using your own data and a real model

1. **Data** — replace the three files in `data/` with your own, keeping the
   same schema:
   - `sft_data.jsonl`: `{"prompt": ..., "response": ...}`
   - `preference_data.jsonl`: `{"prompt": ..., "chosen": ..., "rejected": ...}`
   - `ppo_prompts.jsonl`: `{"prompt": ...}`
2. **Model** — in `configs/config.yaml`, set `base_model` to something like
   `Qwen/Qwen2.5-1.5B-Instruct` or `meta-llama/Llama-3.2-1B-Instruct`.
3. **LoRA** — for anything above ~1B params on a single consumer GPU, set
   `use_lora: true` in the config and adjust `lora.target_modules` for your
   model's attention layer names (`q_proj`/`v_proj` works for
   Llama/Qwen-style architectures).
4. **Scale up PPO** — `ppo.total_episodes: 200` in the config is a toy value
   for the smoke test; a real run typically needs 10,000+.

## Notes on the TRL API

- `SFTTrainer` / `RewardTrainer` / `DPOTrainer` are the stable, current APIs.
- `PPOTrainer` / `PPOConfig` live under `trl.experimental.ppo` — Hugging Face
  moved PPO into their experimental namespace and it's marked as subject to
  change. For that reason `requirements.txt` pins `trl==1.8.0` exactly (the
  version this repo was verified against, July 2026); re-check
  [the PPO trainer docs](https://huggingface.co/docs/trl/main/en/ppo_trainer)
  before bumping the pin.
- The rest of the stack uses version floors: a fresh install resolves to
  transformers 5.x / datasets 4.x, and the whole pipeline was smoke-tested
  against torch 2.13.0 / transformers 5.13.0 / trl 1.8.0. `torch` stays a
  loose floor on purpose so Colab keeps its preinstalled CUDA build.
- If you'd rather skip PPO's complexity (separate reward model + RL loop),
  `src/04_dpo_train.py` (Direct Preference Optimization) trains directly on
  the same `chosen`/`rejected` preference pairs and is the more common
  practical default today. This repo trains it as a baseline alongside PPO
  so the two can be compared with `src/05_eval_compare.py`.

## Berlin CKAN synthetic metadata experiment

The repo also includes a small reproducible workflow for turning Berlin CKAN
package metadata into the three pipeline datasets. It is meant for testing the
pipeline, not for creating a high-quality human preference dataset.

Step 1: fetch the raw CKAN dump.

```bash
python src/fetch_ckan_dump.py --rows 1000
```

This writes `data/berlin_ckan_package_search.json`. The script defaults to the
working CKAN backend, `https://datenregister.berlin.de/api/3/action/package_search`.
The `--rows 1000` flag requests all available packages (the Berlin portal has ~2,616
packages; CKAN's `package_search` endpoint caps responses at 1000 rows per request).

Step 2: generate synthetic RLHF data.

```bash
python src/prepare_ckan_data.py
```

This writes five files with a 600/200/100/100 package split:

- `data/ckan_sft_data.jsonl` — 600 rows ({"prompt", "response"})
- `data/ckan_preference_data.jsonl` — ~1200 rows (2 preference pairs per training package)
- `data/ckan_ppo_prompts.jsonl` — 200 held-out prompts
- `data/ckan_rm_eval.jsonl` — 100 held-out preference pairs for reward-model eval accuracy
- `data/ckan_gen_eval.jsonl` — 100 held-out prompts for before/after generation comparison

Step 3: understand how the synthetic labels are created.

- SFT rows use a clean, structured rendering of the real CKAN package as the
  `response`.
- Preference rows use the same clean rendering as `chosen`, paired with two types
  of degraded renderings as negatives: 3 easy degradation modes (vague/short) and
  3 hard negatives (truncated-clean, field-swapped-from-another-package,
  unknown-heavy). This grading prevents the reward model from learning a simple
  length heuristic.
- Prompts use 5 deterministic template variants chosen per package, so there is
  structural variety in the inputs.
- PPO and eval rows contain only held-out prompts; the model generates responses
  during PPO and eval, and the reward model scores them.

Step 4: run the pipeline against the CKAN files.

```bash
python src/01_sft.py --config configs/ckan_config.yaml
python src/02_reward_model.py --config configs/ckan_config.yaml
python src/03_ppo_train.py --config configs/ckan_config.yaml
python src/04_dpo_train.py --config configs/ckan_config.yaml   # optional DPO baseline
python src/05_eval_compare.py --config configs/ckan_config.yaml
```

The first three stages train SFT → reward model → PPO. The DPO stage trains a
Direct Preference Optimization baseline from the SFT checkpoint on the same
preference data. The final eval stage generates responses from all three model
checkpoints on the held-out prompts, scores them with the trained reward model,
and prints mean reward per model plus side-by-side samples.

### Cloud GPU notes

For the CKAN workflow at scale:

- **Recommended GPUs**: L4, A10G, or A100 (all support bfloat16). The free Colab
  T4 has no bfloat16 support; the training scripts auto-detect this via
  `common.precision_kwargs()` (`torch.cuda.is_bf16_supported()`) and fall back
  to fp16 automatically, so no manual edits are needed on a T4.
- **Memory**: With the 0.5B base model, full-parameter PPO fits on a single 24 GB
  GPU (PPO holds 4 models in memory: policy, reference, reward, value). For 1.5B+
  parameter models, set `use_lora: true` in `configs/ckan_config.yaml`.
- **Response length**: `ppo.response_length` is 384 because the median clean CKAN
  record is ~280 tokens (p90 ~360) — a shorter budget caps the reward the policy
  can possibly reach. If you shrink it to save memory, expect a reward ceiling.
- **Success criteria**: The reward model eval should show high accuracy but below
  100% (hard negatives should cost a few points). PPO mean reward should rise
  monotonically with bounded KL divergence. Most importantly, `05_eval_compare.py`
  should show PPO and DPO mean reward clearly above SFT.

## What was and wasn't verified here

- All scripts compile cleanly (`python -m py_compile`).
- `configs/config.yaml` parses and has the sections each script expects.
- All three `data/*.jsonl` files parse and match the schema each script
  reads.
- `common.load_config()` resolves paths correctly regardless of caller cwd.
- `python src/smoke_test.py --config configs/ckan_config.yaml` has been run
  end-to-end (CPU, tiny model, 32-row data slices) and all five stages — SFT,
  reward modeling (incl. held-out eval), PPO, DPO, and the eval comparison —
  complete without errors against the pinned stack: trl 1.8.0,
  transformers 5.13.0, torch 2.13.0 (the versions a fresh
  `pip install -r requirements.txt` resolves to as of July 2026).
- The CKAN data pipeline (`fetch_ckan_dump.py` and `prepare_ckan_data.py`) has
  been regenerated at the 1000-package scale, and the datasets were audited:
  disjoint splits, valid schemas, 5 prompt templates, hard negatives present
  (field-swapped rejected responses are full-length and structured).
- Records deliberately contain no resource URLs: TRL 1.x's `RewardTrainer`
  *drops* (not truncates) pairs longer than `max_length`, and URL-heavy
  records blew past 512 tokens — prompt+chosen now tokenizes at median ~310 /
  p90 ~390 tokens (Qwen tokenizer), so only a ~3% tail is filtered.

Not verified: a real (non-tiny) training run on GPU.
