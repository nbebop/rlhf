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
GPU time on a real run.

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

- `SFTTrainer` / `RewardTrainer` are the stable, current APIs.
- `PPOTrainer` / `PPOConfig` currently live under `trl.experimental.ppo` —
  Hugging Face moved PPO into their experimental namespace and it's marked
  as subject to change. `requirements.txt` pins `trl>=0.29` where this move
  already happened; re-check
  [the PPO trainer docs](https://huggingface.co/docs/trl/main/en/ppo_trainer)
  if you upgrade `trl` and something breaks.
- If you'd rather skip PPO's complexity (separate reward model + RL loop),
  `trl.DPOTrainer` (Direct Preference Optimization) trains directly on the
  same `chosen`/`rejected` preference pairs and is the more common practical
  default today. Worth considering if PPO's instability becomes a blocker.

## What was and wasn't verified here

- All scripts compile cleanly (`python -m py_compile`).
- `configs/config.yaml` parses and has the sections each script expects.
- All three `data/*.jsonl` files parse and match the schema each script
  reads.
- `common.load_config()` resolves paths correctly regardless of caller cwd.
- `python src/smoke_test.py` has been run end-to-end (CPU, tiny model) and
  all three stages — SFT, reward modeling, PPO — complete without errors.

Not verified: a real (non-tiny) training run on GPU.
