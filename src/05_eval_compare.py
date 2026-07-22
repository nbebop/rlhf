"""
Stage 5: Evaluation -- compare SFT vs PPO vs DPO checkpoints.

Generates responses from each available checkpoint on the held-out
generation-eval prompts (`cfg["gen_eval_path"]`), scores each prompt+response
pair with the stage-2 reward model, and prints the mean reward per model
plus side-by-side samples. This is the showcase script that demonstrates
whether RL/preference fine-tuning actually moved the policy in the direction
the reward model rewards.

Run:
    python src/05_eval_compare.py --config configs/ckan_config.yaml
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from common import load_config

CANDIDATES = ("sft", "ppo", "dpo")
BATCH_SIZE = 8
REWARD_MAX_LENGTH = 512


def _check_safetensors_file(path):
    """Compare the file size against the size the safetensors header implies.

    Reads only the header (a few KB), not the tensor data, so this is cheap
    even for large checkpoints. Catches saves that were interrupted partway
    through writing the data section (e.g. a Colab disconnect or an OOM kill
    mid-`save_pretrained`) -- exactly the "broken halfway" case that would
    otherwise surface as a cryptic error deep inside `from_pretrained`.
    """
    size = os.path.getsize(path)
    if size == 0:
        return ["model.safetensors is empty (0 bytes)"]
    try:
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            if 8 + header_len > size:
                return ["model.safetensors header is truncated"]
            header = json.loads(f.read(header_len))
    except (OSError, json.JSONDecodeError) as e:
        return [f"model.safetensors header is unreadable ({e})"]

    data_end = max(
        (v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"),
        default=0,
    )
    expected_size = 8 + header_len + data_end
    if size != expected_size:
        return [
            f"model.safetensors size mismatch: file is {size} bytes, header implies "
            f"{expected_size} (likely truncated by an interrupted save)"
        ]
    return []


def check_checkpoint_health(ckpt_dir, check_tokenizer=False):
    """Sanity-check a checkpoint directory before spending GPU time on it.

    Returns a list of human-readable issues; an empty list means the
    checkpoint looks complete and loadable. Does not fully load the model.

    `check_tokenizer` should only be set for directories this script actually
    loads a tokenizer from (sft/reward) -- PPO/DPO's own re-saved tokenizer
    copy is never read (see the comment in main()), so a missing one there
    isn't a real problem.
    """
    issues = []

    if check_tokenizer:
        has_tokenizer_json = os.path.isfile(os.path.join(ckpt_dir, "tokenizer.json"))
        has_tokenizer_config = os.path.isfile(os.path.join(ckpt_dir, "tokenizer_config.json"))
        if not (has_tokenizer_json or has_tokenizer_config):
            issues.append("no tokenizer files found (tokenizer.json / tokenizer_config.json) -- "
                           "training run may have been interrupted before its final save")

    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(config_path):
        issues.append("missing config.json")
    else:
        try:
            with open(config_path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            issues.append(f"config.json is not valid JSON ({e})")

    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    single_path = os.path.join(ckpt_dir, "model.safetensors")
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")

    if os.path.isfile(index_path):
        try:
            with open(index_path) as f:
                index = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(f"model.safetensors.index.json is not valid JSON ({e})")
            index = {}
        shard_files = sorted(set(index.get("weight_map", {}).values()))
        if not shard_files:
            issues.append("weight index lists no shards")
        for shard in shard_files:
            shard_path = os.path.join(ckpt_dir, shard)
            if not os.path.isfile(shard_path):
                issues.append(f"missing shard {shard}")
            elif os.path.getsize(shard_path) == 0:
                issues.append(f"shard {shard} is empty (0 bytes)")
    elif os.path.isfile(single_path):
        issues.extend(_check_safetensors_file(single_path))
    elif os.path.isfile(bin_path):
        if os.path.getsize(bin_path) == 0:
            issues.append("pytorch_model.bin is empty (0 bytes)")
    else:
        issues.append("no weight file found (model.safetensors / pytorch_model.bin / *.index.json)")

    return issues


def generate_responses(ckpt_dir, tokenizer, prompts, max_new_tokens, device):
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    responses = []
    with torch.inference_mode():
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[:, prompt_len:]
            responses.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return responses


def score_responses(reward_model, reward_tokenizer, prompts, responses, device):
    rewards = []
    with torch.inference_mode():
        for i in range(0, len(prompts), BATCH_SIZE):
            texts = [p + "\n" + r for p, r in zip(prompts[i:i + BATCH_SIZE], responses[i:i + BATCH_SIZE])]
            inputs = reward_tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=REWARD_MAX_LENGTH
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = reward_model(**inputs).logits.squeeze(-1)
            rewards.extend(logits.tolist())
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--check-only", action="store_true",
                         help="Run the checkpoint health check and exit, without generating or scoring.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dir_by_name = {
        "sft": cfg["sft_output_dir"],
        "ppo": cfg["ppo_output_dir"],
        "dpo": cfg.get("dpo_output_dir"),
        "reward": cfg["reward_output_dir"],
    }

    print("Checkpoint health check:")
    available = []
    healthy_by_name = {}
    for name in CANDIDATES + ("reward",):
        ckpt_dir = dir_by_name[name]
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            print(f"  {name}: NOT FOUND ({ckpt_dir!r})")
            healthy_by_name[name] = False
            continue
        issues = check_checkpoint_health(ckpt_dir, check_tokenizer=(name in ("sft", "reward")))
        if issues:
            print(f"  {name}: BROKEN at {ckpt_dir}")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  {name}: ok ({ckpt_dir})")
        healthy_by_name[name] = not issues
        if name != "reward" and not issues:
            available.append((name, ckpt_dir))

    if args.check_only:
        return

    if not healthy_by_name["reward"]:
        raise SystemExit("Reward model checkpoint is missing or broken -- can't score any responses. "
                          "Re-run src/02_reward_model.py.")
    if not available:
        raise SystemExit("No healthy SFT/PPO/DPO checkpoints available to evaluate. "
                          "Run at least one training stage first (or re-run it if it was interrupted).")

    dataset = load_dataset("json", data_files=cfg["gen_eval_path"], split="train")
    prompts = dataset["prompt"]
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]

    # PPO/DPO don't modify the tokenizer -- load it once from the SFT
    # checkpoint (guaranteed to exist and be complete) instead of trusting
    # each stage's own re-saved copy, which may be missing if that stage's
    # training run was interrupted before its final save.
    tokenizer = AutoTokenizer.from_pretrained(cfg["sft_output_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    reward_tokenizer = AutoTokenizer.from_pretrained(cfg["reward_output_dir"])
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    reward_model = AutoModelForSequenceClassification.from_pretrained(cfg["reward_output_dir"], num_labels=1)
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model.to(device)
    reward_model.eval()

    max_new_tokens = cfg["ppo"]["response_length"]

    responses_by_model = {}
    rewards_by_model = {}
    for name, ckpt_dir in available:
        responses = generate_responses(ckpt_dir, tokenizer, prompts, max_new_tokens, device)
        rewards = score_responses(reward_model, reward_tokenizer, prompts, responses, device)
        responses_by_model[name] = responses
        rewards_by_model[name] = rewards

    print("\nmodel | mean reward | std | n")
    for name, _ in available:
        rewards = torch.tensor(rewards_by_model[name])
        mean = rewards.mean().item()
        std = rewards.std().item() if rewards.numel() > 1 else 0.0
        print(f"{name} | {mean:.4f} | {std:.4f} | {rewards.numel()}")

    num_samples = min(args.num_samples, len(prompts))
    for i in range(num_samples):
        print(f"\n--- sample {i + 1} ---")
        print(f"prompt: {prompts[i]}")
        for name, _ in available:
            response = responses_by_model[name][i]
            reward = rewards_by_model[name][i]
            print(f"[{name}] (reward={reward:.4f}): {response}")


if __name__ == "__main__":
    main()
