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
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from common import load_config

CANDIDATES = ("sft", "ppo", "dpo")
BATCH_SIZE = 8
REWARD_MAX_LENGTH = 512


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
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset("json", data_files=cfg["gen_eval_path"], split="train")
    prompts = dataset["prompt"]
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]

    dir_by_name = {
        "sft": cfg["sft_output_dir"],
        "ppo": cfg["ppo_output_dir"],
        "dpo": cfg.get("dpo_output_dir"),
    }

    available = []
    for name in CANDIDATES:
        ckpt_dir = dir_by_name[name]
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            print(f"WARNING: {name} checkpoint not found at {ckpt_dir!r}, skipping")
            continue
        available.append((name, ckpt_dir))

    if not available:
        raise SystemExit("No checkpoints available to evaluate. Run at least one training stage first.")

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
