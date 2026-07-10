"""
Fast end-to-end sanity check for the pipeline, using a tiny model and 1 step
per stage. Run this after `pip install -r requirements.txt` and before
kicking off a real (slow, GPU) run, to catch wiring bugs cheaply.

Run:
    python src/smoke_test.py
"""

import argparse
import copy
import os
import shutil
import sys
import tempfile

from common import load_config

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"  # a few MB, CPU-friendly
SMOKE_MAX_ROWS = 32  # wiring check only; full datasets make CPU smoke runs take 30+ min


def _truncate_jsonl(path, dest_dir, max_rows=SMOKE_MAX_ROWS):
    dest = os.path.join(dest_dir, os.path.basename(path))
    with open(path, encoding="utf-8") as src, open(dest, "w", encoding="utf-8") as out:
        for i, line in enumerate(src):
            if i >= max_rows:
                break
            out.write(line)
    return dest


def make_smoke_config(cfg, data_dir):
    smoke = copy.deepcopy(cfg)
    smoke["base_model"] = TINY_MODEL
    for key in ("sft_data_path", "preference_data_path", "ppo_prompts_path",
                "rm_eval_path", "gen_eval_path"):
        if smoke.get(key):
            smoke[key] = _truncate_jsonl(smoke[key], data_dir)
    smoke["sft_output_dir"] = cfg["sft_output_dir"] + "_smoke"
    smoke["reward_output_dir"] = cfg["reward_output_dir"] + "_smoke"
    smoke["ppo_output_dir"] = cfg["ppo_output_dir"] + "_smoke"
    smoke["sft"]["num_train_epochs"] = 1
    smoke["reward"]["num_train_epochs"] = 1
    smoke["ppo"]["total_episodes"] = 8
    smoke["ppo"]["per_device_train_batch_size"] = 2
    smoke["ppo"]["mini_batch_size"] = 1
    smoke["ppo"]["response_length"] = 8
    if "dpo" in smoke:
        smoke["dpo_output_dir"] = cfg["dpo_output_dir"] + "_smoke"
        smoke["dpo"]["num_train_epochs"] = 1
    return smoke


def run(config_path=None):
    cfg = load_config(config_path)
    smoke_data_dir = tempfile.mkdtemp(prefix="rlhf_smoke_data_")
    smoke_cfg = make_smoke_config(cfg, smoke_data_dir)
    # Optional stages only exist for configs that define them (e.g. the CKAN
    # config); the default config still smoke-tests the classic three stages.
    has_dpo = "dpo" in smoke_cfg
    has_gen_eval = bool(smoke_cfg.get("gen_eval_path"))
    total = 3 + has_dpo + has_gen_eval
    output_dirs = [smoke_cfg["sft_output_dir"], smoke_cfg["reward_output_dir"],
                   smoke_cfg["ppo_output_dir"]]

    print(f"=== Stage 1/{total}: SFT (tiny model, 1 epoch) ===")
    import importlib
    sft_mod = importlib.import_module("01_sft")
    _run_stage(sft_mod, smoke_cfg)

    print(f"=== Stage 2/{total}: Reward model (tiny model, 1 epoch) ===")
    reward_mod = importlib.import_module("02_reward_model")
    _run_stage(reward_mod, smoke_cfg)

    print(f"=== Stage 3/{total}: PPO (8 episodes) ===")
    ppo_mod = importlib.import_module("03_ppo_train")
    _run_stage(ppo_mod, smoke_cfg)

    if has_dpo:
        print(f"=== Stage 4/{total}: DPO baseline (tiny model, 1 epoch) ===")
        dpo_mod = importlib.import_module("04_dpo_train")
        _run_stage(dpo_mod, smoke_cfg)
        output_dirs.append(smoke_cfg["dpo_output_dir"])

    if has_gen_eval:
        print(f"=== Stage {total}/{total}: Eval comparison (4 prompts) ===")
        eval_mod = importlib.import_module("05_eval_compare")
        _run_stage(eval_mod, smoke_cfg,
                   extra_argv=["--max-prompts", "4", "--num-samples", "1"])

    print(f"\nSmoke test passed: all {total} stages ran without errors.")
    for d in output_dirs:
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(smoke_data_dir, ignore_errors=True)


def _run_stage(module, cfg, extra_argv=None):
    # Each stage script does `from common import load_config`, which binds
    # its own local name -- patch that local binding, not common's, or the
    # patch is invisible to the already-imported module.
    original = module.load_config
    original_argv = sys.argv
    module.load_config = lambda *_a, **_k: cfg
    if extra_argv is not None:
        # Stage mains parse sys.argv; replace it so smoke-only flags apply
        # and smoke_test's own --config flag is not re-parsed by the stage.
        sys.argv = [original_argv[0]] + extra_argv
    try:
        module.main()
    finally:
        module.load_config = original
        sys.argv = original_argv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    sys.path.insert(0, "")
    run(args.config)
