"""
Build small synthetic RLHF datasets from a Berlin CKAN package_search dump.

This script turns the raw dump created by fetch_ckan_dump.py into five
JSONL files used by the repo's training and evaluation pipeline:

1. SFT rows:
   {"prompt": ..., "response": ...}
   The response is a clean, structured rendering of one real CKAN package.

2. Preference rows:
   {"prompt": ..., "chosen": ..., "rejected": ...}
   The chosen response is the same clean rendering. The rejected response is
   a deterministic degraded rendering of the same package, with useful fields
   removed, shortened, made less structured, or (for the "hard" negatives)
   kept fully structured but truncated, mixed up with another package's
   fields, or hollowed out to mostly "Unknown" values. Each training package
   contributes TWO preference rows: one with an easy rejected response and
   one with a hard rejected response, so the reward model sees both kinds of
   mistakes.

3. PPO prompt rows:
   {"prompt": ...}
   These are held-out package prompts only. PPO will generate responses during
   training and score them with the reward model.

4. RM eval rows:
   {"prompt": ..., "chosen": ..., "rejected": ...}
   A held-out preference set (disjoint from training and PPO packages) used to
   evaluate the reward model. One pair per package, alternating easy and hard
   rejected responses by package index so the eval set covers both
   difficulties.

5. Gen eval rows:
   {"prompt": ...}
   A held-out, prompt-only set (disjoint from all of the above) used to
   evaluate generation quality after PPO.

Prompts are drawn deterministically from one of five templates, selected by
`index % 5` within each split, so wording varies across packages without any
randomness. A package's SFT row and its preference rows always use the same
template (they share the same in-split index); PPO and gen-eval prompts use
the same template selection scheme within their own splits.

Package layout is sequential and disjoint across the whole dump:

    [0, train_count)                                          -> SFT + preference rows
    [train_count, train_count + ppo_count)                    -> PPO prompts
    [train_count + ppo_count, ... + rm_eval_count)             -> RM eval preference rows
    [... + rm_eval_count, ... + gen_eval_count)                -> gen eval prompts

Run from the project root after fetching the dump:

    python src/fetch_ckan_dump.py
    python src/prepare_ckan_data.py

By default this writes:

    data/ckan_sft_data.jsonl
    data/ckan_preference_data.jsonl
    data/ckan_ppo_prompts.jsonl
    data/ckan_rm_eval.jsonl
    data/ckan_gen_eval.jsonl

Use --write-defaults if you want to overwrite the pipeline's default training
data files: data/sft_data.jsonl, data/preference_data.jsonl, and
data/ppo_prompts.jsonl. The two eval files are not affected by
--write-defaults; they are always written to data/ckan_rm_eval.jsonl and
data/ckan_gen_eval.jsonl.
"""

import argparse
import json
import os
from typing import Any


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_DUMP_PATH = os.path.join(DATA_DIR, "berlin_ckan_package_search.json")

DEFAULT_SFT_PATH = os.path.join(DATA_DIR, "ckan_sft_data.jsonl")
DEFAULT_PREFERENCE_PATH = os.path.join(DATA_DIR, "ckan_preference_data.jsonl")
DEFAULT_PPO_PATH = os.path.join(DATA_DIR, "ckan_ppo_prompts.jsonl")
DEFAULT_RM_EVAL_PATH = os.path.join(DATA_DIR, "ckan_rm_eval.jsonl")
DEFAULT_GEN_EVAL_PATH = os.path.join(DATA_DIR, "ckan_gen_eval.jsonl")

PIPELINE_SFT_PATH = os.path.join(DATA_DIR, "sft_data.jsonl")
PIPELINE_PREFERENCE_PATH = os.path.join(DATA_DIR, "preference_data.jsonl")
PIPELINE_PPO_PATH = os.path.join(DATA_DIR, "ppo_prompts.jsonl")

PROMPT_TEMPLATES = [
    "Write a clear, complete metadata summary for the Berlin open dataset "
    "'{title}'.",
    "Summarize the metadata of the Berlin open-data package '{title}' in a "
    "structured, labeled format.",
    "Produce a complete metadata record for the dataset '{title}' from the "
    "Berlin open data portal.",
    "Describe the dataset '{title}' published on daten.berlin.de, covering "
    "all available metadata fields.",
    "Erstelle eine vollständige, strukturierte Metadaten-Übersicht für den "
    "Berliner Datensatz '{title}'.",
]
DESCRIPTION_CHAR_LIMIT = 500
SHORT_DESCRIPTION_CHAR_LIMIT = 120
MAX_TAGS = 8
MAX_RESOURCES = 6
NUM_DEGRADED_MODES = 6
TRUNCATED_CLEAN_LINE_COUNT = 5
FIELD_SWAP_DONOR_OFFSET = 7
FIELD_SWAP_DONOR_FALLBACK_OFFSET = 8


def text(value: Any, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    normalized = " ".join(str(value).split())
    return normalized if normalized else fallback


def truncate_words(value: str, limit: int) -> str:
    value = text(value, fallback="")
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0] + "..."


def load_packages(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        dump = json.load(f)

    if not dump.get("success"):
        raise ValueError(f"CKAN dump does not have success=true: {path}")

    packages = dump.get("result", {}).get("results", [])
    if not isinstance(packages, list):
        raise ValueError("CKAN dump result.results is not a list")

    return [pkg for pkg in packages if package_title(pkg) != "Unknown"]


def package_title(pkg: dict[str, Any]) -> str:
    return text(pkg.get("title") or pkg.get("name"))


def package_tags(pkg: dict[str, Any]) -> list[str]:
    tags = []
    for tag in pkg.get("tags") or []:
        name = text(tag.get("display_name") or tag.get("name"), fallback="")
        if name:
            tags.append(name)
    return tags[:MAX_TAGS]


def package_resource_summaries(pkg: dict[str, Any]) -> list[str]:
    # Deliberately no URLs: they are unlearnable for held-out packages, and
    # they inflate records past the trainers' max_length -- TRL's
    # RewardTrainer *drops* (not truncates) pairs longer than max_length.
    summaries = []
    for resource in (pkg.get("resources") or [])[:MAX_RESOURCES]:
        name = text(resource.get("name") or resource.get("description"), fallback="Unnamed resource")
        fmt = text(resource.get("format"), fallback="unknown format")
        summaries.append(f"{name} ({fmt})")
    return summaries


def clean_response(pkg: dict[str, Any]) -> str:
    """Render useful CKAN fields as a compact, labeled metadata record."""
    organization = pkg.get("organization") or {}
    tags = package_tags(pkg)
    resources = package_resource_summaries(pkg)

    lines = [
        f"Title: {package_title(pkg)}",
        f"Description: {truncate_words(pkg.get('notes'), DESCRIPTION_CHAR_LIMIT)}",
        f"Organization: {text(organization.get('title') or organization.get('name'))}",
        f"License: {text(pkg.get('license_title') or pkg.get('license_id'))}",
        f"Berlin Dataset Type: {text(pkg.get('berlin_type'))}",
        f"Geographical Coverage: {text(pkg.get('geographical_coverage'))}",
        f"Geographical Granularity: {text(pkg.get('geographical_granularity'))}",
        f"Temporal Coverage: {text(pkg.get('temporal_coverage_from'))} to {text(pkg.get('temporal_coverage_to'))}",
        f"Temporal Granularity: {text(pkg.get('temporal_granularity'))}",
        f"Released: {text(pkg.get('date_released'))}",
        f"Updated: {text(pkg.get('date_updated') or pkg.get('metadata_modified'))}",
        f"Tags: {', '.join(tags) if tags else 'Unknown'}",
        f"Resources: {len(pkg.get('resources') or [])}",
    ]

    if resources:
        lines.append("Resource Details:")
        lines.extend(f"- {resource}" for resource in resources)

    return "\n".join(lines)


def _degraded_summary_only(pkg: dict[str, Any]) -> str:
    """Mode 0 (easy): a bare, unstructured one-liner-ish summary."""
    title = package_title(pkg)
    tags = package_tags(pkg)
    return "\n".join([
        f"Title: {title}",
        "Description: Dataset about Berlin.",
        f"Tags: {', '.join(tags[:2]) if tags else 'Berlin'}",
    ])


def _degraded_missing_fields(pkg: dict[str, Any]) -> str:
    """Mode 1 (easy): a short response that drops most metadata fields."""
    description = truncate_words(pkg.get("notes"), SHORT_DESCRIPTION_CHAR_LIMIT)
    return "\n".join([
        f"Title: {package_title(pkg)}",
        f"Description: {description}",
        "License: Unknown",
        "Resources: Not specified",
    ])


def _degraded_prose(pkg: dict[str, Any]) -> str:
    """Mode 2 (easy): unstructured prose with no labeled fields at all."""
    title = package_title(pkg)
    organization = pkg.get("organization") or {}
    org_name = text(organization.get("title") or organization.get("name"))
    return (
        f"{title} is an open data record from {org_name}. "
        f"It contains public-sector information for Berlin. "
        "Important metadata such as license, coverage, dates, tags, and "
        "resource formats is not included here."
    )


def _degraded_truncated_clean(pkg: dict[str, Any]) -> str:
    """Mode 3 (hard): the clean rendering, cut off after a few lines."""
    lines = clean_response(pkg).split("\n")
    return "\n".join(lines[:TRUNCATED_CLEAN_LINE_COUNT])


def _field_swap_values(pkg: dict[str, Any]) -> tuple[str, str, str]:
    organization = pkg.get("organization") or {}
    return (
        text(organization.get("title") or organization.get("name")),
        text(pkg.get("license_title") or pkg.get("license_id")),
        f"{text(pkg.get('temporal_coverage_from'))} to {text(pkg.get('temporal_coverage_to'))}",
    )


def _degraded_field_swapped(pkg: dict[str, Any], packages: list[dict[str, Any]], index: int) -> str:
    """Mode 4 (hard): fully structured, but Organization/License/Temporal

    Coverage are taken from a different ("donor") package, so the response
    looks complete but is factually wrong.

    CKAN dumps contain long series of near-identical packages, so the donor
    search skips ahead until it finds one whose swapped fields actually differ
    -- otherwise the swap would be a no-op and the pair would collapse to a
    different degradation mode.
    """
    own_values = _field_swap_values(pkg)
    donor = None
    for step in range(FIELD_SWAP_DONOR_OFFSET, FIELD_SWAP_DONOR_OFFSET + len(packages)):
        candidate_index = (index + step) % len(packages)
        if candidate_index == index:
            continue
        candidate = packages[candidate_index]
        if _field_swap_values(candidate) != own_values:
            donor = candidate
            break
    if donor is None:
        # Every package shares the same values; keep the deterministic donor
        # and let degraded_response's chosen != rejected guard handle it.
        donor = packages[(index + FIELD_SWAP_DONOR_FALLBACK_OFFSET) % len(packages)]

    donor_organization = donor.get("organization") or {}
    tags = package_tags(pkg)
    resources = package_resource_summaries(pkg)

    lines = [
        f"Title: {package_title(pkg)}",
        f"Description: {truncate_words(pkg.get('notes'), DESCRIPTION_CHAR_LIMIT)}",
        f"Organization: {text(donor_organization.get('title') or donor_organization.get('name'))}",
        f"License: {text(donor.get('license_title') or donor.get('license_id'))}",
        f"Berlin Dataset Type: {text(pkg.get('berlin_type'))}",
        f"Geographical Coverage: {text(pkg.get('geographical_coverage'))}",
        f"Geographical Granularity: {text(pkg.get('geographical_granularity'))}",
        f"Temporal Coverage: {text(donor.get('temporal_coverage_from'))} to {text(donor.get('temporal_coverage_to'))}",
        f"Temporal Granularity: {text(pkg.get('temporal_granularity'))}",
        f"Released: {text(pkg.get('date_released'))}",
        f"Updated: {text(pkg.get('date_updated') or pkg.get('metadata_modified'))}",
        f"Tags: {', '.join(tags) if tags else 'Unknown'}",
        f"Resources: {len(pkg.get('resources') or [])}",
    ]

    if resources:
        lines.append("Resource Details:")
        lines.extend(f"- {resource}" for resource in resources)

    return "\n".join(lines)


def _degraded_unknown_heavy(pkg: dict[str, Any]) -> str:
    """Mode 5 (hard): fully structured, but the most informative fields are

    hollowed out to "Unknown" (and Resource Details is omitted entirely).
    """
    organization = pkg.get("organization") or {}

    lines = [
        f"Title: {package_title(pkg)}",
        "Description: Unknown",
        f"Organization: {text(organization.get('title') or organization.get('name'))}",
        "License: Unknown",
        f"Berlin Dataset Type: {text(pkg.get('berlin_type'))}",
        "Geographical Coverage: Unknown",
        f"Geographical Granularity: {text(pkg.get('geographical_granularity'))}",
        "Temporal Coverage: Unknown",
        f"Temporal Granularity: {text(pkg.get('temporal_granularity'))}",
        f"Released: {text(pkg.get('date_released'))}",
        f"Updated: {text(pkg.get('date_updated') or pkg.get('metadata_modified'))}",
        "Tags: Unknown",
        f"Resources: {len(pkg.get('resources') or [])}",
    ]
    # Resource Details is omitted entirely, per the "hollowed out" degradation.

    return "\n".join(lines)


def _render_degraded_mode(
    pkg: dict[str, Any],
    index: int,
    mode: int,
    packages: list[dict[str, Any]],
) -> str:
    if mode == 0:
        return _degraded_summary_only(pkg)
    if mode == 1:
        return _degraded_missing_fields(pkg)
    if mode == 2:
        return _degraded_prose(pkg)
    if mode == 3:
        return _degraded_truncated_clean(pkg)
    if mode == 4:
        return _degraded_field_swapped(pkg, packages, index)
    if mode == 5:
        return _degraded_unknown_heavy(pkg)
    raise ValueError(f"Unknown degraded mode: {mode}")


def degraded_response(
    pkg: dict[str, Any],
    index: int,
    mode: int,
    packages: list[dict[str, Any]],
) -> str:
    """Create plausible but less useful metadata for reward-model training.

    Modes 0-2 are "easy" negatives (unstructured, short, or missing fields).
    Modes 3-5 are "hard" negatives: they stay structured and plausible
    (truncated-clean, field-swapped, unknown-heavy) so the reward model can't
    just rely on response length or the presence of field labels.

    If a degraded rendering ever happens to equal the clean response (e.g. a
    package with almost no metadata), fall back to the next mode so that
    chosen != rejected always holds.
    """
    clean = clean_response(pkg)
    tried = set()
    current_mode = mode % NUM_DEGRADED_MODES
    while current_mode not in tried:
        tried.add(current_mode)
        candidate = _render_degraded_mode(pkg, index, current_mode, packages)
        if candidate != clean:
            return candidate
        current_mode = (current_mode + 1) % NUM_DEGRADED_MODES

    raise ValueError(
        f"All degraded modes matched the clean response for package index {index}"
    )


def prompt_for(pkg: dict[str, Any], index: int) -> str:
    template = PROMPT_TEMPLATES[index % len(PROMPT_TEMPLATES)]
    return template.format(title=package_title(pkg))


def write_jsonl(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows(
    packages: list[dict[str, Any]],
    train_count: int,
    ppo_count: int,
    rm_eval_count: int,
    gen_eval_count: int,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    if train_count <= 0 or ppo_count <= 0 or rm_eval_count <= 0 or gen_eval_count <= 0:
        raise ValueError(
            "train_count, ppo_count, rm_eval_count, and gen_eval_count must all be positive"
        )

    total_needed = train_count + ppo_count + rm_eval_count + gen_eval_count
    if len(packages) < total_needed:
        raise ValueError(
            f"Need {total_needed} packages total (train={train_count}, "
            f"ppo={ppo_count}, rm_eval={rm_eval_count}, gen_eval={gen_eval_count}), "
            f"found {len(packages)}"
        )

    train_packages = packages[:train_count]
    ppo_packages = packages[train_count:train_count + ppo_count]
    rm_eval_packages = packages[
        train_count + ppo_count: train_count + ppo_count + rm_eval_count
    ]
    gen_eval_packages = packages[
        train_count + ppo_count + rm_eval_count:
        train_count + ppo_count + rm_eval_count + gen_eval_count
    ]

    sft_rows = []
    preference_rows = []
    for index, pkg in enumerate(train_packages):
        prompt = prompt_for(pkg, index)
        chosen = clean_response(pkg)
        easy_mode = index % 3
        hard_mode = 3 + index % 3
        easy_rejected = degraded_response(pkg, index, easy_mode, train_packages)
        hard_rejected = degraded_response(pkg, index, hard_mode, train_packages)

        sft_rows.append({"prompt": prompt, "response": chosen})
        preference_rows.append({"prompt": prompt, "chosen": chosen, "rejected": easy_rejected})
        preference_rows.append({"prompt": prompt, "chosen": chosen, "rejected": hard_rejected})

    ppo_rows = [{"prompt": prompt_for(pkg, index)} for index, pkg in enumerate(ppo_packages)]

    rm_eval_rows = []
    for index, pkg in enumerate(rm_eval_packages):
        prompt = prompt_for(pkg, index)
        chosen = clean_response(pkg)
        mode = (index % 3) if index % 2 == 0 else 3 + index % 3
        rejected = degraded_response(pkg, index, mode, rm_eval_packages)
        rm_eval_rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    gen_eval_rows = [
        {"prompt": prompt_for(pkg, index)} for index, pkg in enumerate(gen_eval_packages)
    ]

    return sft_rows, preference_rows, ppo_rows, rm_eval_rows, gen_eval_rows


def validate_rows(
    sft_rows: list[dict[str, str]],
    preference_rows: list[dict[str, str]],
    ppo_rows: list[dict[str, str]],
    rm_eval_rows: list[dict[str, str]],
    gen_eval_rows: list[dict[str, str]],
) -> None:
    for row in sft_rows:
        if sorted(row) != ["prompt", "response"]:
            raise ValueError(f"Invalid SFT row keys: {row.keys()}")
    for row in preference_rows + rm_eval_rows:
        if sorted(row) != ["chosen", "prompt", "rejected"]:
            raise ValueError(f"Invalid preference row keys: {row.keys()}")
        if row["chosen"] == row["rejected"]:
            raise ValueError("Preference row has identical chosen and rejected responses")
    for row in ppo_rows + gen_eval_rows:
        if sorted(row) != ["prompt"]:
            raise ValueError(f"Invalid prompt-only row keys: {row.keys()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare synthetic SFT, preference, PPO, and eval data from a CKAN dump."
    )
    parser.add_argument("--dump", default=DEFAULT_DUMP_PATH, help="Input CKAN JSON dump path.")
    parser.add_argument(
        "--train-count",
        type=int,
        default=600,
        help="Number of packages to use for SFT and preference rows. Defaults to 600.",
    )
    parser.add_argument(
        "--ppo-count",
        type=int,
        default=200,
        help="Number of held-out packages to use for PPO prompts. Defaults to 200.",
    )
    parser.add_argument(
        "--rm-eval-count",
        type=int,
        default=100,
        help="Number of held-out packages to use for RM eval preference rows. Defaults to 100.",
    )
    parser.add_argument(
        "--gen-eval-count",
        type=int,
        default=100,
        help="Number of held-out packages to use for gen eval prompts. Defaults to 100.",
    )
    parser.add_argument(
        "--write-defaults",
        action="store_true",
        help="Write data/sft_data.jsonl, data/preference_data.jsonl, and data/ppo_prompts.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packages = load_packages(args.dump)
    sft_rows, preference_rows, ppo_rows, rm_eval_rows, gen_eval_rows = build_rows(
        packages=packages,
        train_count=args.train_count,
        ppo_count=args.ppo_count,
        rm_eval_count=args.rm_eval_count,
        gen_eval_count=args.gen_eval_count,
    )
    validate_rows(sft_rows, preference_rows, ppo_rows, rm_eval_rows, gen_eval_rows)

    if args.write_defaults:
        sft_path = PIPELINE_SFT_PATH
        preference_path = PIPELINE_PREFERENCE_PATH
        ppo_path = PIPELINE_PPO_PATH
    else:
        sft_path = DEFAULT_SFT_PATH
        preference_path = DEFAULT_PREFERENCE_PATH
        ppo_path = DEFAULT_PPO_PATH

    rm_eval_path = DEFAULT_RM_EVAL_PATH
    gen_eval_path = DEFAULT_GEN_EVAL_PATH

    write_jsonl(sft_path, sft_rows)
    write_jsonl(preference_path, preference_rows)
    write_jsonl(ppo_path, ppo_rows)
    write_jsonl(rm_eval_path, rm_eval_rows)
    write_jsonl(gen_eval_path, gen_eval_rows)

    print(f"Read {len(packages)} CKAN packages from {args.dump}")
    print(f"Wrote {len(sft_rows)} SFT rows to {sft_path}")
    print(f"Wrote {len(preference_rows)} preference rows to {preference_path}")
    print(f"Wrote {len(ppo_rows)} PPO prompts to {ppo_path}")
    print(f"Wrote {len(rm_eval_rows)} RM eval preference rows to {rm_eval_path}")
    print(f"Wrote {len(gen_eval_rows)} gen eval prompts to {gen_eval_path}")


if __name__ == "__main__":
    main()
