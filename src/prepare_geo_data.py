"""
One-off data-prep script for the geo-metadata experiment.

Fetches real geospatial dataset metadata records from NASA's CMR (Common
Metadata Repository) public search API and turns them into the three JSONL
files consumed by the pipeline: geo_sft_data.jsonl, geo_preference_data.jsonl,
and geo_ppo_prompts.jsonl.

"Good" vs "bad" is defined by how many of a fixed set of optional-in-CMR
metadata fields (spatial extent, platforms, archive center, processing
level, collection type, consortia) a given collection record actually has
filled in -- title/abstract/temporal-start are present on virtually every
CMR record, so they don't discriminate, but these six fields vary a lot in
practice. Within each topic query, the most-complete record becomes the
"chosen"/SFT example and the least-complete becomes "rejected" -- this is a
real, observed completeness gap rather than an invented one.

Not part of the recurring SFT/reward/PPO pipeline -- run this once (or
re-run to refresh the data) before using configs/geo_config.yaml:

    python src/prepare_geo_data.py
"""

import json
import os
import time

import requests

CMR_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Topics used for SFT + preference data (need a completeness spread).
TOPICS = [
    "elevation",
    "soil moisture",
    "land cover",
    "sea surface temperature",
    "glacier",
    "air quality",
    "wetlands",
    "seismic hazard",
    "coral reef",
    "snow cover",
    "coastal erosion",
    "urban heat island",
]

# Disjoint topic set, held out purely as PPO rollout-seed prompts.
PPO_TOPICS = [
    "ocean acidification",
    "permafrost",
    "tropical cyclone",
    "river discharge",
    "aerosol optical depth",
    "vegetation index",
    "lake water level",
    "groundwater",
]

# Fields present on ~every CMR record -- always rendered, don't discriminate.
ALWAYS_FIELDS = ("title", "summary", "time_start")

# Fields CMR only sometimes fills in -- these drive the completeness score
# and are omitted from the rendered record when absent.
OPTIONAL_FIELDS = ("boxes", "platforms", "archive_center", "processing_level_id",
                   "collection_data_type", "consortiums")

PROMPT_TEMPLATE = "Write an ISO-19115-style metadata record for a dataset about {topic}."

# CMR abstracts can run to hundreds of words; capping keeps total record
# length well under the pipeline's 512-token max so the trailing structured
# fields (the actual completeness signal) don't get truncated or cause the
# whole row to be filtered out by RewardTrainer's max_length cutoff.
ABSTRACT_CHAR_LIMIT = 400


def fetch_topic(topic, page_size=8):
    resp = requests.get(CMR_URL, params={"keyword": topic, "page_size": page_size}, timeout=30)
    resp.raise_for_status()
    return resp.json()["feed"]["entry"]


def completeness(entry):
    return sum(1 for f in OPTIONAL_FIELDS if entry.get(f))


def normalize(entry):
    """Render a CMR entry as a labeled, colon-separated text block. Optional
    fields are simply omitted when CMR doesn't have them -- that omission is
    what produces real incompleteness, no fabrication needed."""
    if not entry.get("title") or not entry.get("summary"):
        return None

    abstract = " ".join(entry["summary"].split())  # collapse newlines/whitespace
    if len(abstract) > ABSTRACT_CHAR_LIMIT:
        abstract = abstract[:ABSTRACT_CHAR_LIMIT].rsplit(" ", 1)[0] + "..."

    lines = [
        f"Title: {entry['title']}",
        f"Abstract: {abstract}",
    ]
    if entry.get("boxes"):
        lines.append(f"Spatial Extent (S W N E): {entry['boxes'][0]}")
    if entry.get("time_start"):
        end = entry.get("time_end", "present")
        lines.append(f"Temporal Extent: {entry['time_start']} to {end}")
    if entry.get("platforms"):
        lines.append(f"Platforms: {', '.join(entry['platforms'])}")
    if entry.get("archive_center"):
        lines.append(f"Archive Center: {entry['archive_center']}")
    if entry.get("processing_level_id"):
        lines.append(f"Processing Level: {entry['processing_level_id']}")
    if entry.get("collection_data_type"):
        lines.append(f"Collection Type: {entry['collection_data_type']}")
    if entry.get("consortiums"):
        lines.append(f"Associated Consortia: {', '.join(entry['consortiums'])}")
    return "\n".join(lines)


def build_sft_and_preference_rows():
    sft_rows, pref_rows = [], []
    for topic in TOPICS:
        try:
            entries = fetch_topic(topic)
        except requests.RequestException as e:
            print(f"  skip topic {topic!r}: {e}")
            continue
        time.sleep(0.3)  # be polite to the public API

        if not entries:
            continue

        scored = sorted(entries, key=completeness, reverse=True)
        best, worst = scored[0], scored[-1]
        prompt = PROMPT_TEMPLATE.format(topic=topic)

        best_text = normalize(best)
        if best_text is None:
            continue
        sft_rows.append({"prompt": prompt, "response": best_text})

        if completeness(worst) < completeness(best):
            worst_text = normalize(worst)
            if worst_text is not None:
                pref_rows.append({"prompt": prompt, "chosen": best_text, "rejected": worst_text})
    return sft_rows, pref_rows


def build_ppo_prompt_rows():
    rows = []
    for topic in PPO_TOPICS:
        rows.append({"prompt": PROMPT_TEMPLATE.format(topic=topic)})
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    print("Fetching SFT + preference data from CMR...")
    sft_rows, pref_rows = build_sft_and_preference_rows()
    ppo_rows = build_ppo_prompt_rows()

    write_jsonl(os.path.join(DATA_DIR, "geo_sft_data.jsonl"), sft_rows)
    write_jsonl(os.path.join(DATA_DIR, "geo_preference_data.jsonl"), pref_rows)
    write_jsonl(os.path.join(DATA_DIR, "geo_ppo_prompts.jsonl"), ppo_rows)

    print(f"Wrote {len(sft_rows)} SFT rows, {len(pref_rows)} preference rows, "
          f"{len(ppo_rows)} PPO prompt rows to {DATA_DIR}")


if __name__ == "__main__":
    main()
