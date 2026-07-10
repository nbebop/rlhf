"""
Fetch a JSON dump from Berlin's CKAN package_search endpoint.

Run from the project root:

    python src/fetch_ckan_dump.py

By default this writes the raw CKAN response for the first 100 packages to:

    data/berlin_ckan_package_search.json
"""

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# daten.berlin.de is the public portal; datenregister.berlin.de is the
# CKAN backend that serves /api/3/action/package_search.
DEFAULT_ENDPOINT = "https://datenregister.berlin.de/api/3/action/package_search"
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "berlin_ckan_package_search.json")


def fetch_ckan_dump(endpoint: str, rows: int) -> dict:
    query = urllib.parse.urlencode({"rows": rows})
    url = f"{endpoint}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "rlhf-ckan-dump/1.0",
        },
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = response.read().decode(charset)

    data = json.loads(payload)
    if not data.get("success"):
        raise RuntimeError(f"CKAN API returned success=false: {data!r}")
    return data


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump Berlin CKAN package_search JSON to a local file."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of package rows to request from CKAN. Defaults to 100.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"CKAN package_search endpoint. Defaults to {DEFAULT_ENDPOINT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        data = fetch_ckan_dump(args.endpoint, args.rows)
    except urllib.error.URLError as e:
        raise SystemExit(f"Failed to fetch CKAN data: {e}") from e

    write_json(args.output, data)

    result = data.get("result", {})
    print(
        f"Wrote {len(result.get('results', []))} package records "
        f"(total available: {result.get('count', 'unknown')}) to {args.output}"
    )


if __name__ == "__main__":
    main()
