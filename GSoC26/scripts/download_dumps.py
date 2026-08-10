#!/usr/bin/env python3
"""Download DBpedia dumps (labels, redirects, disambiguations — English) from the
DBpedia Databus into ./dumps/.

URLs are resolved live against the Databus SPARQL endpoint, so this always
fetches the latest published version of each artifact. Partial downloads are
resumed via HTTP Range requests.
"""
import argparse
import bz2
import gzip
import os
import sys

import requests

DATABUS_SPARQL = "https://databus.dbpedia.org/sparql"
ARTIFACTS = ["labels", "redirects", "disambiguations"]
DUMPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps")

SPARQL_TEMPLATE = """
PREFIX dcat:    <http://www.w3.org/ns/dcat#>
PREFIX dct:     <http://purl.org/dc/terms/>
PREFIX databus: <https://dataid.dbpedia.org/databus#>
SELECT ?url ?version WHERE {{
  ?dataset databus:artifact <https://databus.dbpedia.org/dbpedia/generic/{artifact}> ;
           dct:hasVersion ?version ;
           dcat:distribution ?dist .
  ?dist dcat:downloadURL ?url .
  FILTER(REGEX(STR(?url), "lang=en\\\\.ttl\\\\.(bz2|gz)$"))
  {version_filter}
}}
ORDER BY DESC(?version)
LIMIT 1
"""


def resolve_latest_url(artifact: str, version: str | None = None) -> tuple[str, str]:
    """Return (download_url, version) for the newest — or a pinned — English release."""
    version_filter = f'FILTER(STR(?version) = "{version}")' if version else ""
    resp = requests.get(
        DATABUS_SPARQL,
        params={
            "query": SPARQL_TEMPLATE.format(artifact=artifact, version_filter=version_filter),
            "format": "json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    bindings = resp.json()["results"]["bindings"]
    if not bindings:
        raise RuntimeError(f"Databus returned no English .ttl file for artifact {artifact!r}")
    return bindings[0]["url"]["value"], bindings[0]["version"]["value"]


def download(url: str, dest: str) -> None:
    """Download url to dest, resuming from a partial file if one exists."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))

    if total and have == total:
        print(f"  already complete ({have:,} bytes), skipping")
        return
    if have > total > 0:
        print(f"  local file larger than remote ({have:,} > {total:,}), restarting")
        os.remove(dest)
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if have and r.status_code != 206:
            # server ignored the Range request — start over
            print("  server does not support resume, restarting from 0")
            have = 0
            r.raise_for_status()
            mode = "wb"
        else:
            r.raise_for_status()
            mode = "ab" if have else "wb"
        done = have
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {done:,}/{total:,} bytes ({pct:5.1f}%)", end="", flush=True)
        print()


def open_dump(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def count_lines(path: str) -> int:
    n = 0
    with open_dump(path) as f:
        for _ in f:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", nargs="+", default=ARTIFACTS,
                    help="Databus generic-group artifact names (default: %(default)s)")
    ap.add_argument("--version", default=None,
                    help="pin a release version (e.g. 2022.12.01) instead of latest")
    ap.add_argument("--no-count", action="store_true",
                    help="skip the final line count (useful for very large dumps)")
    args = ap.parse_args()

    os.makedirs(DUMPS_DIR, exist_ok=True)
    downloaded = []
    for artifact in args.artifacts:
        url, version = resolve_latest_url(artifact, args.version)
        dest = os.path.join(DUMPS_DIR, url.rsplit("/", 1)[-1])
        print(f"{artifact} (version {version})\n  {url}\n  -> {dest}")
        download(url, dest)
        downloaded.append((artifact, dest))

    print("\n=== summary ===")
    for artifact, dest in downloaded:
        size = os.path.getsize(dest)
        print(f"{artifact}: {dest}")
        print(f"  size: {size:,} bytes ({size / 1e6:.1f} MB)")
        if not args.no_count:
            print("  counting lines...", flush=True)
            print(f"  lines: {count_lines(dest):,}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\ninterrupted — rerun to resume")
