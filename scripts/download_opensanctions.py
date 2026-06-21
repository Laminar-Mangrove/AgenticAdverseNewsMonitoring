#!/usr/bin/env python3
"""
Download OpenSanctions bulk data for PEP and sanction list screening.

Uses the pre-filtered `sanctions` and `peps` collections in the lightweight
`targets.simple.csv` format (name + aliases + metadata), which is far smaller
than the full FtM JSON export.

Free for non-commercial use. Commercial use requires a data license:
https://www.opensanctions.org/licensing/
"""
import sys
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "opensanctions"

# collection name -> local filename. The `sanctions` and `peps` collections are
# already topic-filtered, so every row in each file belongs to that category.
COLLECTIONS = {
    "sanctions": "sanction_targets.csv",
    "peps": "pep_targets.csv",
}

BASE_URL = "https://data.opensanctions.org/datasets/latest/{collection}/targets.simple.csv"

# For a lighter/faster demo, swap "sanctions" -> "us_ofac_sdn" (~37k vs ~100k)
# and skip "peps" (peps is ~940k rows). Pass collection names as CLI args to
# override, e.g.:  python scripts/download_opensanctions.py us_ofac_sdn


def _download(url: str, dest: Path) -> bool:
    """Stream a URL to disk in chunks (avoids loading huge files into memory)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "adverse-news-classifier/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r  {dest.name}: {done/1e6:6.1f} / {total/1e6:6.1f} MB ({pct}%)", end="")
                    else:
                        print(f"\r  {dest.name}: {done/1e6:6.1f} MB", end="")
        print()
        tmp.replace(dest)
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def main(collections: dict[str, str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading OpenSanctions bulk data (targets.simple.csv)...")
    print(f"Output: {OUTPUT_DIR}\n")

    ok = 0
    for collection, filename in collections.items():
        url = BASE_URL.format(collection=collection)
        print(f"[{collection}] {url}")
        if _download(url, OUTPUT_DIR / filename):
            ok += 1

    print(f"\nDone. {ok}/{len(collections)} files downloaded.")
    if ok == 0:
        print("Manual download: https://www.opensanctions.org/datasets/")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        # Override: treat each arg as a collection/dataset name
        selected = {name: f"{name}_targets.csv" for name in args}
    else:
        selected = COLLECTIONS
    main(selected)
