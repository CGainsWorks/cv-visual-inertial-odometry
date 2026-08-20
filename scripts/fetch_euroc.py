#!/usr/bin/env python
"""Download and extract the EuRoC MAV dataset.

The per-sequence downloads at ``robotics.ethz.ch`` that every VIO tutorial
links to are gone: ETH moved the dataset to their Research Collection in late
2025, and it is now served as two large bundles rather than eleven small ones.
This script resolves the current bitstream URLs through the repository's own
API instead of hard-coding them, so it keeps working if the identifiers change
again, and it extracts only the ASL-format archives, the ROS bags are roughly
half the bytes and nothing in this repository reads them.

    python scripts/fetch_euroc.py --out ~/datasets/euroc

About 19 GB of download and 19 GB on disk after extraction, for Machine Hall
and Vicon Room 1.  Resumable: run it again and it continues rather than
starting over.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

#: The Research Collection item.  Resolving the handle rather than hard-coding
#: bitstream ids means a re-upload does not silently break this.
HANDLE = "20.500.11850/690084"
API = "https://www.research-collection.ethz.ch/server/api"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

LICENCE = """EuRoC MAV dataset, ETH Zurich Autonomous Systems Lab.

Burri, Nikolic, Gohl, Schneider, Rehder, Omari, Achtelik and Siegwart,
"The EuRoC micro aerial vehicle datasets", International Journal of Robotics
Research, 2016.  doi:10.1177/0278364915620033

Distributed by the ETH Research Collection under Creative Commons Attribution
3.0 Unported (CC BY 3.0).  Commercial use is permitted with attribution.
https://doi.org/10.3929/ethz-b-000690084
"""


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def bitstreams() -> dict:
    """Map bundle filename to ``(url, size)``, via the DSpace API."""
    item = json.loads(_get(f"{API}/pid/find?id=hdl:{HANDLE}"))
    bundles = json.loads(_get(f"{API}/core/items/{item['uuid']}/bundles"))
    original = next(
        b for b in bundles["_embedded"]["bundles"] if b["name"] == "ORIGINAL"
    )
    listing = json.loads(
        _get(f"{API}/core/bundles/{original['uuid']}/bitstreams?size=200")
    )
    return {
        b["name"]: (b["_links"]["content"]["href"], int(b["sizeBytes"]))
        for b in listing["_embedded"]["bitstreams"]
    }


def download(url: str, path: Path, size: int) -> None:
    """Resumable download, using a byte-range request to continue."""
    have = path.stat().st_size if path.exists() else 0
    if have >= size:
        print(f"  {path.name}: already complete")
        return
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"  {path.name}: resuming at {have / 1e9:.1f} GB")
    request = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(request, timeout=120) as response:
        with open(path, "ab" if have else "wb") as fh:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                have += len(chunk)
                if have % (256 << 20) < (1 << 20):
                    rate = have / max(time.time() - t0, 1e-9) / 1e6
                    print(f"    {have / 1e9:5.1f} / {size / 1e9:.1f} GB "
                          f"({rate:.0f} MB/s)", flush=True)


def extract(bundle: Path, out: Path) -> None:
    """Pull the inner ASL archives out, skipping the ROS bags."""
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as outer:
        inner = [
            n for n in outer.namelist()
            if n.endswith(".zip") and not Path(n).name.startswith("._")
        ]
        for name in sorted(inner):
            sequence = Path(name).stem
            destination = out / sequence
            if (destination / "mav0" / "cam0" / "data.csv").exists():
                print(f"  {sequence}: already extracted")
                continue
            t0 = time.time()
            with outer.open(name) as fh:
                blob = io.BytesIO(fh.read())
            with zipfile.ZipFile(blob) as archive:
                archive.extractall(destination)
            n = sum(1 for _ in (destination / "mav0" / "cam0" / "data").glob("*.png"))
            print(f"  {sequence}: {n} images in {time.time() - t0:.0f} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "datasets" / "euroc"))
    ap.add_argument("--bundles", nargs="*",
                    default=["machine_hall.zip", "vicon_room1.zip"])
    ap.add_argument("--keep-archives", action="store_true",
                    help="do not delete the bundles after extracting")
    args = ap.parse_args()

    root = Path(args.out)
    archives = root / "archives"
    sequences = root / "sequences"
    archives.mkdir(parents=True, exist_ok=True)
    (root / "LICENCE.txt").write_text(LICENCE, encoding="utf-8")

    print("resolving download URLs from the ETH Research Collection")
    available = bitstreams()
    for name in args.bundles:
        if name not in available:
            print(f"  {name} not found; available: {', '.join(sorted(available))}")
            return 2

    for name in args.bundles:
        url, size = available[name]
        print(f"{name} ({size / 1e9:.1f} GB)")
        download(url, archives / name, size)

    for name in args.bundles:
        print(f"extracting {name}")
        extract(archives / name, sequences)
        if not args.keep_archives:
            (archives / name).unlink(missing_ok=True)

    found = sorted(
        d.name for d in sequences.iterdir()
        if (d / "mav0" / "cam0" / "data.csv").exists()
    )
    print(f"\n{len(found)} sequences under {sequences}:")
    for name in found:
        print(f"  {name}")
    print(f"\nlicence recorded in {root / 'LICENCE.txt'}")
    print(f"set EUROC_ROOT={sequences} or pass --data to the scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
