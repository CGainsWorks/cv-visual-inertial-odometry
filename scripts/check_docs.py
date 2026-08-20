#!/usr/bin/env python
"""Do the documents still agree with the results they quote?

Three checks, all of which fail the build rather than warn:

1. **Tables.**  Every generated table in the README and ``docs/results.md`` is
   re-rendered from ``results/`` and compared.  A stale table is a document
   claiming a number the code no longer produces.

2. **Loose numbers.**  Figures written in prose, a sentence saying "0.122 m
   on V1_01", are matched against the result files by a small registry of
   claims below.  Prose drifts more easily than tables and is harder to spot.

3. **Links.**  Every relative link in the documents points at a file that
   exists.

This is what ``make docs-check`` runs.  It is deliberately not part of the CI
job, because CI has no dataset and therefore no results to check against; it is
run on a machine that has just produced the results, before committing them.

    python scripts/check_docs.py --results results
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from make_tables import DOCUMENTS, apply, render  # noqa: E402

#: Claims made in prose, as ``(label, document, regex, extractor, tolerance)``.
#: The regex must have one capturing group holding the number.  The extractor
#: pulls the corresponding value out of the loaded results.
PROSE_CLAIMS = [
    (
        "headline VIO ATE",
        "README.md",
        r"mean absolute trajectory error of \*\*([0-9.]+) m\*\*",
        lambda runs, res: _mean(runs, "vio", "ate"),
        0.0006,
    ),
    (
        "headline VIO scale error",
        "README.md",
        r"scale to within \*\*([0-9.]+) per cent\*\*",
        lambda runs, res: 100.0 * _mean(runs, "vio", "scale_error"),
        0.06,
    ),
    (
        "headline runtime",
        "README.md",
        r"about \*\*([0-9.]+) ms per frame\*\*",
        lambda runs, res: _mean(runs, "vio", "ms"),
        1.0,
    ),
]


def _mean(runs, mode, field):
    import numpy as np

    values = []
    for name in runs:
        record = runs[name].get(mode)
        if record is None or not record.get("initialised", True):
            continue
        metrics = record["metrics"]
        alignment = "ate_se3" if mode == "vio" else "ate_sim3"
        if field == "ate":
            values.append(metrics[alignment]["ate_translation_m"]["rmse"])
        elif field == "scale_error":
            values.append(abs(metrics["ate_sim3"]["alignment_scale"] - 1.0))
        elif field == "ms":
            values.append(record["timing"]["ms_per_frame"])
    return float(np.mean(values)) if values else float("nan")


def check_links(document: Path) -> list:
    if not document.exists():
        return []
    text = document.read_text(encoding="utf-8")
    broken = []
    for match in re.finditer(r"\[[^\]]*\]\(([^)#]+)(?:#[^)]*)?\)", text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (document.parent / target).resolve()
        if not resolved.exists():
            broken.append(target)
    return broken


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--skip-prose", action="store_true")
    args = ap.parse_args()

    results = Path(args.results)
    problems = []

    #, 1. tables -------------------------------------------------------
    tables = render(results)
    for document in DOCUMENTS:
        stale = apply(document, tables, write=False)
        for name in stale:
            problems.append(
                f"{document.relative_to(ROOT)}: table '{name}' is stale; "
                f"run `make tables`"
            )

    #, 2. prose --------------------------------------------------------
    if not args.skip_prose:
        from make_tables import _load

        runs = _load(results)
        if runs:
            for label, doc_name, pattern, extractor, tolerance in PROSE_CLAIMS:
                document = ROOT / doc_name
                if not document.exists():
                    continue
                text = document.read_text(encoding="utf-8")
                match = re.search(pattern, text)
                if not match:
                    problems.append(f"{doc_name}: claim '{label}' not found")
                    continue
                stated = float(match.group(1))
                actual = extractor(runs, results)
                if abs(stated - actual) > tolerance:
                    problems.append(
                        f"{doc_name}: '{label}' says {stated:g}, results give "
                        f"{actual:g}"
                    )
        else:
            print(f"no results under {results}; prose checks skipped")

    #, 3. links --------------------------------------------------------
    for document in list(DOCUMENTS) + sorted((ROOT / "docs").glob("*.md")):
        for target in check_links(document):
            problems.append(f"{document.relative_to(ROOT)}: broken link {target}")

    if problems:
        print("documents disagree with the results:")
        for problem in sorted(set(problems)):
            print(f"  {problem}")
        return 1
    print("documents agree with the results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
