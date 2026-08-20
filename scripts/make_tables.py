#!/usr/bin/env python
"""Regenerate every table in the README and docs from the result files.

Tables live between markers::

    <!-- table:euroc -->
    ...generated...
    <!-- /table:euroc -->

so a document is edited by prose and a table is edited by re-running the
experiment.  ``scripts/check_docs.py`` runs the same generators and fails if a
document has drifted from the results it quotes, which is how the claim that
every number is reproducible stays true rather than becoming aspirational.

    python scripts/make_tables.py --results results
    python scripts/make_tables.py --check          # do not write, just report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DOCUMENTS = [ROOT / "README.md", ROOT / "docs" / "results.md"]


def _load(results: Path) -> dict:
    runs = {}
    for path in sorted(results.glob("*_v*.json")):
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in ("vio", "vo"):
            continue
        runs.setdefault(parts[0], {})[parts[1]] = json.loads(
            path.read_text(encoding="utf-8")
        )
    return runs


def _rows(runs: dict):
    out = []
    for name in sorted(runs):
        entry = {"sequence": name}
        for mode in ("vio", "vo"):
            record = runs[name].get(mode)
            if record is None:
                continue
            metrics = record["metrics"]
            alignment = "ate_se3" if mode == "vio" else "ate_sim3"
            entry[mode] = {
                "ate": metrics[alignment]["ate_translation_m"]["rmse"],
                "ate_median": metrics[alignment]["ate_translation_m"]["median"],
                "rot": metrics[alignment]["ate_rotation_deg"]["rmse"],
                "scale": metrics["ate_sim3"]["alignment_scale"],
                "rpe1": metrics[f"rpe_{alignment[4:]}_1m"]["rpe_translation_m"]["rmse"],
                "drift": 100.0 * metrics[f"drift_{alignment[4:]}"]["final_per_metre"],
                "distance": metrics[f"drift_{alignment[4:]}"]["distance_m"],
                "poses": record["frames_with_pose"],
                "frames": record["frames_processed"],
                "ms": record["timing"]["ms_per_frame"],
                "initialised": record["initialised"],
            }
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Generators.  Each returns the markdown that goes between its markers.
# ---------------------------------------------------------------------------


def table_euroc(runs: dict) -> str:
    rows = _rows(runs)
    if not rows:
        return "_No results yet.  Run `make experiments`._"
    lines = [
        "| sequence | length | VIO ATE (m) | VIO scale err | VIO rot (deg) "
        "| VO ATE, sim3 (m) | VO scale |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        vio = r.get("vio")
        vo = r.get("vo")
        length = (vio or vo or {}).get("distance", float("nan"))
        lines.append(
            f"| `{r['sequence']}` | {length:.0f} m | "
            + (f"**{vio['ate']:.3f}** | {100 * abs(vio['scale'] - 1):.1f}% | "
               f"{vio['rot']:.2f} | " if vio else "n/a | n/a | n/a | ")
            + (f"{vo['ate']:.3f} | {vo['scale']:.2f}x |" if vo else "n/a | n/a |")
        )
    if any("vio" in r for r in rows):
        vios = [r["vio"] for r in rows if "vio" in r]
        lines.append(
            f"| **mean** | | **{np.mean([v['ate'] for v in vios]):.3f}** | "
            f"{np.mean([100 * abs(v['scale'] - 1) for v in vios]):.1f}% | "
            f"{np.mean([v['rot'] for v in vios]):.2f} | "
            + (f"{np.mean([r['vo']['ate'] for r in rows if 'vo' in r]):.3f} | |"
               if any("vo" in r for r in rows) else "n/a | |")
        )
    return "\n".join(lines)


def table_drift(runs: dict) -> str:
    rows = _rows(runs)
    if not rows:
        return "_No results yet.  Run `make experiments`._"
    lines = [
        "| sequence | distance | VIO RPE over 1 m | VIO drift | VO RPE over 1 m "
        "| VO drift |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        vio, vo = r.get("vio"), r.get("vo")
        length = (vio or vo or {}).get("distance", float("nan"))
        lines.append(
            f"| `{r['sequence']}` | {length:.0f} m | "
            + (f"{vio['rpe1']:.4f} m | {vio['drift']:.2f} %/m | " if vio else "n/a | n/a | ")
            + (f"{vo['rpe1']:.4f} m | {vo['drift']:.2f} %/m |" if vo else "n/a | n/a |")
        )
    return "\n".join(lines)


def table_timing(runs: dict) -> str:
    rows = []
    for name in sorted(runs):
        record = runs[name].get("vio")
        if record is None:
            continue
        t = record["timing"]
        share = t["keyframes"] / max(t["frames"], 1)
        rows.append({
            "sequence": name,
            "track": t["ms_track_per_frame"],
            "pre": t["ms_preintegrate_per_frame"],
            "tri": t["ms_triangulate_per_keyframe"] * share,
            "solve": t["ms_optimise_per_keyframe"] * share,
            "marg": t["ms_marginalise_per_keyframe"] * share,
            "total": t["ms_per_frame"],
            "hz": t["effective_hz"],
            "kf": 100.0 * share,
        })
    if not rows:
        return "_No results yet.  Run `make experiments`._"
    lines = [
        "| sequence | tracking | preintegration | triangulation | window solve "
        "| marginalisation | total | effective |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['sequence']}` | {r['track']:.1f} | {r['pre']:.1f} | "
            f"{r['tri']:.1f} | {r['solve']:.1f} | {r['marg']:.1f} | "
            f"**{r['total']:.0f} ms** | {r['hz']:.1f} Hz |"
        )
    lines.append(
        f"| **mean** | {np.mean([r['track'] for r in rows]):.1f} | "
        f"{np.mean([r['pre'] for r in rows]):.1f} | "
        f"{np.mean([r['tri'] for r in rows]):.1f} | "
        f"{np.mean([r['solve'] for r in rows]):.1f} | "
        f"{np.mean([r['marg'] for r in rows]):.1f} | "
        f"**{np.mean([r['total'] for r in rows]):.0f} ms** | "
        f"{np.mean([r['hz'] for r in rows]):.1f} Hz |"
    )
    lines.append("")
    lines.append("All figures are milliseconds per *frame* on one core; the "
                 "per-keyframe stages are scaled by the keyframe rate.")
    return "\n".join(lines)


def table_ablation_integration(runs: dict, ablations: dict) -> str:
    data = ablations.get("integration")
    if not data or "rows" not in data:
        return "_No ablation yet.  Run `make ablate`._"
    lines = [
        "| block length | Euler position | midpoint position | Euler rotation "
        "| midpoint rotation |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    spans = sorted({r["span_s"] for r in data["rows"]})
    for span in spans:
        e = next(r for r in data["rows"] if r["scheme"] == "euler" and r["span_s"] == span)
        m = next(r for r in data["rows"] if r["scheme"] == "midpoint" and r["span_s"] == span)
        lines.append(
            f"| {span:.2f} s | {e['position_m']:.2e} m | {m['position_m']:.2e} m | "
            f"{e['rotation_rad']:.2e} rad | {m['rotation_rad']:.2e} rad |"
        )
    return "\n".join(lines)


def table_ablation_frontend(runs: dict, ablations: dict) -> str:
    data = ablations.get("frontend")
    if not data or "rows" not in data:
        return "_No ablation yet.  Run `make ablate`._"
    lines = [
        "| stride | effective rate | tracked, no gyro | tracked, with gyro | gain "
        "| guess error, no gyro | with gyro |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in data["rows"]:
        lines.append(
            f"| {r['stride']} | {r['fps']:.1f} Hz | {r['tracked_no_gyro']:.1f} | "
            f"{r['tracked_with_gyro']:.1f} | {r['gain_percent']:+.1f}% | "
            f"{r['guess_error_no_gyro_px']:.2f} px | "
            f"{r['guess_error_with_gyro_px']:.2f} px |"
        )
    return "\n".join(lines)


def table_ablation_backend(runs: dict, ablations: dict) -> str:
    data = ablations.get("backend")
    if not data:
        return "_No ablation yet.  Run `make ablate`._"
    lines = []
    group = data.get("group")
    if group:
        lines += [
            f"Vectorised reprojection group, {group['observations']} observations "
            "in the window:",
            "",
            "| path | per linearisation | agreement |",
            "| --- | ---: | ---: |",
            f"| scalar factors | {group['ms_scalar']:.1f} ms | reference |",
            f"| vectorised group | **{group['ms_group']:.1f} ms** "
            f"({group['speedup']:.0f}x) | "
            f"{group['max_jacobian_difference']:.0e} in the Jacobian |",
            "",
        ]
    rows = data.get("marginalisation")
    if rows:
        lines += [
            "Marginalising the departing keyframe's inertial states, or not:",
            "",
            "| marginalise | ATE (m) | rotation (deg) | scale error | per keyframe |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for r in rows:
            lines.append(
                f"| {'yes' if r['marginalise'] else 'no'} | {r['ate_rmse_m']:.4f} | "
                f"{r['rotation_rmse_deg']:.2f} | "
                f"{r['scale_error_percent']:.2f}% | {r['ms_per_keyframe']:.0f} ms |"
            )
    return "\n".join(lines)


def table_ablation_window(runs: dict, ablations: dict) -> str:
    data = ablations.get("window")
    if not data or "rows" not in data:
        return "_No ablation yet.  Run `make ablate`._"
    lines = [
        "| window | ATE (m) | scale error | per frame |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in data["rows"]:
        lines.append(
            f"| {r['window']} keyframes | {r['ate_rmse_m']:.4f} | "
            f"{r['scale_error_percent']:.2f}% | {r['ms_per_frame']:.0f} ms |"
        )
    return "\n".join(lines)


def table_loop(runs: dict, ablations: dict, results: Path) -> str:
    rows = []
    for path in sorted(results.glob("*_loop.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        return "_No loop-closure runs yet.  Run `make loop`._"
    lines = [
        "| sequence | revisits | closures | correct | precision | recall "
        "| ATE before | ATE after |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        before = r["ate_before"]["ate_translation_m"]["rmse"]
        after = (r["ate_after"]["ate_translation_m"]["rmse"]
                 if r.get("ate_after") else None)
        lines.append(
            f"| `{r['sequence']}` | {r['revisit_opportunities']} | "
            f"{r['closures_emitted']} | {r['closures_correct']} | "
            f"{r['precision']:.2f} | {r['recall']:.2f} | {before:.3f} m | "
            + (f"**{after:.3f} m** |" if after is not None else "n/a |")
        )
    return "\n".join(lines)


def table_calibration(runs: dict, ablations: dict, results: Path) -> str:
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(results.glob("*_calibration.json"))]
    if not rows:
        return "_No calibration runs yet.  Run `make calibrate`._"
    lines = [
        "| sequence | rotation error | Procrustes residual | offset injected "
        "| offset recovered | error | correlation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        t = r["temporal"]
        lines.append(
            f"| `{r['sequence']}` | {r['rotation']['error_deg']:.3f} deg | "
            f"{r['rotation']['procrustes_residual_rad']:.1e} rad | "
            f"{1e3 * t['injected_offset_s']:+.1f} ms | "
            f"{1e3 * t['estimated_offset_s']:+.1f} ms | "
            f"{1e3 * t['error_s']:+.2f} ms | {t['correlation']:.3f} |"
        )
    return "\n".join(lines)


GENERATORS = {
    "euroc": lambda runs, abl, res: table_euroc(runs),
    "drift": lambda runs, abl, res: table_drift(runs),
    "timing": lambda runs, abl, res: table_timing(runs),
    "integration": lambda runs, abl, res: table_ablation_integration(runs, abl),
    "frontend": lambda runs, abl, res: table_ablation_frontend(runs, abl),
    "backend": lambda runs, abl, res: table_ablation_backend(runs, abl),
    "window": lambda runs, abl, res: table_ablation_window(runs, abl),
    "loop": table_loop,
    "calibration": table_calibration,
}


def render(results: Path) -> dict:
    runs = _load(results)
    ablation_path = results / "ablations.json"
    ablations = (
        json.loads(ablation_path.read_text(encoding="utf-8"))
        if ablation_path.exists() else {}
    )
    return {name: fn(runs, ablations, results) for name, fn in GENERATORS.items()}


def apply(document: Path, tables: dict, write: bool):
    """Replace each marked block.  Returns the names of blocks that changed."""
    if not document.exists():
        return []
    text = document.read_text(encoding="utf-8")
    changed = []
    for name, body in tables.items():
        pattern = re.compile(
            rf"(<!-- table:{name} -->\n)(.*?)(\n<!-- /table:{name} -->)",
            re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            continue
        if match.group(2).strip() != body.strip():
            changed.append(name)
            text = pattern.sub(lambda m: m.group(1) + body + m.group(3), text, count=1)
    if write and changed:
        document.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--check", action="store_true",
                    help="report drift without writing, and exit non-zero")
    args = ap.parse_args()

    tables = render(Path(args.results))
    total = 0
    for document in DOCUMENTS:
        changed = apply(document, tables, write=not args.check)
        total += len(changed)
        if changed:
            verb = "would update" if args.check else "updated"
            print(f"{verb} {document.relative_to(ROOT)}: {', '.join(changed)}")
    if not total:
        print("all tables already agree with the results")
    return 1 if (args.check and total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
