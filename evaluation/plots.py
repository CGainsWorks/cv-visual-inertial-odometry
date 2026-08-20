#!/usr/bin/env python
"""Every figure the README and docs/results.md show.

Nothing here computes a metric.  It reads the trajectory files and the result
records that ``scripts/run_sequence.py`` wrote and draws them, so a figure and
the number quoted next to it cannot drift apart.

    python evaluation/plots.py --results results --out docs/plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vio.evaluation.metrics import align_trajectory, associate, ate  # noqa: E402
from vio.io.trajectory import read_tum  # noqa: E402

VIO_COLOUR = "#1f6feb"
VO_COLOUR = "#d97706"
GT_COLOUR = "#4b5563"
LOOP_COLOUR = "#16a34a"


def _style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def _aligned(est_path: Path, gt_path: Path, mode: str):
    est_t, est_p = read_tum(est_path)
    gt_t, gt_p = read_tum(gt_path)
    alignment = "se3" if mode == "vio" else "sim3"
    ei, ri = associate(est_t, gt_t, 0.02)
    if len(ei) < 3:
        return None
    e = [est_p[i] for i in ei]
    g = [gt_p[i] for i in ri]
    aligned, _, _, scale = align_trajectory(e, g, alignment)
    times = np.asarray(est_t)[ei]
    errors = np.linalg.norm(
        np.stack([a.t for a in aligned]) - np.stack([p.t for p in g]), axis=1
    )
    return times, aligned, g, errors, scale, alignment


def plot_sequence(name: str, results: Path, out: Path) -> bool:
    """Top-down trajectory and error over time, for one sequence."""
    gt_path = results / f"{name}_gt.tum"
    if not gt_path.exists():
        return False
    panels = {}
    for mode in ("vio", "vo"):
        path = results / f"{name}_{mode}.tum"
        if path.exists():
            got = _aligned(path, gt_path, mode)
            if got:
                panels[mode] = got
    if not panels:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    reference = next(iter(panels.values()))[2]
    ref = np.stack([p.t for p in reference])
    ax.plot(ref[:, 0], ref[:, 1], color=GT_COLOUR, lw=2.2, label="ground truth",
            zorder=1)
    for mode, colour in (("vio", VIO_COLOUR), ("vo", VO_COLOUR)):
        if mode not in panels:
            continue
        _, aligned, _, _, scale, alignment = panels[mode]
        p = np.stack([a.t for a in aligned])
        label = ("visual-inertial (rigid alignment)" if mode == "vio"
                 else f"visual only (similarity, scale {scale:.2f})")
        ax.plot(p[:, 0], p[:, 1], color=colour, lw=1.1, label=label, zorder=2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{name}: top-down trajectory")
    ax.legend(loc="best", fontsize=7.5)

    ax = axes[1]
    for mode, colour in (("vio", VIO_COLOUR), ("vo", VO_COLOUR)):
        if mode not in panels:
            continue
        times, _, _, errors, _, alignment = panels[mode]
        ax.plot(times - times[0], errors, color=colour, lw=1.0,
                label=f"{mode} ({alignment}), rmse {np.sqrt((errors ** 2).mean()):.3f} m")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("absolute position error (m)")
    ax.set_title("error against ground truth")
    ax.legend(loc="best", fontsize=7.5)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    return True


def plot_summary(results: Path, out: Path) -> bool:
    """ATE across every sequence, both configurations, side by side."""
    rows = []
    for path in sorted(results.glob("*_vio.json")):
        name = path.stem[: -len("_vio")]
        record = json.loads(path.read_text(encoding="utf-8"))
        entry = {
            "sequence": name,
            "vio": record["metrics"]["ate_se3"]["ate_translation_m"]["rmse"],
            "vio_scale": abs(record["metrics"]["ate_sim3"]["alignment_scale"] - 1.0),
        }
        vo_path = results / f"{name}_vo.json"
        if vo_path.exists():
            vo = json.loads(vo_path.read_text(encoding="utf-8"))
            entry["vo"] = vo["metrics"]["ate_sim3"]["ate_translation_m"]["rmse"]
            entry["vo_scale"] = abs(
                vo["metrics"]["ate_sim3"]["alignment_scale"] - 1.0
            )
        rows.append(entry)
    if not rows:
        return False
    rows.sort(key=lambda r: r["sequence"])

    names = [r["sequence"].replace("_", " ") for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax = axes[0]
    ax.bar(x - 0.2, [r["vio"] for r in rows], 0.4, color=VIO_COLOUR,
           label="visual-inertial, rigid alignment")
    ax.bar(x + 0.2, [r.get("vo", np.nan) for r in rows], 0.4, color=VO_COLOUR,
           label="visual only, similarity alignment")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("ATE rmse (m)")
    ax.set_title("Absolute trajectory error")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.bar(x - 0.2, [100 * r["vio_scale"] for r in rows], 0.4, color=VIO_COLOUR,
           label="visual-inertial")
    ax.bar(x + 0.2, [100 * r.get("vo_scale", np.nan) for r in rows], 0.4,
           color=VO_COLOUR, label="visual only")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("scale error (%)")
    ax.set_yscale("log")
    ax.set_title("Scale error: what the IMU buys")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "summary.png", bbox_inches="tight")
    plt.close(fig)
    return True


def plot_timing(results: Path, out: Path) -> bool:
    """Where the milliseconds go, per stage, averaged over sequences."""
    stages = [
        ("ms_track_per_frame", "feature tracking", "#1f6feb"),
        ("ms_preintegrate_per_frame", "preintegration", "#7c3aed"),
        ("ms_optimise_per_keyframe", "window solve", "#d97706"),
        ("ms_triangulate_per_keyframe", "triangulation", "#16a34a"),
        ("ms_marginalise_per_keyframe", "marginalisation", "#dc2626"),
    ]
    rows = []
    for path in sorted(results.glob("*_vio.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        timing = record["timing"]
        keyframe_fraction = timing["keyframes"] / max(timing["frames"], 1)
        entry = {"sequence": record["sequence"]}
        for key, _, _ in stages:
            value = timing.get(key, 0.0)
            if key.endswith("per_keyframe"):
                value *= keyframe_fraction
            entry[key] = value
        rows.append(entry)
    if not rows:
        return False

    names = [r["sequence"].replace("_", " ") for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    bottom = np.zeros(len(rows))
    for key, label, colour in stages:
        values = np.array([r[key] for r in rows])
        ax.bar(x, values, 0.6, bottom=bottom, label=label, color=colour)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("milliseconds per frame")
    ax.set_title("Where the time goes (visual-inertial, single core)")
    ax.axhline(50.0, color="black", ls="--", lw=0.9)
    ax.text(len(rows) - 0.4, 52.0, "20 Hz budget", fontsize=7.5, ha="right")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "timing.png", bbox_inches="tight")
    plt.close(fig)
    return True


def plot_loop(results: Path, out: Path) -> bool:
    """Trajectory before and after the pose graph, for every loop run."""
    made = False
    for path in sorted(results.glob("*_loop.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        name = record["sequence"]
        mode = record["mode"]
        gt_path = results / f"{name}_gt.tum"
        before_path = results / f"{name}_{mode}.tum"
        after_path = results / f"{name}_{mode}_loop.tum"
        if not (gt_path.exists() and before_path.exists() and after_path.exists()):
            continue
        gt_t, gt_p = read_tum(gt_path)
        alignment = "se3" if mode == "vio" else "sim3"

        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ref = np.stack([p.t for p in gt_p])
        ax.plot(ref[:, 0], ref[:, 1], color=GT_COLOUR, lw=2.2, label="ground truth")
        for traj, colour, label in (
            (before_path, VIO_COLOUR, "before loop closure"),
            (after_path, LOOP_COLOUR, "after loop closure"),
        ):
            est_t, est_p = read_tum(traj)
            ei, ri = associate(est_t, gt_t, 0.02)
            if len(ei) < 3:
                continue
            aligned, _, _, _ = align_trajectory(
                [est_p[i] for i in ei], [gt_p[i] for i in ri], alignment
            )
            p = np.stack([a.t for a in aligned])
            rmse = ate(est_t, est_p, gt_t, gt_p, mode=alignment).translation.rmse
            ax.plot(p[:, 0], p[:, 1], color=colour, lw=1.1,
                    label=f"{label} ({rmse:.3f} m)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{name}: {record['closures_emitted']} closures, "
                     f"precision {record['precision']:.2f}")
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}_loop.png", bbox_inches="tight")
        plt.close(fig)
        made = True
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "plots"))
    args = ap.parse_args()
    _style()

    results = Path(args.results)
    out = Path(args.out)
    names = sorted({p.stem[: -len("_gt")] for p in results.glob("*_gt.tum")})

    made = 0
    for name in names:
        made += plot_sequence(name, results, out)
    made += plot_summary(results, out)
    made += plot_timing(results, out)
    made += plot_loop(results, out)

    if not made:
        print(f"nothing to plot under {results}")
        return 1
    print(f"wrote {made} figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
