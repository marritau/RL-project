"""Robustness diagnostic for TVS similarity matrices.

The TVS paper reports that small negative eigenvalues can be clamped to zero and
checks this by adding diagonal jitter to the kernel matrix.  This script mirrors
that appendix-style check for any saved `similarity_matrix.npy`.

Example:
    python scripts/analyze_tvs_robustness.py \
      --matrix results/tvs_all/vi/similarity_matrix.npy \
      --output-dir results/tvs_all/vi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from pacman_rldp.diversity import temporal_vendi_score_from_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TVS robustness under diagonal jitter.")
    parser.add_argument("--matrix", required=True, help="Path to similarity_matrix.npy")
    parser.add_argument("--output-dir", default=None, help="Where to save robustness JSON/PNG")
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        help="Diagonal jitter values, matching the paper by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix)
    output_dir = Path(args.output_dir) if args.output_dir else matrix_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.load(matrix_path)
    baseline_tvs, baseline_eigs = temporal_vendi_score_from_similarity(matrix)
    rows = []
    for eps in args.epsilons:
        jittered = matrix + float(eps) * np.eye(matrix.shape[0], dtype=matrix.dtype)
        tvs, eigs = temporal_vendi_score_from_similarity(jittered)
        rel_change_pct = 0.0 if baseline_tvs == 0 else 100.0 * (tvs - baseline_tvs) / baseline_tvs
        rows.append({
            "epsilon": float(eps),
            "tvs": float(tvs),
            "relative_change_percent": float(rel_change_pct),
            "min_eigenvalue_after_normalization": float(np.min(eigs)) if len(eigs) else 0.0,
        })

    payload = {
        "matrix_path": str(matrix_path),
        "baseline_tvs": float(baseline_tvs),
        "baseline_min_eigenvalue_after_clamp": float(np.min(baseline_eigs)) if len(baseline_eigs) else 0.0,
        "rows": rows,
    }
    with (output_dir / "tvs_jitter_robustness.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [str(row["epsilon"]) for row in rows]
    ys = [row["relative_change_percent"] for row in rows]
    ax.bar(xs, ys)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("TVS robustness to diagonal jitter")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Relative TVS change (%)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "tvs_jitter_robustness.png", dpi=160)
    plt.close(fig)

    print(f"Baseline TVS: {baseline_tvs:.6f}")
    print(f"Saved: {output_dir / 'tvs_jitter_robustness.json'}")
    print(f"Saved: {output_dir / 'tvs_jitter_robustness.png'}")


if __name__ == "__main__":
    main()
