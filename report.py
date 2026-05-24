"""
report.py
---------
Generates report figures and tables from raw benchmark results CSVs.
Includes:
- ARPD summary table (Table 6 / Table 16 in paper)
- Wilcoxon p-value matrix (QIG vs others)
- Boxplots of RPD distribution for representative instances
- Convergence curves (optional, requires re-running algorithms on selected instances)

Usage: python report.py --results "results/raw_results_t60_*.csv" --out figures/

Results CSVs:
    figures/
    ├── boxplot_taillard.png
    ├── boxplot_vrf_small.png
    ├── boxplot_vrf_large.png
    ├── convergence_grid.png
    ├── arpd_bar.png
    ├── arpd_table.csv
    └── wilcoxon_pvalues.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import (
    load_results,
    compute_arpd_table,
    arpd_summary,
    wilcoxon_matrix,
    print_summary,
)
from visualize import (
    boxplot_rpd,
    arpd_bar,
    convergence_curve,
    convergence_grid,
    collect_convergence_from_algo,
    operator_usage_timeline,
    qtable_heatmap,
    _sort_algos,
)


# Helpers

TAILLARD_SETS = [
    "ta001", "ta011", "ta021", "ta031", "ta041",
    "ta051", "ta061", "ta071", "ta081", "ta091", "ta101", "ta111",
]
# Tên đại diện mỗi group (n×m)
TAILLARD_GROUPS = {
    "ta001": "20×5",  "ta011": "20×10", "ta021": "20×20",
    "ta031": "50×5",  "ta041": "50×10", "ta051": "50×20",
    "ta061": "100×5", "ta071": "100×10","ta081": "100×20",
    "ta091": "200×10","ta101": "200×20","ta111": "500×20",
}


def pick_representative_instances(df: pd.DataFrame, source_prefix: str, n_pick: int = 6) -> List[str]:
    """Chọn n_pick instances đại diện cho việc vẽ boxplot."""
    all_insts = sorted(df[df["instance"].str.lower().str.startswith(source_prefix)]["instance"].unique())
    if len(all_insts) <= n_pick:
        return all_insts
    # Phân bổ đều
    indices = np.linspace(0, len(all_insts) - 1, n_pick, dtype=int)
    return [all_insts[i] for i in indices]


def group_by_nm(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột nm_group = 'n×m' để nhóm instances."""
    df = df.copy()
    df["nm_group"] = df["n"].astype(str) + "×" + df["m"].astype(str)
    return df


# Main report

def generate_report(
    results_pattern: str,
    out_dir: Path,
    run_convergence: bool = False,
    dataset_root: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from: {results_pattern}")
    df = load_results(results_pattern)
    print(f"Loaded {len(df):,} rows | instances={df['instance'].nunique()} | algos={df['algorithm'].nunique()}")

    algorithms = _sort_algos(df["algorithm"].unique().tolist())
    print(f"Algorithms: {algorithms}\n")

    # ARPD Table
    print("Computing ARPD table...")
    arpd_df = compute_arpd_table(df)
    arpd_df.to_csv(out_dir / "arpd_table.csv", index=False, float_format="%.6f")
    print(f"Saved: {out_dir/'arpd_table.csv'}")

    summary = arpd_summary(arpd_df)
    print("\nOverall ARPD (AvS) per algorithm:")
    print(summary.to_string(index=False, float_format="%.4f"))

    # Wilcoxon
    print("\nComputing Wilcoxon tests (QIG vs others)...")
    pvals, verdicts = wilcoxon_matrix(df, algorithms)
    pvals.to_csv(out_dir / "wilcoxon_pvalues.csv", float_format="%.6f")
    verdicts.to_csv(out_dir / "wilcoxon_verdicts.csv")
    print(f"Saved: {out_dir/'wilcoxon_pvalues.csv'}")

    if "QIG" in algorithms:
        print("\n  QIG vs:")
        for algo in algorithms:
            if algo != "QIG":
                p = pvals.loc["QIG", algo]
                v = verdicts.loc["QIG", algo]
                print(f"    {algo:<8}: p={p:.4f}  {v}")

    # Figure 1: ARPD Bar
    fig = arpd_bar(arpd_df, metric="AvS", title="Overall ARPD (AvS) by Algorithm")
    fig.savefig(out_dir / "arpd_bar_avs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_dir/'arpd_bar_avs.png'}")

    fig = arpd_bar(arpd_df, metric="BS", title="Overall ARPD (BS) by Algorithm")
    fig.savefig(out_dir / "arpd_bar_bs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Boxplot RPD (Taillard)
    tai_insts = df[df["instance"].str.lower().str.startswith("ta")]["instance"].unique()
    if len(tai_insts) > 0:
        # Chọn 6 instances đại diện
        picks = pick_representative_instances(df, "ta", n_pick=6)
        fig = boxplot_rpd(
            df, instance_sets=picks, algorithms=algorithms,
            figsize=(max(12, len(picks) * 2 + 2), 5),
            title="RPD Distribution - Taillard Instances",
        )
        fig.savefig(out_dir / "boxplot_taillard.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_dir/'boxplot_taillard.png'}")

    # Figure 3: Boxplot RPD (VRF-small)
    vrf_small_insts = df[df["instance"].str.upper().str.startswith("VFR1")]["instance"].unique()
    if len(vrf_small_insts) > 0:
        picks = pick_representative_instances(df, "vfr1", n_pick=4)
        if not picks:
            picks = pick_representative_instances(df, "vrf1", n_pick=4)
        if picks:
            fig = boxplot_rpd(
                df, instance_sets=picks, algorithms=algorithms,
                figsize=(max(10, len(picks) * 2 + 2), 5),
                title="RPD Distribution - VRF-hard-small",
            )
            fig.savefig(out_dir / "boxplot_vrf_small.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_dir/'boxplot_vrf_small.png'}")

    # Figure 4: Boxplot RPD (VRF-large)
    for prefix in ["vfr5", "vfr8", "vrf5", "vrf8"]:
        large_picks = pick_representative_instances(df, prefix, n_pick=4)
        if large_picks:
            break
    if large_picks:
        fig = boxplot_rpd(
            df, instance_sets=large_picks, algorithms=algorithms,
            figsize=(max(10, len(large_picks) * 2 + 2), 5),
            title="RPD Distribution - VRF-hard-large",
        )
        fig.savefig(out_dir / "boxplot_vrf_large.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_dir/'boxplot_vrf_large.png'}")

    # Figure 5: Convergence Grid
    if run_convergence and dataset_root:
        print("\nCollecting convergence data (this may take a while)...")
        _convergence_report(df, algorithms, dataset_root, out_dir)

    print(f"\n✓ Report complete → {out_dir}")


def _convergence_report(
    df: pd.DataFrame,
    algorithms: List[str],
    dataset_root: str,
    out_dir: Path,
    n_runs: int = 5,
) -> None:
    """Chạy lại một số instances để thu convergence data rồi vẽ grid."""
    from benchmark import load_project_datasets, time_limit_ms as tlms

    data = load_project_datasets(dataset_root)
    # Chọn 4 instances đại diện từ Taillard
    tai = data["taillard"]
    target_names = []
    for inst in tai:
        if inst.upper_bound and inst.n in [50, 100, 200, 500] and inst.m == 20:
            target_names.append(inst.name)
            if len(target_names) == 4:
                break

    conv_by_instance: Dict[str, Dict] = {}

    for name in target_names[:4]:
        inst = next((x for x in tai if x.name == name), None)
        if not inst:
            continue
        limit = tlms(inst.n, inst.m, t=60)
        print(f"  Collecting convergence for {name} (n={inst.n}, m={inst.m}) ...")
        conv_by_instance[name] = {}
        for algo in algorithms:
            conv_by_instance[name][algo] = collect_convergence_from_algo(
                inst.p, algo, None, n_runs=n_runs, time_limit_ms_val=limit
            )

    if conv_by_instance:
        fig = convergence_grid(conv_by_instance, title="Convergence Rate (Taillard)")
        fig.savefig(out_dir / "convergence_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_dir/'convergence_grid.png'}")

def parse_args():
    p = argparse.ArgumentParser(description="Generate report figures from benchmark results")
    p.add_argument("--results", required=True, help="Glob pattern to raw CSV(s), e.g. 'results/*.csv'")
    p.add_argument("--out",     default="figures", help="Output directory for figures")
    p.add_argument("--convergence", action="store_true",
                   help="Also collect & plot convergence curves (re-runs algorithm, slow)")
    p.add_argument("--root",   default="datasets",
                   help="Dataset root (needed only with --convergence)")
    return p.parse_args()


def main():
    args = parse_args()
    generate_report(
        results_pattern=args.results,
        out_dir=Path(args.out),
        run_convergence=args.convergence,
        dataset_root=args.root if args.convergence else None,
    )


if __name__ == "__main__":
    main()
