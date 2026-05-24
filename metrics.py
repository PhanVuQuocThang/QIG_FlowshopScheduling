"""
metrics.py
----------
Tính RPD, ARPD, Best-Solution gap, Wilcoxon signed-rank test
từ raw CSV log được tạo bởi experiment_runner.py.

Usage:
    from metrics import load_results, compute_arpd_table, wilcoxon_matrix
    df = load_results("results/raw_results_t60_*.csv")
    table = compute_arpd_table(df)
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# Load raw log

def load_results(pattern: str) -> pd.DataFrame:
    """
    Read all CSV files matching the pattern, concatenate into a single DataFrame.
    Expected columns:
        instance, n, m, algorithm, run_id, seed, cmax, upper_bound, rpd, cpu_sec, time_scale
    """
    files = sorted(glob.glob(pattern))
    if not files:
        p = Path(pattern)
        if p.exists():
            files = [str(p)]
        else:
            raise FileNotFoundError(f"No files matched: {pattern}")

    dfs = []
    for f in files:
        df_f = pd.read_csv(f, dtype={"upper_bound": "Int64"})
        dfs.append(df_f)

    df = pd.concat(dfs, ignore_index=True)

    # Cleanup
    df["rpd"] = pd.to_numeric(df["rpd"], errors="coerce")
    df["cpu_sec"] = pd.to_numeric(df["cpu_sec"], errors="coerce")
    df["cmax"] = pd.to_numeric(df["cmax"], errors="coerce")

    return df


# RPD / ARPD

def compute_rpd(cmax: float, upper_bound: float) -> Optional[float]:
    """RPD = (cmax - UB) / UB * 100"""
    if upper_bound is None or upper_bound <= 0:
        return None
    return 100.0 * (cmax - upper_bound) / upper_bound


def compute_arpd_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ARPD table: for each (instance, algorithm), compute:
    - AvS: Average RPD across runs
    - BS: RPD of the best solution found across runs
    - T(s): Average CPU time across runs
    Also include n, m for reference.
    Returns a DataFrame with columns:
        instance | algorithm | AvS | BS | T(s) | n | m
    """
    rows = []

    # Best solution per (instance, algorithm)
    best_per_inst_algo = (
        df.groupby(["instance", "algorithm"])["cmax"].min().rename("best_cmax").reset_index()
    )
    ub_map = (
        df.dropna(subset=["upper_bound"])
        .groupby("instance")["upper_bound"]
        .first()
        .to_dict()
    )
    size_map = (
        df.groupby("instance")[["n", "m"]].first().to_dict("index")
    )

    for (instance, algorithm), group in df.groupby(["instance", "algorithm"]):
        ub = ub_map.get(instance)
        avs_rpd = group["rpd"].mean() if ub else None

        best_cmax = best_per_inst_algo.loc[
            (best_per_inst_algo["instance"] == instance) &
            (best_per_inst_algo["algorithm"] == algorithm),
            "best_cmax"
        ].values[0]
        bs_rpd = compute_rpd(best_cmax, ub) if ub else None

        avg_cpu = group["cpu_sec"].mean()
        nm = size_map.get(instance, {})

        rows.append({
            "instance":  instance,
            "n":         nm.get("n"),
            "m":         nm.get("m"),
            "algorithm": algorithm,
            "AvS":       avs_rpd,
            "BS":        bs_rpd,
            "T(s)":      avg_cpu,
            "n_runs":    len(group),
        })

    return pd.DataFrame(rows)


def arpd_summary(arpd_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        arpd_df.groupby("algorithm")[["AvS", "BS", "T(s)"]]
        .mean()
        .reset_index()
        .sort_values("AvS")
    )
    return summary


# Wilcoxon test

def wilcoxon_pvalue(
    df: pd.DataFrame,
    algo_a: str,
    algo_b: str,
    metric: str = "rpd",
) -> Tuple[float, str]:
    """
    Wilcoxon signed-rank test (two-sided) and compare algo_a vs algo_b.
    Returns (p_value, verdict):
        verdict = 'A>B' if A is better and significant
                = 'B>A' if B is better and significant
                = '~'   if there is no significant difference (p >= 0.05)
    """
    # mean per instance for each algorithm
    a_vals = (
        df[df["algorithm"] == algo_a]
        .groupby("instance")[metric]
        .mean()
        .sort_index()
    )
    b_vals = (
        df[df["algorithm"] == algo_b]
        .groupby("instance")[metric]
        .mean()
        .sort_index()
    )

    # Chỉ giữ instances có cả 2 thuật toán
    common = a_vals.index.intersection(b_vals.index)
    if len(common) < 2:
        return float("nan"), "insufficient data"

    a = a_vals.loc[common].values
    b = b_vals.loc[common].values

    diff = a - b
    if np.all(diff == 0):
        return 1.0, "~"

    try:
        stat, p = wilcoxon(diff, alternative="two-sided")
    except ValueError:
        return float("nan"), "error"

    if p >= 0.05:
        return p, "~"
    elif a.mean() < b.mean():
        return p, f"{algo_a}>{algo_b}"
    else:
        return p, f"{algo_b}>{algo_a}"


def wilcoxon_matrix(
    df: pd.DataFrame,
    algorithms: Optional[List[str]] = None,
    metric: str = "rpd",
) -> pd.DataFrame:
    if algorithms is None:
        algorithms = sorted(df["algorithm"].unique())

    pvals = pd.DataFrame(index=algorithms, columns=algorithms, dtype=float)
    verdicts = pd.DataFrame(index=algorithms, columns=algorithms, dtype=str)

    for a in algorithms:
        for b in algorithms:
            if a == b:
                pvals.loc[a, b] = np.nan
                verdicts.loc[a, b] = "-"
            else:
                p, v = wilcoxon_pvalue(df, a, b, metric)
                pvals.loc[a, b] = p
                verdicts.loc[a, b] = v

    return pvals, verdicts


# Convenience summary

def print_summary(df: pd.DataFrame, title: str = "") -> None:
    """Print a summary of the results."""
    arpd_df = compute_arpd_table(df)
    summary = arpd_summary(arpd_df)
    if title:
        print(f"\n{'='*50}")
        print(f" {title}")
        print('='*50)
    print(summary.to_string(index=False, float_format="%.4f"))
    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python metrics.py <raw_results_csv_glob>")
        sys.exit(1)
    df = load_results(sys.argv[1])
    print_summary(df, title=f"Summary: {sys.argv[1]}")

    algos = sorted(df["algorithm"].unique())
    print("Wilcoxon p-values (QIG vs others):")
    for a in algos:
        if a == "QIG":
            continue
        p, v = wilcoxon_pvalue(df, "QIG", a)
        print(f"  QIG vs {a:<6}: p={p:.4f}  {v}")
