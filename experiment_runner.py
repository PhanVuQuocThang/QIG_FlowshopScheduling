"""
experiment_runner.py
--------------------
Chạy tự động benchmark cho QIG, RIG, IIG_1..4 trên Taillard / VRF-hard.
Mỗi instance chạy N_RUNS lần độc lập, thu log kết quả vào CSV.

Usage (quick test):
    python experiment_runner.py --root datasets --runs 3 --t 60 --out results/

Usage (full paper):
    python experiment_runner.py --root datasets --runs 30 --t 60 --out results/
    python experiment_runner.py --root datasets --runs 30 --t 90 --out results/
    python experiment_runner.py --root datasets --runs 30 --t 120 --out results/
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from benchmark import (
    BenchmarkSuite,
    PFSPInstance,
    RunResult,
    load_project_datasets,
    time_limit_ms,
    write_results_csv,
)
from ig import IteratedGreedyAlgorithm


def make_solver(strategy: str = "individual", d: int = 1) -> Callable:
    """
    Returns a solver function compatible with benchmark_algorithm.
    The returned solver takes:
        - p: problem instance data
        - seed: optional random seed
        - time_limit_ms: time limit in milliseconds
    It returns the best permutation found and its makespan (Cmax).
    """
    def solver(p, seed: int = 42, time_limit_ms: int = 5000):
        random.seed(seed)
        np.random.seed(seed)
        algo = IteratedGreedyAlgorithm(p, strategy=strategy, d=d)
        algo.execute(
            stopping_criterion="CPU_time",
            runtime_in_miliseconds=time_limit_ms,
            max_iteration=float("inf"),
        )
        return algo.best_solution.perm, algo.best_solution.cmax

    solver.__name__ = f"IIG{d}" if strategy == "individual" else strategy.upper()
    return solver


# List of all algorithms to benchmark
def build_algorithms() -> Dict[str, Callable]:
    algos = {}
    for d in [1, 2, 3, 4]:
        algos[f"IIG{d}"] = make_solver(strategy="individual", d=d)
    algos["RIG"]  = make_solver(strategy="random")
    algos["QIG"]  = make_solver(strategy="qlearning")
    return algos


# Benchmarking logic
@dataclass
class RunRecord:
    """A log entry: instance × algorithm × run."""
    instance:   str
    n:          int
    m:          int
    algorithm:  str
    run_id:     int
    seed:       int
    cmax:       int
    upper_bound: Optional[int]
    rpd:        Optional[float]   # (cmax - UB) / UB * 100
    cpu_sec:    float
    time_scale: int               # t ∈ {60, 90, 120}
    # convergence tracking (best cmax at each iteration)
    convergence: List[int] = field(default_factory=list)


def run_one(
    inst: PFSPInstance,
    algo_name: str,
    solver: Callable,
    run_id: int,
    seed: int,
    t_scale: int,
) -> RunRecord:
    """Run a single experiment, return RunRecord."""
    limit = time_limit_ms(inst.n, inst.m, t=t_scale)

    t0 = time.perf_counter()
    best_perm, best_cmax = solver(inst.p, seed=seed, time_limit_ms=limit)
    cpu = time.perf_counter() - t0

    rpd_val = inst.rpd(best_cmax) if inst.upper_bound else None

    return RunRecord(
        instance=inst.name,
        n=inst.n,
        m=inst.m,
        algorithm=algo_name,
        run_id=run_id,
        seed=seed,
        cmax=best_cmax,
        upper_bound=inst.upper_bound,
        rpd=rpd_val,
        cpu_sec=cpu,
        time_scale=t_scale,
    )


def run_benchmark(
    suite: BenchmarkSuite,
    algorithms: Dict[str, Callable],
    n_runs: int = 30,
    seed0: int = 1000,
    t_scale: int = 60,
    log_dir: Path = Path("results"),
    verbose: bool = True,
) -> List[RunRecord]:
    """
    Run the full benchmark.
    Write results incrementally to CSV (to avoid losing data on crash).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / f"raw_results_t{t_scale}_{timestamp}.csv"

    fieldnames = [
        "instance", "n", "m", "algorithm", "run_id", "seed",
        "cmax", "upper_bound", "rpd", "cpu_sec", "time_scale",
    ]

    all_records: List[RunRecord] = []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total = len(suite) * len(algorithms) * n_runs
        done  = 0

        for inst in suite:
            for algo_name, solver in algorithms.items():
                for run_id in range(n_runs):
                    seed = seed0 + run_id

                    rec = run_one(inst, algo_name, solver, run_id, seed, t_scale)
                    all_records.append(rec)

                    writer.writerow({
                        "instance":    rec.instance,
                        "n":           rec.n,
                        "m":           rec.m,
                        "algorithm":   rec.algorithm,
                        "run_id":      rec.run_id,
                        "seed":        rec.seed,
                        "cmax":        rec.cmax,
                        "upper_bound": rec.upper_bound if rec.upper_bound else "",
                        "rpd":         f"{rec.rpd:.8f}" if rec.rpd is not None else "",
                        "cpu_sec":     f"{rec.cpu_sec:.6f}",
                        "time_scale":  rec.time_scale,
                    })
                    f.flush()

                    done += 1
                    if verbose:
                        rpd_str = f"{rec.rpd:.4f}%" if rec.rpd is not None else "N/A"
                        print(
                            f"[{done:>5}/{total}] {inst.name:<20} {algo_name:<8} "
                            f"run={run_id:>2}  cmax={rec.cmax}  RPD={rpd_str}  "
                            f"cpu={rec.cpu_sec:.2f}s"
                        )

    if verbose:
        print(f"\nResults saved to {csv_path}")

    return all_records

def parse_args():
    p = argparse.ArgumentParser(description="QIG/RIG/IIG benchmark runner")
    p.add_argument("--root",    default="datasets",  help="Dataset root folder")
    p.add_argument("--dataset", default="all",
                   choices=["all", "taillard", "vrf_small", "vrf_large"],
                   help="Which dataset to run")
    p.add_argument("--runs",    type=int, default=30, help="Independent runs per instance")
    p.add_argument("--t", type=int, default=60)
    p.add_argument("--out",     default="results",   help="Output directory")
    p.add_argument("--algos",   default="all",
                   help="Comma-separated list: IIG1,IIG2,IIG3,IIG4,RIG,QIG  (default: all)")
    p.add_argument("--seed0",   type=int, default=1000, help="Base seed")
    p.add_argument("--quiet",   action="store_true",    help="Suppress per-run prints")
    p.add_argument("--limit", type=int, default=None, help="Limit number of instances for debugging")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading datasets from '{args.root}' ...")
    data = load_project_datasets(args.root)
    suite = data[args.dataset]
    if args.limit is not None:
        suite = BenchmarkSuite(list(suite)[:args.limit])
    print(suite.summary())
    print()

    all_algos = build_algorithms()
    if args.algos.lower() == "all":
        algorithms = all_algos
    else:
        selected = [a.strip() for a in args.algos.split(",")]
        algorithms = {k: v for k, v in all_algos.items() if k in selected}
        if not algorithms:
            raise ValueError(f"[Error] No valid algorithm in --algos={args.algos}. "
                             f"Valid: {list(all_algos.keys())}")

    print(f"Algorithms : {list(algorithms.keys())}")
    print(f"Runs/inst  : {args.runs}")
    print(f"Time scale : t={args.t}")
    print(f"Output     : {args.out}\n")

    run_benchmark(
        suite=suite,
        algorithms=algorithms,
        n_runs=args.runs,
        seed0=args.seed0,
        t_scale=args.t,
        log_dir=Path(args.out),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
