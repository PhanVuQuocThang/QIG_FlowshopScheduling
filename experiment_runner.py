"""
experiment_runner_resume.py
---------------------------
Resume-capable benchmark runner for QIG/RIG/IIG.

It skips runs already present in previous raw_results CSV files, so you can
continue after a crash/power loss without re-running completed jobs.

Typical usage:
    python experiment_runner_resume.py --root datasets --dataset taillard --runs 15 --t 60 \
        --algos IIG1,IIG2,IIG3,RIG,QIG --out results/taillard_full_t60 --resume "results/taillard_full_t60/*.csv"
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from benchmark import BenchmarkSuite, PFSPInstance, load_project_datasets, time_limit_ms
from ig import IteratedGreedyAlgorithm


def make_solver(strategy: str = "individual", d: int = 1) -> Callable:
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


def build_algorithms() -> Dict[str, Callable]:
    algos = {}
    for d in [1, 2, 3, 4]:
        algos[f"IIG{d}"] = make_solver(strategy="individual", d=d)
    algos["RIG"] = make_solver(strategy="random")
    algos["QIG"] = make_solver(strategy="qlearning")
    return algos


@dataclass
class RunRecord:
    instance: str
    n: int
    m: int
    algorithm: str
    run_id: int
    seed: int
    cmax: int
    upper_bound: Optional[int]
    rpd: Optional[float]
    cpu_sec: float
    time_scale: int


def _expand_patterns(patterns: Iterable[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            files.extend(matches)
        elif Path(pat).exists():
            files.append(pat)
    return sorted(set(files))


def load_completed_keys(patterns: Iterable[str]) -> Set[Tuple[str, str, int]]:
    """
    Read previous raw_results CSVs and return keys:
        (instance, algorithm, run_id)
    """
    files = _expand_patterns(patterns)
    completed: Set[Tuple[str, str, int]] = set()

    for f in files:
        try:
            df = pd.read_csv(f, usecols=["instance", "algorithm", "run_id"])
        except Exception as exc:
            print(f"[WARN] cannot read resume file {f}: {exc}")
            continue

        for row in df.itertuples(index=False):
            try:
                completed.add((str(row.instance), str(row.algorithm), int(row.run_id)))
            except Exception:
                continue

    return completed


def run_one(
    inst: PFSPInstance,
    algo_name: str,
    solver: Callable,
    run_id: int,
    seed: int,
    t_scale: int,
) -> RunRecord:
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
        cmax=int(best_cmax),
        upper_bound=inst.upper_bound,
        rpd=rpd_val,
        cpu_sec=cpu,
        time_scale=t_scale,
    )


def run_benchmark_resume(
    suite: BenchmarkSuite,
    algorithms: Dict[str, Callable],
    n_runs: int,
    seed0: int,
    t_scale: int,
    log_dir: Path,
    completed: Set[Tuple[str, str, int]],
    verbose: bool = True,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / f"raw_results_t{t_scale}_resume_{timestamp}.csv"

    fieldnames = [
        "instance", "n", "m", "algorithm", "run_id", "seed",
        "cmax", "upper_bound", "rpd", "cpu_sec", "time_scale",
    ]

    total_target = len(suite) * len(algorithms) * n_runs
    total_skipped = 0
    total_to_run = 0

    planned = []
    for inst in suite:
        for algo_name, solver in algorithms.items():
            for run_id in range(n_runs):
                key = (inst.name, algo_name, run_id)
                if key in completed:
                    total_skipped += 1
                    continue
                planned.append((inst, algo_name, solver, run_id))
                total_to_run += 1

    print(f"Target runs      : {total_target}")
    print(f"Already completed: {total_skipped}")
    print(f"Remaining to run : {total_to_run}")
    print(f"Resume output    : {csv_path}\n")

    done = 0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for inst, algo_name, solver, run_id in planned:
            seed = seed0 + run_id
            rec = run_one(inst, algo_name, solver, run_id, seed, t_scale)

            writer.writerow({
                "instance": rec.instance,
                "n": rec.n,
                "m": rec.m,
                "algorithm": rec.algorithm,
                "run_id": rec.run_id,
                "seed": rec.seed,
                "cmax": rec.cmax,
                "upper_bound": rec.upper_bound if rec.upper_bound else "",
                "rpd": f"{rec.rpd:.8f}" if rec.rpd is not None else "",
                "cpu_sec": f"{rec.cpu_sec:.6f}",
                "time_scale": rec.time_scale,
            })
            f.flush()

            done += 1
            if verbose:
                rpd_str = f"{rec.rpd:.4f}%" if rec.rpd is not None else "N/A"
                print(
                    f"[{done:>5}/{total_to_run}] {inst.name:<20} {algo_name:<8} "
                    f"run={run_id:>2}  cmax={rec.cmax}  RPD={rpd_str}  "
                    f"cpu={rec.cpu_sec:.2f}s"
                )

    print(f"\nResume results saved to {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Resume-capable QIG/RIG/IIG benchmark runner")
    p.add_argument("--root", default="datasets", help="Dataset root folder")
    p.add_argument("--dataset", default="all", choices=["all", "taillard", "vrf_small", "vrf_large", "taillard_ins_subset_1"])
    p.add_argument("--runs", type=int, default=30, help="Independent runs per instance")
    p.add_argument("--t", type=int, default=60)
    p.add_argument("--out", default="results", help="Output directory")
    p.add_argument("--algos", default="all", help="Comma-separated list: IIG1,IIG2,IIG3,IIG4,RIG,QIG")
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--resume",
        default=None,
        help='CSV path/glob to skip completed runs, e.g. "results/taillard_full_t60/*.csv". '
             "If omitted, automatically uses --out/*.csv.",
    )
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
        selected = [a.strip() for a in args.algos.split(",") if a.strip()]
        algorithms = {k: v for k, v in all_algos.items() if k in selected}
        if not algorithms:
            raise ValueError(f"No valid algorithm in --algos={args.algos}. Valid: {list(all_algos.keys())}")

    resume_pattern = args.resume if args.resume is not None else str(Path(args.out) / "*.csv")
    completed = load_completed_keys([resume_pattern])

    print(f"Algorithms : {list(algorithms.keys())}")
    print(f"Runs/inst  : {args.runs}")
    print(f"Time scale : t={args.t}")
    print(f"Output     : {args.out}")
    print(f"Resume from: {resume_pattern}")
    print(f"Completed keys found: {len(completed)}\n")

    run_benchmark_resume(
        suite=suite,
        algorithms=algorithms,
        n_runs=args.runs,
        seed0=args.seed0,
        t_scale=args.t,
        log_dir=Path(args.out),
        completed=completed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
