"""
benchmark.py
---
PFSP benchmark loader for the project folder structure:

datasets/
├── bounds/
│   ├── Taillard_UB_Schedules_v9.csv
│   ├── VFRlarge_UB_Schedules_v9.csv
│   └── VFRsmall_UB_Schedules_v9.csv
├── taillard_instances/
│   ├── ta001
│   ├── ta002
│   └── ...
└── vrf_instances/
    ├── Large/
    │   ├── VFR100_20_1_Gap.txt
    │   └── ...
    └── Small/
        ├── VFR10_5_1_Gap.txt
        └── ...

Supported raw formats
---------------------
1) Taillard files in datasets/taillard_instances:
   Header:
       n m seed UB LB
   Then m rows, each row has n processing times.
   Internal matrix is p[machine][job].

2) VRF/VFR *_Gap.txt files:
   Header:
       n m
   Then n rows, each row represents one job and contains m pairs:
       machine_id processing_time machine_id processing_time ...
   Example:
       0 45 1 31 2 54 3 54 4 64
   Internal matrix is converted to p[machine][job].

3) CSV files in datasets/bounds:
   These only contain metadata / UB / best-known permutation.
   They do NOT contain processing times.

Main entry point
----------------
    from benchmark import load_project_datasets
    data = load_project_datasets("datasets")
    taillard = data["taillard"]
    vrf_small = data["vrf_small"]
    vrf_large = data["vrf_large"]
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Data classes
@dataclass
class BoundsRecord:
    name: str
    n: int
    m: int
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    optimal: Optional[bool] = None
    ub_found_by: Optional[str] = None
    permutation: Optional[List[int]] = None


@dataclass
class PFSPInstance:
    """
    One PFSP instance.

    p[machine][job] is the processing-time matrix used by NEH/IG/QIG.
    Jobs and machines are stored 0-based internally.
    The best_permutation from CSV is kept as read, usually 1-based.
    """

    name: str
    n: int
    m: int
    p: List[List[int]]
    source: str = "unknown"
    path: Optional[str] = None

    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    optimal: Optional[bool] = None
    ub_found_by: Optional[str] = None
    best_permutation: Optional[List[int]] = None

    meta: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_processing_times(self.p, self.n, self.m)

    def __repr__(self) -> str:
        attrs = []
        if self.upper_bound is not None:
            attrs.append(f"UB={self.upper_bound}")
        if self.lower_bound is not None:
            attrs.append(f"LB={self.lower_bound}")
        if self.optimal is not None:
            attrs.append(f"optimal={int(self.optimal)}")
        suffix = f" [{', '.join(attrs)}]" if attrs else ""
        return (
            f"PFSPInstance(name='{self.name}', n={self.n}, m={self.m}, "
            f"source='{self.source}'{suffix})"
        )

    def cmax(self, permutation: Sequence[int], assume: str = "auto") -> int:
        return makespan(self.p, permutation, assume=assume)

    def rpd(self, cmax_value: int, reference: str = "ub") -> Optional[float]:
        ref = self.upper_bound if reference.lower() == "ub" else self.lower_bound
        if ref is None or ref <= 0:
            return None
        return 100.0 * (cmax_value - ref) / ref

    def best_known_cmax(self) -> Optional[int]:
        if not self.best_permutation:
            return None
        return self.cmax(self.best_permutation, assume="auto")

    def check_bound(self) -> Tuple[bool, Optional[int], Optional[int]]:
        """Return (ok, computed_cmax_from_csv_permutation, csv_ub)."""
        if self.best_permutation is None or self.upper_bound is None:
            return False, self.best_known_cmax(), self.upper_bound
        c = self.best_known_cmax()
        return c == self.upper_bound, c, self.upper_bound

# PFSP utilities
def validate_processing_times(p: Sequence[Sequence[int]], n: int, m: int) -> None:
    if len(p) != m:
        raise ValueError(f"Expected {m} machine rows, got {len(p)}.")
    for i, row in enumerate(p):
        if len(row) != n:
            raise ValueError(f"Machine row {i}: expected {n} jobs, got {len(row)}.")
        if any(int(x) < 0 for x in row):
            raise ValueError(f"Negative processing time in machine row {i}.")


def _to_zero_based_perm(permutation: Sequence[int], n: int, assume: str = "auto") -> List[int]:
    perm = [int(x) for x in permutation]
    if len(perm) != n:
        raise ValueError(f"Permutation length mismatch: expected {n}, got {len(perm)}.")

    assume = assume.lower()
    if assume == "zero":
        z = perm
    elif assume == "one":
        z = [x - 1 for x in perm]
    elif assume == "auto":
        s = set(perm)
        if s == set(range(n)):
            z = perm
        elif s == set(range(1, n + 1)):
            z = [x - 1 for x in perm]
        else:
            raise ValueError("Cannot infer permutation indexing; expected 0..n-1 or 1..n.")
    else:
        raise ValueError("assume must be: auto, zero, one")

    if set(z) != set(range(n)):
        raise ValueError("Invalid permutation.")
    return z


def makespan(p: Sequence[Sequence[int]], permutation: Sequence[int], assume: str = "auto") -> int:
    """Compute Cmax for PFSP with p[machine][job]."""
    if not p:
        return 0
    m = len(p)
    n = len(p[0])
    perm = _to_zero_based_perm(permutation, n=n, assume=assume)

    c = [0] * m
    for job in perm:
        c[0] += int(p[0][job])
        for machine in range(1, m):
            c[machine] = max(c[machine], c[machine - 1]) + int(p[machine][job])
    return c[-1]


def rpd(cmax_value: int, reference_value: int) -> float:
    if reference_value <= 0:
        raise ValueError("reference_value must be positive.")
    return 100.0 * (cmax_value - reference_value) / reference_value


def time_limit_ms(n: int, m: int, t: int = 60) -> int:
    """Paper phase-2 time limit: T = n*m/2*t milliseconds."""
    return int(n * m / 2 * t)


# Name normalization and bounds CSV
def _clean(x: object) -> str:
    return "" if x is None else str(x).strip()


def _int_or_none(x: object) -> Optional[int]:
    s = _clean(x)
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _bool_or_none(x: object) -> Optional[bool]:
    v = _int_or_none(x)
    return None if v is None else bool(v)


def _perm_or_none(x: object) -> Optional[List[int]]:
    s = _clean(x)
    if not s:
        return None
    vals = re.findall(r"-?\d+", s)
    return [int(v) for v in vals] if vals else None


def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", Path(str(name)).stem.upper())


def name_keys(name: str) -> set[str]:
    """
    Robust aliases:
      ta001           <-> Ta001
      tai001          <-> Ta001
      VFR100_20_1_Gap <-> VFR100_20_1
      VRF...          <-> VFR...
    """
    k = normalize_name(name)
    keys = {k}

    # Strip GAP suffix in VFR *_Gap.txt files.
    if k.endswith("GAP"):
        keys.add(k[:-3])

    # Taillard aliases: TA001 <-> TAI001.
    for x in list(keys):
        if x.startswith("TAI"):
            keys.add("TA" + x[3:])
        if x.startswith("TA") and not x.startswith("TAI"):
            keys.add("TAI" + x[2:])

    # VFR/VRF spelling aliases.
    for x in list(keys):
        if x.startswith("VFR"):
            keys.add("VRF" + x[3:])
        if x.startswith("VRF"):
            keys.add("VFR" + x[3:])

    return keys


def load_bounds_csv(filepath: str | Path) -> Dict[str, BoundsRecord]:
    """Load Name;n;m;LB;UB;Optimal;UBFoundBy;Permutation CSV."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    records: Dict[str, BoundsRecord] = {}
    with filepath.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)

        delimiter = ";"
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delimiter = dialect.delimiter
        except csv.Error:
            pass

        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            return records
        reader.fieldnames = [_clean(h).lstrip("\ufeff") for h in reader.fieldnames]

        for raw in reader:
            row = {_clean(k).lstrip("\ufeff"): _clean(v) for k, v in raw.items()}
            name = row.get("Name") or row.get("name")
            if not name:
                continue

            rec = BoundsRecord(
                name=name,
                n=_int_or_none(row.get("n")) or _int_or_none(row.get("N")) or 0,
                m=_int_or_none(row.get("m")) or _int_or_none(row.get("M")) or 0,
                lower_bound=_int_or_none(row.get("LB")),
                upper_bound=_int_or_none(row.get("UB")),
                optimal=_bool_or_none(row.get("Optimal")),
                ub_found_by=_clean(row.get("UBFoundBy")),
                permutation=_perm_or_none(row.get("Permutation")),
            )

            for key in name_keys(name):
                records[key] = rec

    return records


def attach_bounds(
    instances: Iterable[PFSPInstance],
    bounds: Dict[str, BoundsRecord],
    strict_size: bool = True,
) -> List[PFSPInstance]:
    out = []
    for inst in instances:
        rec = None
        for key in name_keys(inst.name):
            if key in bounds:
                rec = bounds[key]
                break

        if rec is not None:
            size_ok = (
                not strict_size
                or (rec.n in {0, inst.n} and rec.m in {0, inst.m})
            )
            if size_ok:
                inst.lower_bound = rec.lower_bound
                inst.upper_bound = rec.upper_bound
                inst.optimal = rec.optimal
                inst.ub_found_by = rec.ub_found_by
                inst.best_permutation = rec.permutation

        out.append(inst)
    return out

# Raw instance parsers
def _ints_from_text(text: str) -> List[int]:
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r"#[^\n]*", " ", text)
    return [int(x) for x in re.findall(r"-?\d+", text)]


def parse_taillard_file(filepath: str | Path, name: Optional[str] = None) -> PFSPInstance:
    """
    Parse datasets/taillard_instances/ta001 style.

    Format:
        n m seed UB LB
        m rows x n processing times
    """
    filepath = Path(filepath)
    nums = _ints_from_text(filepath.read_text(encoding="utf-8", errors="replace"))
    if len(nums) < 5:
        raise ValueError(f"{filepath}: not enough integers for Taillard header.")

    n, m, seed, ub, lb = nums[:5]
    need = 5 + n * m
    if n <= 0 or m <= 0 or len(nums) < need:
        raise ValueError(f"{filepath}: incomplete Taillard instance.")

    vals = nums[5:need]
    p = [vals[i * n : (i + 1) * n] for i in range(m)]

    return PFSPInstance(
        name=name or filepath.stem,
        n=n,
        m=m,
        p=p,
        source="taillard",
        path=str(filepath),
        lower_bound=lb if lb > 0 else None,
        upper_bound=ub if ub > 0 else None,
        meta={"seed": str(seed), "raw_format": "taillard_machine_major"},
    )


def parse_vrf_gap_file(
    filepath: str | Path,
    name: Optional[str] = None,
    source: Optional[str] = None,
) -> PFSPInstance:
    """
    Parse VFR*_Gap.txt.

    Format:
        n m
        n rows, one row per job:
            machine_id processing_time machine_id processing_time ...

    Output p[machine][job].
    """
    filepath = Path(filepath)
    lines = [
        ln.strip()
        for ln in filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip()
    ]
    if not lines:
        raise ValueError(f"{filepath}: empty file.")

    header = [int(x) for x in re.findall(r"-?\d+", lines[0])]
    if len(header) < 2:
        raise ValueError(f"{filepath}: first line must contain n and m.")
    n, m = header[:2]

    if len(lines) - 1 < n:
        raise ValueError(f"{filepath}: expected {n} job rows, got {len(lines) - 1}.")

    p = [[0 for _ in range(n)] for _ in range(m)]

    for job in range(n):
        nums = [int(x) for x in re.findall(r"-?\d+", lines[1 + job])]
        if len(nums) != 2 * m:
            raise ValueError(
                f"{filepath}: job row {job} must contain {2*m} integers "
                f"({m} machine/time pairs), got {len(nums)}."
            )
        seen = set()
        for k in range(m):
            machine = nums[2 * k]
            proc = nums[2 * k + 1]
            if not (0 <= machine < m):
                raise ValueError(f"{filepath}: invalid machine id {machine} in job row {job}.")
            if machine in seen:
                raise ValueError(f"{filepath}: duplicated machine id {machine} in job row {job}.")
            seen.add(machine)
            p[machine][job] = proc

    stem = filepath.stem
    if stem.upper().endswith("_GAP"):
        stem = stem[:-4]

    return PFSPInstance(
        name=name or stem,
        n=n,
        m=m,
        p=p,
        source=source or guess_source(filepath),
        path=str(filepath),
        meta={"raw_format": "vrf_gap_job_pair"},
    )


def guess_source(path: str | Path) -> str:
    s = str(path).lower()
    stem = Path(path).stem.lower()
    if "taillard" in s or re.match(r"ta\d+", stem) or re.match(r"tai\d+", stem):
        return "taillard"
    if "small" in s:
        return "vrf_small"
    if "large" in s:
        return "vrf_large"
    if "vfr" in stem or "vrf" in stem:
        return "vrf"
    return "unknown"


def guess_format(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    if stem.endswith("_gap") or "vfr" in stem or "vrf" in stem:
        return "vrf_gap"
    if re.match(r"ta\d+", stem) or re.match(r"tai\d+", stem):
        return "taillard"
    return "auto"


def load_instance_file(filepath: str | Path, fmt: str = "auto") -> PFSPInstance:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    if fmt == "auto":
        fmt = guess_format(filepath)
        if fmt == "auto":
            # Fallback by shape: if first line has 2 ints and next line has pairs -> vrf_gap.
            lines = [ln.strip() for ln in filepath.read_text(errors="replace").splitlines() if ln.strip()]
            first = [int(x) for x in re.findall(r"-?\d+", lines[0])] if lines else []
            second = [int(x) for x in re.findall(r"-?\d+", lines[1])] if len(lines) > 1 else []
            if len(first) == 2 and len(second) == 2 * first[1]:
                fmt = "vrf_gap"
            else:
                fmt = "taillard"

    fmt = fmt.lower()
    if fmt in {"taillard", "ta"}:
        return parse_taillard_file(filepath)
    if fmt in {"vrf_gap", "vfr_gap", "vrf", "vfr"}:
        return parse_vrf_gap_file(filepath)

    raise ValueError(f"Unsupported fmt={fmt!r}")

# Suite and project loaders
class BenchmarkSuite:
    def __init__(self, instances: Iterable[PFSPInstance] = ()):
        self._instances = list(instances)

    def __len__(self) -> int:
        return len(self._instances)

    def __iter__(self) -> Iterator[PFSPInstance]:
        return iter(self._instances)

    def __getitem__(self, idx):
        return self._instances[idx]

    def __repr__(self) -> str:
        return f"BenchmarkSuite({len(self)} instances)"

    @property
    def instances(self) -> List[PFSPInstance]:
        return list(self._instances)

    def attach_bounds(self, bounds_csv_or_map: str | Path | Dict[str, BoundsRecord]) -> "BenchmarkSuite":
        bounds = (
            bounds_csv_or_map
            if isinstance(bounds_csv_or_map, dict)
            else load_bounds_csv(bounds_csv_or_map)
        )
        attach_bounds(self._instances, bounds)
        return self

    def filter(
        self,
        min_n: int = 0,
        max_n: int = 10**9,
        min_m: int = 0,
        max_m: int = 10**9,
        source: Optional[str] = None,
        name_pattern: Optional[str] = None,
        has_ub: Optional[bool] = None,
    ) -> "BenchmarkSuite":
        pat = re.compile(name_pattern, re.I) if name_pattern else None

        def keep(x: PFSPInstance) -> bool:
            if not (min_n <= x.n <= max_n and min_m <= x.m <= max_m):
                return False
            if source is not None and x.source != source:
                return False
            if pat is not None and not pat.search(x.name):
                return False
            if has_ub is not None and ((x.upper_bound is not None) != has_ub):
                return False
            return True

        return BenchmarkSuite([x for x in self._instances if keep(x)])

    def summary(self) -> str:
        counts = Counter((x.source, x.n, x.m) for x in self._instances)
        lines = [f"{'source':<12} {'n':>6} {'m':>6} {'count':>8} {'with_UB':>8}"]
        lines.append("-" * 46)
        for (src, n, m), cnt in sorted(counts.items(), key=lambda z: (z[0][0], z[0][1], z[0][2])):
            with_ub = sum(
                1 for x in self._instances
                if x.source == src and x.n == n and x.m == m and x.upper_bound is not None
            )
            lines.append(f"{src:<12} {n:>6} {m:>6} {cnt:>8} {with_ub:>8}")
        lines.append("-" * 46)
        lines.append(f"total instances: {len(self._instances)}")
        return "\n".join(lines)

    def check_bounds_consistency(self, limit: int = 20) -> List[Tuple[str, bool, Optional[int], Optional[int]]]:
        bad = []
        for inst in self._instances:
            if inst.upper_bound is None or inst.best_permutation is None:
                continue
            ok, c, ub = inst.check_bound()
            if not ok:
                bad.append((inst.name, ok, c, ub))
                if len(bad) >= limit:
                    break
        return bad


def _files_in(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file()])


def load_taillard_instances(
    folder: str | Path,
    bounds_csv: Optional[str | Path] = None,
) -> BenchmarkSuite:
    instances = []
    for path in _files_in(folder):
        try:
            instances.append(parse_taillard_file(path))
        except Exception as exc:
            print(f"[WARN] skip {path}: {exc}")

    suite = BenchmarkSuite(instances)
    if bounds_csv is not None and Path(bounds_csv).exists():
        suite.attach_bounds(bounds_csv)
    return suite


def load_vrf_gap_instances(
    folder: str | Path,
    bounds_csv: Optional[str | Path] = None,
    source: Optional[str] = None,
) -> BenchmarkSuite:
    instances = []
    for path in _files_in(folder):
        try:
            instances.append(parse_vrf_gap_file(path, source=source or guess_source(path)))
        except Exception as exc:
            print(f"[WARN] skip {path}: {exc}")

    suite = BenchmarkSuite(instances)
    if bounds_csv is not None and Path(bounds_csv).exists():
        suite.attach_bounds(bounds_csv)
    return suite


def load_project_datasets(root: str | Path = "datasets") -> Dict[str, BenchmarkSuite]:
    """
    Load all datasets using your current folder tree.

    Returns:
        {
            "taillard": BenchmarkSuite(...),
            "vrf_small": BenchmarkSuite(...),
            "vrf_large": BenchmarkSuite(...),
            "all": BenchmarkSuite(...)
        }
    """
    root = Path(root)
    bounds = root / "bounds"

    taillard = load_taillard_instances(
        root / "taillard_instances",
        bounds / "Taillard_UB_Schedules_v9.csv",
    )
    vrf_small = load_vrf_gap_instances(
        root / "vrf_instances" / "Small",
        bounds / "VFRsmall_UB_Schedules_v9.csv",
        source="vrf_small",
    )
    vrf_large = load_vrf_gap_instances(
        root / "vrf_instances" / "Large",
        bounds / "VFRlarge_UB_Schedules_v9.csv",
        source="vrf_large",
    )

    all_suite = BenchmarkSuite(
        taillard.instances + vrf_small.instances + vrf_large.instances
    )

    return {
        "taillard": taillard,
        "vrf_small": vrf_small,
        "vrf_large": vrf_large,
        "all": all_suite,
    }


# =============================================================================
# Result helpers for benchmark runs
# =============================================================================

@dataclass
class RunResult:
    instance: str
    algorithm: str
    run_id: int
    cmax: int
    cpu_time: float
    gap: Optional[float] = None
    seed: Optional[int] = None


def write_results_csv(results: Iterable[RunResult], filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fields = ["instance", "algorithm", "run_id", "seed", "cmax", "gap", "cpu_time"]

    with filepath.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "instance": r.instance,
                "algorithm": r.algorithm,
                "run_id": r.run_id,
                "seed": "" if r.seed is None else r.seed,
                "cmax": r.cmax,
                "gap": "" if r.gap is None else f"{r.gap:.8f}",
                "cpu_time": f"{r.cpu_time:.6f}",
            })


def benchmark_algorithm(
    suite: BenchmarkSuite,
    algorithm,
    algorithm_name: str,
    runs: int = 30,
    seed0: int = 1000,
    time_limit_t: Optional[int] = None,
) -> List[RunResult]:
    """
    Generic runner.

    Expected algorithm signature:
        best_perm, best_cmax = algorithm(p, seed=seed, time_limit_ms=limit_ms)

    If your function does not accept time_limit_ms, wrap it with a small adapter.
    """
    results: List[RunResult] = []

    for inst in suite:
        for run_id in range(runs):
            seed = seed0 + run_id
            limit = None if time_limit_t is None else time_limit_ms(inst.n, inst.m, t=time_limit_t)

            tic = time.perf_counter()
            if limit is None:
                best_perm, best_cmax = algorithm(inst.p, seed=seed)
            else:
                best_perm, best_cmax = algorithm(inst.p, seed=seed, time_limit_ms=limit)
            cpu = time.perf_counter() - tic

            results.append(
                RunResult(
                    instance=inst.name,
                    algorithm=algorithm_name,
                    run_id=run_id,
                    seed=seed,
                    cmax=int(best_cmax),
                    gap=inst.rpd(int(best_cmax)),
                    cpu_time=cpu,
                )
            )
    return results


# =============================================================================
# CLI
# =============================================================================

def _cli() -> None:
    parser = argparse.ArgumentParser(description="PFSP benchmark dataset loader")
    parser.add_argument("--root", default="datasets", help="Dataset root folder")
    parser.add_argument("--which", default="all", choices=["all", "taillard", "vrf_small", "vrf_large"])
    parser.add_argument("--check-bounds", action="store_true", help="Check Cmax(csv permutation) == UB")
    parser.add_argument("--show-first", action="store_true")
    args = parser.parse_args()

    data = load_project_datasets(args.root)
    suite = data[args.which]

    print(suite.summary())

    if args.show_first and len(suite) > 0:
        inst = suite[0]
        print("\nFirst instance:")
        print(" ", inst)
        print("  matrix shape:", len(inst.p), "x", len(inst.p[0]))
        print("  first machine first 10 jobs:", inst.p[0][:10])
        print("  UB:", inst.upper_bound)
        if inst.best_permutation:
            print("  Cmax(best permutation):", inst.best_known_cmax())

    if args.check_bounds:
        bad = suite.check_bounds_consistency(limit=20)
        if bad:
            print("\nBound mismatches:")
            for row in bad:
                print(" ", row)
        else:
            print("\nBounds check: no mismatch found among instances with UB + permutation.")


if __name__ == "__main__":
    _cli()
