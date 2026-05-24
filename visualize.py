"""
visualize.py
------------
Usage:
    from visualize import boxplot_rpd, convergence_curve
    fig = boxplot_rpd(df, instance_set="tai_100_20")
    fig.savefig("boxplot.png", dpi=150)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":   120,
})

ALGO_COLORS = {
    "QIG":  "#E63946",   # đỏ - thuật toán đề xuất
    "RIG":  "#457B9D",   # xanh dương
    "IIG1": "#2A9D8F",   # xanh lá
    "IIG2": "#E9C46A",   # vàng
    "IIG3": "#F4A261",   # cam
    "IIG4": "#A8DADC",   # xanh nhạt
}

def _algo_color(algo: str) -> str:
    return ALGO_COLORS.get(algo, "#888888")

def _sort_algos(algos: Sequence[str]) -> List[str]:
    order = ["QIG", "RIG", "IIG1", "IIG2", "IIG3", "IIG4"]
    sorted_a = [a for a in order if a in algos]
    rest = [a for a in algos if a not in order]
    return sorted_a + rest


# 1. Boxplot RPD 

def boxplot_rpd(
    df: pd.DataFrame,
    instance_sets: Optional[List[str]] = None,
    algorithms:    Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
    title: str = "",
) -> plt.Figure:
    """
    Boxplot RPD on representative instances.
    """
    if algorithms is None:
        algorithms = _sort_algos(df["algorithm"].unique())
    if instance_sets is None:
        instance_sets = sorted(df["instance"].unique())

    n_sets = len(instance_sets)
    fig, axes = plt.subplots(1, n_sets, figsize=figsize, sharey=False)
    if n_sets == 1:
        axes = [axes]

    for ax, inst in zip(axes, instance_sets):
        data_by_algo = []
        labels = []
        colors = []

        for algo in algorithms:
            vals = df[(df["instance"] == inst) & (df["algorithm"] == algo)]["rpd"].dropna()
            if len(vals) > 0:
                data_by_algo.append(vals.values)
                labels.append(algo)
                colors.append(_algo_color(algo))

        bp = ax.boxplot(
            data_by_algo,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Thêm điểm mean (red circle, như paper Figure 3)
        for i, vals in enumerate(data_by_algo, start=1):
            ax.plot(i, np.mean(vals), "ro", markersize=5, zorder=5)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(inst, fontsize=10)
        ax.set_ylabel("RPD (%)" if axes.index(ax) == 0 else "")  # type: ignore
        ax.axhline(0, color="gray", lw=0.7, ls="--")

    fig.suptitle(title or "RPD Distribution by Instance", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# 2. Convergence Curve

def convergence_curve(
    convergence_data: Dict[str, Dict[str, List[float]]],
    instance_name: str = "",
    figsize: Tuple[int, int] = (8, 4),
    title: str = "",
) -> plt.Figure:
    """
    Plot convergence curve (best Cmax by iteration) for multiple algorithms on the same instance.
    convergence_data: {
    "QIG": {"mean": [...], "std": [...]},
    "RIG": {"mean": [...], "std": [...]},
    ...}
    """
    fig, ax = plt.subplots(figsize=figsize)
    algorithms = _sort_algos(list(convergence_data.keys()))

    for algo in algorithms:
        data = convergence_data[algo]
        mean = np.array(data["mean"])
        iters = np.arange(len(mean))
        color = _algo_color(algo)
        lw = 2.5 if algo == "QIG" else 1.5

        ax.plot(iters, mean, label=algo, color=color, linewidth=lw)
        if "std" in data:
            std = np.array(data["std"])
            ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Best Cmax")
    ax.set_title(title or f"Convergence: {instance_name}", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def collect_convergence_from_algo(
    p: list,
    algo_name: str,
    make_solver_fn,
    n_runs: int = 5,
    time_limit_ms_val: int = 5000,
    seed0: int = 1000,
) -> Dict[str, List[float]]:
    """
    Run multiple times of a given algorithm on the same instance to collect convergence data (best Cmax at each iteration).
    """
    import random
    from ig import IteratedGreedyAlgorithm

    all_series = []
    for run_id in range(n_runs):
        seed = seed0 + run_id
        random.seed(seed)
        np.random.seed(seed)

        # Đọc strategy và d từ tên
        if algo_name == "QIG":
            strategy, d = "qlearning", 1
        elif algo_name == "RIG":
            strategy, d = "random", 1
        elif algo_name.startswith("IIG"):
            strategy, d = "individual", int(algo_name[3:])
        else:
            raise ValueError(f"Unknown algo: {algo_name}")

        algo = IteratedGreedyAlgorithm(p, strategy=strategy, d=d)
        algo.execute(
            stopping_criterion="CPU_time",
            runtime_in_miliseconds=time_limit_ms_val,
            max_iteration=float("inf"),
        )
        all_series.append(algo.best_fitness_list)

    # Pad to same length
    max_len = max(len(s) for s in all_series)
    padded = [s + [s[-1]] * (max_len - len(s)) for s in all_series]
    arr = np.array(padded, dtype=float)

    return {
        "mean": arr.mean(axis=0).tolist(),
        "std":  arr.std(axis=0).tolist(),
    }


# 3. Operator Usage

def operator_usage_timeline(
    actions_sequence: List[int],
    operator_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 2.5),
    title: str = "",
    window: int = 50,
) -> plt.Figure:
    """
    plot operator usage over time
    """
    if operator_labels is None:
        n_ops = max(actions_sequence) + 1
        operator_labels = [f"d={i+1}" for i in range(n_ops)]

    n_ops = len(operator_labels)
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"][:n_ops]

    # One-hot encode
    arr = np.zeros((len(actions_sequence), n_ops))
    for t, a in enumerate(actions_sequence):
        arr[t, a] = 1.0

    # Rolling mean
    if window > 1 and len(actions_sequence) >= window:
        rolled = pd.DataFrame(arr).rolling(window, min_periods=1).mean().values
    else:
        rolled = arr

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(actions_sequence))

    for i, (label, color) in enumerate(zip(operator_labels, colors)):
        lw = 2 if i == 0 else 1.5
        ax.plot(x, rolled[:, i], label=label, color=color, linewidth=lw, alpha=0.85)

    ax.set_xlabel("Episode index")
    ax.set_ylabel("Usage frequency (rolling avg)")
    ax.set_title(title or "Operator usage over time (QIG)", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# 4. ARPD Bar Chart

def arpd_bar(
    arpd_df: pd.DataFrame,
    metric: str = "AvS",
    figsize: Tuple[int, int] = (8, 4),
    title: str = "",
) -> plt.Figure:
    """
    Bar chart: comparison of ARPD
    """
    summary = arpd_df.groupby("algorithm")[metric].mean().reset_index()
    summary = summary.sort_values(metric)
    algorithms = summary["algorithm"].tolist()
    values = summary[metric].tolist()
    colors = [_algo_color(a) for a in algorithms]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(algorithms, values, color=colors, alpha=0.82, edgecolor="black", linewidth=0.6)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel(f"Average {metric} (%)")
    ax.set_title(title or f"Overall {metric} Comparison", fontweight="bold")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# 5. Q-table Heatmap 

def qtable_heatmap(
    q_matrix: np.ndarray,
    operator_labels: Optional[List[str]] = None,
    state_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (5, 2.5),
    title: str = "",
) -> plt.Figure:
    """
    Heatmap Q-table after training.
    """
    if operator_labels is None:
        operator_labels = [f"d={i+1}" for i in range(q_matrix.shape[1])]
    if state_labels is None:
        state_labels = ["s=0 (stuck)", "s=1 (improving)"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(q_matrix, cmap="RdYlGn", aspect="auto")
    fig.colorbar(im, ax=ax, label="Q-value")

    ax.set_xticks(range(q_matrix.shape[1]))
    ax.set_xticklabels(operator_labels)
    ax.set_yticks(range(q_matrix.shape[0]))
    ax.set_yticklabels(state_labels)

    # Annotate cells
    for i in range(q_matrix.shape[0]):
        for j in range(q_matrix.shape[1]):
            ax.text(j, i, f"{q_matrix[i, j]:.4f}", ha="center", va="center",
                    fontsize=10, color="black")

    ax.set_title(title or "Q-table (final state)", fontweight="bold")
    fig.tight_layout()
    return fig


# 6. Multi-instance convergence grid

def convergence_grid(
    conv_by_instance: Dict[str, Dict[str, Dict[str, List[float]]]],
    figsize_per: Tuple[int, int] = (5, 3),
    title: str = "Convergence Rate",
) -> plt.Figure:
    """
    plot convergence curves for multiple instances in a grid layout. conv_by_instance: {
    "instance1": {
        "QIG": {"mean": [...], "std": [...]},
        "RIG": {"mean": [...], "std": [...]},
        ...
    },
    "instance2": {
        ...
    }
    """
    instances = list(conv_by_instance.keys())
    n = len(instances)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fw = figsize_per[0] * ncols
    fh = figsize_per[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
    axes = np.array(axes).flatten()

    for idx, inst in enumerate(instances):
        ax = axes[idx]
        data = conv_by_instance[inst]
        algorithms = _sort_algos(list(data.keys()))

        for algo in algorithms:
            mean = np.array(data[algo]["mean"])
            iters = np.arange(len(mean))
            color = _algo_color(algo)
            lw = 2.2 if algo == "QIG" else 1.4
            ax.plot(iters, mean, label=algo, color=color, linewidth=lw)

        ax.set_title(inst, fontsize=10)
        ax.set_xlabel("Iterations", fontsize=9)
        ax.set_ylabel("Best Cmax", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    for idx in range(len(instances), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


if __name__ == "__main__":
    # Fake convergence data
    np.random.seed(42)
    algos = ["QIG", "RIG", "IIG1", "IIG2", "IIG3"]
    fake_conv: Dict[str, Dict[str, List[float]]] = {}
    for i, algo in enumerate(algos):
        start = 6000 - i * 50
        noise = np.random.randn(200) * 20
        mean_curve = np.maximum(
            start - np.linspace(0, 400 + i * 30, 200) + np.cumsum(noise) * 0.1,
            5000
        )
        fake_conv[algo] = {
            "mean": mean_curve.tolist(),
            "std":  (np.abs(noise) * 2).tolist(),
        }

    fig = convergence_curve(fake_conv, instance_name="tai_100_20 (demo)")
    fig.savefig("/tmp/demo_convergence.png", dpi=120)
    print("Saved /tmp/demo_convergence.png")

    # Fake Q-table
    q = np.array([[0.012, 0.045, 0.023], [0.031, 0.018, 0.062]])
    fig2 = qtable_heatmap(q, title="Q-table after 1000 episodes (demo)")
    fig2.savefig("/tmp/demo_qtable.png", dpi=120)
    print("Saved /tmp/demo_qtable.png")

    # Fake operator usage
    actions = np.random.choice([0, 1, 2], size=300, p=[0.5, 0.3, 0.2]).tolist()
    fig3 = operator_usage_timeline(actions, title="Operator usage demo")
    fig3.savefig("/tmp/demo_operator.png", dpi=120)
    print("Saved /tmp/demo_operator.png")
