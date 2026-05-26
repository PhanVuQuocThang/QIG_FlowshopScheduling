"""
demo.py
---
QIG Flowshop Scheduling Demo
Streamlit web app: convergence race + Gantt chart
Run: streamlit run demo.py
"""
import random, threading, time, tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ig import IteratedGreedyAlgorithm
from solution import _makespan_np
from benchmark import parse_taillard_file, parse_vrf_gap_file

st.set_page_config(page_title="QIG Flowshop Scheduler", page_icon="", layout="wide")

ALGO_CONFIGS = {
    "QIG":  dict(strategy="qlearning",  d=1, color="#E63946"),
    "RIG":  dict(strategy="random",     d=1, color="#457B9D"),
    "IIG1": dict(strategy="individual", d=1, color="#2A9D8F"),
    "IIG2": dict(strategy="individual", d=2, color="#E9C46A"),
    "IIG3": dict(strategy="individual", d=3, color="#F4A261"),
}
JOB_COLORS = px.colors.qualitative.Plotly + px.colors.qualitative.Safe
CHART_HEIGHT = 400

# - Helpers -
def gen_instance(n: int, m: int, seed: int) -> np.ndarray:
    return np.random.RandomState(seed).randint(1, 100, size=(m, n)).astype(np.int64)

def detect_and_parse(content: bytes, filename: str):
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as f:
        f.write(content); tmp_path = f.name
    try:
        inst = parse_taillard_file(tmp_path, name=Path(filename).stem)
        return np.array(inst.p, dtype=np.int64), inst.n, inst.m, inst.name, inst.upper_bound, inst.lower_bound
    except Exception:
        pass
    try:
        inst = parse_vrf_gap_file(tmp_path, name=Path(filename).stem)
        return np.array(inst.p, dtype=np.int64), inst.n, inst.m, inst.name, inst.upper_bound, inst.lower_bound
    except Exception as e:
        raise ValueError(f"Cannot parse '{filename}': {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def run_one(name: str, cfg: dict, p_np: np.ndarray, time_ms: int, seed: int) -> dict:
    """Run a single algo once. Returns result dict."""
    random.seed(seed); np.random.seed(seed)
    algo = IteratedGreedyAlgorithm(p_np.tolist(), strategy=cfg["strategy"], d=cfg["d"])
    algo.execute(stopping_criterion="CPU_time", runtime_in_miliseconds=time_ms,
                 max_iteration=float("inf"))
    return {
        "fitness_list": algo.best_fitness_list[2:],  # strip 2 pre-loop NEH entries
        "best_perm":    algo.best_solution.perm,
        "best_cmax":    algo.best_solution.cmax,
        "iters":        algo.iterations,
    }

def run_algo_multi(name: str, cfg: dict, p_np: np.ndarray, time_ms: int,
                   n_runs: int, seed0: int, results: dict):
    """Run algo n_runs times, store aggregated results."""
    cmaxes, iters_list, all_fitness = [], [], []
    best_perm_overall, best_cmax_overall = None, float("inf")

    for run_id in range(n_runs):
        r = run_one(name, cfg, p_np, time_ms, seed0 + run_id)
        cmaxes.append(r["best_cmax"])
        iters_list.append(r["iters"])
        all_fitness.append(r["fitness_list"])
        if r["best_cmax"] < best_cmax_overall:
            best_cmax_overall = r["best_cmax"]
            best_perm_overall = r["best_perm"]
        # Update partial results after each run so live chart works
        results[name] = _aggregate(name, cmaxes, iters_list, all_fitness,
                                   best_perm_overall, best_cmax_overall)

def _aggregate(name, cmaxes, iters_list, all_fitness, best_perm, best_cmax):
    # Pad fitness lists to same length then average
    max_len = max(len(f) for f in all_fitness)
    padded  = [f + [f[-1]] * (max_len - len(f)) for f in all_fitness]
    arr     = np.array(padded, dtype=float)
    return {
        "fitness_list": arr.mean(axis=0).tolist(),   # mean convergence
        "best_perm":    best_perm,
        "best_cmax":    best_cmax,                   # BS
        "avs_cmax":     float(np.mean(cmaxes)),      # AvS (raw cmax mean)
        "iters":        int(np.mean(iters_list)),
        "n_runs_done":  len(cmaxes),
    }

# - Chart builders -
def build_convergence(results: dict, selected: List[str], upper_bound=None) -> go.Figure:
    fig = go.Figure()
    for name in selected:
        if name not in results: continue
        fl = results[name]["fitness_list"]
        if not fl: continue
        step = max(1, len(fl) // 3000)
        x = list(range(0, len(fl), step))
        y = [fl[i] for i in x]
        cfg = ALGO_CONFIGS[name]
        n_done = results[name].get("n_runs_done", 1)
        label  = f"{name} (run {n_done})" if n_done > 1 else name
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=label,
            line=dict(color=cfg["color"], width=2.5 if name == "QIG" else 1.8),
        ))
    if upper_bound:
        fig.add_hline(y=upper_bound, line_dash="dot", line_color="#888",
                      annotation_text=f"UB={upper_bound}",
                      annotation_position="bottom right",
                      annotation_font=dict(color="#888", size=10))
    fig.update_layout(
        xaxis_title="Iteration", yaxis_title="Best Cmax (mean across runs)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        height=CHART_HEIGHT,
        margin=dict(l=60, r=20, t=40, b=50),
        plot_bgcolor="#1a1a1a", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#333")
    fig.update_yaxes(showgrid=True, gridcolor="#333")
    return fig

def build_gantt(p_np: np.ndarray, perm: List[int], title: str = "") -> go.Figure:
    m = p_np.shape[0]
    c = np.zeros((m, len(perm)), dtype=np.int64)
    for j_idx, j in enumerate(perm):
        for i in range(m):
            if i == 0 and j_idx == 0:   c[i,j_idx] = p_np[i,j]
            elif i == 0:                 c[i,j_idx] = c[i,j_idx-1] + p_np[i,j]
            elif j_idx == 0:             c[i,j_idx] = c[i-1,j_idx] + p_np[i,j]
            else:                        c[i,j_idx] = max(c[i-1,j_idx], c[i,j_idx-1]) + p_np[i,j]
    fig = go.Figure()
    machines = [f"M{i+1}" for i in range(m)]
    for j_idx, j in enumerate(perm):
        color  = JOB_COLORS[j_idx % len(JOB_COLORS)]
        starts = [int(c[i, j_idx] - p_np[i, j]) for i in range(m)]
        widths = [int(p_np[i, j]) for i in range(m)]
        labels = [f"J{j+1}" if widths[i] >= 3 else "" for i in range(m)]
        fig.add_trace(go.Bar(
            name=f"J{j+1}", x=widths, y=machines, base=starts,
            orientation="h",
            marker_color=color, marker_line=dict(color="white", width=0.5),
            text=labels, textposition="inside", insidetextanchor="middle",
            textfont=dict(size=9, color="white"), showlegend=False,
            hovertemplate=f"Job {j+1}<br>Start: %{{base}}<br>Duration: %{{x}}<extra></extra>",
        ))
    cmax = int(c[-1, -1])
    fig.add_vline(x=cmax, line_dash="dash", line_color="#E63946", line_width=2,
                  annotation_text=f"Cmax = {cmax}",
                  annotation_font=dict(color="#E63946", size=12),
                  annotation_position="top right")
    fig.update_layout(
        barmode="overlay",
        title=dict(text=title, font=dict(size=13)) if title else None,
        xaxis=dict(title="Time", showgrid=True, gridcolor="#333", zeroline=False),
        yaxis=dict(title="Machine", categoryorder="array",
                   categoryarray=machines[::-1], showgrid=False),
        height=CHART_HEIGHT,
        margin=dict(l=60, r=20, t=40 if title else 20, b=40),
        plot_bgcolor="#1a1a1a", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    return fig

def empty_fig(msg: str) -> go.Figure:
    return go.Figure().update_layout(
        height=CHART_HEIGHT, plot_bgcolor="#1a1a1a", paper_bgcolor="#0e1117",
        annotations=[dict(text=msg, showarrow=False,
                          font=dict(size=15, color="#555"),
                          xref="paper", yref="paper", x=0.5, y=0.5)],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

def rpd(cmax, ub):
    return round((cmax / ub - 1) * 100, 4) if ub else None

# - UI -
st.title("QIG Flowshop Scheduling - Live Demo")
st.markdown("*Permutation Flowshop Scheduling with Q-learning based Iterated Greedy (Karimi-Mamaghan et al., 2023)*")

with st.sidebar:
    st.header("Instance")
    input_mode = st.radio("Source", ["Random", "Upload file"], horizontal=True)

    p_np, inst_name, upper_bound, lower_bound = None, "random", None, None
    n_jobs, n_mach = 20, 5

    if input_mode == "Random":
        n_jobs    = st.slider("Jobs (n)", 5, 200, 50, step=5)
        n_mach    = st.slider("Machines (m)", 2, 20, 10)
        inst_seed = st.number_input("Instance seed", value=42, step=1)
        p_np      = gen_instance(n_jobs, n_mach, int(inst_seed))
        inst_name = f"random_{n_jobs}x{n_mach}"
        st.caption(f"Processing times: {p_np.min()} - {p_np.max()}")
    else:
        uploaded = st.file_uploader("Upload instance file", type=None,
                                    help="Taillard or VRF Gap format - auto-detected")
        if uploaded is not None:
            try:
                p_np, n_jobs, n_mach, inst_name, upper_bound, lower_bound = \
                    detect_and_parse(uploaded.read(), uploaded.name)
                st.success(f"{inst_name} - {n_jobs} jobs x {n_mach} machines")
                if upper_bound: st.caption(f"Upper bound: {upper_bound}")
                if lower_bound: st.caption(f"Lower bound: {lower_bound}")
            except Exception as e:
                st.error(f"Parse error: {e}")
        else:
            st.info("Upload a Taillard or VRF instance file")

    st.divider()
    st.header("Algorithms")
    selected_algos = []
    for name, cfg in ALGO_CONFIGS.items():
        default = name in ["QIG", "RIG", "IIG1"]
        if st.checkbox(name, value=default,
                       help=f"strategy={cfg['strategy']}, d={cfg['d']}"):
            selected_algos.append(name)

    st.divider()
    st.header("Run settings")
    time_limit_s = st.slider("Time limit per run (s)", 1, 120, 10)
    n_runs       = st.slider("Number of runs", 1, 30, 1,
                             help="1 = single run (fastest). More runs = stable AvS/BS stats.")
    run_seed     = st.number_input("Base seed", value=42, step=1)
    run_btn      = st.button("Run", type="primary", use_container_width=True,
                             disabled=(p_np is None or not selected_algos))

# - Layout -
col_conv, col_gantt = st.columns(2, gap="medium")
with col_conv:
    st.subheader("Convergence Race")
    conv_ph = st.empty()
with col_gantt:
    st.subheader("Best Schedule (Gantt)")
    gantt_ph = st.empty()

status_ph   = st.empty()
progress_ph = st.empty()
results_ph  = st.empty()

conv_ph.plotly_chart(empty_fig("Press Run to start"), use_container_width=True, key="c0")
gantt_ph.plotly_chart(empty_fig("Gantt appears after run"), use_container_width=True, key="g0")

# - Run -
if run_btn and p_np is not None:
    time_ms = time_limit_s * 1000
    results: Dict = {}
    total_work = len(selected_algos) * n_runs

    # Each algo runs n_runs sequentially in its own thread
    threads = [
        threading.Thread(
            target=run_algo_multi,
            args=(name, ALGO_CONFIGS[name], p_np, time_ms, n_runs, int(run_seed), results),
            daemon=True,
        )
        for name in selected_algos
    ]
    for t in threads: t.start()

    t_start = time.time()
    total_time_est = time_limit_s * n_runs
    key = 0

    while any(t.is_alive() for t in threads):
        elapsed  = time.time() - t_start
        done_runs = sum(r.get("n_runs_done", 0) for r in results.values())
        frac     = min(done_runs / total_work, 0.99) if total_work > 0 else 0
        progress_ph.progress(frac)
        status_ph.markdown(
            f"Running... `{elapsed:.0f}s`"
            f"  |  {done_runs}/{total_work} algo-runs done"
            f"  |  {len(results)}/{len(selected_algos)} algos started"
        )
        if results:
            conv_ph.plotly_chart(
                build_convergence(results, selected_algos, upper_bound),
                use_container_width=True, key=f"c{key}"
            )
            best_name = min(results, key=lambda k: results[k]["best_cmax"])
            gantt_ph.plotly_chart(
                build_gantt(p_np, results[best_name]["best_perm"],
                            title=f"Best so far: {best_name} - Cmax={results[best_name]['best_cmax']}"),
                use_container_width=True, key=f"g{key}"
            )
        key += 1
        time.sleep(0.5)

    for t in threads: t.join()
    progress_ph.progress(1.0)
    status_ph.success(
        f"Done! {inst_name} | {n_jobs} jobs x {n_mach} machines"
        f" | {n_runs} run(s) per algo"
    )

    # Final charts
    conv_ph.plotly_chart(
        build_convergence(results, selected_algos, upper_bound),
        use_container_width=True, key="c_final"
    )
    best_name = min(results, key=lambda k: (results[k]["best_cmax"], results[k]["iters"]))
    gantt_ph.plotly_chart(
        build_gantt(p_np, results[best_name]["best_perm"],
                    title=f"Best: {best_name} - Cmax={results[best_name]['best_cmax']}"),
        use_container_width=True, key="g_final"
    )

    # Results table
    with results_ph.container():
        st.subheader("Results")

        rows = sorted(
            [dict(
                Algorithm  = n,
                BS         = results[n]["best_cmax"],
                AvS        = round(results[n].get("avs_cmax", results[n]["best_cmax"]), 2),
                Runs       = results[n].get("n_runs_done", 1),
                Iters      = results[n]["iters"],
                BS_RPD     = rpd(results[n]["best_cmax"], upper_bound),
                AvS_RPD    = rpd(results[n].get("avs_cmax", results[n]["best_cmax"]), upper_bound),
            ) for n in selected_algos if n in results],
            key=lambda r: (r["BS"], r["AvS"])
        )

        cols = st.columns(len(rows))
        for col, row in zip(cols, rows):
            is_best = row == rows[0]
            label   = f"🥇 {row['Algorithm']}" if is_best else row["Algorithm"]

            # Primary ranking metric: BS (paper Table 3-5, col BS)
            bs_rpd_str = f" (RPD {row['BS_RPD']:.4f}%)" if row["BS_RPD"] is not None else ""
            bs_str = f"BS = {row['BS']}{bs_rpd_str}"

            # Secondary: AvS; iters shown for reference only, not used in ranking
            avs_rpd_str = f" (RPD {row['AvS_RPD']:.4f}%)" if row["AvS_RPD"] is not None else ""
            delta_str = f"AvS = {row['AvS']}{avs_rpd_str}  |  {row['Iters']:,} iters/run  |  {row['Runs']} run(s)"

            col.metric(label=label, value=bs_str, delta=delta_str, delta_color="off",
                       help="Ranked by BS then AvS (per paper). Iters shown for reference only.")