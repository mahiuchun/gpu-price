"""Figures for the findings. Reads only files already written by the runs.

    uv run python -m gpuval.figures

Colours: three validated categorical slots plus a neutral for the reference forecast; every series
is also named by a direct label or legend, never by colour alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data

OUT = data.REPO / "findings" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(data.REPO))


def fig_blind_weeks(rep: dict) -> None:
    """Blind skill by week (real world): drift, the engine with no research, the scientist's live version."""
    b = rep["worlds"]["real"]["gpt"]
    weeks = sorted(b["by_week"])
    series = [("Drift (reference)", NEUTRAL, [b["by_week"][w]["drift"] for w in weeks]),
              ("Engine, no research", BLUE, [b["by_week"][w]["v0"] for w in weeks]),
              ("Engine + AI scientist's kept ideas", ORANGE, [b["by_week"][w][f"v{w}"] for w in weeks])]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.62})
    x = np.arange(len(weeks))
    wbar = 0.24
    for i, (name, col, vals) in enumerate(series):
        ax.bar(x + (i - 1) * (wbar + 0.02), vals, wbar, color=col, label=name, edgecolor=SURFACE, linewidth=2)
        for xx, v in zip(x + (i - 1) * (wbar + 0.02), vals):
            ax.text(xx, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    labels = {"1": "Week 1\n17–23 Oct", "2": "Week 2\n24–30 Oct", "3": "Week 3\n31 Oct–6 Nov"}
    ax.set_xticks(x, [labels.get(w, w) for w in weeks])
    ax.set_ylabel("Skill against a flat last value\n(higher is better)")
    ax.set_title("Skill on three blind weeks")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5, columnspacing=1.2, handlelength=1.2)
    rows = [("Engine, no research", BLUE, b["against_drift"]["v0_no_research"]),
            ("Engine + AI scientist", ORANGE, b["against_drift"]["live"]),
            ("Engine + random changes", AQUA, rep["worlds"]["real"]["random"]["against_drift"]["live"])]
    for i, (name, col, d) in enumerate(rows[::-1]):
        ax2.plot([d["lo90"] * 100, d["hi90"] * 100], [i, i], color=col, linewidth=2, solid_capstyle="round")
        ax2.plot(d["skill"] * 100, i, "o", color=col, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax2.text(d["hi90"] * 100 + 0.6, i, f"{d['skill'] * 100:+.1f}%", va="center", fontsize=9, color=INK)
    ax2.axvline(0, color=INK2, linewidth=1)
    ax2.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax2.set_xlabel("All three weeks, %  (dot = estimate, bar = 90% interval)")
    ax2.set_title("Error cut against drift")
    ax2.grid(axis="y", visible=False)
    ax2.set_xlim(right=max(r[2]["hi90"] for r in rows) * 100 + 6)
    _save(fig, "fig1_blind_weeks.png")


def fig_real_vs_placebo(rep: dict) -> None:
    rows = rep["real_vs_placebo"]["gpt"]
    order = ["real"] + sorted(r["world"] for r in rows if r["world"] != "real")
    by = {r["world"]: r for r in rows}
    names = {"real": "Real world"} | {f"placebo{i}": f"Shuffled world {i}" for i in range(1, 9)}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    for ax, key, title, xl in ((axes[0], "v0_vs_drift", "Engine against drift (no research)", "Error cut against drift, %"),
                               (axes[1], "research_gain", "What the scientist's kept ideas added", "Change in error from research, %  (right = better)")):
        for i, w in enumerate(order[::-1]):
            col = BLUE if w == "real" else NEUTRAL
            ax.plot(by[w][key] * 100, i, "o", color=col, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(by[w][key] * 100 + 0.35, i, f"{by[w][key] * 100:+.1f}", va="center", fontsize=9, color=INK)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(range(len(order)), [names[w] for w in order[::-1]])
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    lo = min(by[w]["research_gain"] for w in order) * 100
    axes[1].set_xlim(lo - 1.5, max(1.0, max(by[w]["research_gain"] for w in order) * 100) + 2.0)
    axes[0].set_xlim(0, max(by[w]["v0_vs_drift"] for w in order) * 100 + 2.5)
    _save(fig, "fig2_real_vs_shuffled.png")


def fig_what_moves() -> None:
    p = data.REPO / "findings" / "data" / "movement_by_class.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).set_index("cls")
    names = {"neocloud_rate_card": "Neocloud list prices", "neocloud_spot": "Neocloud spot",
             "hyperscaler_spot": "Hyperscaler spot", "marketplace_ask": "Marketplace asks"}
    order = [c for c in ("neocloud_rate_card", "neocloud_spot", "marketplace_ask", "hyperscaler_spot") if c in d.index]
    parts = [("from_list_changes", "Offers joining or leaving the list", BLUE),
             ("from_frequent_movers", "Offers that reprice often", ORANGE),
             ("from_rare_movers", "Offers that rarely reprice", AQUA)]
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for key, label, col in parts:
        vals = d.loc[order, key].to_numpy() * 100
        ax.barh(y, vals, left=left, height=0.5, color=col, label=label, edgecolor=SURFACE, linewidth=2)
        for yy, l, v in zip(y, left, vals):
            if v >= 9:
                ax.text(l + v / 2, yy, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        left += vals
    ax.set_yticks(y, [f"{names[c]}  ({int(d.loc[c, 'n_series'])} series)" for c in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of day-to-day movement in the published median")
    ax.set_title("What moves a published median price")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=9)
    _save(fig, "fig3_what_moves_the_median.png")


def fig_rollforward() -> None:
    p = data.REPO / "findings" / "data" / "rollout_by_horizon.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    x = np.arange(len(d))
    ax.plot(x, d.inside, "-o", color=BLUE, linewidth=2, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label="Engine rolled forward day by day")
    ax.plot(x, d.drift_inside, "-o", color=NEUTRAL, linewidth=2, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label="Drift (reference)")
    ax.axhline(0.8, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(d) - 1, 0.815, "target 80%", ha="right", fontsize=9, color=INK2)
    ax.text(x[-1] + 0.12, d.inside.iloc[-1], f"{d.inside.iloc[-1]:.0%}", va="center", fontsize=9)
    ax.text(x[0] - 0.12, d.inside.iloc[0], f"{d.inside.iloc[0]:.0%}", va="center", ha="right", fontsize=9)
    ax.set_xticks(x, [f"{h}\n({n} series)" for h, n in zip(d.h, d.n_series)])
    ax.set_xlabel("Days ahead (series with enough history to test)")
    ax.set_ylabel("Truth inside the forecast range")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(d) - 0.4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Rolled forward, the ranges stop being trustworthy within a week or two")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, "fig4_roll_forward.png")


def fig_rl() -> None:
    log = data.REPO / "rl" / "runs" / "full" / "log.json"
    if not log.exists():
        return
    L = json.loads(log.read_text())
    r = pd.DataFrame(L["rounds"])
    blind_p = data.REPO / "rl" / "runs" / "blind.json"
    fig, axes = plt.subplots(1, 2 if blind_p.exists() else 1, figsize=(12 if blind_p.exists() else 6.5, 3.6),
                             gridspec_kw={"wspace": 0.6} if blind_p.exists() else None)
    ax = axes[0] if blind_p.exists() else axes
    real = [x.get("real", np.nan) for x in r.by_kind]
    plac = [x.get("placebo", np.nan) for x in r.by_kind]
    ax.plot(r["round"], real, "-", color=BLUE, linewidth=2, label="Practice worlds from real data")
    ax.plot(r["round"], plac, "-", color=NEUTRAL, linewidth=2, label="Shuffled practice worlds")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Training round (64 tries each)")
    ax.set_ylabel("Average error cut per idea, %")
    ax.set_title("RL training: reward on practice worlds")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    if blind_p.exists():
        B = json.loads(blind_p.read_text())
        ax2 = axes[1]
        arms = list(B["arms"])
        cols = {"small model, untrained": NEUTRAL, "small model, RL-trained": BLUE, "random control": AQUA}
        vals = [B["arms"][a]["all"]["mean_change_pct"] for a in arms]
        y = np.arange(len(arms))[::-1]
        ax2.barh(y, vals, height=0.45, color=[cols.get(a, ORANGE) for a in arms], edgecolor=SURFACE, linewidth=2)
        for yy, v, a in zip(y, vals, arms):
            ax2.text(max(v, 0) + 0.06, yy, f"{v:+.2f}%  ({B['arms'][a]['all']['share_helpful']:.0%} of ideas helped)",
                     va="center", ha="left", fontsize=9)
        ax2.axvline(0, color=INK2, linewidth=1)
        ax2.set_yticks(y, [a[0].upper() + a[1:] for a in arms])
        ax2.set_xlabel("Average error cut per idea on blind weeks, %")
        ax2.set_title("Blind test: 48 single ideas per proposer")
        ax2.grid(axis="y", visible=False)
        span = max(abs(min(vals)), abs(max(vals)), 0.5)
        ax2.set_xlim(-span * 0.6, span * 2.9)
    _save(fig, "fig5_rl.png")


def fig_three_tests() -> None:
    """Engine, ML model and AI agents against drift in three separate tests, with 90% intervals."""
    d = data.REPO / "findings" / "data"
    if not (d / "aws_table.json").exists():
        return
    t = json.loads((d / "periods_table.json").read_text())["blocks"]
    ad = json.loads((d / "periods_agent_days.json").read_text())
    aws = json.loads((d / "aws_table.json").read_text())
    before, test = t["before the test weeks (21 Aug to 15 Oct)"], t["test weeks (16 Oct to 5 Nov)"]
    groups = ["Our 37 prices\n8 earlier weeks", "Our 37 prices\n3 test weeks", "AWS spot prices\n4 years"]
    series = [("Engine", BLUE, [before["engine"]["vs_drift"], test["engine"]["vs_drift"], aws["engine"]["vs_drift"]]),
              ("ML model (Chronos-2)", ORANGE, [before["ml"]["vs_drift"], test["ml"]["vs_drift"], aws["ml"]["vs_drift"]]),
              ("Five AI agents", AQUA, [ad["before"]["agents"]["vs_drift"], ad["test"]["agents"]["vs_drift"], aws["agents"]["vs_drift"]])]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(groups)); w = 0.25
    for i, (name, col, vals) in enumerate(series):
        xs = x + (i - 1) * (w + 0.02)
        v = np.array([100 * q["skill"] for q in vals]); lo = np.array([100 * q["lo90"] for q in vals]); hi = np.array([100 * q["hi90"] for q in vals])
        ax.bar(xs, v, w, color=col, label=name, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
        for xx, vv, hh in zip(xs, v, hi):
            ax.text(xx, max(hh, vv, 0) + 1.2, f"{vv:+.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Less error than drift\n(bar = estimate, line = 90% interval)")
    ax.set_title("Three separate tests")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    _save(fig, "fig6_three_tests.png")


def fig_self_improvement() -> None:
    """The two improvement loops that held up: the coach's changes to the agent crowd, and RL training
    of an open model. Everything is scored on days the loop never saw."""
    c, r = data.REPO / "runs" / "coach", data.REPO / "rl" / "runs"
    if not (c / "confirm.json").exists() or not (r / "fc1" / "eval.json").exists():
        return
    exam, conf = json.loads((c / "exam.json").read_text()), json.loads((c / "confirm.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})

    def bars(ax, xs, vals, col, label, w, alpha=1.0, hatch=None):
        v = np.array([100 * q["skill"] for q in vals]); lo = np.array([100 * q["lo90"] for q in vals]); hi = np.array([100 * q["hi90"] for q in vals])
        ax.bar(xs, v, w, color=col, label=label, edgecolor=SURFACE, linewidth=2, alpha=alpha, hatch=hatch)
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
        for xx, vv, hh in zip(xs, v, hi):
            ax.text(xx, max(hh, vv, 0) + 1.0, f"{vv:+.0f}%", ha="center", va="bottom", fontsize=9, color=INK)

    blocks = [("Our prices\n5 fresh exam days", exam["ours"]), ("Our prices\n14 other days", conf["our prices, all 14 days"]),
              ("AWS\n3 fresh exam days", exam["aws"]), ("AWS\n12 other days", conf["AWS, 12 days"])]
    x = np.arange(len(blocks)); w = 0.36
    bars(a, x - w / 2 - 0.01, [k["original_vs_drift"] for _, k in blocks], NEUTRAL, "Five agents as first written", w)
    bars(a, x + w / 2 + 0.01, [k["improved_vs_drift"] for _, k in blocks], AQUA, "After the coach's two kept changes", w)
    a.set_xticks(x, [f"{n}\n({k['n']:,} forecasts)" for n, k in blocks], fontsize=8.5)
    a.axhline(0, color=INK2, linewidth=1)
    a.set_ylabel("Less error than drift\n(bar = estimate, line = 90% interval)")
    a.set_title("A coach AI rewrites the agent crowd")
    a.grid(axis="x", visible=False)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=9)

    runs = [(n, col, json.loads((r / d / "eval.json").read_text())["table"]) for n, col, d in (("Qwen3-8B", BLUE, "fc1"), ("Qwen3.5-9B", ORANGE, "fc2")) if (r / d / "eval.json").exists()]
    groups = [("ours", "Our 37 prices\n(never trained on)"), ("aws", "AWS, Fridays in 2026\n(after the training dates)")]
    x = np.arange(len(groups)); w = 0.8 / (2 * len(runs))
    for i, (name, col, t) in enumerate(runs):
        bars(b, x + (2 * i - len(runs) + 0.5) * (w + 0.01), [t[g]["base_vs_drift"] for g, _ in groups], col, f"{name} before training", w, alpha=0.4, hatch="//")
        bars(b, x + (2 * i - len(runs) + 1.5) * (w + 0.01), [t[g]["trained_vs_drift"] for g, _ in groups], col, f"{name} after RL training", w)
    b.set_xticks(x, [f"{lab}\n({runs[0][2][g]['n']:,} forecasts)" for g, lab in groups], fontsize=8.5)
    b.axhline(0, color=INK2, linewidth=1)
    b.set_title("RL training teaches an open model to forecast")
    b.grid(axis="x", visible=False)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=9)
    _save(fig, "fig7_self_improvement.png")


def fig_modal_rl() -> None:
    """Every RL training run of the open models, Tinker and Modal, on the same test days: how much the
    result depends on the random seed, the learning rate, the amount of practice and the model."""
    path = data.REPO / "findings" / "data" / "modal_rl.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    runs = d["runs"]

    def rate(lr: float) -> str:
        mant, exp = f"{lr:.1e}".split("e")
        return f"{float(mant):g}e{int(exp)}"

    def untrained(t: dict) -> list:
        return [(None, {w: {"trained_vs_drift": t[w]["base_vs_drift"]} for w in ("ours", "aws")})]

    cols = []  # (label, colour, [(seed, block)])
    t8, t9 = d["tinker"]["Qwen/Qwen3-8B"], d["tinker"]["Qwen/Qwen3.5-9B"]
    cols.append(("Untrained", NEUTRAL, untrained(t8)))
    cols.append(("Tinker\nrate 4e-5\n1 run", BLUE, [(0, t8)]))
    for lr in sorted({r["lr"] for r in runs if r["plan"] == "sweep1"}):
        tag = "\n(chosen)" if abs(lr - d["chosen_lr"]) < 1e-12 else ""
        cols.append((f"Modal\nrate {rate(lr)}{tag}", BLUE, [(r["seed"], r) for r in runs if r["plan"] == "sweep1" and r["lr"] == lr]))
    longer = [(r["seed"], r) for r in runs if r["plan"] == "longer"]
    if longer:
        cols.append(("Modal\nrate 1e-4\n3x practice", BLUE, longer))
    split = len(cols)  # Qwen3-8B to the left, Qwen3.5-9B to the right
    cols.append(("Untrained", NEUTRAL, untrained(t9)))
    cols.append(("Tinker\nrate 4e-5\n1 run", ORANGE, [(0, t9)]))
    nine = [r for r in runs if r["base_model"].endswith("Qwen3.5-9B")]
    for lr in sorted({r["lr"] for r in nine}):
        cols.append((f"Modal\nrate {rate(lr)}", ORANGE, [(r["seed"], r) for r in nine if r["lr"] == lr]))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.6), sharex=True)
    for ax, w, title in ((axes[0], "ours", "Our 37 prices (never practised on)"), (axes[1], "aws", "AWS, Fridays in 2026")):
        for i, (label, col, members) in enumerate(cols):
            for j, (seed, b) in enumerate(sorted(members, key=lambda m: (m[0] is None, m[0]))):
                q = b[w]["trained_vs_drift"]
                x = i + (j - (len(members) - 1) / 2) * 0.3
                left = len(members) > 1 and j == 0  # the first of a pair is labelled on its left, so labels never meet
                ax.errorbar([x], [100 * q["skill"]], yerr=[[100 * (q["skill"] - q["lo90"])], [100 * (q["hi90"] - q["skill"])]], fmt="none", ecolor=INK2, elinewidth=1.1, capsize=2.5)
                ax.scatter([x], [100 * q["skill"]], s=62, color=col, marker="o" if seed in (None, 0) else "s", edgecolor=SURFACE, linewidth=1.5, zorder=3)
                ax.text(x + (-0.09 if left else 0.09), 100 * q["skill"], f"{100 * q['skill']:+.0f}", fontsize=8.5, va="center", ha="right" if left else "left", color=INK)
        eng = 100 * t8[w]["engine_vs_drift"]["skill"]
        ax.axhline(eng, color=AQUA, linewidth=1.3, linestyle="--")
        ax.text(1.005, eng, f"engine\n{eng:+.0f}%", color=INK2, fontsize=8.5, va="center", ha="left", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.axvline(split - 0.5, color=INK2, linewidth=0.8, linestyle=":")
        for x0, name in ((-0.5, "Qwen3-8B"), (split - 0.5, "Qwen3.5-9B")):
            ax.text(x0 + 0.08, 0.985, name, transform=ax.get_xaxis_transform(), fontsize=9.5, fontweight="bold", color=INK2, va="top", ha="left")
        ax.set_xticks(range(len(cols)), [c[0] for c in cols], fontsize=8.5)
        ax.set_xlim(-0.5, len(cols) - 0.5)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.08 * (hi - lo))  # room for the model names
        ax.set_title(title)
        ax.set_ylabel("Less error than drift, %")
        ax.grid(axis="x", visible=False)
    axes[1].scatter([], [], s=62, color=INK2, marker="o", label="one training run, first random seed (same practice prices as the Tinker run)")
    axes[1].scatter([], [], s=62, color=INK2, marker="s", label="second random seed")
    axes[1].plot([], [], color=INK2, linewidth=1.1, label="90% interval")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    fig.subplots_adjust(hspace=0.16)
    _save(fig, "fig8_rl_runs.png")


def fig_range_memory() -> None:
    """The stretch the first crowd's range memory chose day by day, and what it was worth."""
    p = data.REPO / "findings" / "data" / "range_memory.json"
    if not p.exists():
        return
    rm = json.loads(p.read_text())
    path = pd.DataFrame(rm["crowd"]["path"])
    days = sorted(path.day.unique())
    pos = {d: i for i, d in enumerate(days)}
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7), gridspec_kw={"width_ratios": [1, 1, 1, 1.25], "wspace": 0.32})
    for ax, h in zip(axes[:3], (1, 3, 7)):
        g = path[path.h == h].sort_values("day")
        g = g[g.n_memory > 0] if h == 7 else g
        coach = rm["rule"]["coach_stretch"][str(h)]
        ax.axhline(coach, color=NEUTRAL, linewidth=1.6, linestyle=(0, (4, 3)))
        ax.text(0.98, coach - 0.07, f"coach, fixed: {coach:g}", ha="right", va="top", fontsize=8.5, color=INK2, transform=ax.get_yaxis_transform())
        ax.set_xlim(-0.8, len(days) - 0.2)
        ax.plot([pos[d] for d in g.day], g.stretch, "-o", color=BLUE, linewidth=2, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1)
        last = g.iloc[-1]
        ax.text(pos[last.day], last.stretch + 0.1, f"{last.stretch:g}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_ylim(0.4, 3.2)
        ticks = list(range(0, len(days), 6))
        ax.set_xticks(ticks, [f"{int(days[i][8:])} {'Sep' if days[i][5:7] == '09' else 'Oct' if days[i][5:7] == '10' else 'Nov'}" for i in ticks], fontsize=8.5)
        ax.set_title(f"{h} day{'s' if h > 1 else ''} ahead", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Range stretch chosen by the memory\n(1 = ranges as the agents gave them)")
    b = rm["crowd"]["all"]
    arms = [("raw", "As the agents\ngave it", NEUTRAL), ("coach", "Coach's\nfixed stretch", AQUA), ("mem", "Range\nmemory", BLUE)]
    ax = axes[3]
    for i, (k, name, col) in enumerate(arms):
        q = b["vs_drift"][k]
        v, lo, hi = 100 * q["skill"], 100 * q["lo90"], 100 * q["hi90"]
        ax.bar(i, v, 0.6, color=col, edgecolor=SURFACE, linewidth=2)
        ax.errorbar([i], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
        ax.text(i, hi + 0.8, f"{v:+.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(i, 1.0, f"inside\n{100 * b['inside'][k]:.0f}%", ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold")
    ax.set_xticks(range(3), [n for _, n, _ in arms], fontsize=8.5)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylabel("Less error than drift")
    ax.set_ylim(0, 26)
    ax.set_title(f"First crowd, {b['n']:,} forecasts, {b['n_days']} days", fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.suptitle("A memory of its own misses: ranges closer to the 80% target, no clear gain in error", x=0.065, ha="left", fontsize=11.5, fontweight="bold", y=1.04)
    _save(fig, "fig9_range_memory.png")


def fig_euro() -> None:
    """One provider's dollar price follows the euro: it changes after working days only."""
    p = data.REPO / "findings" / "data" / "checks.json"
    if not p.exists():
        return
    e = json.loads(p.read_text())["euro"]
    t = e["scaleway_price_changed_by_weekday"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.3})
    names = list(t)
    vals = [100 * t[d]["changed"] / t[d]["of"] for d in names]
    a.bar(range(len(names)), vals, 0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
    for i, d in enumerate(names):
        a.text(i, vals[i] + 2, f"{t[d]['changed']} of {t[d]['of']}", ha="center", va="bottom", fontsize=9, color=INK)
    a.set_xticks(range(len(names)), names)
    a.set_ylim(0, 112)
    a.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    a.set_ylabel("Days the price differed\nfrom the day before")
    a.set_title("Scaleway's dollar price, by weekday of the reading", fontsize=10)
    a.grid(axis="x", visible=False)
    f = e["one_day_forecasts_where_scaleway_is_the_middle"]
    keys = [("target Tue to Sat", "Target day\nTue to Sat"), ("target Sun or Mon", "Target day\nSun or Mon")]
    v2 = [100 * f[k]["price_moved"] / f[k]["n"] for k, _ in keys]
    b.bar(range(2), v2, 0.5, color=BLUE, edgecolor=SURFACE, linewidth=2)
    for i, (k, _) in enumerate(keys):
        b.text(i, v2[i] + 2, f"{f[k]['price_moved']} of {f[k]['n']}", ha="center", va="bottom", fontsize=9, color=INK)
    b.set_xticks(range(2), [n for _, n in keys])
    b.set_ylim(0, 112)
    b.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    b.set_ylabel("1-day forecasts where the\npublished price moved")
    b.set_title("Where Scaleway is the middle offer", fontsize=10)
    b.grid(axis="x", visible=False)
    fig.suptitle("A scored price that follows the euro: it moves the morning after a working day", x=0.07, ha="left", fontsize=11.5, fontweight="bold", y=1.03)
    _save(fig, "fig10_euro_weekday.png")


def fig_coverage() -> None:
    """Truth inside the range, split by whether the price moved at all."""
    p = data.REPO / "findings" / "data" / "checks.json"
    if not p.exists():
        return
    c = json.loads(p.read_text())["coverage"]["exam_days"]
    rows = [("drift", "Drift"), ("engine", "Engine"), ("first_crowd", "Five agents, as first written"), ("improved_crowd", "Five agents, after the coach")]
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    y = np.arange(len(rows))[::-1]
    hgt = 0.34
    for off, key, col, label in ((hgt / 2 + 0.02, "unchanged", NEUTRAL, f"Price did not change ({c['n'] - c['n_moved']} forecasts)"),
                                 (-hgt / 2 - 0.02, "moved", BLUE, f"Price moved ({c['n_moved']} forecasts)")):
        vals = [100 * c[k][key] for k, _ in rows]
        ax.barh(y + off, vals, height=hgt, color=col, label=label, edgecolor=SURFACE, linewidth=2)
        for yy, v in zip(y + off, vals):
            ax.text(v + 1, yy, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.axvline(80, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(81, len(rows) - 1.5, "target 80%", fontsize=9, color=INK2, va="center", bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_yticks(y, [f"{n}\n(all: {100 * c[k]['all']:.0f}%)" for k, n in rows], fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("Outcomes that landed inside the forecast range")
    ax.set_title("An overall figure near 80% hides what happens when the price moves")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    _save(fig, "fig11_coverage_when_moved.png")


def fig_agent_memory() -> None:
    """The five agents with and without a memory of their own forecasts: error against drift, and each arm against no memory."""
    p = data.REPO / "findings" / "data" / "agent_memory.json"
    if not p.exists():
        return
    b = json.loads(p.read_text())["blocks"]["all days"]
    fig, (a, c) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.3})
    arms = [("recorded", "No memory\n(answers of 14 Nov)", NEUTRAL), ("none", "No memory\n(asked again)", NEUTRAL),
            ("record", "Own scored\nrecord", BLUE), ("lesson", "Record plus\na written lesson", ORANGE)]
    for i, (k, name, col) in enumerate(arms):
        q = b["vs_drift"][k]
        v, lo, hi = 100 * q["skill"], 100 * q["lo90"], 100 * q["hi90"]
        a.bar(i, v, 0.6, color=col, edgecolor=SURFACE, linewidth=2, alpha=0.55 if k == "recorded" else 1.0)
        a.errorbar([i], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
        a.text(i, hi + 0.8, f"{v:+.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
        if k in b["inside"]:
            a.text(i, 1.0, f"inside\n{100 * b['inside'][k]:.0f}%", ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold")
    a.set_xticks(range(len(arms)), [n for _, n, _ in arms], fontsize=8.5)
    a.axhline(0, color=INK2, linewidth=1)
    a.set_ylabel("Less error than drift\n(bar = estimate, line = 90% interval)")
    a.set_title(f"Each arm against drift ({b['n']:,} forecasts, {b['n_days']} days)", fontsize=10)
    a.grid(axis="x", visible=False)
    rows = [("none_vs_recorded", "Asked again, against the\nanswers of 14 Nov (no change made)", NEUTRAL),
            ("record_vs_none", "Own scored record,\nagainst no memory", BLUE), ("lesson_vs_none", "Record plus lesson,\nagainst no memory", ORANGE),
            ("lesson_vs_record", "Lesson, against\nrecord alone", ORANGE)]
    for i, (k, name, col) in enumerate(rows[::-1]):
        q = b[k]
        c.plot([100 * q["lo90"], 100 * q["hi90"]], [i, i], color=col, linewidth=2, solid_capstyle="round")
        c.plot(100 * q["skill"], i, "o", color=col, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        c.text(100 * q["hi90"] + 0.3, i, f"{100 * q['skill']:+.1f}%", va="center", fontsize=9, color=INK)
    c.axvline(0, color=INK2, linewidth=1)
    c.set_yticks(range(len(rows)), [n for _, n, _ in rows[::-1]], fontsize=8.5)
    c.set_xlabel("Change in error on the same forecasts (right of zero = better)")
    c.set_title("One arm against another", fontsize=10)
    c.grid(axis="y", visible=False)
    c.margins(x=0.18)
    fig.suptitle("Do the agents forecast better when they can see their own past forecasts?", x=0.07, ha="left", fontsize=11.5, fontweight="bold", y=1.03)
    _save(fig, "fig12_agent_memory.png")


def main() -> int:
    rep = json.loads((data.REPO / "runs" / "record" / "report.json").read_text())
    fig_blind_weeks(rep)
    fig_real_vs_placebo(rep)
    fig_what_moves()
    fig_rollforward()
    fig_rl()
    fig_three_tests()
    fig_self_improvement()
    fig_modal_rl()
    fig_range_memory()
    fig_euro()
    fig_coverage()
    fig_agent_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
