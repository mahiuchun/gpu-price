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


def main() -> int:
    rep = json.loads((data.REPO / "runs" / "record" / "report.json").read_text())
    fig_blind_weeks(rep)
    fig_real_vs_placebo(rep)
    fig_what_moves()
    fig_rollforward()
    fig_rl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
