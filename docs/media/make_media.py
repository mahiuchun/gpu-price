"""README visuals. Reads only files already in the repo (no data pack, no keys, no network).

    uv run python docs/media/make_media.py

Writes into docs/media/:
  hero_scoreboard.png   every forecaster on the same fresh exam days, against drift
  forecast_6nov.png     the forecast handed in: 1/3/7-day ranges for the 37 prices
  engine_replay.gif/mp4 animated explainer of the engine (illustrative numbers, not real offers)
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "media"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(REPO))


# --------------------------------------------------------------------------- hero scoreboard
def hero_scoreboard() -> None:
    d = json.loads((REPO / "findings/data/fresh_days.json").read_text())["ours"]["vs_drift"]
    rows = [
        ("Qwen3-8B, untrained", "qwen3_8b_before_rl", NEUTRAL),
        ("Qwen3.5-9B, untrained", "qwen3_5_9b_before_rl", NEUTRAL),
        ("Engine (no AI)", "engine", BLUE),
        ("Qwen3-8B after RL", "qwen3_8b_after_rl", ORANGE),
        ("Qwen3.5-9B after RL", "qwen3_5_9b_after_rl", ORANGE),
        ("Five AI agents, as first written", "gpt_crowd_original", AQUA),
        ("Five AI agents + coach's 2 changes\n(the forecast handed in)", "gpt_crowd_improved", AQUA),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(rows))
    for i, (label, key, col) in enumerate(rows):
        s = d[key]
        est, lo, hi = 100 * s["skill"], 100 * s["lo90"], 100 * s["hi90"]
        final = key == "gpt_crowd_improved"
        ax.plot([lo, hi], [i, i], color=col, lw=6 if final else 4, alpha=0.28, solid_capstyle="round")
        ax.scatter([est], [i], s=150 if final else 90, color=col, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(hi + 1.0, i, f"{est:+.0f}%", va="center", fontsize=11 if final else 10,
                fontweight="bold" if final else "normal", color=INK if final else INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.text(0.4, len(rows) - 0.45, "drift (reference)", fontsize=8.5, color=INK2, va="bottom")
    ax.set_yticks(y, [r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[1] == "gpt_crowd_improved":
            t.set_fontweight("bold"); t.set_color(INK)
    ax.set_xlim(-14, 36)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Less error than drift, %   (dot = estimate, bar = 90% interval)")
    ax.set_title("Every forecaster on the same 479 fresh forecasts of our 37 GPU prices")
    fig.text(0.01, -0.04, "Pinball loss as the organisers' scorer computes it. Exam days no agent or coach had seen. "
             "Source: findings/data/fresh_days.json", fontsize=8, color=INK2)
    _save(fig, "hero_scoreboard.png")


# --------------------------------------------------------------------------- forecast handed in
def forecast_chart() -> None:
    f = pd.read_csv(REPO / "submission/forecast_2026-11-06.csv")
    f["gpu"] = f.series_id.str.split(".").str[2].str.replace("_", " ")
    f["cls"] = f.series_id.str.split(".").str[1].str.replace("_", " ")
    f["src"] = f.series_id.str.split(".").str[0]
    base = f[f.horizon_days == 1].set_index("series_id").q50
    f["lo"] = 100 * (f.q10 / f.series_id.map(base) - 1)
    f["mid"] = 100 * (f.q50 / f.series_id.map(base) - 1)
    f["hi"] = 100 * (f.q90 / f.series_id.map(base) - 1)
    w7 = f[f.horizon_days == 7].set_index("series_id")
    order = (w7.hi - w7.lo).sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 10.5))
    cols = {1: NEUTRAL, 3: BLUE, 7: ORANGE}
    off = {1: -0.25, 3: 0, 7: 0.25}
    for i, sid in enumerate(order):
        for h in (1, 3, 7):
            r = f[(f.series_id == sid) & (f.horizon_days == h)].iloc[0]
            ax.plot([r.lo, r.hi], [i + off[h]] * 2, color=cols[h], lw=3, solid_capstyle="round")
            ax.scatter([r.mid], [i + off[h]], color=cols[h], s=14, zorder=3, edgecolor=SURFACE, linewidth=0.6)
    lab = [f"{f[f.series_id == s].gpu.iloc[0]}  ·  {f[f.series_id == s].cls.iloc[0]}  ·  "
           f"{f[f.series_id == s].src.iloc[0].split('_')[0]}  (${base[s]:.2f}/h)" for s in order]
    ax.set_yticks(range(len(order)), lab, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Change from the 1-day middle forecast, %   (line = 10th to 90th percentile, dot = middle)")
    ax.set_title("The forecast handed in: 37 GPU rental prices from 6 Nov 2026")
    for h in (1, 3, 7):
        ax.plot([], [], color=cols[h], lw=3, label=f"{h} day{'s' if h > 1 else ''} ahead")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(-0.8, len(order) - 0.2)
    _save(fig, "forecast_6nov.png")


# --------------------------------------------------------------------------- engine animation
def engine_animation() -> None:
    """Illustrative: 9 offers, one past (d, d+h) pair at a time is replayed onto today's list."""
    rng = np.random.default_rng(7)
    today = np.array([1.79, 1.95, 2.05, 2.10, 2.19, 2.29, 2.40, 2.55, 2.89])
    names = [f"offer {c}" for c in "ABCDEFGHI"]
    n_pairs = 24
    pairs = []
    for _ in range(n_pairs):
        move = np.ones_like(today)
        rep = rng.random(len(today)) < 0.25                    # repriced
        move[rep] = 1 + rng.normal(0, 0.06, rep.sum())
        exits = rng.random(len(today)) < 0.08                 # dropped off the list
        join = rng.random() < 0.3                             # an offer rejoined
        pairs.append((move, exits, join, rng.uniform(1.7, 2.6)))

    def replay(k):
        move, exits, join, jp = pairs[k]
        p = today * move
        p = p[~exits]
        if join:
            p = np.append(p, jp)
        return p, move, exits, join, jp

    medians = [np.median(replay(k)[0]) for k in range(n_pairs)]
    binned = np.round(np.array(medians) / 0.02) * 0.02
    max_stack = max(int((binned == v).sum()) for v in np.unique(binned))
    q10, q50, q90 = np.quantile(medians, [0.1, 0.5, 0.9])

    fig = plt.figure(figsize=(10, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.25)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    hold = 4
    frames = [("intro", 0)] * 6 + [(("pair", k)) for k in range(n_pairs) for _ in range(hold if k < 4 else 2)] + [("done", 0)] * 18

    def draw(fr):
        kind, k = fr
        ax.clear(); ax2.clear()
        x = np.arange(len(today))
        ax.set_xlim(-0.8, len(today) + 0.6)
        ax.set_ylim(1.55, 3.1)
        ax.set_ylabel("$ per GPU-hour")
        ax.set_xticks(list(x) + [len(today)], names + ["joins"], fontsize=8, rotation=30)
        ax.grid(axis="x", visible=False)
        ax.axhline(np.median(today), color=INK2, lw=1, ls=":")
        ax.text(0.02, 0.97, f"····  today's median  ${np.median(today):.2f}", fontsize=8.5, color=INK2,
                transform=ax.transAxes, va="top")
        ax.bar(x, today, 0.62, color=GRID, edgecolor=NEUTRAL, linewidth=0.8)
        if kind == "intro":
            ax.set_title("1. Today's list of offers for one GPU")
            ax.bar(x[4], today[4], 0.62, color=BLUE)
            ax.text(x[4], today[4] + 0.03, "middle", ha="center", fontsize=8, color=BLUE, fontweight="bold")
        else:
            kk = k if kind == "pair" else n_pairs - 1
            p, move, exits, join, jp = replay(kk)
            ax.set_title(f"2. Replay past pair of days #{kk + 1} onto today's list")
            for i in x:
                if exits[i]:
                    ax.bar(i, today[i], 0.62, color=SURFACE, edgecolor=ORANGE, linewidth=1.5, hatch="///")
                    ax.text(i, today[i] + 0.03, "left", ha="center", fontsize=7.5, color=ORANGE)
                elif abs(move[i] - 1) > 1e-9:
                    ax.bar(i, today[i] * move[i], 0.62, color=ORANGE, alpha=0.85)
                    ax.text(i, today[i] * move[i] + 0.03, f"{100 * (move[i] - 1):+.0f}%", ha="center", fontsize=7.5, color=ORANGE)
            if join:
                ax.bar(len(today), jp, 0.62, color=AQUA)
                ax.text(len(today), jp + 0.03, "rejoined", ha="center", fontsize=7.5, color=AQUA)
            m = np.median(p)
            ax.axhline(m, color=BLUE, lw=2)
            ax.text(0.02, 0.91, f"——  replayed median  ${m:.3f}", fontsize=8.5, color=BLUE, va="top",
                    fontweight="bold", transform=ax.transAxes)
        # right panel: distribution of replayed medians
        ax2.set_title("3. Replayed medians → forecast range")
        ax2.set_xlim(1.95, 2.45)
        ax2.set_ylim(0, max_stack + 3)
        ax2.set_yticks([])
        ax2.set_xlabel("median in h days, $")
        ax2.grid(axis="y", visible=False)
        upto = 0 if kind == "intro" else (k + 1 if kind == "pair" else n_pairs)
        if upto:
            vals = np.round(np.array(medians[:upto]) / 0.02) * 0.02
            for v in np.unique(vals):
                c = int((vals == v).sum())
                ax2.scatter([v] * c, np.arange(c) + 0.5, s=60, color=BLUE, edgecolor=SURFACE, zorder=3)
        if kind == "done":
            ax2.axvspan(q10, q90, color=ORANGE, alpha=0.15)
            ax2.axvline(q50, color=ORANGE, lw=2)
            ax2.text(0.5, 0.96, f"forecast:  q10 ${q10:.2f} · q50 ${q50:.2f} · q90 ${q90:.2f}", ha="center",
                     fontsize=8.5, color=ORANGE, fontweight="bold", transform=ax2.transAxes,
                     bbox=dict(facecolor=SURFACE, edgecolor=ORANGE, boxstyle="round,pad=0.3"))
        fig.suptitle("How the engine forecasts a median price  (illustrative numbers)", x=0.06, ha="left",
                     fontsize=13, fontweight="bold", y=1.0)

    anim = FuncAnimation(fig, draw, frames=frames, interval=220)
    gif = OUT / "engine_replay.gif"
    anim.save(gif, writer=PillowWriter(fps=5), dpi=90)
    print("wrote", gif.relative_to(REPO))
    if shutil.which("ffmpeg"):
        mp4 = OUT / "engine_replay.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(gif), "-movflags", "faststart",
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)], check=True)
        print("wrote", mp4.relative_to(REPO))
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hero_scoreboard()
    forecast_chart()
    engine_animation()
