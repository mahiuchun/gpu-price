"""The agent crowd on six models, scored on the same 479 exam forecasts. No AI calls.

    uv run python runs/other_models/table.py

Reads the recorded gpt-6.1-sol run (runs/coach/exam_cells.parquet, our prices only) and each other model's
answers as scored by run_crowd.py (<model>/cells.parquet). Error is pinball loss in USD, as starter/scorer.py
computes it; ranges are 90%, from resampling prices and target days (gpuval.report.skill_ci). Writes
cells.parquet (every forecast's error for every model) and table.json (every comparison).

Each model has two crowds: the five agents as first written, and the crowd with the coach's two kept changes
(one rewritten agent, wider ranges), which the coach chose while working with gpt-6.1-sol.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gpuval import report

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# short name -> (folder with the answers, name in the tables); gpt-6.1-sol is the recorded run
MODELS = {"qwen": ("Qwen3.5-9B", "Qwen3.5-9B, untrained"), "luna": ("gpt-6-luna", "gpt-6-luna"), "sol": (None, "gpt-6.1-sol"),
          "astra": ("gpt-6-astra", "gpt-6-astra"), "sonnet": ("claude-sonnet-5-5", "Claude Sonnet 5.5"), "opus": ("claude-opus-5-5", "Claude Opus 5.5")}


def main() -> None:
    m = pd.read_parquet(REPO / "runs" / "coach" / "exam_cells.parquet")
    m = m[m.world == "ours"].copy()
    df = m[["id", "series_id", "h", "target", "y", "last", "usd_drift", "usd_engine"]].copy()
    df["origin"] = df.id.str.split("|").str[2]
    df["usd_sol_orig"], df["usd_sol_impr"] = m.usd_orig.values, m.usd_impr.values
    # "Inside the range" is judged on the range as a forecast file holds it, to 6 decimals. Unrounded, the average
    # of five identical answers can sit a hair off the value itself, and a price that did not move counts as a miss.
    df["in_sol_orig"] = ((m.y >= m.o_lo.round(6)) & (m.y <= m.o_hi.round(6))).values
    df["in_sol_impr"] = ((m.y >= m.i_lo.round(6)) & (m.y <= m.i_hi.round(6))).values
    for k, (folder, name) in MODELS.items():
        if folder is None:
            continue
        c = pd.read_parquet(HERE / folder / "cells.parquet")
        c = c.rename(columns={"orig_usd_loss": f"usd_{k}_orig", "impr_usd_loss": f"usd_{k}_impr"})
        for tag in ("orig", "impr"):
            c[f"in_{k}_{tag}"] = (c.y >= c[f"{tag}_lo"].round(6)) & (c.y <= c[f"{tag}_hi"].round(6))
        n = len(df)
        df = df.merge(c[["id", "h", f"usd_{k}_orig", f"usd_{k}_impr", f"in_{k}_orig", f"in_{k}_impr"]], on=["id", "h"], how="left")
        assert len(df) == n and df[f"usd_{k}_orig"].notna().all(), f"{name}: answers do not cover the {n} forecasts"
    # the open model after RL training (rl/runs/fc2), on the same forecasts, for comparison with the untrained crowd
    rl = pd.read_parquet(REPO / "rl" / "runs" / "fc2" / "eval_cells.parquet")[["id", "h", "base", "trained"]]
    df = df.merge(rl.rename(columns={"base": "usd_rl_qwen_untrained", "trained": "usd_rl_qwen_trained"}), on=["id", "h"], how="left")

    ci = lambda g, a, b: report.skill_ci(g, a, b)
    f = lambda v: f"{100 * v['skill']:+.1f}% ({100 * v['lo90']:+.1f}% to {100 * v['hi90']:+.1f}%)"
    out = {"n": int(len(df)), "n_series": int(df.series_id.nunique()), "n_days": int(df.origin.nunique()),
           "nothing_moved": int((df.y == df["last"]).sum()), "engine_vs_drift": ci(df, "usd_engine", "usd_drift"),
           "engine_vs_drift_by_h": {int(h): ci(g, "usd_engine", "usd_drift") for h, g in df.groupby("h")}, "models": {}}
    for k, (folder, name) in MODELS.items():
        o, i = f"usd_{k}_orig", f"usd_{k}_impr"
        v = {"first_written_vs_drift": ci(df, o, "usd_drift"), "coach_changes_vs_drift": ci(df, i, "usd_drift"),
             "coach_changes_vs_first_written": ci(df, i, o), "first_written_vs_engine": ci(df, o, "usd_engine"),
             "coach_changes_vs_engine": ci(df, i, "usd_engine"),
             "inside": {"first_written": float(df[f"in_{k}_orig"].mean()), "coach_changes": float(df[f"in_{k}_impr"].mean())},
             "by_h": {int(h): {"n": int(len(g)), "first_written_vs_drift": ci(g, o, "usd_drift"), "coach_changes_vs_drift": ci(g, i, "usd_drift"),
                               "inside_first_written": float(g[f"in_{k}_orig"].mean()), "inside_coach_changes": float(g[f"in_{k}_impr"].mean())}
                      for h, g in df.groupby("h")},
             "by_day": {d: float(1 - g[o].sum() / g.usd_drift.sum()) for d, g in df.groupby("origin")}}
        if folder:
            v["first_written_vs_gpt_6_1_sol"] = ci(df, o, "usd_sol_orig")
            s = json.loads((HERE / folder / "spend.json").read_text())
            v["run"] = {k2: s[k2] for k2 in ("usd", "calls", "input_tokens", "output_tokens", "thinking_tokens", "bad_answers")}
        out["models"][name] = v
    out["pairs"] = {"opus_vs_sonnet_first_written": ci(df, "usd_opus_orig", "usd_sonnet_orig"),
                    "rl_trained_qwen_vs_untrained_crowd_on_qwen": ci(df, "usd_rl_qwen_trained", "usd_qwen_orig"),
                    "rl_trained_qwen_vs_drift": ci(df, "usd_rl_qwen_trained", "usd_drift")}

    df.to_parquet(HERE / "cells.parquet", index=False)
    (HERE / "table.json").write_text(json.dumps(out, indent=1))

    print(f"{out['n']} forecasts, {out['n_series']} prices, {out['n_days']} days ({out['nothing_moved']} where nothing moved). "
          f"Percent less error than drift (90% range). Engine, no AI: {f(out['engine_vs_drift'])}")
    for name, v in out["models"].items():
        print(f"\n{name}")
        print(f"  as first written {f(v['first_written_vs_drift'])}, inside {v['inside']['first_written']:.3f} | with the coach's changes "
              f"{f(v['coach_changes_vs_drift'])}, inside {v['inside']['coach_changes']:.3f}")
        print(f"  coach's changes against first written {f(v['coach_changes_vs_first_written'])} | first written against the engine "
              f"{f(v['first_written_vs_engine'])} | coach's changes against the engine {f(v['coach_changes_vs_engine'])}")
        print("  by horizon, first written: " + " | ".join(f"{h}d {f(x['first_written_vs_drift'])} inside {x['inside_first_written']:.2f}" for h, x in v["by_h"].items()))
        if "run" in v:
            print(f"  against the same crowd on gpt-6.1-sol {f(v['first_written_vs_gpt_6_1_sol'])} | {v['run']['calls']} calls, ${v['run']['usd']:.2f}")
    print("\npairs: " + " | ".join(f"{k} {f(v)}" for k, v in out["pairs"].items()))


if __name__ == "__main__":
    main()
