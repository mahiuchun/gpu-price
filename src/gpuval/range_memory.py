"""Range memory: a forecaster sets how wide its ranges are from its own past misses. No AI calls.

    uv run python -m gpuval.range_memory

On each forecast day, for each horizon, the memory holds every earlier forecast by the same
forecaster whose outcome has been published by that day. One number per horizon, the stretch, is
chosen to give the lowest total error on the memory, and is applied to that day's ranges:

    low  -> mid * (low / mid) ** stretch          high -> mid * (high / mid) ** stretch

This is the same stretch the coach set once by hand (1.0 / 1.25 / 1.65), learned day by day
instead. Every forecast is scored with a stretch that was fixed before its outcome existed.

The rule (grid, minimum memory, scoring days) is written in PROTOCOL.md, deviation 12, and was
committed before the first run. Everything is rebuilt from recorded answers:

    first crowd      runs/agents/{extra,test}_answers.json, runs/coach/answers.json
    engine           recomputed from the pack at default settings
    open models      rl/runs/{fc1,fc2}/eval_cells.parquet (before and after RL training)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import coach, data, harness, report
from .core import HORIZONS, cell_loss, day_of, iso_of
from .engine import default_spec

GRID = np.round(np.arange(0.50, 3.0001, 0.05), 2)
MIN_MEMORY = 50                       # fewer resolved forecasts than this at a horizon: no stretch
COACH = {1: 1.0, 3: 1.25, 7: 1.65}    # the coach's fixed stretch, for comparison
ENGINE_FROM = "2026-08-21"
OUT = data.REPO / "findings" / "data"

# the 19 days the first crowd answered outside the coach's practice and check days
SCORE_DAYS = ["2026-09-24", "2026-09-27", "2026-09-30", "2026-10-03", "2026-10-06", "2026-10-09", "2026-10-12",
              "2026-10-16", "2026-10-19", "2026-10-22", "2026-10-25", "2026-10-28", "2026-10-31", "2026-11-03",
              *coach.DAYS["exam"]["ours"]]
COACH_OWN_DAYS = coach.DAYS["practice"]["ours"] + coach.DAYS["check"]["ours"]   # in the memory, never scored


def stretched(lo, mid, hi, s):
    lo, mid, hi = (np.asarray(v, dtype=float) for v in (lo, mid, hi))
    return mid * (np.minimum(lo, mid) / mid) ** s, mid * (np.maximum(hi, mid) / mid) ** s


def fit(mem: pd.DataFrame) -> float:
    """The stretch with the lowest total pinball loss (USD) on the memory; 1.0 if the memory is thin."""
    if len(mem) < MIN_MEMORY:
        return 1.0
    y, lo, mid, hi = (mem[c].to_numpy(float) for c in ("y", "lo", "mid", "hi"))
    best_s, best = 1.0, np.inf
    for s in sorted(GRID, key=lambda v: (abs(v - 1.0), v)):      # ties go to the stretch nearest 1
        a, b = stretched(lo, mid, hi, s)
        tot = float(cell_loss(y, a, mid, b).sum())
        if tot < best - 1e-12:
            best_s, best = float(s), tot
    return best_s


def walk(df: pd.DataFrame) -> pd.DataFrame:
    """Go through the forecast days in order. Needs series_id, origin, h, target, y, lo, mid, hi."""
    out = df.sort_values(["origin", "series_id", "h"]).reset_index(drop=True).copy()
    out["s"], out["n_memory"] = 1.0, 0
    for d in sorted(out.origin.unique()):
        for h in HORIZONS:
            mem = out[(out.h == h) & (out.target <= d)]          # outcome published on or before day d
            m = (out.origin == d) & (out.h == h)
            out.loc[m, "s"], out.loc[m, "n_memory"] = fit(mem), len(mem)
    for tag, s in (("raw", 1.0), ("coach", out.h.map(COACH).to_numpy(float)), ("mem", out.s.to_numpy(float))):
        lo, hi = stretched(out.lo, out.mid, out.hi, s)
        out[f"lo_{tag}"], out[f"hi_{tag}"] = lo, hi
        out[f"loss_{tag}"] = cell_loss(out.y, lo, out.mid, hi)
        out[f"in_{tag}"] = (out.y >= lo - 1e-12) & (out.y <= hi + 1e-12)
    out["moved"] = np.abs(out.y - out["last"]) > 1e-9
    return out


# ------------------------------------------------------------------ frames ----
def truth_cells(full: dict) -> pd.DataFrame:
    """Every scorable forecast of our prices with its outcome, drift and the engine at default settings."""
    c = harness.run_spec(full, default_spec(), harness.base_cells(full))
    return c.rename(columns={"loss": "loss_engine", "q10": "e_lo", "q50": "e_mid", "q90": "e_hi"})


def crowd_frame(cells: pd.DataFrame) -> pd.DataFrame:
    """The first crowd's recorded answers on every day of our prices it was asked about."""
    rows = []
    for name in ("extra_answers", "test_answers"):
        for j in json.loads((data.REPO / "runs" / "agents" / f"{name}.json").read_text()):
            for h in HORIZONS:
                lo, mid, hi = j["crowd1"][str(h)]
                rows.append((j["series_id"], day_of(j["origin"]), h, lo, mid, hi))
    ans = json.loads((coach.OUT / "answers.json").read_text())
    roles = list(coach.base_config()["roles"].values())
    sids = sorted(cells.series_id.unique())
    for iso in COACH_OWN_DAYS + coach.DAYS["exam"]["ours"]:
        for sid in sids:
            keys = [coach.Answers.key(r, f"ours|{sid}|{iso}") for r in roles]
            if not all(k in ans for k in keys):
                continue
            q = np.exp(np.mean(np.log(np.array([[ans[k]["q"][str(h)] for h in HORIZONS] for k in keys], dtype=float)), axis=0))
            for i, h in enumerate(HORIZONS):
                rows.append((sid, day_of(iso), h, min(q[i, 0], q[i, 1]), q[i, 1], max(q[i, 2], q[i, 1])))
    f = pd.DataFrame(rows, columns=["series_id", "origin", "h", "lo", "mid", "hi"])
    assert not f.duplicated(["series_id", "origin", "h"]).any()
    return cells.merge(f, on=["series_id", "origin", "h"], how="inner")


def engine_frame(cells: pd.DataFrame) -> pd.DataFrame:
    c = cells[cells.origin >= day_of(ENGINE_FROM)]
    return c.assign(lo=c.e_lo, mid=c.e_mid, hi=c.e_hi)


def open_frame(cells: pd.DataFrame, run: str, which: str) -> pd.DataFrame | None:
    p = data.REPO / "rl" / "runs" / run / "eval_cells.parquet"
    if not p.exists():
        return None
    q = pd.read_parquet(p)
    q = q[q.world == "ours"].assign(origin=lambda d: d.origin.map(day_of))
    q = q[["series_id", "origin", "h", f"{which}_lo", f"{which}_mid", f"{which}_hi"]]
    q.columns = ["series_id", "origin", "h", "lo", "mid", "hi"]
    return cells.merge(q, on=["series_id", "origin", "h"], how="inner")


# ----------------------------------------------------------------- tables ----
def _ci(g, a, b):
    return report.skill_ci(g, a, b)


def block(g: pd.DataFrame, arms=("raw", "coach", "mem")) -> dict:
    out = {"n": int(len(g)), "n_days": int(g.origin.nunique()), "n_prices": int(g.series_id.nunique()),
           "vs_drift": {a: _ci(g, f"loss_{a}", "loss_drift") for a in arms},
           "inside": {a: float(g[f"in_{a}"].mean()) for a in arms},
           "inside_when_moved": {a: float(g[g.moved][f"in_{a}"].mean()) for a in arms},
           "inside_when_unchanged": {a: float(g[~g.moved][f"in_{a}"].mean()) for a in arms},
           "n_moved": int(g.moved.sum()),
           "memory_vs_none": _ci(g, "loss_mem", "loss_raw")}
    if "coach" in arms:
        out["memory_vs_coach"] = _ci(g, "loss_mem", "loss_coach")
        out["coach_vs_none"] = _ci(g, "loss_coach", "loss_raw")
    return out


def by_horizon(g: pd.DataFrame, arms=("raw", "coach", "mem")) -> dict:
    return {int(h): block(gh, arms) for h, gh in g.groupby("h")}


def stretch_path(w: pd.DataFrame) -> list[dict]:
    p = w.groupby(["origin", "h"]).agg(s=("s", "first"), n_memory=("n_memory", "first")).reset_index()
    return [{"day": iso_of(r.origin), "h": int(r.h), "stretch": float(r.s), "n_memory": int(r.n_memory)} for r in p.itertuples()]


def main() -> int:
    f = report.fmt_ci
    full = data.build_world(data.DATA_END)
    cells = truth_cells(full)
    score = {day_of(d) for d in SCORE_DAYS}
    out = {"rule": {"grid": [float(GRID[0]), float(GRID[-1]), 0.05], "min_memory": MIN_MEMORY, "coach_stretch": COACH,
                    "score_days": SCORE_DAYS, "memory_only_days": COACH_OWN_DAYS}}

    # --- the first crowd -------------------------------------------------------------------
    w = walk(crowd_frame(cells))
    g = w[w.origin.isin(score)]
    late = g[g.n_memory >= MIN_MEMORY]
    out["crowd"] = {"all": block(g), "by_h": by_horizon(g), "where_memory_existed": block(late),
                    "engine_vs_drift_same_forecasts": _ci(g, "loss_engine", "loss_drift"),
                    "memory_vs_engine": _ci(g, "loss_mem", "loss_engine"), "path": stretch_path(w)}
    w.to_parquet(OUT / "range_memory_crowd_cells.parquet", index=False)
    b = out["crowd"]["all"]
    print(f"FIRST CROWD, {b['n']} forecasts on {b['n_days']} days ({b['n_moved']} where the price moved)")
    for a, name in (("raw", "as the agents gave it"), ("coach", "coach's fixed stretch"), ("mem", "range memory")):
        print(f"  {name:24s} vs drift {f(b['vs_drift'][a])} | inside {b['inside'][a]:.3f} | inside when moved {b['inside_when_moved'][a]:.3f} "
              f"| inside when unchanged {b['inside_when_unchanged'][a]:.3f}")
    print(f"  memory vs none {f(b['memory_vs_none'])} | coach vs none {f(b['coach_vs_none'])} | memory vs coach {f(b['memory_vs_coach'])}")
    for h, v in out["crowd"]["by_h"].items():
        print(f"    {h}d (n {v['n']}): memory vs none {f(v['memory_vs_none'])} | coach vs none {f(v['coach_vs_none'])} | memory vs coach {f(v['memory_vs_coach'])} "
              f"| inside none/coach/memory {v['inside']['raw']:.2f}/{v['inside']['coach']:.2f}/{v['inside']['mem']:.2f}")
    v = out["crowd"]["where_memory_existed"]
    print(f"  only forecasts made with a memory ({v['n']} on {v['n_days']} days): memory vs none {f(v['memory_vs_none'])} | memory vs coach {f(v['memory_vs_coach'])}")
    path = pd.DataFrame(out["crowd"]["path"]).pivot(index="day", columns="h", values="stretch")
    print("  stretch chosen, by day and horizon:\n" + path.to_string())

    # --- the engine ------------------------------------------------------------------------
    we = walk(engine_frame(cells))
    ge = we[we.origin.isin(score)]
    blind = we[(we.origin >= day_of("2026-10-16")) & (we.origin <= day_of("2026-11-05"))]
    out["engine"] = {"crowd_days": block(ge, ("raw", "mem")), "all_days_16oct_5nov": block(blind, ("raw", "mem")),
                     "by_h_all_days": by_horizon(blind, ("raw", "mem")), "path": stretch_path(we[we.origin >= day_of("2026-09-24")])}
    for name, key in (("the crowd's 19 days", "crowd_days"), ("every day 16 Oct to 5 Nov", "all_days_16oct_5nov")):
        v = out["engine"][key]
        print(f"ENGINE, {name} ({v['n']} forecasts): as it is {f(v['vs_drift']['raw'])} | with memory {f(v['vs_drift']['mem'])} | memory vs none {f(v['memory_vs_none'])} "
              f"| inside {v['inside']['raw']:.3f} -> {v['inside']['mem']:.3f} | when moved {v['inside_when_moved']['raw']:.3f} -> {v['inside_when_moved']['mem']:.3f}")

    # --- the open models: is range width all that RL taught? -----------------------------------
    out["open_models"] = {}
    for run, label in (("fc1", "Qwen3-8B"), ("fc2", "Qwen3.5-9B")):
        base, trained = open_frame(cells, run, "base"), open_frame(cells, run, "trained")
        if base is None:
            continue
        wb, wt = walk(base), walk(trained)
        k = ["series_id", "origin", "h"]
        m = wb[k + ["target", "loss_drift", "loss_engine", "loss_raw", "loss_mem", "in_raw", "in_mem", "moved"]].merge(
            wt[k + ["loss_raw", "loss_mem", "in_raw", "in_mem"]], on=k, suffixes=("_base", "_trained"))
        r = {"n": int(len(m)), "n_days": int(m.origin.nunique()),
             "untrained_vs_drift": _ci(m, "loss_raw_base", "loss_drift"), "untrained_plus_memory_vs_drift": _ci(m, "loss_mem_base", "loss_drift"),
             "trained_vs_drift": _ci(m, "loss_raw_trained", "loss_drift"), "trained_plus_memory_vs_drift": _ci(m, "loss_mem_trained", "loss_drift"),
             "trained_vs_untrained": _ci(m, "loss_raw_trained", "loss_raw_base"),
             "untrained_plus_memory_vs_untrained": _ci(m, "loss_mem_base", "loss_raw_base"),
             "trained_vs_untrained_plus_memory": _ci(m, "loss_raw_trained", "loss_mem_base"),
             "trained_plus_memory_vs_trained": _ci(m, "loss_mem_trained", "loss_raw_trained"),
             "inside": {"untrained": float(m.in_raw_base.mean()), "untrained_plus_memory": float(m.in_mem_base.mean()),
                        "trained": float(m.in_raw_trained.mean()), "trained_plus_memory": float(m.in_mem_trained.mean())},
             "path_untrained": stretch_path(wb)}
        out["open_models"][label] = r
        print(f"{label} ({r['n']} forecasts, {r['n_days']} days): untrained {f(r['untrained_vs_drift'])} | untrained + memory {f(r['untrained_plus_memory_vs_drift'])} "
              f"| trained {f(r['trained_vs_drift'])} | trained + memory {f(r['trained_plus_memory_vs_drift'])}")
        print(f"   trained vs untrained {f(r['trained_vs_untrained'])} | memory alone vs untrained {f(r['untrained_plus_memory_vs_untrained'])} "
              f"| trained vs untrained + memory {f(r['trained_vs_untrained_plus_memory'])} | inside {r['inside']}")
    (OUT / "range_memory.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote", (OUT / "range_memory.json").relative_to(data.REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
