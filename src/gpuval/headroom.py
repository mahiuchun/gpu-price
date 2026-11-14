"""How much room is there above drift, and how much of it did we take?

    uv run python -m gpuval.headroom

Everything here is scored on the three blind weeks of the real world, WITH hindsight, so none of it
is a forecast: these are ceilings to compare the real results against.

  1. Knowing the answer (ground truth): zero error.
  2. Knowing only WHETHER the price will move: flat forecast when it will not, our engine when it will.
  3. Picking, for every forecast, whichever of our engine and drift turned out better.
  4. The best engine settings found by searching the dials directly on the blind weeks: the most a
     perfect scientist could have reached inside the search space the AI scientist was given.
"""
from __future__ import annotations

import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from . import data, harness
from .core import HORIZONS, cell_loss, day_of
from .engine import CLASSES, SPEC_SCHEMA, apply_patch, default_spec
from .timemachine import RECORD_STEPS, plus_days

WEEKS: list[tuple[dict, pd.DataFrame, dict]] = []

GRID = {
    "lookback_days": [14, 21, 30, 45, 90, 120], "halflife_days": [7, 14, 21, 30, 60], "replay_exits": [False],
    "replay_entries": [False], "momentum": [0.25, 0.5, 1.0], "momentum_window": [3, 14], "active_share": [0.1, 0.6],
    "blend_drift": [0.25, 0.5, 0.75, 1.0], "width_scale": [0.5, 0.75, 1.25, 1.5, 2.0], "center": ["last"],
    "min_scenarios": [5, 20, 40], "floor_up": [0.005, 0.01, 0.02, 0.04], "floor_down": [0.005, 0.01, 0.02, 0.04],
    "peer_pull": [0.1, 0.3], "peer_window": [7],
}
SCOPES = ["all"] + [f"class:{c}" for c in CLASSES] + [f"horizon:{h}" for h in HORIZONS]


def _patch(scope: str, dial: str, value) -> dict:
    if scope == "all":
        return {dial: value}
    kind, name = scope.split(":")
    return {("by_class" if kind == "class" else "by_horizon"): {name: {dial: value}}}


def _build() -> None:
    full = data.build_world(data.DATA_END)
    for T in RECORD_STEPS:
        end = plus_days(T, 7)
        w = data.cut_world(full, end)
        c = harness.base_cells(w)
        WEEKS.append((w, c[(c.origin >= day_of(T)) & (c.target <= day_of(end))], {}))


def _run(spec: dict) -> pd.DataFrame:
    return pd.concat([harness.run_spec(w, spec, c, eng) for w, c, eng in WEEKS], ignore_index=True)


def _loss(spec: dict) -> float:
    return float(_run(spec).loss.sum())


def _try(args) -> tuple[float, str, str, object]:
    spec, scope, dial, value = args
    return _loss(apply_patch(spec, _patch(scope, dial, value))), scope, dial, value


def main() -> int:
    _build()
    base = _run(default_spec())
    L_last, L_drift, L_eng = float(base.loss_last.sum()), float(base.loss_drift.sum()), float(base.loss.sum())
    cut = lambda L: 1 - L / L_drift           # error cut against drift
    skill = lambda L: 1 - L / L_last          # skill against a flat last value
    moved = base.y.to_numpy() != base["last"].to_numpy()
    out = {"n_forecasts": int(len(base)), "share_where_price_moved": float(moved.mean()),
           "share_of_drift_error_on_moved": float(base.loss_drift[moved].sum() / L_drift),
           "share_of_engine_error_on_moved": float(base.loss[moved].sum() / L_eng), "rows": []}

    def row(name: str, L: float, note: str = "") -> None:
        out["rows"].append({"forecast": name, "error_cut_vs_drift": cut(L), "skill_vs_last": skill(L), "note": note})

    row("Drift", L_drift)
    row("Our engine, default settings (a real forecast)", L_eng)
    row("Hindsight: flat when the price will not move, our engine when it will", float(base.loss[moved].sum()))
    row("Hindsight: flat when it will not move, drift when it will", float(base.loss_drift[moved].sum()))
    row("Hindsight: per forecast, the better of our engine and drift", float(np.minimum(base.loss, base.loss_drift).sum()))
    # middle exactly right, range width as the engine gave it
    half_lo, half_hi = base.q50 - base.q10, base.q90 - base.q50
    row("Hindsight: middle exactly right, our engine's range width kept",
        float(cell_loss(base.y, base.y - half_lo, base.y, base.y + half_hi).sum()))
    # best settings in hindsight: greedy search over single dial changes, scored on the blind weeks themselves
    spec, best, steps = default_spec(), L_eng, []
    cands = [(s, d, v) for d, vals in GRID.items() for v in vals for s in SCOPES]
    with mp.get_context("fork").Pool(32) as pool:
        for _ in range(4):
            res = pool.map(_try, [(spec, s, d, v) for s, d, v in cands], chunksize=4)
            L, s, d, v = min(res, key=lambda r: r[0])
            if L >= best * (1 - 1e-4):
                break
            spec, best = apply_patch(spec, _patch(s, d, v)), L
            steps.append({"scope": s, "dial": d, "value": v, "error_cut_vs_drift": cut(L)})
            row(f"Hindsight: best {len(steps)} dial change(s) of our engine", L,
                "; ".join(f"{x['scope']} {x['dial']}={x['value']}" for x in steps))
    row("Ground truth (knowing the answer)", 0.0)
    out["hindsight_dial_steps"] = steps
    (data.REPO / "findings" / "data" / "headroom.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"{out['n_forecasts']} blind forecasts; the price moved in {out['share_where_price_moved']:.0%} of them; "
          f"{out['share_of_drift_error_on_moved']:.0%} of drift's error and {out['share_of_engine_error_on_moved']:.0%} "
          f"of the engine's error is on those")
    for r in out["rows"]:
        print(f"  {r['forecast']:68s} error cut vs drift {r['error_cut_vs_drift']:+7.1%} | skill {r['skill_vs_last']:+.3f}"
              + (f" | {r['note']}" if r["note"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
