"""Turn a time-machine run into the numbers the findings quote.

    uv run python -m gpuval.report --name record

Reads runs/<name>/<world>_<arm>/{cells.parquet,state.json}; writes runs/<name>/report.json and
prints the tables. Uncertainty: every comparison is resampled by series and by target day
(2,000 resamples, 90% interval), as fixed in PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data

N_BOOT = 2000


def _mats(df: pd.DataFrame, cols: list[str]) -> dict:
    s_codes, s_idx = np.unique(df.series_id, return_inverse=True)
    d_codes, d_idx = np.unique(df.target, return_inverse=True)
    out = {}
    for c in cols:
        m = np.zeros((len(s_codes), len(d_codes)))
        np.add.at(m, (s_idx, d_idx), df[c].to_numpy(float))
        out[c] = m
    return out


def skill_ci(df: pd.DataFrame, ours: str, ref: str, seed: int = 0) -> dict:
    """Skill of `ours` relative to `ref` (1 - error ratio) with a 90% interval from resampling
    series and target days."""
    if len(df) == 0:
        return {"n_cells": 0}
    M = _mats(df, [ours, ref])
    A, B = M[ours], M[ref]
    point = 1.0 - A.sum() / B.sum() if B.sum() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    ns, nd = A.shape
    ws = rng.multinomial(ns, np.ones(ns) / ns, size=N_BOOT).astype(float)
    wd = rng.multinomial(nd, np.ones(nd) / nd, size=N_BOOT).astype(float)
    num = np.einsum("bs,sd,bd->b", ws, A, wd)
    den = np.einsum("bs,sd,bd->b", ws, B, wd)
    stat = 1.0 - num / np.where(den > 0, den, np.nan)
    lo, hi = np.nanquantile(stat, [0.05, 0.95])
    return {"n_cells": int(len(df)), "n_series": int(ns), "n_days": int(nd), "skill": float(point),
            "lo90": float(lo), "hi90": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


def avg_series_skill(df: pd.DataFrame, ours: str, ref: str = "loss_last") -> float:
    per = df.groupby("series_id").agg(l=(ours, "mean"), b=(ref, "mean"))
    sk = (1 - per.l / per.b)[per.b > 0]
    return float(sk.mean()) if len(sk) else float("nan")


def live(df: pd.DataFrame) -> pd.DataFrame:
    """The live system: in week j the forecasts come from the version frozen at step j."""
    out = df.copy()
    out["loss_live"] = np.nan
    out["in_live"] = False
    for j in sorted(out.week.unique()):
        m = out.week == j
        out.loc[m, "loss_live"] = out.loc[m, f"loss_v{j}"]
        out.loc[m, "in_live"] = out.loc[m, f"in_v{j}"]
    return out


def load_run(name: str) -> tuple[dict, dict, dict]:
    root = data.REPO / "runs" / name
    meta = json.loads((root / "meta.json").read_text())
    cells, states = {}, {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        world, arm = d.name.rsplit("_", 1)
        if (d / "cells.parquet").exists():
            cells[(world, arm)] = pd.read_parquet(d / "cells.parquet")
        if (d / "state.json").exists():
            states[(world, arm)] = json.loads((d / "state.json").read_text())
    return meta, cells, states


def world_block(df: pd.DataFrame) -> dict:
    lv = live(df)
    out = {
        "n_cells": int(len(lv)),
        "total_skill": {
            "drift": skill_ci(lv, "loss_drift", "loss_last"),
            "v0_no_research": skill_ci(lv, "loss_v0", "loss_last"),
            "live": skill_ci(lv, "loss_live", "loss_last"),
        },
        "per_series_average": {
            "drift": avg_series_skill(lv, "loss_drift"), "v0_no_research": avg_series_skill(lv, "loss_v0"),
            "live": avg_series_skill(lv, "loss_live"),
        },
        "against_drift": {
            "v0_no_research": skill_ci(lv, "loss_v0", "loss_drift"),
            "live": skill_ci(lv, "loss_live", "loss_drift"),
        },
        "research_gain_live_vs_v0": skill_ci(lv, "loss_live", "loss_v0"),
        "inside_range": {"v0_no_research": float(lv.in_v0.mean()), "live": float(lv.in_live.mean())},
        "by_week": {}, "by_horizon": {}, "by_class": {},
    }
    for j, g in lv.groupby("week"):
        row = {"n_cells": int(len(g)), "drift": float(1 - g.loss_drift.sum() / g.loss_last.sum())}
        for k in range(0, int(j) + 1):
            row[f"v{k}"] = float(1 - g[f"loss_v{k}"].sum() / g.loss_last.sum())
        out["by_week"][str(j)] = row
    for h, g in lv.groupby("h"):
        out["by_horizon"][str(h)] = {"n_cells": int(len(g)), "live_vs_drift": skill_ci(g, "loss_live", "loss_drift"),
                                     "v0_vs_drift": skill_ci(g, "loss_v0", "loss_drift")}
    for c, g in lv.groupby("cls"):
        out["by_class"][c] = {"n_cells": int(len(g)), "share_of_error": float(g.loss_last.sum() / lv.loss_last.sum()),
                              "live_vs_drift": skill_ci(g, "loss_live", "loss_drift"),
                              "v0_vs_drift": skill_ci(g, "loss_v0", "loss_drift")}
    return out


def ledger_block(state: dict) -> dict:
    led = state.get("ledger", [])
    kept = [e for e in led if e["kept"]]
    dials: dict = {}
    for e in kept:
        for c in e["changes"]:
            dials[c["dial"]] = dials.get(c["dial"], 0) + 1
    return {"ideas": len(led), "kept": len(kept), "no_change": sum(bool(e.get("no_change")) for e in led),
            "failed_development": sum(not e["dev_ok"] and not e.get("no_change") for e in led),
            "failed_hidden_check": sum(e["dev_ok"] and not e["kept"] for e in led),
            "kept_dials": dials, "final_spec": state.get("spec"), "usage": state.get("usage", {})}


def build(name: str) -> dict:
    meta, cells, states = load_run(name)
    rep = {"run": {k: meta.get(k) for k in ("name", "steps", "arms", "placebos", "ideas", "model", "reasoning",
                                            "started_utc", "finished_utc", "modal")},
           "worlds": {}, "ledgers": {}}
    for (world, arm), df in cells.items():
        rep["worlds"].setdefault(world, {})[arm] = world_block(df)
    for (world, arm), st in states.items():
        rep["ledgers"].setdefault(world, {})[arm] = ledger_block(st)
    # real against placebo: is the scientist's gain in the real world unusual?
    comp = {}
    for arm in meta.get("arms", []):
        rows = []
        for world in rep["worlds"]:
            b = rep["worlds"][world].get(arm)
            if not b:
                continue
            rows.append({"world": world, "v0_vs_drift": b["against_drift"]["v0_no_research"]["skill"],
                         "live_vs_drift": b["against_drift"]["live"]["skill"],
                         "research_gain": b["research_gain_live_vs_v0"]["skill"],
                         "kept": rep["ledgers"].get(world, {}).get(arm, {}).get("kept")})
        comp[arm] = rows
    rep["real_vs_placebo"] = comp
    return rep


def fmt_ci(d: dict) -> str:
    if not d or not d.get("n_cells"):
        return "n/a"
    return f"{d['skill']:+.3f} [{d['lo90']:+.3f}, {d['hi90']:+.3f}]"


def print_report(rep: dict) -> None:
    for world, arms in rep["worlds"].items():
        for arm, b in arms.items():
            t = b["total_skill"]
            print(f"\n== {world} / {arm}: {b['n_cells']} blind cells")
            print(f"  total skill vs last value : drift {fmt_ci(t['drift'])} | v0 {fmt_ci(t['v0_no_research'])} | live {fmt_ci(t['live'])}")
            pa = b["per_series_average"]
            print(f"  per-series average        : drift {pa['drift']:+.3f} | v0 {pa['v0_no_research']:+.3f} | live {pa['live']:+.3f}")
            ad = b["against_drift"]
            print(f"  error cut against drift   : v0 {fmt_ci(ad['v0_no_research'])} | live {fmt_ci(ad['live'])}")
            print(f"  research gain (live vs v0): {fmt_ci(b['research_gain_live_vs_v0'])}")
            print(f"  truth inside range        : v0 {b['inside_range']['v0_no_research']:.2f} | live {b['inside_range']['live']:.2f}")
            print("  by week: " + " | ".join(f"w{j}: " + ", ".join(f"{k} {v:+.3f}" for k, v in r.items() if k != "n_cells")
                                            for j, r in b["by_week"].items()))
            led = rep["ledgers"].get(world, {}).get(arm, {})
            if led:
                print(f"  ideas {led['ideas']}: kept {led['kept']}, failed development {led['failed_development']}, "
                      f"failed hidden check {led['failed_hidden_check']}, no change {led['no_change']}; kept dials {led['kept_dials']}")
    print("\n== real against placebo")
    for arm, rows in rep["real_vs_placebo"].items():
        print(f"  arm {arm}")
        for r in rows:
            print(f"    {r['world']:9s} v0 vs drift {r['v0_vs_drift']:+.3f} | live vs drift {r['live_vs_drift']:+.3f} | "
                  f"research gain {r['research_gain']:+.3f} | ideas kept {r['kept']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    rep = build(a.name)
    out = data.REPO / "runs" / a.name / "report.json"
    out.write_text(json.dumps(rep, indent=1, default=str))
    print_report(rep)
    print(f"\nwrote {out.relative_to(data.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
