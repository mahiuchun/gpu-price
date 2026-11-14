"""Score the forecasters on many weekly periods, not just the three test weeks.

    uv run python -m gpuval.periods build      # cells, engine, drift, and the pretrained model for every period
    uv run python -m gpuval.periods agents     # the five-agent crowd on extra days (round 1 only)
    uv run python -m gpuval.periods table      # per-period comparison

A period is one week of forecast days. A forecast made on day d uses data up to d and is scored on
the published price 1, 3 or 7 days later, as long as that date falls inside the period's 7 days.
Periods start on Fridays: 21 Aug ... 30 Oct 2026 (eleven of them). The last three are the test
weeks used everywhere else in this repo.
"""
from __future__ import annotations

import io
import json
import sys

import numpy as np
import pandas as pd

from . import data, harness, report
from .core import HORIZONS, cell_loss, day_of, iso_of
from .engine import default_spec
from .timemachine import plus_days

STARTS = [plus_days("2026-08-21", 7 * k) for k in range(11)]
OUT = data.REPO / "findings" / "data"
EXTRA_AGENT_DAYS = ["2026-09-24", "2026-09-27", "2026-09-30", "2026-10-03", "2026-10-06", "2026-10-09", "2026-10-12"]
TEST_AGENT_DAYS = ["2026-10-16", "2026-10-19", "2026-10-22", "2026-10-25", "2026-10-28", "2026-10-31", "2026-11-03"]


def all_cells(full: dict) -> pd.DataFrame:
    c = harness.base_cells(full)
    c = c[(c.origin >= day_of(STARTS[0])) & (c.origin <= day_of(STARTS[-1]) + 6)].copy()
    start = day_of(STARTS[0]) + 7 * ((c.origin - day_of(STARTS[0])) // 7)
    c["period"] = start.map(iso_of)
    return c[c.target <= start + 7].reset_index(drop=True)


def build() -> None:
    import modal

    from .ml_challenger import app, chronos_predict, contexts

    full = data.build_world(data.DATA_END)
    cells = all_cells(full)
    res = harness.run_spec(full, default_spec(), cells)
    ctx = contexts(cells, full)
    buf = io.BytesIO()
    ctx.to_parquet(buf)
    print(f"{len(cells)} forecasts in {cells.period.nunique()} periods; {ctx.id.nunique()} histories to the pretrained model", flush=True)
    with app.run():
        pred = pd.read_parquet(io.BytesIO(chronos_predict.remote(buf.getvalue())))
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    pred["series_id"] = pred["id"].str.rsplit("|", n=1).str[0]
    pred["origin"] = pred["id"].str.rsplit("|", n=1).str[1].astype(int)
    pred["target"] = pred["timestamp"].map(pd.Timestamp.toordinal)
    q = pred.rename(columns={"0.1": "c10", "0.5": "c50", "0.9": "c90"})[["series_id", "origin", "target", "c10", "c50", "c90"]]
    m = res.merge(q, on=["series_id", "origin", "target"], how="left")
    for c, d in (("c10", "d10"), ("c50", "d50"), ("c90", "d90")):
        m[c] = m[c].fillna(m[d])
    m["c50"] = m.c50.clip(lower=1e-6)
    m["c10"] = np.minimum(m.c10, m.c50).clip(lower=1e-6)
    m["c90"] = np.maximum(m.c90, m.c50)
    m = m.rename(columns={"loss": "loss_engine"})
    m["loss_ml"] = cell_loss(m.y, m.c10, m.c50, m.c90)
    gm = lambda a, b: np.exp(0.5 * np.log(a) + 0.5 * np.log(b))
    e50 = gm(m.q50, m.c50)
    m["loss_engine_ml"] = cell_loss(m.y, np.minimum(gm(m.q10, m.c10), e50), e50, np.maximum(gm(m.q90, m.c90), e50))
    m.to_parquet(OUT / "periods_cells.parquet", index=False)
    print("wrote", OUT / "periods_cells.parquet")


def agents_extra() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    from . import agents

    load_dotenv(data.REPO / ".env")
    client = OpenAI()
    full = data.build_world(data.DATA_END)
    cells = pd.read_parquet(OUT / "periods_cells.parquet")
    chron = cells[["series_id", "origin", "h", "c10", "c50", "c90"]]
    jobs = []
    for o in EXTRA_AGENT_DAYS:
        jobs += agents.run_origin(full, o, chron, client, 24, rounds=1)
        print(o, "done", len(jobs), flush=True)
        (agents.OUT / "extra_answers.json").write_text(json.dumps(jobs, indent=1, default=str))


def _agent_rows() -> pd.DataFrame:
    from . import agents

    rows = []
    for name in ("test_answers.json", "extra_answers.json"):
        p = agents.OUT / name
        if not p.exists():
            continue
        for j in json.loads(p.read_text()):
            for h in HORIZONS:
                lo, mid, hi = j["crowd1"][str(h)] if str(h) in j["crowd1"] else j["crowd1"][h]
                rows.append({"series_id": j["series_id"], "origin": day_of(j["origin"]), "h": h, "a10": lo, "a50": mid, "a90": hi})
    return pd.DataFrame(rows)


def table() -> dict:
    m = pd.read_parquet(OUT / "periods_cells.parquet")
    a = _agent_rows()
    m = m.merge(a, on=["series_id", "origin", "h"], how="left")
    has = m.a50.notna()
    m.loc[has, "loss_agents"] = cell_loss(m.y[has], m.a10[has], m.a50[has], m.a90[has])
    gm3 = lambda a, b, c: np.exp((np.log(a) + np.log(b) + np.log(c)) / 3)
    mid = gm3(m.a50[has], m.q50[has], m.c50[has])
    m.loc[has, "loss_all3"] = cell_loss(m.y[has], np.minimum(gm3(m.a10[has], m.q10[has], m.c10[has]), mid), mid,
                                        np.maximum(gm3(m.a90[has], m.q90[has], m.c90[has]), mid))
    methods = {"engine": "loss_engine", "ml": "loss_ml", "engine_ml": "loss_engine_ml", "agents": "loss_agents", "all3": "loss_all3"}
    out = {"periods": [], "agent_days": EXTRA_AGENT_DAYS + TEST_AGENT_DAYS}
    for p, g in m.groupby("period"):
        row = {"start": p, "end": plus_days(p, 7), "n": int(len(g)), "n_prices": int(g.series_id.nunique()),
               "drift_skill": float(1 - g.loss_drift.sum() / g.loss_last.sum()),
               "moved_share": float((g.y != g["last"]).mean())}
        for k, col in methods.items():
            gg = g[g[col].notna()]
            if len(gg) < 30:
                row[k] = None
                continue
            row[k] = {"n": int(len(gg)), "vs_drift": float(1 - gg[col].sum() / gg.loss_drift.sum()),
                      "engine_same_cells": float(1 - gg.loss_engine.sum() / gg.loss_drift.sum())}
        out["periods"].append(row)
    # pooled blocks: everything before the test weeks, the test weeks, and all together
    blocks = {"before the test weeks (21 Aug to 15 Oct)": m.period < "2026-10-16", "test weeks (16 Oct to 5 Nov)": m.period >= "2026-10-16",
              "all eleven weeks": m.period >= "2026-01-01"}
    out["blocks"] = {}
    for name, mask in blocks.items():
        g = m[mask]
        b = {"n": int(len(g))}
        for k, col in methods.items():
            gg = g[g[col].notna()]
            if len(gg) < 30:
                continue
            b[k] = {"n": int(len(gg)), "vs_drift": report.skill_ci(gg, col, "loss_drift"),
                    "vs_engine": report.skill_ci(gg, col, "loss_engine") if k != "engine" else None,
                    "per_price_avg": report.avg_series_skill(gg, col)}
        out["blocks"][name] = b
    (OUT / "periods_table.json").write_text(json.dumps(out, indent=1, default=str))
    f = lambda d: "   -  " if not d else f"{100 * d['vs_drift']:+6.1f}"
    print(f"{'period (forecast days)':24s} {'n':>5s} {'prices':>6s} {'moved':>6s} | {'engine':>7s} {'ML':>7s} {'eng+ML':>7s} {'agents':>7s} {'all 3':>7s}   (% less error than drift)")
    for r in out["periods"]:
        print(f"{r['start']} to {plus_days(r['start'], 6)[5:]:8s} {r['n']:5d} {r['n_prices']:6d} {r['moved_share']:6.2f} | {f(r['engine'])} {f(r['ml'])} {f(r['engine_ml'])} {f(r['agents'])} {f(r['all3'])}")
    for name, b in out["blocks"].items():
        print(f"\n{name}: {b['n']} forecasts")
        for k in methods:
            if k in b:
                v = b[k]
                print(f"   {k:10s} n={v['n']:5d}  vs drift {report.fmt_ci(v['vs_drift'])}" + (f" | vs engine {report.fmt_ci(v['vs_engine'])}" if v["vs_engine"] else "") + f" | per-price avg {v['per_price_avg']:+.3f}")
    return out


if __name__ == "__main__":
    {"build": build, "agents": agents_extra, "table": table}[sys.argv[1]]()
