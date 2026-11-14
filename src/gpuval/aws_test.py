"""Test the forecasters on four years of AWS spot prices (and on our four long-running prices).

    uv run python -m gpuval.aws_test build     # AWS world, weekly periods, engine, drift, pretrained model
    uv run python -m gpuval.aws_test agents    # the five-agent crowd on twelve days spread over the four years
    uv run python -m gpuval.aws_test table     # comparison by year

The AWS figure for a machine type is the median spot price across its availability zones, so each
zone plays the part of a provider on a price list and the engine applies unchanged. These are not
the prices scored in the competition; they are a much longer second test bed. Nothing is carried
across the archive's gap (1 Mar to 30 Jun 2026).
"""
from __future__ import annotations

import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from . import data, harness, report
from .core import HORIZONS, cell_loss, day_of, iso_of
from .engine import default_spec
from .timemachine import plus_days

INSTANCES = ["p4d.24xlarge", "p4de.24xlarge", "p5.48xlarge", "p5e.48xlarge", "p5en.48xlarge", "p3.2xlarge", "p3.8xlarge",
             "p3.16xlarge", "p3dn.24xlarge", "p2.xlarge", "g4dn.xlarge", "g4dn.12xlarge", "g5.xlarge", "g5.12xlarge",
             "g5.48xlarge", "g6.xlarge", "g6.12xlarge", "g6e.xlarge", "g6e.12xlarge"]
BASE = "2022-08-05"          # first Friday with about two months of history behind it
GAP_END = "2026-07-01"
AGENT_DAYS = ["2022-11-04", "2023-03-03", "2023-07-07", "2023-11-03", "2024-03-01", "2024-07-05", "2024-11-01",
              "2025-03-07", "2025-07-04", "2025-11-07", "2026-02-06", "2026-09-04"]
OUT = data.REPO / "findings" / "data"
WORK = data.CACHE / "aws_test"


def aws_world() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    series, catalog = WORK / "series.csv", WORK / "catalog.csv"
    if not series.exists():
        subprocess.run([sys.executable, "-B", str(data.PACK / "starter" / "build_series.py"), "--pack", str(data.PACK),
                        "--no-daily-statistics", "--aws-instances", ",".join(INSTANCES), "--out", str(series),
                        "--catalog", str(catalog)], check=True, capture_output=True)
    s = pd.read_csv(series, usecols=["series_id", "date", "value"])
    s["day"] = pd.to_datetime(s["date"]).map(pd.Timestamp.toordinal).astype(np.int64)
    cat = pd.read_csv(catalog)
    world = {"wall": "2026-10-31", "name": "aws", "series": {}}
    for r in cat[cat.series_id.str.endswith("az_median")].itertuples():
        g = s[s.series_id == r.series_id].sort_values("day")
        members = str(r.members).split(";")
        mat = s[s.series_id.isin(members)].pivot(index="day", columns="series_id", values="value").reindex(g.day.to_numpy())
        world["series"][r.series_id] = {
            "days": g.day.to_numpy(np.int64), "value": g.value.to_numpy(float), "P": mat.to_numpy(float),
            "slots": [c.split(".")[-1] for c in mat.columns],
            "meta": {"source_id": "AWS spot (each offer on the list is one availability zone)", "instrument_class": "hyperscaler_spot",
                     "gpu_model": f"{r.instance_type} ({r.gpu_model})", "pricing_tier": "spot"}}
    return world


def aws_cells(world: dict) -> pd.DataFrame:
    c = harness.base_cells(world)
    c = c[c.origin >= day_of(BASE)].copy()
    start = day_of(BASE) + 7 * ((c.origin - day_of(BASE)) // 7)
    c["period"] = start.map(iso_of)
    c["weekday0"] = c.origin == start
    return c[c.target <= start + 7].reset_index(drop=True)


def _contexts(cells: pd.DataFrame, world: dict) -> pd.DataFrame:
    rows = []
    for (sid, origin), _ in cells[cells.weekday0].groupby(["series_id", "origin"]):
        s = world["series"][sid]
        k = int(np.searchsorted(s["days"], origin, side="right"))
        lo = max(origin - 365, day_of(GAP_END) if origin >= day_of(GAP_END) else 0)
        j = int(np.searchsorted(s["days"], lo))
        ser = pd.Series(s["value"][j:k], index=pd.to_datetime([iso_of(d) for d in s["days"][j:k]])).asfreq("D").ffill()
        if len(ser) >= 14:
            rows.append(pd.DataFrame({"id": f"{sid}|{origin}", "timestamp": ser.index, "target": ser.to_numpy(float)}))
    return pd.concat(rows, ignore_index=True)


def build() -> None:
    from .ml_challenger import app, chronos_predict

    world = aws_world()
    v = data.verify_world(world)
    print(f"{len(world['series'])} AWS machine types; median of zones reproduces the published figure on {v.match_share.mean():.1%} of days", flush=True)
    cells = aws_cells(world)
    res = harness.run_spec(world, default_spec(), cells).rename(columns={"loss": "loss_engine"})
    print(f"{len(res)} forecasts in {res.period.nunique()} weekly periods, {res.period.min()} to {res.period.max()}", flush=True)
    ctx = _contexts(res, world)
    buf = io.BytesIO()
    ctx.to_parquet(buf)
    print(f"{ctx.id.nunique()} histories to the pretrained model", flush=True)
    with app.run():
        pred = pd.read_parquet(io.BytesIO(chronos_predict.remote(buf.getvalue())))
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    pred["series_id"] = pred["id"].str.rsplit("|", n=1).str[0]
    pred["origin"] = pred["id"].str.rsplit("|", n=1).str[1].astype(int)
    pred["target"] = pred["timestamp"].map(pd.Timestamp.toordinal)
    q = pred.rename(columns={"0.1": "c10", "0.5": "c50", "0.9": "c90"})[["series_id", "origin", "target", "c10", "c50", "c90"]]
    m = res.merge(q, on=["series_id", "origin", "target"], how="left")
    ok = m.c50.notna()
    c50 = m.c50[ok].clip(lower=1e-6)
    m.loc[ok, "loss_ml"] = cell_loss(m.y[ok], np.minimum(m.c10[ok], c50).clip(lower=1e-6), c50, np.maximum(m.c90[ok], c50))
    m.to_parquet(OUT / "aws_cells.parquet", index=False)
    data.save_world(world, WORK / "world.pkl")
    print("wrote", OUT / "aws_cells.parquet")


def agents_run() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    from . import agents

    load_dotenv(data.REPO / ".env")
    client = OpenAI()
    world = data.load_world(WORK / "world.pkl")
    world["wall"] = "2026-10-31"
    cells = pd.read_parquet(OUT / "aws_cells.parquet")
    chron = cells[cells.c50.notna()][["series_id", "origin", "h", "c10", "c50", "c90"]]
    jobs = []
    for o in AGENT_DAYS:
        jobs += agents.run_origin(world, o, chron, client, 24, rounds=1)
        print(o, "done", len(jobs), flush=True)
        (agents.OUT / "aws_answers.json").write_text(json.dumps(jobs, indent=1, default=str))


def table() -> dict:
    from . import agents

    m = pd.read_parquet(OUT / "aws_cells.parquet")
    m["year"] = m.period.str[:4]
    out = {"n": int(len(m)), "n_types": int(m.series_id.nunique()), "n_periods": int(m.period.nunique()),
           "first": m.period.min(), "last": plus_days(m.period.max(), 6)}
    wk = m.groupby("period").agg(e=("loss_engine", "sum"), d=("loss_drift", "sum"))
    wk = wk[wk.d > 0]
    out["weeks_engine_ahead"] = int((wk.e < wk.d).sum())
    out["weeks_total"] = int(len(wk))
    out["engine"] = {"vs_drift": report.skill_ci(m, "loss_engine", "loss_drift"), "per_type_avg": report.avg_series_skill(m, "loss_engine"),
                     "drift_per_type_avg": report.avg_series_skill(m, "loss_drift")}
    out["by_year"] = []
    for y, g in m.groupby("year"):
        gm = g[g.loss_ml.notna()]
        out["by_year"].append({"year": y, "n": int(len(g)), "types": int(g.series_id.nunique()),
                               "engine_vs_drift": float(1 - g.loss_engine.sum() / g.loss_drift.sum()),
                               "ml_n": int(len(gm)), "ml_vs_drift": float(1 - gm.loss_ml.sum() / gm.loss_drift.sum()) if len(gm) else None,
                               "engine_on_ml_cells": float(1 - gm.loss_engine.sum() / gm.loss_drift.sum()) if len(gm) else None})
    gm = m[m.loss_ml.notna()]
    out["ml"] = {"n": int(len(gm)), "vs_drift": report.skill_ci(gm, "loss_ml", "loss_drift"), "vs_engine": report.skill_ci(gm, "loss_ml", "loss_engine"),
                 "engine_same_cells": report.skill_ci(gm, "loss_engine", "loss_drift")}
    p = agents.OUT / "aws_answers.json"
    if p.exists():
        rows = []
        for j in json.loads(p.read_text()):
            for h in HORIZONS:
                lo, mid, hi = j["crowd1"][str(h)]
                rows.append({"series_id": j["series_id"], "origin": day_of(j["origin"]), "h": h, "a10": lo, "a50": mid, "a90": hi})
        a = m.merge(pd.DataFrame(rows), on=["series_id", "origin", "h"], how="inner")
        a["loss_agents"] = cell_loss(a.y, a.a10, a.a50, a.a90)
        out["agents"] = {"n": int(len(a)), "days": int(a.origin.nunique()), "vs_drift": report.skill_ci(a, "loss_agents", "loss_drift"),
                         "vs_engine": report.skill_ci(a, "loss_agents", "loss_engine"), "engine_same_cells": report.skill_ci(a, "loss_engine", "loss_drift"),
                         "ml_same_cells": report.skill_ci(a[a.loss_ml.notna()], "loss_ml", "loss_drift"),
                         "by_day": [{"day": iso_of(o), "n": int(len(g)), "agents": float(1 - g.loss_agents.sum() / g.loss_drift.sum()) if g.loss_drift.sum() > 0 else None,
                                     "engine": float(1 - g.loss_engine.sum() / g.loss_drift.sum()) if g.loss_drift.sum() > 0 else None}
                                    for o, g in a.groupby("origin")]}
    # our own four long-running prices, before the eleven weeks
    full = data.build_world(data.DATA_END)
    c = harness.base_cells(full)
    c = c[c.origin < day_of("2026-08-21")]
    if len(c):
        r = harness.run_spec(full, default_spec(), c)
        out["own_four_early"] = {"n": int(len(r)), "prices": int(r.series_id.nunique()), "first": iso_of(r.origin.min()), "last": iso_of(r.origin.max()),
                                 "engine_vs_drift": report.skill_ci(r, "loss", "loss_drift")}
    (OUT / "aws_table.json").write_text(json.dumps(out, indent=1, default=str))
    f = report.fmt_ci
    print(f"AWS: {out['n']} forecasts, {out['n_types']} machine types, {out['n_periods']} weekly periods, {out['first']} to {out['last']}")
    print(f"engine vs drift: {f(out['engine']['vs_drift'])} | ahead of drift in {out['weeks_engine_ahead']} of {out['weeks_total']} weeks | per-type avg skill: engine {out['engine']['per_type_avg']:+.3f}, drift {out['engine']['drift_per_type_avg']:+.3f}")
    for y in out["by_year"]:
        print(f"  {y['year']}: {y['n']:6d} forecasts, {y['types']:2d} types | engine {100 * y['engine_vs_drift']:+6.1f}%"
              + (f" | Fridays only ({y['ml_n']}): ML {100 * y['ml_vs_drift']:+6.1f}%, engine {100 * y['engine_on_ml_cells']:+6.1f}%" if y["ml_n"] else ""))
    print(f"ML model (Friday forecasts, {out['ml']['n']}): vs drift {f(out['ml']['vs_drift'])} | vs engine {f(out['ml']['vs_engine'])} | engine same forecasts {f(out['ml']['engine_same_cells'])}")
    if "agents" in out:
        ag = out["agents"]
        print(f"AI agents ({ag['n']} forecasts on {ag['days']} days): vs drift {f(ag['vs_drift'])} | vs engine {f(ag['vs_engine'])} | engine same forecasts {f(ag['engine_same_cells'])} | ML same forecasts {f(ag['ml_same_cells'])}")
        print("  by day (agents, engine):", [(d["day"], None if d["agents"] is None else round(100 * d["agents"], 1), None if d["engine"] is None else round(100 * d["engine"], 1)) for d in ag["by_day"]])
    if "own_four_early" in out:
        o = out["own_four_early"]
        print(f"our own long-running prices before 21 Aug: {o['n']} forecasts on {o['prices']} prices, {o['first']} to {o['last']}: engine vs drift {f(o['engine_vs_drift'])}")
    return out


if __name__ == "__main__":
    {"build": build, "agents": agents_run, "table": table}[sys.argv[1]]()
