"""A ready-made ML forecaster as a challenger: Chronos-2, a model pretrained on other time series.

    uv run python -m gpuval.ml_challenger

It is used as it comes (no training on our data, nothing tuned). For every blind forecast in the
three test weeks it is shown the published price history up to the origin day and asked for the
10%, 50% and 90% points 1, 3 and 7 days ahead. It runs on a Modal GPU and is scored on exactly the
same cells as the engine and drift.
"""
from __future__ import annotations

import io
import json

import modal
import numpy as np
import pandas as pd

from . import data, harness, report
from .core import cell_loss, day_of, iso_of
from .timemachine import RECORD_STEPS, plus_days

MODEL = "amazon/chronos-2"
app = modal.App("gpuval-ml-challenger")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("chronos-forecasting==2.3.2", "pandas[pyarrow]>=2.0", "torch>=2.2"))


@app.function(image=image, gpu="T4", timeout=1800, memory=8192)
def chronos_predict(context_parquet: bytes, prediction_length: int = 7) -> bytes:
    import pandas as pd
    import torch
    from chronos import Chronos2Pipeline

    ctx = pd.read_parquet(io.BytesIO(context_parquet))
    pipe = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda" if torch.cuda.is_available() else "cpu")
    pred = pipe.predict_df(ctx, prediction_length=prediction_length, quantile_levels=[0.1, 0.5, 0.9],
                           id_column="id", timestamp_column="timestamp", target="target")
    buf = io.BytesIO()
    pred.to_parquet(buf)
    return buf.getvalue()


ARMS = ("cross", "cov", "cross_cov")    # the second look, added 15 Nov (PROTOCOL.md, deviation 12)
COVS = ("cov_cheapest", "cov_lower_quartile", "cov_upper_quartile", "cov_offers")


@app.function(image=image, gpu="T4", timeout=3600, memory=16384)
def chronos_predict_grouped(context_parquet: bytes, arms: list[str], prediction_length: int = 7) -> bytes:
    """The model run the way its paper also describes: prices that share a forecast day read each
    other (cross-learning), and facts about the price list go in as past-only covariates. One call
    per forecast day, so a batch never holds a later reading of any price."""
    import pandas as pd
    import torch
    from chronos import Chronos2Pipeline

    ctx = pd.read_parquet(io.BytesIO(context_parquet))
    pipe = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda" if torch.cuda.is_available() else "cpu")
    covs = [c for c in ctx.columns if c.startswith("cov_")]
    out = []
    for arm in arms:
        for grp, g in ctx.groupby("group"):
            df = g.drop(columns=["group"] + ([] if "cov" in arm else covs))
            pred = pipe.predict_df(df, prediction_length=prediction_length, quantile_levels=[0.1, 0.5, 0.9],
                                   id_column="id", timestamp_column="timestamp", target="target",
                                   cross_learning="cross" in arm, batch_size=1024)
            out.append(pred[["id", "timestamp", "0.1", "0.5", "0.9"]].assign(arm=arm))
    buf = io.BytesIO()
    pd.concat(out, ignore_index=True).to_parquet(buf)
    return buf.getvalue()


def grouped_contexts(cells: pd.DataFrame, full: dict) -> pd.DataFrame:
    """Histories up to each forecast day, grouped by that day, with four facts about the list."""
    rows = []
    for (sid, origin), _ in cells.groupby(["series_id", "origin"]):
        s = full["series"][sid]
        k = int(np.searchsorted(s["days"], origin, side="right"))
        P = s["P"][:k]
        with np.errstate(all="ignore"):
            f = pd.DataFrame({"target": s["value"][:k], COVS[0]: np.nanmin(P, axis=1), COVS[1]: np.nanquantile(P, 0.25, axis=1),
                              COVS[2]: np.nanquantile(P, 0.75, axis=1), COVS[3]: (~np.isnan(P)).sum(axis=1).astype(float)},
                             index=pd.to_datetime([iso_of(d) for d in s["days"][:k]]))
        f = f.asfreq("D").ffill()
        rows.append(f.reset_index(names="timestamp").assign(id=f"{sid}|{origin}", group=int(origin)))
    return pd.concat(rows, ignore_index=True)


def crossrun() -> int:
    """Score the three new arms on the same forecasts as the plain run (findings/data/periods_cells.parquet)."""
    import warnings

    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    dest = data.REPO / "findings" / "data"
    cells = pd.read_parquet(dest / "periods_cells.parquet")
    full = data.build_world(data.DATA_END)
    ctx = grouped_contexts(cells, full)
    buf = io.BytesIO()
    ctx.to_parquet(buf)
    print(f"{len(cells)} forecasts, {ctx.id.nunique()} histories in {ctx.group.nunique()} forecast-day batches, {len(ctx)} rows sent", flush=True)
    with modal.enable_output(), app.run():
        pred = pd.read_parquet(io.BytesIO(chronos_predict_grouped.remote(buf.getvalue(), list(ARMS))))
    pred["series_id"] = pred["id"].str.rsplit("|", n=1).str[0]
    pred["origin"] = pred["id"].str.rsplit("|", n=1).str[1].astype(int)
    pred["target"] = pd.to_datetime(pred["timestamp"]).map(pd.Timestamp.toordinal)
    m = cells.copy()
    for arm in ARMS:
        q = pred[pred.arm == arm].rename(columns={"0.1": "a10", "0.5": "a50", "0.9": "a90"})[["series_id", "origin", "target", "a10", "a50", "a90"]]
        j = m[["series_id", "origin", "target"]].merge(q, on=["series_id", "origin", "target"], how="left")
        assert j.a50.notna().all(), f"{arm}: {int(j.a50.isna().sum())} forecasts missing"
        a50 = j.a50.clip(lower=1e-6).to_numpy()
        m[f"{arm}_10"], m[f"{arm}_50"], m[f"{arm}_90"] = np.minimum(j.a10.to_numpy(), a50).clip(1e-6), a50, np.maximum(j.a90.to_numpy(), a50)
        m[f"loss_{arm}"] = cell_loss(m.y, m[f"{arm}_10"], m[f"{arm}_50"], m[f"{arm}_90"])
    m.to_parquet(dest / "ml_second_look_cells.parquet", index=False)
    out = {"model": MODEL, "arms": list(ARMS), "covariates": list(COVS), "blocks": {}}
    f = report.fmt_ci
    for name, g in (("8 earlier weeks", m[m.period < "2026-10-16"]), ("3 test weeks", m[m.period >= "2026-10-16"]), ("all 11 weeks", m)):
        b = {"n": int(len(g)), "plain_vs_drift": report.skill_ci(g, "loss_ml", "loss_drift"), "engine_vs_drift": report.skill_ci(g, "loss_engine", "loss_drift")}
        for arm in ARMS:
            b[arm] = {"vs_drift": report.skill_ci(g, f"loss_{arm}", "loss_drift"), "vs_plain": report.skill_ci(g, f"loss_{arm}", "loss_ml"),
                      "vs_engine": report.skill_ci(g, f"loss_{arm}", "loss_engine"),
                      "inside": float(((g.y >= g[f"{arm}_10"]) & (g.y <= g[f"{arm}_90"])).mean())}
        out["blocks"][name] = b
        print(f"{name} ({b['n']}): plain {f(b['plain_vs_drift'])} | engine {f(b['engine_vs_drift'])}")
        for arm in ARMS:
            print(f"   {arm:10s} vs drift {f(b[arm]['vs_drift'])} | vs plain {f(b[arm]['vs_plain'])} | vs engine {f(b[arm]['vs_engine'])} | inside {b[arm]['inside']:.2f}")
    wk = m.groupby("period")[["loss_drift", "loss_ml"] + [f"loss_{a}" for a in ARMS]].sum()
    out["weeks_ahead_of_drift"] = {"plain": int((wk.loss_ml < wk.loss_drift).sum()), **{a: int((wk[f"loss_{a}"] < wk.loss_drift).sum()) for a in ARMS}, "weeks": int(len(wk))}
    print("weeks ahead of drift:", out["weeks_ahead_of_drift"])
    (dest / "ml_second_look.json").write_text(json.dumps(out, indent=1))
    return 0


def blind_cells() -> tuple[pd.DataFrame, dict]:
    full = data.build_world(data.DATA_END)
    frames = []
    for j, T in enumerate(RECORD_STEPS, start=1):
        end = plus_days(T, 7)
        w = data.cut_world(full, end)
        c = harness.base_cells(w)
        c = c[(c.origin >= day_of(T)) & (c.target <= day_of(end))].copy()
        c["week"] = j
        frames.append(c)
    return pd.concat(frames, ignore_index=True), full


def contexts(cells: pd.DataFrame, full: dict) -> pd.DataFrame:
    """One daily history per (series, origin): values up to the origin, missing days carried forward."""
    rows = []
    for (sid, origin), _ in cells.groupby(["series_id", "origin"]):
        s = full["series"][sid]
        k = int(np.searchsorted(s["days"], origin, side="right"))
        ser = pd.Series(s["value"][:k], index=pd.to_datetime([iso_of(d) for d in s["days"][:k]]))
        ser = ser.asfreq("D").ffill()
        rows.append(pd.DataFrame({"id": f"{sid}|{origin}", "timestamp": ser.index, "target": ser.to_numpy(float)}))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    cells, full = blind_cells()
    ctx = contexts(cells, full)
    buf = io.BytesIO()
    ctx.to_parquet(buf)
    print(f"{len(cells)} blind cells, {ctx.id.nunique()} histories, {len(ctx)} rows sent to the model", flush=True)
    with modal.enable_output(), app.run():
        pred = pd.read_parquet(io.BytesIO(chronos_predict.remote(buf.getvalue())))
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    pred["sid"] = pred["id"].str.rsplit("|", n=1).str[0]
    pred["origin"] = pred["id"].str.rsplit("|", n=1).str[1].astype(int)
    pred["target"] = pred["timestamp"].map(pd.Timestamp.toordinal)
    q = pred.rename(columns={"0.1": "c10", "0.5": "c50", "0.9": "c90"})[["sid", "origin", "target", "c10", "c50", "c90"]]
    m = cells.merge(q, left_on=["series_id", "origin", "target"], right_on=["sid", "origin", "target"], how="left")
    missing = int(m.c50.isna().sum())
    for c, d in (("c10", "d10"), ("c50", "d50"), ("c90", "d90")):   # any cell the model did not return falls back to drift
        m[c] = m[c].fillna(m[d])
    m["c50"] = m.c50.clip(lower=1e-6)
    m["c10"] = np.minimum(m.c10, m.c50).clip(lower=1e-6)
    m["c90"] = np.maximum(m.c90, m.c50)
    m["loss_ml"] = cell_loss(m.y, m.c10, m.c50, m.c90)
    eng = pd.read_parquet(data.REPO / "runs" / "record" / "real_gpt" / "cells.parquet")[["series_id", "origin", "h", "loss_v0"]]
    m = m.merge(eng, on=["series_id", "origin", "h"], how="left")
    out = {
        "model": MODEL, "n_cells": int(len(m)), "cells_without_model_output": missing,
        "total_skill": {"drift": report.skill_ci(m, "loss_drift", "loss_last"), "engine": report.skill_ci(m, "loss_v0", "loss_last"),
                        "ml_model": report.skill_ci(m, "loss_ml", "loss_last")},
        "error_cut_vs_drift": {"engine": report.skill_ci(m, "loss_v0", "loss_drift"), "ml_model": report.skill_ci(m, "loss_ml", "loss_drift")},
        "ml_vs_engine": report.skill_ci(m, "loss_ml", "loss_v0"),
        "per_series_average": {"drift": report.avg_series_skill(m, "loss_drift"), "engine": report.avg_series_skill(m, "loss_v0"),
                               "ml_model": report.avg_series_skill(m, "loss_ml")},
        "inside_range": {"ml_model": float(((m.y >= m.c10) & (m.y <= m.c90)).mean())},
        "by_horizon": {str(h): {"ml_vs_drift": report.skill_ci(g, "loss_ml", "loss_drift"), "engine_vs_drift": report.skill_ci(g, "loss_v0", "loss_drift")}
                       for h, g in m.groupby("h")},
        "by_week": {str(wk): {"drift": float(1 - g.loss_drift.sum() / g.loss_last.sum()), "engine": float(1 - g.loss_v0.sum() / g.loss_last.sum()),
                              "ml_model": float(1 - g.loss_ml.sum() / g.loss_last.sum())} for wk, g in m.groupby("week")},
    }
    dest = data.REPO / "findings" / "data"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ml_challenger.json").write_text(json.dumps(out, indent=1))
    m[["series_id", "cls", "origin", "h", "target", "week", "y", "last", "c10", "c50", "c90", "loss_ml", "loss_v0", "loss_drift", "loss_last"]] \
        .to_parquet(dest / "ml_challenger_cells.parquet", index=False)
    f = report.fmt_ci
    print(f"cells {out['n_cells']} (model returned nothing for {missing})")
    print("total skill vs last value : drift", f(out["total_skill"]["drift"]), "| engine", f(out["total_skill"]["engine"]), "| ML model", f(out["total_skill"]["ml_model"]))
    print("error cut against drift   : engine", f(out["error_cut_vs_drift"]["engine"]), "| ML model", f(out["error_cut_vs_drift"]["ml_model"]))
    print("ML model against engine   :", f(out["ml_vs_engine"]))
    print("per-series average        :", {k: round(v, 3) for k, v in out["per_series_average"].items()}, "| ML inside range:", round(out["inside_range"]["ml_model"], 3))
    print("by horizon (vs drift)     :", {h: (round(v["ml_vs_drift"]["skill"], 3), round(v["engine_vs_drift"]["skill"], 3)) for h, v in out["by_horizon"].items()}, "(ML, engine)")
    print("by week (skill)           :", {w: {k: round(x, 3) for k, x in v.items()} for w, v in out["by_week"].items()})
    return 0


def final(origin: str = data.DATA_END, out: str | None = None) -> int:
    """The model's numbers for the forecast day itself: `versions/chronos2_quantiles_<origin>.csv`.

        uv run python -m gpuval.ml_challenger final [origin] [out.csv]

    One of the five agents (the one that weighs reference forecasts) is shown these beside drift and
    the engine (`agents.py --final`). Histories are built exactly as `contexts` builds them for the tests.
    Added 15 Nov: the file was first written by hand-run code that was not committed."""
    from .core import HORIZONS

    full = data.build_world(origin)
    data.assert_walled(full, origin)
    od = day_of(origin)
    rows = []
    for sid, s in full["series"].items():
        if int(s["days"][-1]) != od:
            continue
        ser = pd.Series(s["value"], index=pd.to_datetime([iso_of(d) for d in s["days"]])).asfreq("D").ffill()
        rows.append(pd.DataFrame({"id": sid, "timestamp": ser.index, "target": ser.to_numpy(float)}))
    buf = io.BytesIO()
    pd.concat(rows, ignore_index=True).to_parquet(buf)
    with modal.enable_output(), app.run():
        pred = pd.read_parquet(io.BytesIO(chronos_predict.remote(buf.getvalue())))
    pred["h"] = pd.to_datetime(pred["timestamp"]).map(pd.Timestamp.toordinal) - od
    q = pred[pred.h.isin(HORIZONS)].rename(columns={"id": "series_id", "0.1": "c10", "0.5": "c50", "0.9": "c90"})
    q = q[["series_id", "h", "c10", "c50", "c90"]].sort_values(["series_id", "h"])
    dest = data.REPO / "versions" / f"chronos2_quantiles_{origin}.csv" if out is None else out
    q.to_csv(dest, index=False)
    print(f"wrote {dest}: {len(q)} rows, {q.series_id.nunique()} prices")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(final(*sys.argv[2:4]) if sys.argv[1:2] == ["final"] else crossrun() if sys.argv[1:2] == ["crossrun"] else main())
