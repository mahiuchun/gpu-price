"""The test harness: enumerate forecast cells, run a spec, score it. Fixed code, never edited by the AI.

A cell is one (series, origin day, horizon). The forecast for a cell uses only rows dated on or
before its origin; the truth is the official value exactly `horizon` days later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import HORIZONS, cell_loss, day_of, drift_forecast, iso_of
from .engine import build_engines, validate_spec

MIN_HISTORY_DAYS = 14  # an origin needs at least this much history behind it


def base_cells(world: dict, horizons=HORIZONS) -> pd.DataFrame:
    """Every scorable cell in the world with the truth, the last value and the drift forecast.
    Origins are days on which the series was actually observed (no stale origins)."""
    rows = []
    for sid, s in world["series"].items():
        days, vals = s["days"], s["value"]
        cls = s["meta"]["instrument_class"]
        for i0 in range(len(days)):
            o = int(days[i0])
            if o - int(days[0]) < MIN_HISTORY_DAYS:
                continue
            for h in horizons:
                k = int(np.searchsorted(days, o + h))
                if k >= len(days) or days[k] != o + h:
                    continue
                d10, d50, d90 = drift_forecast(days[:i0 + 1], vals[:i0 + 1], h)
                rows.append((sid, cls, i0, o, h, o + h, float(vals[k]), float(vals[i0]), d10, d50, d90))
    df = pd.DataFrame(rows, columns=["series_id", "cls", "i0", "origin", "h", "target", "y", "last",
                                     "d10", "d50", "d90"])
    df["loss_last"] = cell_loss(df.y, df["last"], df["last"], df["last"])
    df["loss_drift"] = cell_loss(df.y, df.d10, df.d50, df.d90)
    return df


def run_spec(world: dict, spec: dict, cells: pd.DataFrame, engines: dict | None = None,
             fallback: set | None = None) -> pd.DataFrame:
    """Forecast every cell in `cells` with the engine under `spec`; returns cells + q10/q50/q90/loss.
    Series in `fallback` use the drift forecast instead."""
    spec = validate_spec(spec)
    engines = engines if engines is not None else {}
    if not engines:
        engines.update(build_engines(world))
    fallback = fallback or set()
    out = cells.copy()
    q = np.full((len(out), 3), np.nan)
    src = np.empty(len(out), dtype=object)
    pos = {idx: n for n, idx in enumerate(out.index)}
    for sid, g in out.groupby("series_id", sort=False):
        if sid in fallback:
            for idx, r in g.iterrows():
                q[pos[idx]] = (r.d10, r.d50, r.d90)
                src[pos[idx]] = "drift_series"
            continue
        eng = engines[sid]
        for i0, gg in g.groupby("i0", sort=False):
            hs = tuple(int(x) for x in gg.h.unique())
            f = eng.forecast(int(i0), spec, hs)
            for idx, h in zip(gg.index, gg.h):
                q[pos[idx]] = f[int(h)][:3]
                src[pos[idx]] = f[int(h)][3]
    out["q10"], out["q50"], out["q90"], out["src"] = q[:, 0], q[:, 1], q[:, 2], src
    out["loss"] = cell_loss(out.y, out.q10, out.q50, out.q90)
    return out


def _skill(loss: float, base: float):
    return None if (base is None or base == 0 or not np.isfinite(base)) else float(1.0 - loss / base)


def summarize(df: pd.DataFrame, loss_col: str = "loss", qcols=("q10", "q50", "q90")) -> dict:
    """Scores in the organisers' terms. Skill is against the flat last value, as in scorer.py."""
    if len(df) == 0:
        return {"n_cells": 0}
    loss, base, drift = float(df[loss_col].mean()), float(df.loss_last.mean()), float(df.loss_drift.mean())
    per = df.groupby("series_id").agg(l=(loss_col, "mean"), b=("loss_last", "mean"))
    sk = (1.0 - per.l / per.b)[per.b > 0]
    lo, mid, hi = (df[c] for c in qcols)
    return {
        "n_cells": int(len(df)), "n_series": int(df.series_id.nunique()),
        "loss": loss, "loss_last": base, "loss_drift": drift,
        "skill_total": _skill(loss, base),              # "pooled": total dollars of error
        "skill_avg_series": float(sk.mean()) if len(sk) else None,   # each series counts the same
        "skill_median_series": float(sk.median()) if len(sk) else None,
        "skill_vs_drift": _skill(loss, drift),
        "drift_skill_total": _skill(drift, base),
        "below": float((df.y < lo).mean()), "inside": float(((df.y >= lo) & (df.y <= hi)).mean()),
        "above": float((df.y > hi).mean()),
    }


def split_by_target(cells: pd.DataFrame, lo_excl: str | None, hi_incl: str | None,
                    origin_min: str | None = None) -> pd.DataFrame:
    """Cells whose target date lies in (lo_excl, hi_incl], optionally with origin >= origin_min."""
    m = np.ones(len(cells), dtype=bool)
    if lo_excl:
        m &= cells.target.to_numpy() > day_of(lo_excl)
    if hi_incl:
        m &= cells.target.to_numpy() <= day_of(hi_incl)
    if origin_min:
        m &= cells.origin.to_numpy() >= day_of(origin_min)
    return cells[m]


def diagnostics(df: pd.DataFrame, top: int = 6) -> str:
    """A compact plain-text report of where a spec wins and loses (shown to the AI scientist)."""
    lines = []
    s = summarize(df)
    f = lambda v: "n/a" if v is None else f"{v:+.3f}"
    lines.append(f"ALL: cells={s['n_cells']} skill_total={f(s['skill_total'])} skill_avg_series={f(s['skill_avg_series'])} "
                 f"vs_drift={f(s['skill_vs_drift'])} (drift itself: {f(s['drift_skill_total'])}) "
                 f"truth below/inside/above range = {s['below']:.2f}/{s['inside']:.2f}/{s['above']:.2f} (target 0.10/0.80/0.10)")
    tot = df.loss_last.sum()
    lines.append("By class and horizon (share = share of total error at stake; vs_drift > 0 means better than drift):")
    for (cls, h), g in df.groupby(["cls", "h"]):
        t = summarize(g)
        lines.append(f"  {cls:19s} h={h}: cells={t['n_cells']:4d} share={g.loss_last.sum()/tot:.2f} skill_total={f(t['skill_total'])} "
                     f"vs_drift={f(t['skill_vs_drift'])} below/inside/above={t['below']:.2f}/{t['inside']:.2f}/{t['above']:.2f} "
                     f"engine_used={np.mean(g.src == 'engine'):.2f}")
    per = df.groupby("series_id").agg(l=("loss", "sum"), d=("loss_drift", "sum"), b=("loss_last", "sum"))
    per["share"] = per.b / tot
    per["vs_drift"] = 1 - per.l / per.d.replace(0, np.nan)
    lines.append(f"Biggest series by error at stake (top {top}):")
    for sid, r in per.sort_values("share", ascending=False).head(top).iterrows():
        lines.append(f"  {sid.replace('.median', ''):62s} share={r.share:.2f} vs_drift={f(None if pd.isna(r.vs_drift) else r.vs_drift)}")
    worst = per[per.d > 0].sort_values("vs_drift").head(3)
    lines.append("Series where the engine is worst relative to drift: " +
                 "; ".join(f"{sid.replace('.median', '')} ({r.vs_drift:+.2f})" for sid, r in worst.iterrows()))
    return "\n".join(lines)


def to_submission(df: pd.DataFrame, team_id: str = "team", method_note: str = "") -> pd.DataFrame:
    """Rows in the starter kit's forecast-file format."""
    out = pd.DataFrame({
        "series_id": df.series_id, "origin_date": df.origin.map(iso_of), "horizon_days": df.h.astype(int),
        "q10": df.q10.round(6), "q50": df.q50.round(6), "q90": df.q90.round(6),
        "target_date": df.target.map(iso_of), "team_id": team_id, "method_note": method_note or df.get("src", ""),
    })
    return out
