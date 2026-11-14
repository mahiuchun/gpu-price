"""Data layer: build a "world" from the pack, never reading past a wall date.

A world holds, for every scored series, the official daily value (the pack's daily median)
and the table of individual offers behind it: one column per offer slot, one row per day.

    world = {"wall": "2026-10-16", "name": "real", "series": {series_id: {
        "days":   int64[n_days]            calendar day ordinals with an observation
        "value":  float64[n_days]          official daily median, USD per GPU-hour
        "P":      float64[n_days, n_slots] offer prices, NaN where the slot was not listed
        "slots":  list[str]                slot keys (provider|sku|region|snapshot#|dup#)
        "meta":   dict                     source_id, instrument_class, gpu_model, pricing_tier
    }}}

A slot is one row of a site's price list followed through time. The official series is the
median over that day's rows, so nanmedian(P[day]) reproduces it (checked in `verify_world`).
"""
from __future__ import annotations

import copy
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .core import day_of, iso_of

REPO = Path(__file__).resolve().parents[2]
PACK = Path(os.environ.get("GPUVAL_PACK", REPO / "gpu-price-pack-v0.1.1-handover"))
CACHE = REPO / "runs" / "_cache"
HEADLINE_FILE = REPO / "data" / "headline_series.txt"
DATA_END = "2026-11-06"  # the pack's cutoff


def headline_ids() -> list[str]:
    return [l.strip() for l in HEADLINE_FILE.read_text().splitlines() if l.strip()]


def _starter_series(wall: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official series and catalog from the starter's own builder, truncated at the wall."""
    out = CACHE / f"starter_{wall}"
    series, catalog = out / "series.csv", out / "catalog.csv"
    if not (series.exists() and catalog.exists()):
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-B", str(PACK / "starter" / "build_series.py"), "--pack", str(PACK),
               "--cutoff", wall, "--no-aws", "--out", str(series), "--catalog", str(catalog)]
        subprocess.run(cmd, check=True, capture_output=True)
    return pd.read_csv(series, low_memory=False), pd.read_csv(catalog, low_memory=False)


def _offer_rows(wall: str, keys: pd.DataFrame) -> pd.DataFrame:
    """Offer rows (dt <= wall) for the given (source_id, instrument_class, gpu_model, pricing_tier)."""
    import duckdb

    con = duckdb.connect()
    con.register("keys", keys)
    frames = []
    for src in sorted(keys.source_id.unique()):
        rel = f"read_parquet('{PACK}/rental_observations/source_id={src}/*/*.parquet', hive_partitioning=true)"
        q = f"""
            select o.dt, o.snapshot_ts_utc, o.source_id, o.instrument_class, o.gpu_model, o.pricing_tier,
                   o.provider, coalesce(o.sku_raw, '') as sku, coalesce(o.region_raw, '') as region,
                   o.usd_per_gpu_hour as price
            from {rel} o
            join keys k on k.source_id = o.source_id and k.instrument_class = o.instrument_class
                       and k.gpu_model = o.gpu_model and k.pricing_tier = o.pricing_tier
            where o.dt <= DATE '{wall}' and o.usd_per_gpu_hour is not null
        """
        frames.append(con.sql(q).df())
    df = pd.concat(frames, ignore_index=True)
    df["day"] = pd.to_datetime(df["dt"]).map(pd.Timestamp.toordinal).astype(np.int64)
    return df


def build_world(wall: str = DATA_END, series_ids: list[str] | None = None, name: str = "real") -> dict:
    """Build the real world from the pack using only rows dated on or before `wall`."""
    if wall > DATA_END:
        raise ValueError(f"wall {wall} is after the pack cutoff {DATA_END}")
    series_ids = series_ids or headline_ids()
    sdf, cat = _starter_series(wall)
    cat = cat[cat.series_id.isin(series_ids)]
    keys = cat[["series_id", "source_id", "instrument_class", "gpu_model", "pricing_tier"]].copy()
    rows = _offer_rows(wall, keys.drop(columns="series_id").drop_duplicates())
    rows = rows.merge(keys, on=["source_id", "instrument_class", "gpu_model", "pricing_tier"])
    # snapshot number within the day (sites with several snapshots a day contribute every row)
    rows["snap"] = rows.groupby(["series_id", "day"])["snapshot_ts_utc"].rank(method="dense").astype(int)
    rows = rows.sort_values(["series_id", "day", "snap", "provider", "sku", "region", "price"], kind="stable")
    rows["dup"] = rows.groupby(["series_id", "day", "snap", "provider", "sku", "region"]).cumcount()
    rows["slot"] = (rows.provider + "|" + rows.sku + "|" + rows.region + "|s" + rows.snap.astype(str)
                    + "|d" + rows.dup.astype(str))
    world = {"wall": wall, "name": name, "series": {}}
    sdf = sdf[sdf.series_id.isin(series_ids)]
    for sid in series_ids:
        s = sdf[sdf.series_id == sid].sort_values("date")
        if s.empty:
            continue
        days = pd.to_datetime(s["date"]).map(pd.Timestamp.toordinal).to_numpy(np.int64)
        r = rows[rows.series_id == sid]
        mat = r.pivot(index="day", columns="slot", values="price").reindex(days)
        m = cat[cat.series_id == sid].iloc[0]
        world["series"][sid] = {
            "days": days, "value": s["value"].to_numpy(float), "P": mat.to_numpy(float),
            "slots": list(mat.columns),
            "meta": {k: str(m[k]) for k in ("source_id", "instrument_class", "gpu_model", "pricing_tier")},
        }
    return world


def verify_world(world: dict) -> pd.DataFrame:
    """How well nanmedian over the offer table reproduces the official daily value."""
    out = []
    for sid, s in world["series"].items():
        with np.errstate(all="ignore"):
            rec = np.round(np.nanmedian(s["P"], axis=1), 4)
        ok = np.isclose(rec, s["value"], atol=1.01e-4)
        out.append({"series_id": sid, "n_days": len(s["days"]), "n_slots": len(s["slots"]),
                    "match_share": float(np.mean(ok)), "max_abs_diff": float(np.nanmax(np.abs(rec - s["value"])))})
    return pd.DataFrame(out)


def cut_world(world: dict, wall: str) -> dict:
    """A copy of the world that contains nothing dated after `wall`."""
    wd = day_of(wall)
    out = {"wall": wall, "name": world["name"], "series": {}}
    for sid, s in world["series"].items():
        k = int(np.searchsorted(s["days"], wd, side="right"))
        if k == 0:
            continue
        P = s["P"][:k]
        keep = ~np.all(np.isnan(P), axis=0)
        out["series"][sid] = {"days": s["days"][:k].copy(), "value": s["value"][:k].copy(), "P": P[:, keep].copy(),
                              "slots": [x for x, kk in zip(s["slots"], keep) if kk], "meta": dict(s["meta"])}
    return out


def assert_walled(world: dict, wall: str) -> None:
    wd = day_of(wall)
    for sid, s in world["series"].items():
        if len(s["days"]) and int(s["days"][-1]) > wd:
            raise RuntimeError(f"LEAK: {sid} has data dated {iso_of(s['days'][-1])}, after the wall {wall}")


def make_placebo(world: dict, seed: int, name: str | None = None, shuffle_from: str | None = None) -> dict:
    """A placebo world: same offers, same listing pattern, same set of day-to-day price changes,
    but the ORDER of those changes in time is shuffled (one shuffle per series, shared by all its
    offers so co-movement survives). Any pattern in the timing of changes is destroyed.

    `shuffle_from`: keep history before this date as it really was and shuffle only from there on
    (used so placebo worlds share the real starting state).
    The official value is recomputed as the median of the shuffled offer table.
    """
    rng = np.random.default_rng(seed)
    out = {"wall": world["wall"], "name": name or f"placebo{seed}", "series": {}}
    for sid in sorted(world["series"]):
        s = world["series"][sid]
        P, days = s["P"], s["days"]
        n, m = P.shape
        logP = np.log(P)
        start = 1 if shuffle_from is None else max(1, int(np.searchsorted(days, day_of(shuffle_from))))
        steps = np.arange(start, n)  # step i is the move from day i-1 to day i
        perm = steps.copy()
        rng.shuffle(perm)
        src_of = dict(zip(steps.tolist(), perm.tolist()))
        fake = np.full_like(logP, np.nan)
        fake[:start] = logP[:start]
        last_fake = np.full(m, np.nan)
        for i in range(start):
            pres = ~np.isnan(fake[i])
            last_fake[pres] = fake[i, pres]
        for i in range(start, n):
            j = src_of[i]
            chg = logP[j] - logP[j - 1]              # the change that really happened at step j
            chg = np.where(np.isnan(chg), 0.0, chg)  # offer not listed on both days there: no change
            present = ~np.isnan(logP[i])             # listing pattern stays as it really was
            cont = present & ~np.isnan(last_fake)
            new = present & np.isnan(last_fake)
            fake[i, cont] = last_fake[cont] + chg[cont]
            fake[i, new] = logP[i, new]              # first ever appearance: enter at the real price
            last_fake[present] = fake[i, present]
        Pf = np.exp(fake)
        with np.errstate(all="ignore"):
            val = np.round(np.nanmedian(Pf, axis=1), 4)
        out["series"][sid] = {"days": days.copy(), "value": val, "P": Pf, "slots": list(s["slots"]),
                              "meta": dict(s["meta"])}
    return out


def save_world(world: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(world, f, protocol=4)


def load_world(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clone(world: dict) -> dict:
    return copy.deepcopy(world)
