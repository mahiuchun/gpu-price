"""Two analyses of the offer lists themselves.

1. `decompose`: what actually moves a published median from one day to the next?
   Each day-to-day change of the median is split, in order, into
     - list changes   : offers dropping off or joining the list,
     - frequent movers: repricing by offers that change price often,
     - rare movers    : repricing by offers that rarely change price.
2. `rollout_test`: roll the engine forward one day at a time, feeding each simulated day back in,
   and measure how far ahead its ranges can be trusted.

    uv run python -m gpuval.insight --wall 2026-11-06 --out findings/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .core import QS, cell_loss, drift_forecast, iso_of
from .engine import build_engines

FREQUENT_SHARE = 0.30  # an offer is a "frequent mover" if it repriced on at least 30% of the days it was listed


def _med(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) else np.nan


def decompose(world: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-series shares of median movement by cause, and the list of frequent movers."""
    rows, movers = [], []
    for sid, s in world["series"].items():
        P, days = s["P"], s["days"]
        n, m = P.shape
        pres = ~np.isnan(P)
        both = pres[1:] & pres[:-1]
        changed = both & (np.abs(P[1:] - P[:-1]) > 1e-9)
        n_both = both.sum(axis=0)
        share = np.divide(changed.sum(axis=0), n_both, out=np.zeros(m), where=n_both > 0)
        frequent = (share >= FREQUENT_SHARE) & (n_both >= 5)
        tot = {"list": 0.0, "frequent": 0.0, "rare": 0.0}
        n_moves = 0
        for i in range(1, n):
            a, b = P[i - 1], P[i]
            common = pres[i - 1] & pres[i]
            if not common.any():
                continue
            m_a, m_d = _med(a), _med(b)
            if not (m_a > 0 and m_d > 0) or abs(np.log(m_d / m_a)) < 1e-12:
                continue
            n_moves += 1
            m_b = _med(np.where(common, a, np.nan))                     # exits removed
            mixed = np.where(common & frequent, b, np.where(common, a, np.nan))
            m_c1 = _med(mixed)                                          # frequent movers repriced
            m_c = _med(np.where(common, b, np.nan))                     # everyone repriced
            tot["list"] += abs(np.log(m_b / m_a)) + abs(np.log(m_d / m_c))
            tot["frequent"] += abs(np.log(m_c1 / m_b))
            tot["rare"] += abs(np.log(m_c / m_c1))
        total = sum(tot.values())
        prov = pd.Series([x.split("|")[0] for x in s["slots"]])
        listed = pres.any(axis=0)
        rows.append({
            "series_id": sid, "cls": s["meta"]["instrument_class"], "source": s["meta"]["source_id"],
            "gpu": s["meta"]["gpu_model"], "n_days": n, "n_offers": int(listed.sum()),
            "n_providers": int(prov[listed].nunique()), "n_frequent_offers": int((frequent & listed).sum()),
            "n_frequent_providers": int(prov[frequent & listed].nunique()),
            "share_of_days_median_moved": n_moves / max(n - 1, 1),
            "movement_total": total,
            "from_list_changes": tot["list"] / total if total else np.nan,
            "from_frequent_movers": tot["frequent"] / total if total else np.nan,
            "from_rare_movers": tot["rare"] / total if total else np.nan,
            "median_offer_change_rate": float(np.median(share[listed & (n_both >= 5)])) if (listed & (n_both >= 5)).any() else np.nan,
        })
        for j in np.where(frequent & listed)[0]:
            movers.append({"series_id": sid, "provider": s["slots"][j].split("|")[0], "offer": s["slots"][j],
                           "repriced_share_of_days": float(share[j]), "days_listed": int(pres[:, j].sum())})
    return pd.DataFrame(rows), pd.DataFrame(movers)


def rollout_paths(eng, i0: int, horizon: int, n_paths: int, lookback: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate the list forward day by day. Each simulated day replays, for every offer, what it did
    on one randomly chosen real day in the lookback window, and the result is fed into the next day.
    Returns log(median_t / median_0) with shape (n_paths, horizon)."""
    days, logP = eng.days, eng.logP
    lo = int(np.searchsorted(days, days[i0] - lookback, side="left"))
    a = np.arange(max(lo, 0), i0)
    a = a[days[a + 1] - days[a] == 1]            # consecutive calendar days only
    if len(a) < 10:
        return np.empty((0, horizon))
    x0 = logP[i0]
    X = np.broadcast_to(x0, (n_paths, len(x0))).copy()
    last = np.broadcast_to(eng.last_known[i0], X.shape).copy()
    base = np.nanmedian(np.exp(x0))
    out = np.empty((n_paths, horizon))
    for t in range(horizon):
        pick = rng.choice(a, size=n_paths)
        A, B = logP[pick], logP[pick + 1]
        pa, pb = ~np.isnan(A), ~np.isnan(B)
        here = ~np.isnan(X)
        X = np.where(here & pa & pb, X + (B - A), X)        # reprice
        X = np.where(here & pa & ~pb, np.nan, X)            # drop off the list
        X = np.where(~here & ~pa & pb, last, X)             # (re)join at the last known price
        last = np.where(np.isnan(X), last, X)
        with np.errstate(all="ignore"):
            out[:, t] = np.log(np.nanmedian(np.exp(X), axis=1) / base)
    return out


def rollout_test(world: dict, horizons=(1, 3, 7, 14, 30, 60, 90), n_paths: int = 300, lookback: int = 90,
                 origin_step: int = 7, min_history: int = 45, max_offers: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Roll the engine forward from many origins and compare its ranges with what happened."""
    rng = np.random.default_rng(seed)
    engines = build_engines(world)
    H = max(horizons)
    rows = []
    for sid, eng in engines.items():
        days, vals = eng.days, eng.value
        if eng.logP.shape[1] > max_offers:
            continue  # very large lists are skipped to keep this quick; noted in the findings
        for i0 in range(len(days)):
            o = int(days[i0])
            if o - int(days[0]) < min_history or (o - int(days[0])) % origin_step:
                continue
            reach = [h for h in horizons if o + h <= int(days[-1])]
            if not reach:
                continue
            paths = rollout_paths(eng, i0, max(reach), n_paths, lookback, rng)
            if len(paths) == 0:
                continue
            for h in reach:
                k = int(np.searchsorted(days, o + h))
                if k >= len(days) or days[k] != o + h:
                    continue
                y, last = float(vals[k]), float(vals[i0])
                col = paths[:, h - 1]
                col = col[np.isfinite(col)]
                if len(col) < 50:
                    continue
                q10, q50, q90 = (last * np.exp(np.quantile(col, QS))).tolist()
                d10, d50, d90 = drift_forecast(days[:i0 + 1], vals[:i0 + 1], h)
                rows.append({"series_id": sid, "cls": eng.cls, "origin": iso_of(o), "h": h, "y": y, "last": last,
                             "q10": q10, "q50": q50, "q90": q90,
                             "inside": q10 <= y <= q90, "rel_width": (q90 - q10) / last,
                             "loss": float(cell_loss(y, q10, q50, q90)), "loss_last": float(cell_loss(y, last, last, last)),
                             "loss_drift": float(cell_loss(y, d10, d50, d90)),
                             "drift_inside": d10 <= y <= d90, "abs_move": abs(np.log(y / last))})
    return pd.DataFrame(rows)


def rollout_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("h")
    out = g.agg(n_cells=("y", "size"), n_series=("series_id", "nunique"), n_origins=("origin", "nunique"),
                inside=("inside", "mean"), drift_inside=("drift_inside", "mean"),
                typical_range_width=("rel_width", "median"), typical_actual_move=("abs_move", "median"),
                loss=("loss", "sum"), loss_last=("loss_last", "sum"), loss_drift=("loss_drift", "sum"))
    out["skill_vs_last"] = 1 - out.loss / out.loss_last
    out["drift_skill_vs_last"] = 1 - out.loss_drift / out.loss_last
    return out.drop(columns=["loss", "loss_last", "loss_drift"]).reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wall", default=data.DATA_END)
    ap.add_argument("--out", default=str(data.REPO / "findings" / "data"))
    ap.add_argument("--skip-rollout", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    world = data.build_world(a.wall)
    data.assert_walled(world, a.wall)
    dec, movers = decompose(world)
    dec.to_csv(out / "movement_by_series.csv", index=False)
    movers.to_csv(out / "frequent_movers.csv", index=False)
    w = dec.movement_total
    overall = {k: float((dec[k] * w).sum() / w.sum()) for k in ("from_list_changes", "from_frequent_movers", "from_rare_movers")}
    by_cls = (dec.assign(w=w).groupby("cls")
              .apply(lambda g: pd.Series({k: float((g[k] * g.w).sum() / g.w.sum()) for k in
                                          ("from_list_changes", "from_frequent_movers", "from_rare_movers")} |
                                         {"n_series": len(g), "median_offers": float(g.n_offers.median()),
                                          "median_frequent_offers": float(g.n_frequent_offers.median())}), include_groups=False))
    by_cls.to_csv(out / "movement_by_class.csv")
    top = (movers.groupby("provider").agg(n_series=("series_id", "nunique"), median_share=("repriced_share_of_days", "median"))
           .sort_values("n_series", ascending=False).head(12)) if len(movers) else pd.DataFrame()
    summary = {"wall": a.wall, "overall_shares": overall,
               "median_offer_change_rate": float(dec.median_offer_change_rate.median()),
               "median_share_of_days_median_moved": float(dec.share_of_days_median_moved.median()),
               "top_frequent_providers": top.reset_index().to_dict("records") if len(top) else []}
    pd.set_option("display.width", 200)
    print("What moves the median (all 37 series, weighted by how much each series moves):")
    print({k: round(v, 3) for k, v in overall.items()})
    print(by_cls.round(3).to_string())
    print("Providers that reprice often:")
    print(top.round(2).to_string())
    if not a.skip_rollout:
        ro = rollout_test(world)
        ro.to_csv(out / "rollout_cells.csv", index=False)
        rs = rollout_summary(ro)
        rs.to_csv(out / "rollout_by_horizon.csv", index=False)
        summary["rollout_by_horizon"] = rs.to_dict("records")
        print("Rolling the engine forward (truth inside the 10-90% range; target 0.80):")
        print(rs.round(3).to_string(index=False))
    (out / "insight.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
