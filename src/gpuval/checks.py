"""Checks on the write-up's own claims, from recorded files and the pack. No AI calls, no Modal.

    uv run python -m gpuval.checks          # prints every table; writes findings/data/checks.json

Added on 15 Nov after reading the brief's reference list (PROTOCOL.md, deviation 12). Each block
answers one question a careful reader would ask of FINDINGS.md:

    coverage          how often the range holds when the price really moved, not just overall
    weight            how many prices and days carry the error that the pooled figures add up
    single_agents     whether the different jobs, or the averaging, are what beat the engine
    ml_per_price      the ready-made ML model price by price, and with its middle held at "no change"
    aws_knowledge     the AWS test split at the date the agents' model's knowledge ends
    coach_gain        which prices supply the coach's gain on the exam days
    exits             offers that leave a list: how many come straight back at the same price
    euro              one provider's dollar price that follows the euro, and where it is the middle offer
    aws_flat_zones    AWS zones whose price has not changed for 60 days (carried forward, or gone?)
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from . import aws_test, data, harness, insight, report

warnings.filterwarnings("ignore", message="All-NaN slice encountered")   # a day on which an offer is not listed
from .core import HORIZONS, cell_loss, day_of, iso_of
from .engine import default_spec

OUT = data.REPO / "findings" / "data"
ROLES = ["list_reader", "repricer_watcher", "base_rate_keeper", "combiner", "risk_spotter"]
KNOWLEDGE_END = "2026-04-30"      # the agents' model (gpt-6.1-sol) knows the world to April 2026
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _inside(g: pd.DataFrame, lo: str, hi: str) -> float:
    return float(((g.y >= g[lo] - 1e-12) & (g.y <= g[hi] + 1e-12)).mean()) if len(g) else float("nan")


def _effective(w) -> float:
    """Kish's effective number: 37 equal prices give 37; one price carrying everything gives 1."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.sum() > 0 else 0.0


def _needed(w, share: float = 0.9) -> int:
    w = np.sort(np.asarray(w, dtype=float))[::-1]
    return int(np.searchsorted(np.cumsum(w) / w.sum(), share) + 1)


# ------------------------------------------------------------------ blocks ----
def coverage(exam: pd.DataFrame, cells: pd.DataFrame, rec: pd.DataFrame) -> dict:
    """Truth inside the 10-90% range, split by whether the price moved at all."""
    x = exam.merge(cells[["series_id", "origin", "h", "e_lo", "e_hi", "d10", "d90"]], on=["series_id", "origin", "h"], how="left")
    assert x.e_lo.notna().all()
    x["moved"] = np.abs(x.y - x["last"]) > 1e-9
    arms = {"first_crowd": ("o_lo", "o_hi"), "improved_crowd": ("i_lo", "i_hi"), "engine": ("e_lo", "e_hi"), "drift": ("d10", "d90")}
    out = {"exam_days": {"n": int(len(x)), "n_moved": int(x.moved.sum()),
                         **{k: {"all": _inside(x, *v), "unchanged": _inside(x[~x.moved], *v), "moved": _inside(x[x.moved], *v)} for k, v in arms.items()},
                         "by_h": {int(h): {"n": int(len(g)), "moved_share": float(g.moved.mean()),
                                           **{k: _inside(g[g.moved], *v) for k, v in arms.items()}} for h, g in x.groupby("h")}}}
    b = cells.merge(rec[["series_id", "origin", "h"]], on=["series_id", "origin", "h"])
    b = b.assign(moved=np.abs(b.y - b["last"]) > 1e-9)
    out["blind_weeks"] = {"n": int(len(b)), "n_moved": int(b.moved.sum()),
                          "engine": {"all": _inside(b, "e_lo", "e_hi"), "unchanged": _inside(b[~b.moved], "e_lo", "e_hi"), "moved": _inside(b[b.moved], "e_lo", "e_hi")},
                          "drift": {"all": _inside(b, "d10", "d90"), "unchanged": _inside(b[~b.moved], "d10", "d90"), "moved": _inside(b[b.moved], "d10", "d90")}}
    return out


def weight(rec: pd.DataFrame) -> dict:
    """How concentrated the error is that every pooled figure adds up (three blind weeks, drift's error)."""
    ws, wd = rec.groupby("series_id").loss_drift.sum(), rec.groupby("target").loss_drift.sum()
    per = rec.groupby("series_id").agg(e=("loss_v0", "sum"), d=("loss_drift", "sum"))
    out = {"n": int(len(rec)), "prices": int(len(ws)), "effective_prices": _effective(ws), "prices_for_90pct": _needed(ws),
           "days": int(len(wd)), "effective_days": _effective(wd), "days_for_90pct": _needed(wd),
           "top3_prices": {k.replace(".median", ""): float(v) for k, v in (ws.sort_values(ascending=False) / ws.sum()).head(3).items()},
           "engine_vs_drift_pooled": float(1 - per.e.sum() / per.d.sum()),
           "prices_engine_ahead": int((per.e < per.d).sum()), "prices_engine_behind": int((per.e > per.d).sum()),
           "median_price_engine_vs_drift": float((1 - per.e / per.d)[per.d > 0].median()), "by_h": {}}
    for h, g in rec.groupby("h"):
        w = np.sort(g.loss_drift.to_numpy())[::-1]
        k = max(1, int(round(0.05 * len(w))))
        out["by_h"][int(h)] = {"n": int(len(w)), "origin_days": int(g.origin.nunique()), "prices": int(g.series_id.nunique()),
                               "share_with_no_error": float(np.mean(w == 0)), "top_5pct_cells_hold": float(w[:k].sum() / w.sum()),
                               "engine_vs_drift": report.skill_ci(g, "loss_v0", "loss_drift")}
    return out


def single_agents(test: pd.DataFrame, periods: pd.DataFrame) -> dict:
    """Each agent alone against the average of the five, and how often the five middles agree."""
    rows = []
    for j in json.loads((data.REPO / "runs" / "agents" / "extra_answers.json").read_text()):
        for h in HORIZONS:
            row = {"series_id": j["series_id"], "origin": day_of(j["origin"]), "h": h}
            for r in ROLES:
                row[r + "_q10"], row[r + "_q50"], row[r + "_q90"] = j["r1"][r]["q"][str(h)]
            row["crowd1_q10"], row["crowd1_q50"], row["crowd1_q90"] = j["crowd1"][str(h)]
            rows.append(row)
    e = periods.merge(pd.DataFrame(rows), on=["series_id", "origin", "h"], how="inner")
    for r in ROLES + ["crowd1"]:
        e["loss_" + r] = cell_loss(e.y, e[r + "_q10"], e[r + "_q50"], e[r + "_q90"])
    both = pd.concat([test.assign(loss_engine=test.loss_v0), e], ignore_index=True)
    out = {}
    for name, g in (("7 test days", test.assign(loss_engine=test.loss_v0)), ("7 earlier days", e), ("all 14 days", both)):
        mids = g[[f"{r}_q50" for r in ROLES]].to_numpy()
        out[name] = {"n": int(len(g)),
                     "vs_drift": {r: report.skill_ci(g, "loss_" + r, "loss_drift") for r in ROLES + ["crowd1"]},
                     "vs_engine": {r: report.skill_ci(g, "loss_" + r, "loss_engine") for r in ROLES + ["crowd1"]},
                     "vs_crowd": {r: report.skill_ci(g, "loss_" + r, "loss_crowd1") for r in ROLES},
                     "middles_within_0.1pct": float(np.mean(mids.max(1) / mids.min(1) - 1 <= 0.001)),
                     "all_middles_equal_last_value": float(np.mean(np.all(np.isclose(mids, g["last"].to_numpy()[:, None], rtol=1e-6), axis=1)))}
    return out


def ml_per_price(periods: pd.DataFrame) -> dict:
    """The ML model price by price; and with its middle held at the last value wherever the price's own
    history says the typical change is zero (drift's middle equals the last value). The second is a
    what-if on weeks already looked at, not a forecaster we tested blind."""
    m = periods.copy()
    snap = np.isclose(m.d50, m["last"], rtol=0, atol=1e-12)
    c50 = np.where(snap, m["last"], m.c50)
    m["loss_snap"] = cell_loss(m.y, np.minimum(m.c10, c50), c50, np.maximum(m.c90, c50))
    out = {"share_of_forecasts_held": float(snap.mean()), "blocks": {}}
    for name, g in (("8 earlier weeks", m[m.period < "2026-10-16"]), ("3 test weeks", m[m.period >= "2026-10-16"]), ("all 11 weeks", m)):
        per = g.groupby("series_id").agg(ml=("loss_ml", "sum"), d=("loss_drift", "sum"))
        ok = per[(per.d > 0) & (per.ml > 0)]
        out["blocks"][name] = {"n": int(len(g)), "pooled_vs_drift": report.skill_ci(g, "loss_ml", "loss_drift"),
                               "typical_price_vs_drift": float(1 - np.exp(np.mean(np.log(ok.ml / ok.d)))),   # geometric mean of the error ratio
                               "prices_ahead_of_drift": int((ok.ml < ok.d).sum()), "prices_counted": int(len(ok)),
                               "middle_held_vs_drift": report.skill_ci(g, "loss_snap", "loss_drift"),
                               "middle_held_vs_engine": report.skill_ci(g, "loss_snap", "loss_engine")}
    wk = m.groupby("period").agg(ml=("loss_ml", "sum"), sn=("loss_snap", "sum"), d=("loss_drift", "sum"), e=("loss_engine", "sum"))
    out["weeks_ahead_of_drift"] = {"as_it_comes": int((wk.ml < wk.d).sum()), "middle_held": int((wk.sn < wk.d).sum()),
                                   "engine": int((wk.e < wk.d).sum()), "weeks": int(len(wk))}
    return out


def aws_knowledge(exam: pd.DataFrame) -> dict:
    """The first crowd on AWS days before and after the date its model's knowledge ends."""
    ac = pd.read_parquet(OUT / "aws_cells.parquet")
    rows = []
    for j in json.loads((data.REPO / "runs" / "agents" / "aws_answers.json").read_text()):
        for h in HORIZONS:
            lo, mid, hi = j["crowd1"][str(h)]
            rows.append({"series_id": j["series_id"], "origin": day_of(j["origin"]), "h": h, "a10": lo, "a50": mid, "a90": hi})
    a = ac.merge(pd.DataFrame(rows), on=["series_id", "origin", "h"], how="inner")
    a["agents"] = cell_loss(a.y, a.a10, a.a50, a.a90)
    cols = ["series_id", "origin", "h", "target", "agents", "drift", "engine"]
    a = a.rename(columns={"loss_drift": "drift", "loss_engine": "engine"})[cols]
    xa = exam[exam.world == "aws"].drop(columns=["drift", "engine"]).rename(columns={"usd_orig": "agents", "usd_drift": "drift", "usd_engine": "engine"})[cols]
    both = pd.concat([a, xa], ignore_index=True)
    cut = day_of(KNOWLEDGE_END)
    out = {}
    for name, g in (("12 sampled days", a), ("11 days up to Feb 2026", a[a.origin <= cut]), ("4 days after April 2026", both[both.origin > cut]), ("all 15 days", both)):
        out[name] = {"n": int(len(g)), "days": sorted(iso_of(o) for o in g.origin.unique()),
                     "crowd_vs_drift": report.skill_ci(g, "agents", "drift"), "engine_vs_drift": report.skill_ci(g, "engine", "drift"),
                     "crowd_vs_engine": report.skill_ci(g, "agents", "engine")}
    return out


def coach_gain(exam: pd.DataFrame) -> dict:
    g = (exam.usd_orig - exam.usd_impr).groupby(exam.series_id).sum().sort_values(ascending=False)
    return {"n": int(len(exam)), "gain_share_of_first_crowd_error": float(g.sum() / exam.usd_orig.sum()),
            "top3_share_of_gain": float(g.head(3).sum() / g.sum()), "top3": {k.replace(".median", ""): float(v / g.sum()) for k, v in g.head(3).items()},
            "prices_better": int((g > 0).sum()), "prices_worse": int((g < 0).sum()), "prices_same": int((g == 0).sum()),
            "wider_ranges_alone": report.skill_ci(exam, "usd_stretch", "usd_orig"), "rewritten_agent_on_top": report.skill_ci(exam, "usd_impr", "usd_stretch")}


def exits(full: dict) -> dict:
    """Offers that leave a provider list. A quick return at the same price reads like a missed reading."""
    gaps = []
    for s in full["series"].values():
        if s["meta"]["instrument_class"] != "neocloud_rate_card":
            continue
        P = s["P"]
        pres = ~np.isnan(P)
        n = len(P)
        for j in range(P.shape[1]):
            col = pres[:, j]
            for i in np.where(col[:-1] & ~col[1:])[0] + 1:
                later = np.where(col[i:])[0]
                back = int(later[0]) if len(later) else -1
                gaps.append((s["meta"]["source_id"], back, bool(back > 0 and abs(P[i + back, j] - P[i - 1, j]) < 1e-9), int(n - i)))
    g = pd.DataFrame(gaps, columns=["source", "rows_missing", "same_price", "rows_left"])
    quick = (g.rows_missing > 0) & (g.rows_missing <= 3)
    return {"exits": int(len(g)), "came_back_at_all": int((g.rows_missing > 0).sum()), "back_within_3_readings": int(quick.sum()),
            "back_within_3_at_same_price": int((quick & g.same_price).sum()), "never_back": int((g.rows_missing < 0).sum()),
            "never_back_with_under_4_readings_left": int(((g.rows_missing < 0) & (g.rows_left < 4)).sum()),
            "by_site": {k: {"exits": int(len(v)), "back_within_3": int(((v.rows_missing > 0) & (v.rows_missing <= 3)).sum())} for k, v in g.groupby("source")}}


def euro(full: dict, cells: pd.DataFrame, movers: pd.DataFrame) -> dict:
    """Scaleway prices in euros; the tracking site converts at the ECB's daily rate. Its dollar price
    should then move only after a working day. Where is it the middle offer of a scored price?"""
    out = {"series": {}}
    non_aws = movers[~movers.provider.str.lower().eq("aws")]
    out["frequent_movers"] = {"all": int(len(movers)), "not_aws": int(len(non_aws)), "scaleway": int(non_aws.provider.str.lower().eq("scaleway").sum())}
    table = {d: [0, 0] for d in WEEKDAYS}
    seen = None
    for sid, s in full["series"].items():
        cols = [j for j, name in enumerate(s["slots"]) if name.lower().startswith("scaleway|")]
        if s["meta"]["source_id"] != "gpurentalprices" or not cols:
            continue
        P, days = s["P"], s["days"]
        with np.errstate(all="ignore"):
            sc = np.where((~np.isnan(P[:, cols])).sum(axis=1) == 1, np.nanmax(P[:, cols], axis=1), np.nan)   # one Scaleway offer a day
        n_in = (~np.isnan(P)).sum(axis=1)
        rank_lo = (P < sc[:, None]).sum(axis=1)
        mid = ~np.isnan(sc) & ((rank_lo == (n_in - 1) // 2) | (rank_lo == n_in // 2))
        only = ~np.isnan(sc) & (n_in % 2 == 1) & (rank_lo == (n_in - 1) // 2)        # odd list: the median IS Scaleway's price
        ok = (np.diff(days) == 1) & ~np.isnan(sc[1:]) & ~np.isnan(sc[:-1])
        chg = np.abs(np.diff(sc)) > 1e-9
        wd = (days[1:] + 6) % 7          # date.toordinal() % 7 == 1 is Monday
        if seen is None:                 # the weekday table is the same offer in every list; count it once
            seen = sid
            for k in np.where(ok)[0]:
                table[WEEKDAYS[wd[k]]][0] += int(chg[k]); table[WEEKDAYS[wd[k]]][1] += 1
        v = s["value"]
        okv = np.diff(days) == 1
        mv = np.abs(np.diff(v)) > 1e-9
        work = np.isin(wd, [1, 2, 3, 4, 5])       # Tue..Sat: the morning after an ECB working day
        out["series"][sid.replace(".median", "")] = {
            "days": int(len(days)), "scaleway_listed_days": int((~np.isnan(sc)).sum()),
            "scaleway_a_middle_offer_days": int(mid.sum()), "median_is_scaleway_price_days": int(only.sum()),
            "on_6_nov": {"offers": int(n_in[-1]), "scaleway_price": None if np.isnan(sc[-1]) else float(sc[-1]), "published": float(v[-1]),
                         "scaleway_is_the_middle": bool(only[-1]), "scaleway_among_middle_two": bool(mid[-1])},
            "published_price_moved": {"tue_to_sat": [int(mv[okv & work].sum()), int((okv & work).sum())],
                                      "sun_mon": [int(mv[okv & ~work].sum()), int((okv & ~work).sum())]}}
    out["scaleway_price_changed_by_weekday"] = {d: {"changed": a, "of": b} for d, (a, b) in table.items()}
    out["weekday_table_from"] = seen
    # in forecasting terms: 1-day forecasts whose target is a Sunday or Monday, for prices where Scaleway is the middle offer on the origin day
    rows = []
    for sid, s in full["series"].items():
        cols = [j for j, name in enumerate(s["slots"]) if name.lower().startswith("scaleway|")]
        if s["meta"]["source_id"] != "gpurentalprices" or not cols:
            continue
        P, days = s["P"], s["days"]
        with np.errstate(all="ignore"):
            sc = np.nanmax(P[:, cols], axis=1)
        n_in = (~np.isnan(P)).sum(axis=1)
        rank_lo = (P < sc[:, None]).sum(axis=1)
        is_mid = ~np.isnan(sc) & (n_in % 2 == 1) & (rank_lo == (n_in - 1) // 2)
        rows += [(sid, int(d)) for d, m in zip(days, is_mid) if m]
    k = pd.DataFrame(rows, columns=["series_id", "origin"])
    c = cells[(cells.h == 1) & (cells.origin >= day_of("2026-08-21"))].merge(k, on=["series_id", "origin"])
    c = c.assign(quiet=np.isin((c.target + 6) % 7, [6, 0]))      # target day is Sunday or Monday
    out["one_day_forecasts_where_scaleway_is_the_middle"] = {}
    for name, g in (("target Sun or Mon", c[c.quiet]), ("target Tue to Sat", c[~c.quiet])):
        out["one_day_forecasts_where_scaleway_is_the_middle"][name] = {
            "n": int(len(g)), "price_moved": int((np.abs(g.y - g["last"]) > 1e-9).sum()),
            "mean_error_usd": {"last_value": float(g.loss_last.mean()), "drift": float(g.loss_drift.mean()), "engine": float(g.loss_engine.mean())}}
    return out


def weekday_by_provider(full: dict, top: int = 8) -> dict:
    """How often each provider's offers change price, by the weekday of the reading. Added after the
    analyst (gpuval.analyst) found a weekday pattern that survives taking the euro-priced provider out."""
    rows = []
    for s in full["series"].values():
        P, days = s["P"], s["days"]
        ok = (np.diff(days) == 1)[:, None] & ~np.isnan(P[1:]) & ~np.isnan(P[:-1])
        chg = ok & (np.abs(P[1:] - P[:-1]) > 1e-9)
        wd = (days[1:] + 6) % 7
        prov = np.array([x.split("|")[0].lower() for x in s["slots"]])
        for p in np.unique(prov):
            m = prov == p
            for d in range(7):
                rows.append((p, d, int(ok[wd == d][:, m].sum()), int(chg[wd == d][:, m].sum())))
    df = pd.DataFrame(rows, columns=["provider", "wd", "n", "changed"]).groupby(["provider", "wd"]).sum().reset_index()
    tot = df.groupby("provider").changed.sum().sort_values(ascending=False)
    out = {}
    for p in list(tot.head(top).index) + ["everyone else"]:
        g = df[df.provider == p] if p != "everyone else" else df[~df.provider.isin(tot.head(top).index)]
        g = g.groupby("wd")[["n", "changed"]].sum()
        week, end = g.loc[[0, 1, 2, 3, 4]].sum(), g.loc[[5, 6]].sum()
        out[p] = {"changes": int(g.changed.sum()), "offer_days": int(g.n.sum()),
                  "by_weekday": {WEEKDAYS[d]: (float(g.changed[d] / g.n[d]) if g.n[d] else None) for d in range(7)},
                  "mon_to_fri": float(week.changed / week.n) if week.n else None, "sat_sun": float(end.changed / end.n) if end.n else None}
    return out


def aws_flat_zones() -> dict:
    """The AWS archive holds price changes, not readings, so a zone that stops changing is carried
    forward for ever. We cannot tell a flat price from a zone that has gone."""
    p = aws_test.WORK / "world.pkl"
    if not p.exists():
        return {"skipped": "AWS world not built (uv run python -m gpuval.aws_test build)"}
    aw = data.load_world(p)
    tot = flat = 0
    rows = []
    for sid, s in aw["series"].items():
        P = s["P"]
        pres = ~np.isnan(P)
        same = np.zeros(P.shape, dtype=int)
        for i in range(1, len(P)):
            eq = pres[i] & pres[i - 1] & (np.abs(P[i] - P[i - 1]) < 1e-12)
            same[i] = np.where(eq, same[i - 1] + 1, 0)
        f = same >= 60
        tot += int(pres.sum()); flat += int(f.sum())
        rows.append((sid.replace("aws_spot.", "").replace(".az_median", ""), int(pres.sum()), float(f.sum() / max(pres.sum(), 1))))
    rows.sort(key=lambda r: -r[2])
    return {"zone_days": tot, "unchanged_60_days_or_more": flat, "share": flat / tot, "highest": [{"type": a, "zone_days": b, "share": c} for a, b, c in rows[:5]]}


def beyond_the_37(max_offers: int = 3000) -> dict:
    """The engine on every other daily-median rental price in the pack, same eleven weeks and rules.
    The 37 scored prices were picked by the starter kit for having history, providers and movement;
    the rest mostly have short histories, one or two providers or prices that hardly move.

        uv run python -m gpuval.checks wide          # a few minutes
    """
    cat = pd.read_csv(data.CACHE / f"starter_{data.DATA_END}" / "catalog.csv", low_memory=False)
    cat = cat[(cat.table == "daily_statistics") & cat.series_id.str.endswith(".median")]
    head = set(data.headline_ids())
    world = data.build_world(data.DATA_END, series_ids=sorted(cat.series_id))
    v = data.verify_world(world).set_index("series_id")
    keep = [sid for sid, s in world["series"].items()
            if sid in head or (v.match_share.get(sid, 0) >= 0.99 and s["P"].shape[1] <= max_offers)]
    dropped = sorted(set(world["series"]) - set(keep))
    world["series"] = {sid: world["series"][sid] for sid in keep}
    c = harness.base_cells(world)
    c = c[(c.origin >= day_of("2026-08-21")) & (c.origin <= day_of("2026-11-05"))]
    r = harness.run_spec(world, default_spec(), c)
    r["scored"] = r.series_id.isin(head)
    reasons = cat.set_index("series_id").tier_reason
    out = {"series_in_catalog": int(len(cat)), "series_rebuilt": int(len(keep)), "left_out": dropped, "blocks": {}}
    groups = [("the 37 scored prices", r[r.scored]), ("every other price", r[~r.scored]),
              ("other prices with 3 or more providers", r[~r.scored & ~r.series_id.map(reasons).fillna("").str.contains("median_providers<3")]),
              ("other prices that drift ever misses", None)]
    for name, g in groups:
        if g is None:
            live = r[~r.scored].groupby("series_id").loss_drift.sum()
            g = r[~r.scored & r.series_id.isin(live[live > 0].index)]
        per = g.groupby("series_id").agg(e=("loss", "sum"), d=("loss_drift", "sum"))
        out["blocks"][name] = {"n": int(len(g)), "prices": int(g.series_id.nunique()), "engine_vs_drift": report.skill_ci(g, "loss", "loss_drift"),
                               "prices_engine_ahead": int((per.e < per.d).sum()), "prices_engine_behind": int((per.e > per.d).sum()),
                               "prices_level": int((per.e == per.d).sum()), "share_of_forecasts_where_price_moved": float((np.abs(g.y - g["last"]) > 1e-9).mean())}
    by = r[~r.scored].groupby("cls")
    out["other_prices_by_kind"] = {k: {"n": int(len(g)), "prices": int(g.series_id.nunique()), "engine_vs_drift": report.skill_ci(g, "loss", "loss_drift")} for k, g in by}
    return out


# ------------------------------------------------------------------- main ----
def main_wide() -> int:
    f = report.fmt_ci
    out = beyond_the_37()
    (OUT / "checks_wide.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"BEYOND THE 37: {out['series_rebuilt']} of {out['series_in_catalog']} daily-median prices rebuilt from their offer rows; forecast days 21 Aug to 5 Nov")
    for name, v in out["blocks"].items():
        print(f"  {name:40s} {v['n']:6d} forecasts, {v['prices']:3d} prices: engine vs drift {f(v['engine_vs_drift'])} | ahead on {v['prices_engine_ahead']}, behind on {v['prices_engine_behind']}, "
              f"level on {v['prices_level']} | price moved in {v['share_of_forecasts_where_price_moved']:.2f} of forecasts")
    for k, v in out["other_prices_by_kind"].items():
        print(f"     other prices, {k:20s} {v['n']:6d} forecasts, {v['prices']:3d} prices: {f(v['engine_vs_drift'])}")
    return 0


def main() -> int:
    pd.set_option("display.width", 200)
    f = report.fmt_ci
    full = data.build_world(data.DATA_END)
    cells = harness.run_spec(full, default_spec(), harness.base_cells(full)).rename(
        columns={"loss": "loss_engine", "q10": "e_lo", "q50": "e_mid", "q90": "e_hi"})
    rec = pd.read_parquet(data.REPO / "runs" / "record" / "real_gpt" / "cells.parquet")
    periods = pd.read_parquet(OUT / "periods_cells.parquet")
    test = pd.read_parquet(data.REPO / "runs" / "agents" / "test_cells.parquet")
    exam_all = pd.read_parquet(data.REPO / "runs" / "coach" / "exam_cells.parquet")
    exam = exam_all[exam_all.world == "ours"].copy()
    exam["origin"] = exam.id.str.split("|").str[2].map(day_of)
    _, movers = insight.decompose(full)
    out = {"coverage": coverage(exam, cells, rec), "weight": weight(rec), "single_agents": single_agents(test, periods),
           "ml_per_price": ml_per_price(periods), "aws_knowledge": aws_knowledge(exam_all.assign(origin=exam_all.id.str.split("|").str[2].map(day_of))),
           "coach_gain": coach_gain(exam), "exits": exits(full), "euro": euro(full, cells, movers),
           "weekday_by_provider": weekday_by_provider(full), "aws_flat_zones": aws_flat_zones()}
    (OUT / "checks.json").write_text(json.dumps(out, indent=1, default=str))

    c = out["coverage"]["exam_days"]
    print(f"COVERAGE, coach's 5 exam days of our prices ({c['n']} forecasts, {c['n_moved']} where the price moved). Truth inside the range:")
    for k in ("first_crowd", "improved_crowd", "engine", "drift"):
        print(f"  {k:15s} all {c[k]['all']:.3f} | price unchanged {c[k]['unchanged']:.3f} | price moved {c[k]['moved']:.3f}")
    b = out["coverage"]["blind_weeks"]
    print(f"  three blind weeks ({b['n']}, {b['n_moved']} moved): engine all {b['engine']['all']:.3f}, unchanged {b['engine']['unchanged']:.3f}, moved {b['engine']['moved']:.3f}")
    w = out["weight"]
    print(f"\nWEIGHT, three blind weeks ({w['n']} forecasts): drift's error sits on an effective {w['effective_prices']:.1f} of {w['prices']} prices "
          f"({w['prices_for_90pct']} carry 90%) and {w['effective_days']:.1f} of {w['days']} days ({w['days_for_90pct']} carry 90%)")
    print(f"  engine ahead of drift on {w['prices_engine_ahead']} prices, behind on {w['prices_engine_behind']}; median price {w['median_price_engine_vs_drift']:+.3f}; pooled {w['engine_vs_drift_pooled']:+.3f}")
    for h, v in w["by_h"].items():
        print(f"  {h}d: {v['n']} forecasts from {v['origin_days']} forecast days, {v['prices']} prices | top 5% of forecasts hold {v['top_5pct_cells_hold']:.2f} of the error | engine vs drift {f(v['engine_vs_drift'])}")
    for name, v in out["single_agents"].items():
        print(f"\nSINGLE AGENTS, {name} ({v['n']} forecasts): five middles within 0.1%: {v['middles_within_0.1pct']:.3f}")
        for r in ROLES + ["crowd1"]:
            print(f"  {r:18s} vs drift {f(v['vs_drift'][r])} | vs engine {f(v['vs_engine'][r])}" + (f" | vs the average {f(v['vs_crowd'][r])}" if r in v["vs_crowd"] else ""))
    m = out["ml_per_price"]
    print(f"\nML MODEL per price (middle held at 'no change' on {m['share_of_forecasts_held']:.2f} of forecasts in the what-if):")
    for name, v in m["blocks"].items():
        print(f"  {name}: pooled {f(v['pooled_vs_drift'])} | typical price {v['typical_price_vs_drift']:+.3f} | ahead on {v['prices_ahead_of_drift']} of {v['prices_counted']} prices "
              f"| middle held: {f(v['middle_held_vs_drift'])}, vs engine {f(v['middle_held_vs_engine'])}")
    print("  weeks ahead of drift:", m["weeks_ahead_of_drift"])
    print("\nAWS, first crowd, by whether the day is after the model's knowledge ends:")
    for name, v in out["aws_knowledge"].items():
        print(f"  {name:24s} n={v['n']:4d}: crowd vs drift {f(v['crowd_vs_drift'])} | engine vs drift {f(v['engine_vs_drift'])} | crowd vs engine {f(v['crowd_vs_engine'])}")
    g = out["coach_gain"]
    print(f"\nCOACH GAIN, exam days: top 3 prices supply {g['top3_share_of_gain']:.2f} of it; better on {g['prices_better']} prices, worse on {g['prices_worse']} "
          f"| wider ranges alone {f(g['wider_ranges_alone'])} | rewritten agent on top {f(g['rewritten_agent_on_top'])}")
    print("\nEXITS from provider lists:", {k: v for k, v in out["exits"].items() if k != "by_site"})
    e = out["euro"]
    print("\nEURO: Scaleway's dollar price changed, by weekday:", {d: f"{v['changed']}/{v['of']}" for d, v in e["scaleway_price_changed_by_weekday"].items()})
    print("  frequent movers:", e["frequent_movers"])
    for sid, v in e["series"].items():
        print(f"  {sid:58s} middle offer on {v['scaleway_a_middle_offer_days']:2d} of {v['scaleway_listed_days']} days; 6 Nov: {v['on_6_nov']}; "
              f"published price moved Tue-Sat {v['published_price_moved']['tue_to_sat']}, Sun-Mon {v['published_price_moved']['sun_mon']}")
    for name, v in e["one_day_forecasts_where_scaleway_is_the_middle"].items():
        print(f"  1-day forecasts, Scaleway in the middle, {name}: n={v['n']}, price moved {v['price_moved']}, mean error {v['mean_error_usd']}")
    print("\nREPRICING BY WEEKDAY of the reading (share of an offer's listed days on which its price changed):")
    for p, v in out["weekday_by_provider"].items():
        r3 = lambda x: "-" if x is None else f"{x:.3f}"
        print(f"  {p:16s} {v['changes']:7d} changes | Mon-Fri {r3(v['mon_to_fri'])} | Sat-Sun {r3(v['sat_sun'])} | "
              + " ".join(f"{d} {x:.2f}" if x is not None else f"{d} -" for d, x in v["by_weekday"].items()))
    print("\nAWS FLAT ZONES:", out["aws_flat_zones"])
    print("\nwrote", (OUT / "checks.json").relative_to(data.REPO))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main_wide() if sys.argv[1:] == ["wide"] else main())
