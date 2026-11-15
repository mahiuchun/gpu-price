#!/usr/bin/env python3
"""Long view: what will H100 and A100 GPUs rent for on AWS spot over 1 to 3 years?

Run from the repo root:   uv run python longview/longview.py

Regenerates everything under longview/out/ deterministically (fixed seed):
  1. what happened to the older chip when the newer one arrived (step 1),
  2. a roll-forward test of a step simulator at 30 to 730 days (step 2),
  3. a "new chip arrives" adjustment and whether it helps (step 3),
  4. the outlook from 31 Oct 2026 at +1, +2, +3 years (step 4).

Revised 15 Nov 2026, after reading the brief's references (the long view is not scored against the
7-13 Nov truth):
  * the spot price is set against AWS's on-demand list price (step 0): the list price per region is
    read from the pack, the share of list that spot trades at is reported through time, and every
    outlook path is capped at today's list price (the uncapped paths are kept as "*_nocap");
  * the before/after-arrival comparison is run on every older-chip and successor pair the archive
    supports, beside the same comparison at dates when nothing arrived (step 1b);
  * the successor adjustment is also tested out of sample, carried from the A100 to the H100 (step 3);
  * three paths that fall at a steady hardware-trend rate are written beside the outlook (step 4).

Instrument throughout: AWS EC2 Linux/UNIX SPOT price, USD per GPU-hour (instance
price / 8 GPUs), end-of-day value, median across availability zones.  It is an
administered price that moves in steps; it is not an on-demand list price and
not a resale price.  The archive has no data for 1 Mar to 30 Jun 2026: nothing
is interpolated or carried across that hole.

Choices that matter (all stated in LONGVIEW.md):
  * the simulator resamples past daily changes, conditional on yesterday's move
    and on which side of its usual level (median of history) the price is;
  * days when more than a fifth of a composite's zones were under 90 days old
    are not learned from (new zones open at a floor price and ramp up);
  * each chip's simulated 10-90% range is widened at both ends until it would
    have contained 80% of that chip's own test outcomes;
  * the successor effect is measured on the A100 around the H100's arrival and
    is in-sample when tested on the A100.

Reads only: aws_spot_ticks/, capability/, hardware_fundamentals/ (via the pack's
starter/build_series.py for the daily series) and, for list prices, the AWS on-demand rows of
rental_observations/.  No network.  The pack is found at ./gpu-price-pack-v0.1.1-handover or at
GPUVAL_PACK.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import zlib

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------- config ----
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PACK = os.environ.get("GPUVAL_PACK", os.path.join(ROOT, "gpu-price-pack-v0.1.1-handover"))
OUT = os.path.join(HERE, "out")

SEED = 20261114
HOLE_START, HOLE_END = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-06-30")
HORIZONS = (30, 90, 180, 365, 730)          # roll-forward test
OUTLOOK_HORIZONS = (365, 730, 1095)         # +1, +2, +3 years
N_PATHS_TEST, N_PATHS_OUTLOOK = 2000, 20000
MIN_TRAIN_DAYS = 120                        # history needed before the first test origin
END_WINDOW = 30                             # days in the start/end medians of a rate
MIN_DAYS_TO_ANNUALISE = 180                 # shorter windows get a plain change, not a yearly rate
MIN_WINDOWS_TO_WIDEN = 2                    # independent test windows needed before a horizon's misses set the widening
YOUNG_ZONE_DAYS, YOUNG_ZONE_SHARE = 90, 0.20  # see young_zone_mask()
CAP_SET = "ship_only_inf_cross_benchmark_median_a100ref_sxm80"   # A100 = 1.0 unit of work
EPS = 1e-12

INST = {"A100": "p4d.24xlarge", "A100_80GB": "p4de.24xlarge", "H100": "p5.48xlarge",
        "H200": "p5en.48xlarge", "H200_p5e": "p5e.48xlarge",
        "B200": "p6-b200.48xlarge", "B300": "p6-b300.48xlarge"}

# Chips used only in the all-pairs comparison of step 1b: one standard machine size per chip.
PAIR_INST = {"K80": "p2.xlarge", "V100": "p3.2xlarge", "T4": "g4dn.xlarge", "A10G": "g5.xlarge",
             "L4": "g6.xlarge", "L40S": "g6e.xlarge"}
LINES = {"training": ["K80", "V100", "A100", "H100", "H200", "B200"], "inference": ["T4", "A10G", "L4", "L40S"]}
PAIR_WINDOW, PAIR_MIN_DAYS = 365, 180        # days compared each side of an arrival; fewer than 180 is not annualised
PAIR_ZONE_SHARE = 0.80                       # a day counts when this share of the fixed zone set has a price
PLACEBO_GAP = 90                             # comparison dates this close to any real arrival in the line are left out
# Chips whose on-demand list price AWS cut in June 2025. The sizes for P5, P4d and P4de are checked
# against the pack in list_cut_check(); P5en is as announced only (the pack cannot show it).
LIST_CUT_CHIPS = ("A100", "A100_80GB", "H100", "H200")
AT_CEILING = 0.995                           # a zone at or above this share of its list price is "at the ceiling"
HARDWARE_TRENDS = (-0.18, -0.25, -0.35)      # steady yearly change in rent; see LONGVIEW.md section 4

# chart tokens (validated categorical order: blue, orange, aqua, yellow)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COL = {"A100": "#2a78d6", "H100": "#eb6834", "H200": "#1baf7a", "B200": "#eda100"}


def say(*a):
    print(*a, flush=True)


def pct(log_rate: float) -> float:
    """annualised log rate -> percent per year"""
    return (math.exp(log_rate) - 1.0) * 100.0 if np.isfinite(log_rate) else float("nan")


# ------------------------------------------------------------------- data ----
def build_pack_series() -> None:
    """Daily AWS series from the pack's own starter script (per-AZ closes + az_median)."""
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")   # never write __pycache__ into the pack
    cmd = [sys.executable, "-B", os.path.join(PACK, "starter", "build_series.py"), "--pack", PACK,
           "--no-daily-statistics", "--aws-instances", ",".join(list(INST.values()) + list(PAIR_INST.values())),
           "--out", os.path.join(OUT, "aws_series.csv"),
           "--catalog", os.path.join(OUT, "aws_catalog.csv")]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"build_series.py failed:\n{res.stdout}\n{res.stderr}")
    say("  " + res.stdout.strip())


def load_series():
    s = pd.read_csv(os.path.join(OUT, "aws_series.csv"), usecols=["series_id", "date", "value"], parse_dates=["date"])
    cat = pd.read_csv(os.path.join(OUT, "aws_catalog.csv"), dtype=str, keep_default_na=False)
    zones, pack = {}, {}
    for chip, inst in {**INST, **PAIR_INST}.items():
        pre = f"aws_spot.{inst}."
        g = s[s.series_id.str.startswith(pre)]
        comp = g[g.series_id == pre + "az_median"].set_index("date")["value"].sort_index()
        if len(comp):
            pack[chip] = comp
        z = g[g.series_id != pre + "az_median"].copy()
        z["az"] = z.series_id.str[len(pre):]
        zones[chip] = z.pivot(index="date", columns="az", values="value").sort_index()
    every = {**INST, **PAIR_INST}
    members = {chip: [m.split(".")[-1] for m in
                      cat.loc[cat.series_id == f"aws_spot.{every[chip]}.az_median", "members"].iloc[0].split(";")]
               for chip in pack}
    return zones, pack, members


def first_ticks() -> pd.DataFrame:
    """First Linux/UNIX spot tick per instance type, straight from the raw change events."""
    import duckdb
    files = os.path.join(PACK, "aws_spot_ticks", "*.parquet")
    every = {**INST, **PAIR_INST}
    inst = ", ".join(f"'{v}'" for v in every.values())
    q = f"""SELECT instance_type, min(ts_utc) AS first_tick_utc, max(ts_utc) AS last_tick_utc,
                   arg_min(usd_per_gpu_hour, ts_utc) AS first_usd_per_gpu_hour,
                   arg_min(az_global_id, ts_utc) AS first_zone, count(*) AS n_ticks,
                   count(DISTINCT az_global_id) AS n_zones_ever
            FROM read_parquet('{files}')
            WHERE instance_type IN ({inst}) AND product_description = 'Linux/UNIX'
            GROUP BY 1 ORDER BY 2"""
    con = duckdb.connect()
    df = con.execute(q).df()
    con.close()
    back = {v: k for k, v in every.items()}
    df.insert(0, "chip", df.instance_type.map(back))
    return df


# --------------------------------------------------------- hardware trend ----
def hardware_trend_fit() -> dict:
    """How fast a launch dollar buys more computing, on the pack's copy of Epoch AI's hardware table:
    a straight line through log(speed per launch dollar) against release date, GPUs with a launch price.
    Two speed measures: single-precision (FP32) and each chip's fastest number format ("Max performance").
    The range is the 5% to 95% points of 4,000 resamples of the chips."""
    m = pd.read_csv(os.path.join(PACK, "hardware_fundamentals", "ml_hardware.csv"))
    m = m[m["Type"] == "GPU"]
    t = (pd.to_datetime(m["Release date"], errors="coerce") - pd.Timestamp("2000-01-01")).dt.days / 365.25
    price = pd.to_numeric(m["Release price (USD)"], errors="coerce")
    out = {}
    for name, col in (("fp32_per_launch_dollar", "FP32 (single precision) performance (FLOP/s)"),
                      ("fastest_format_per_launch_dollar", "Max performance")):
        y = np.log(pd.to_numeric(m[col], errors="coerce") / price)
        ok = (np.isfinite(y) & t.notna()).to_numpy()
        x, yy = t.to_numpy()[ok], y.to_numpy()[ok]
        slope = float(np.polyfit(x, yy, 1)[0])
        rng = make_rng("hardware_trend", name)
        bs = np.array([np.polyfit(x[i], yy[i], 1)[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(4000))])
        lo, hi = np.quantile(bs, [0.05, 0.95])
        out[name] = dict(n_gpus=int(len(x)), first_release_year=float(2000 + x.min()), last_release_year=float(2000 + x.max()),
                         cost_pct_per_year=pct(-slope), cost_pct_per_year_slow=pct(-lo), cost_pct_per_year_fast=pct(-hi),
                         halving_years=math.log(2) / slope)
    return out


# ------------------------------------------------------------ list prices ----
_AZ = re.compile(r"^([a-z]{2})(ne|se|nw|sw|n|s|e|w|c)(\d)-az\d+$")
_DIR = {"n": "north", "s": "south", "e": "east", "w": "west", "c": "central", "ne": "northeast",
        "se": "southeast", "nw": "northwest", "sw": "southwest"}


def zone_region(az: str) -> str:
    """'use1-az5' -> 'us-east-1' (the region whose list price applies to a zone)."""
    m = _AZ.match(az)
    return f"{m.group(1)}-{_DIR[m.group(2)]}-{m.group(3)}" if m else ""


def _pack_rows(source: str, select: str, where: str, tail: str) -> pd.DataFrame:
    import duckdb
    files = os.path.join(PACK, "rental_observations", f"source_id={source}", "*", "*.parquet")
    q = f"""SELECT {select} FROM read_parquet('{files}', hive_partitioning=true, union_by_name=true)
            WHERE provider = 'aws' AND instrument_class = 'hyperscaler_list' AND pricing_tier = 'on_demand' AND {where} {tail}"""
    con = duckdb.connect()
    df = con.execute(q).df()
    con.close()
    return df


def list_prices(origin) -> pd.DataFrame:
    """AWS on-demand list price per machine type and region, USD per GPU-hour, from the pack's own
    offer rows (collector afhubbard, instrument hyperscaler_list). One row per (instance, region):
    the last price on or before `origin` and how many different prices that region showed."""
    inst = ", ".join(f"'{v}'" for v in INST.values())
    df = _pack_rows("afhubbard",
                    "sku_raw AS instance_type, region_raw AS region, min(dt) AS first_seen, max(dt) AS last_seen, "
                    "arg_max(usd_per_gpu_hour, snapshot_ts_utc) AS list_usd_per_gpu_hour, "
                    "count(DISTINCT round(usd_per_gpu_hour, 4)) AS n_distinct_prices, count(DISTINCT dt) AS n_days",
                    f"sku_raw IN ({inst}) AND dt <= DATE '{pd.Timestamp(origin).date()}'", "GROUP BY 1, 2 ORDER BY 1, 5, 2")
    back = {v: k for k, v in INST.items()}
    df.insert(0, "chip", df.instance_type.map(back))
    return df


def list_cut_check(origin) -> pd.DataFrame:
    """The June 2025 list cut, as far as the pack can show it. One tracking site (gpurentalprices)
    still carried AWS's earlier on-demand prices on its first days in the pack and then switched to
    the current ones; the two levels give the size of the cut. One row per GPU that shows both."""
    df = _pack_rows("gpurentalprices",
                    "gpu_model, round(usd_per_gpu_hour, 4) AS usd, min(dt) AS first_seen, max(dt) AS last_seen, count(*) AS n_rows",
                    f"gpu_model IN ('H100 SXM 80GB', 'A100 40GB', 'A100 SXM 80GB') AND dt <= DATE '{pd.Timestamp(origin).date()}'",
                    "GROUP BY 1, 2 ORDER BY 1, 3")
    chip_of = {"H100 SXM 80GB": "H100", "A100 40GB": "A100", "A100 SXM 80GB": "A100_80GB"}
    rows = []
    for gpu, g in df.groupby("gpu_model"):
        if len(g) != 2:
            continue                                   # no earlier reading for this GPU: nothing to compare
        old, new = g.iloc[0], g.iloc[1]
        rows.append(dict(chip=chip_of[gpu], gpu_model=gpu, earlier_usd=float(old.usd), earlier_first_seen=str(old.first_seen)[:10],
                         earlier_last_seen=str(old.last_seen)[:10], current_usd=float(new.usd),
                         current_first_seen=str(new.first_seen)[:10], cut_pct=(float(new.usd) / float(old.usd) - 1) * 100,
                         factor=float(new.usd) / float(old.usd)))
    return pd.DataFrame(rows)


def spot_vs_list(zones, pack, members, lp, cut) -> tuple[pd.DataFrame, dict]:
    """Spot as a share of the on-demand list price, zone by zone and month by month.

    Every zone is compared with its own region's list price. The pack's list rows start on
    12 Jan 2026 and show one price per region throughout, so that price is used from the day the
    cut reached the spot ceilings. Before that day the list in force is the current one divided by
    the cut factor (checked: before the cut, zones sat exactly at that higher ceiling). For a chip
    with no cut factor in the pack the months before the cut are left blank."""
    factor = dict(zip(cut.chip, cut.factor))

    def zone_list(chip):
        r = lp[lp.chip == chip].set_index("region").list_usd_per_gpu_hour
        return r, pd.Series({a: r.get(zone_region(a), np.nan) for a in zones[chip].columns}, dtype=float)

    # the day the cut reached spot: the day after the last day any H100 zone traded above today's list price
    above = (zones["H100"] / zone_list("H100")[1] > 1.005).any(axis=1)
    cut_seen = above[above].index.max() + pd.Timedelta(days=1)
    rows, info = [], {"cut_reached_spot_ceilings_on": str(cut_seen.date())}
    for chip in ("H100", "A100", "A100_80GB", "H200"):
        z = zones[chip]
        r, zl = zone_list(chip)
        new = z / zl                                              # share of today's list price
        after = pd.Series(z.index >= cut_seen, index=z.index)
        inforce = new.where(after, new * factor[chip], axis=0) if chip in factor else new.where(after, axis=0)
        mem = [m for m in members[chip] if np.isfinite(zl.get(m, np.nan))]
        comp_list = float(zl[mem].median())
        comp = pack[chip]
        lpc = lp[lp.chip == chip]
        info[chip] = dict(
            regions_with_a_list_price=int(len(r)), list_low=float(r.min()), list_median_of_regions=float(r.median()),
            list_high=float(r.max()), regions_whose_price_changed_in_the_pack=int((lpc.n_distinct_prices > 1).sum()),
            list_first_seen=str(lpc.first_seen.min())[:10], list_last_seen=str(lpc.last_seen.max())[:10],
            composite_zones=len(members[chip]), composite_zones_with_a_list_price=len(mem), composite_list_price=comp_list,
            spot_composite_last_day=float(comp.iloc[-1]), last_day=str(comp.index[-1].date()),
            spot_share_of_list_last_day=float(comp.iloc[-1] / comp_list),
            max_zone_share_of_list_since_cut=float(np.nanmax(new[after.to_numpy()].to_numpy())),
            zone_days_since_cut=int(new[after.to_numpy()].notna().sum().sum()),
            zone_days_above_list_since_cut=int((new[after.to_numpy()] > 1.005).sum().sum()),
            max_zone_share_of_todays_list_before_cut=float(np.nanmax(new[~after.to_numpy()].to_numpy())) if (~after).any() else float("nan"),
            cut_factor_from_earlier_reading=factor.get(chip, float("nan")))
        long = pd.DataFrame({"share": inforce.stack(), "spot": z.stack()}).dropna(subset=["spot"]).reset_index()
        long.columns = ["date", "az", "share", "spot"]
        long["month"] = long.date.dt.strftime("%Y-%m")
        for month, g in long.groupby("month"):
            if month >= "2026-01":
                basis = "list price seen in the pack"
            elif g.date.min() >= cut_seen:
                basis = "same list price, assumed in force since the cut"
            elif g.date.max() < cut_seen:
                basis = "list price before the cut (current price / cut factor)" if chip in factor else "not known"
            else:
                basis = "cut month: both list prices" if chip in factor else "cut month: after the cut only"
            ok = g.share.notna()
            rows.append(dict(chip=chip, instance_type=INST[chip], month=month, n_zones=int(g.az.nunique()), zone_days=len(g),
                             spot_usd_median=float(g.spot.median()), list_basis=basis,
                             spot_share_of_list_median=float(g.share[ok].median()) if ok.any() else np.nan,
                             spot_share_of_list_max=float(g.share[ok].max()) if ok.any() else np.nan,
                             share_of_zone_days_at_ceiling=float((g.share[ok] >= AT_CEILING).mean()) if ok.any() else np.nan))

        def med(a, b):
            m = long[(long.date >= pd.Timestamp(a)) & (long.date <= pd.Timestamp(b))].share.dropna()
            return float(m.median()) if len(m) else float("nan")

        info[chip]["share_of_list_by_period"] = {
            "90 days before the cut": med(cut_seen - pd.Timedelta(days=90), cut_seen - pd.Timedelta(days=1)),
            "Jul 2025 to Feb 2026": med("2025-07-01", "2026-02-28"),
            "Jul to Sep 2026": med("2026-07-01", "2026-09-30"),
            "Oct 2026": med("2026-10-01", "2026-10-31")}
    return pd.DataFrame(rows), info


def founding_composite(z: pd.DataFrame, live_by: str) -> tuple[pd.Series, list[str]]:
    """Median over the zones that already had a price by `live_by` (fixed membership),
    on days when every one of them has a value."""
    first = z.apply(lambda c: c.first_valid_index())
    members = sorted(first[first <= pd.Timestamp(live_by)].index)
    return z[members].dropna().median(axis=1), members


def young_zone_mask(z: pd.DataFrame, members: list[str], index: pd.DatetimeIndex, archive_start) -> pd.Series:
    """True on days the simulator may LEARN from.  A new zone opens at an administrative floor price
    and ramps up for weeks; when many zones open together the zone-median composite climbs for a
    reason that has nothing to do with the market.  Days on which more than YOUNG_ZONE_SHARE of the
    composite's zones are younger than YOUNG_ZONE_DAYS are therefore not used to learn step
    behaviour (they still count as prices: forecast origins and truths are untouched).  Zones already
    present in the archive's first week are treated as old."""
    z = z[members].reindex(index)
    first = z.apply(lambda c: c.first_valid_index())
    present = z.notna()
    young = pd.DataFrame({m: (index >= first[m]) & (index < first[m] + pd.Timedelta(days=YOUNG_ZONE_DAYS))
                             & (first[m] > archive_start + pd.Timedelta(days=7)) for m in members}, index=index)
    share = (young & present).sum(axis=1) / present.sum(axis=1).clip(lower=1)
    return share <= YOUNG_ZONE_SHARE


def capability_units() -> dict:
    cu = pd.read_csv(os.path.join(PACK, "capability", "capability_units.csv"))
    cu = cu[cu.benchmark_set_id == CAP_SET].set_index("gpu_generation")["capability_units_per_accelerator"]
    return {k: float(cu[k]) for k in ("A100", "H100", "H200", "B200", "B300")}


def release_dates() -> dict:
    hw = pd.read_csv(os.path.join(PACK, "hardware_fundamentals", "hardware_fundamentals.csv"))
    names = {"A100": "NVIDIA A100 SXM4 40 GB", "A100_80GB": "NVIDIA A100 SXM4 80 GB", "H100": "NVIDIA H100 SXM5 80GB",
             "H200": "NVIDIA H200 SXM", "B200": "NVIDIA B200", "B300": "NVIDIA B300 (Blackwell Ultra)",
             "VR200": "NVIDIA VR200 (Vera Rubin)"}
    hw = hw.set_index("accelerator")["release_date"]
    return {k: str(hw.get(v, "")) for k, v in names.items()}


# ----------------------------------------------------------------- step 1 ----
def rate(x: pd.Series, a, b, k: int = END_WINDOW) -> dict:
    """Annualised log rate of change of x between dates a..b (inclusive):
    'endpoint' = between the medians of the first and last k days;
    'trend'    = slope of a straight line through log price."""
    seg = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()
    if len(seg) < 2 * k:
        return dict(n_days=len(seg), start_level=np.nan, end_level=np.nan, endpoint=np.nan, trend=np.nan, plain=np.nan)
    lv0, lv1 = float(seg.iloc[:k].median()), float(seg.iloc[-k:].median())
    plain = float(seg.iloc[-1] / seg.iloc[0] - 1.0) * 100.0          # first day to last day, not annualised
    if len(seg) < MIN_DAYS_TO_ANNUALISE:                             # a few months must not be blown up to a year
        return dict(n_days=len(seg), start_level=lv0, end_level=lv1, endpoint=np.nan, trend=np.nan, plain=plain)
    endpoint = (math.log(lv1) - math.log(lv0)) / ((len(seg) - k) / 365.25)
    t = (seg.index - seg.index[0]).days / 365.25
    trend = float(np.polyfit(t, np.log(seg.to_numpy()), 1)[0])
    return dict(n_days=len(seg), start_level=lv0, end_level=lv1, endpoint=endpoint, trend=trend, plain=plain)


def step1(zones, pack, comps, avail, units):
    """Price paths of the older chip around the newer chip's arrival."""
    a_h100, a_h200, a_b200 = avail["H100"], avail["H200"], avail["B200"]
    day = pd.Timedelta(days=1)
    pre_hole = HOLE_START - day
    post_hole = HOLE_END + day
    last = pack["A100"].index[-1]
    win_a = [("before H100", pack["A100"].index[0], a_h100 - day),
             ("after H100: year 1", a_h100, a_h100 + pd.Timedelta(days=364)),
             ("after H100: year 2", a_h100 + pd.Timedelta(days=365), a_h100 + pd.Timedelta(days=729)),
             ("after H100: to the hole", a_h100, pre_hole),
             ("after the hole", post_hole, last)]
    h0 = comps["H100_founding"].index[0]
    win_h = [("before H200", h0, a_h200 - day),
             ("H200 to B200", a_h200, a_b200 - day),
             ("before B200", h0, a_b200 - day),
             ("after H200: to the hole", a_h200, pre_hole),
             ("after B200: to the hole", a_b200, pre_hole),
             ("after the hole", post_hole, last)]
    rows = []

    def add(series_name, x, wins, zone_frame=None, members=None):
        for wname, a, b in wins:
            r = rate(x, a, b)
            row = dict(series=series_name, window=wname, start=str(pd.Timestamp(a).date()), end=str(pd.Timestamp(b).date()),
                       n_days=r["n_days"], start_usd=r["start_level"], end_usd=r["end_level"],
                       pct_change_first_to_last_day=r["plain"],
                       pct_per_year_endpoint=pct(r["endpoint"]), pct_per_year_trend=pct(r["trend"]),
                       log_rate_endpoint=r["endpoint"], log_rate_trend=r["trend"])
            if zone_frame is not None:
                zr = np.array([rate(zone_frame[m], a, b)["endpoint"] for m in members])
                zr = zr[np.isfinite(zr)]
                if len(zr):
                    q = np.quantile(zr, [0.25, 0.5, 0.75])
                    row.update(n_zones=len(zr), zone_pct_q25=pct(q[0]), zone_pct_median=pct(q[1]), zone_pct_q75=pct(q[2]))
            rows.append(row)

    add("A100 p4d az_median (pack composite)", pack["A100"], win_a)
    add("A100 p4d founding zones", comps["A100_founding"], win_a, zones["A100"], comps["A100_members"])
    add("H100 p5 founding zones", comps["H100_founding"], win_h, zones["H100"], comps["H100_members"])
    # the H100 pack composite starts 2024-12-19 inside the launch ramp of 16 new zones: only later windows are meaningful
    add("H100 p5 az_median (pack composite)", pack["H100"], [w for w in win_h if w[0] in ("after B200: to the hole", "after the hole")])
    ramp = "launch to the hole (includes the floor-price ramp)"
    add("H200 p5en az_median (pack composite)", pack["H200"], [(ramp, pack["H200"].index[0], pre_hole), ("after the hole", post_hole, last)])
    add("B200 p6-b200 founding zones", comps["B200_founding"], [(ramp, comps["B200_founding"].index[0], pre_hole),
                                                                ("after the hole", post_hole, last)])
    rates = pd.DataFrame(rows)
    rates.to_csv(os.path.join(OUT, "step1_rates.csv"), index=False, float_format="%.4f")

    # --- the measured "successor effect": change in the A100's annualised log trend
    def lr(series, window, col="log_rate_endpoint"):
        return float(rates.loc[(rates.series == series) & (rates.window == window), col].iloc[0])

    sA, sAf, sH = "A100 p4d az_median (pack composite)", "A100 p4d founding zones", "H100 p5 founding zones"
    delta = lr(sA, "after H100: to the hole") - lr(sA, "before H100")
    alt = {
        "pack composite, straight-line trend": lr(sA, "after H100: to the hole", "log_rate_trend") - lr(sA, "before H100", "log_rate_trend"),
        "founding zones composite": lr(sAf, "after H100: to the hole") - lr(sAf, "before H100"),
        "pack composite, year 1 only": lr(sA, "after H100: year 1") - lr(sA, "before H100"),
        "pack composite, year 2 only": lr(sA, "after H100: year 2") - lr(sA, "before H100"),
    }
    zd = []
    for m in comps["A100_members"]:
        d = rate(zones["A100"][m], a_h100, pre_hole)["endpoint"] - rate(zones["A100"][m], pack["A100"].index[0], a_h100 - day)["endpoint"]
        if np.isfinite(d):
            zd.append(d)
    zq = np.quantile(zd, [0.25, 0.5, 0.75])
    h_delta_h200 = lr(sH, "after H200: to the hole") - lr(sH, "before H200")
    h_delta_b200 = lr(sH, "after B200: to the hole") - lr(sH, "before B200")
    effect = dict(delta_log_per_year=delta, alternatives=alt, zone_q25=float(zq[0]), zone_median=float(zq[1]),
                  zone_q75=float(zq[2]), n_zones=len(zd), h100_delta_at_h200=h_delta_h200, h100_delta_at_b200=h_delta_b200)

    # --- price per unit of work (A100-hour equivalents), monthly medians
    cols = {"A100": pack["A100"], "H100_founding": comps["H100_founding"], "H100": pack["H100"],
            "H200": pack["H200"], "B200_founding": comps["B200_founding"]}
    gen = {"A100": "A100", "H100_founding": "H100", "H100": "H100", "H200": "H200", "B200_founding": "B200"}
    m = pd.DataFrame({k: v.resample("MS").median() for k, v in cols.items()})
    m = m[~((m.index >= HOLE_START) & (m.index <= HOLE_END))]
    pw = pd.DataFrame({f"usd_per_gpu_hour_{k}": m[k] for k in cols})
    for k in cols:
        pw[f"usd_per_work_unit_{k}"] = m[k] / units[gen[k]]
    wcols = [f"usd_per_work_unit_{k}" for k in ("A100", "H100_founding", "H200", "B200_founding")]
    pw["cheapest_usd_per_work_unit"] = pw[wcols].min(axis=1)
    pw["cheapest_chip"] = pw[wcols].idxmin(axis=1).str.replace("usd_per_work_unit_", "")
    pw["a100_premium_vs_h100_per_work"] = pw["usd_per_work_unit_A100"] / pw["usd_per_work_unit_H100_founding"] - 1.0
    pw.index.name = "month"
    pw.to_csv(os.path.join(OUT, "step1_per_work_monthly.csv"), float_format="%.4f")

    def mm(col, a, b):
        return float(pw.loc[a:b, col].median())

    front = {"2022-H2": mm("cheapest_usd_per_work_unit", "2022-07", "2022-12"),
             "2023-H1": mm("cheapest_usd_per_work_unit", "2023-01", "2023-06"),
             "2026 Jan-Feb": mm("cheapest_usd_per_work_unit", "2026-01", "2026-02"),
             "2026 Jul-Oct": mm("cheapest_usd_per_work_unit", "2026-07", "2026-10")}
    per_work = dict(
        frontier=front,
        frontier_pct_per_year_2022H2_to_2026JanFeb=pct(math.log(front["2026 Jan-Feb"] / front["2022-H2"]) / 3.33),
        frontier_pct_per_year_2022H2_to_2026JulOct=pct(math.log(front["2026 Jul-Oct"] / front["2022-H2"]) / 3.9),
        h100_per_work_at_launch=mm("usd_per_work_unit_H100_founding", "2023-10", "2023-12"),
        a100_per_work_at_h100_launch=mm("usd_per_work_unit_A100", "2023-10", "2023-12"),
        a100_premium_2025H2=mm("a100_premium_vs_h100_per_work", "2025-07", "2025-12"),
        a100_premium_2026_JulOct=mm("a100_premium_vs_h100_per_work", "2026-07", "2026-10"),
        oct2026={k: float(pw.loc["2026-10-01", f"usd_per_work_unit_{k}"]) for k in ("A100", "H100", "H200", "B200_founding")},
    )
    return rates, effect, per_work, pw


# ---------------------------------------------------------------- step 1b ----
def pair_composite(z: pd.DataFrame, arrival) -> tuple[pd.Series, int]:
    """The older chip's price around one arrival: the median over a fixed set of zones, namely those
    that already had a price PAIR_WINDOW days before the arrival (for a chip younger than that, in
    its own first 30 days), on days when at least PAIR_ZONE_SHARE of them have a value."""
    first = z.apply(lambda c: c.first_valid_index())
    live_by = max(pd.Timestamp(arrival) - pd.Timedelta(days=PAIR_WINDOW), first.min() + pd.Timedelta(days=30))
    mem = sorted(first[first <= live_by].index)
    zz = z[mem]
    ok = zz.notna().sum(axis=1) >= math.ceil(PAIR_ZONE_SHARE * len(mem))
    return zz[ok].median(axis=1), len(mem)


def before_after(x: pd.Series, t) -> dict | None:
    """Yearly rate of change in the PAIR_WINDOW days before date t and in the PAIR_WINDOW days from t,
    and the difference. Windows never reach across the archive hole; None if either side is too short."""
    t = pd.Timestamp(t)
    day = pd.Timedelta(days=1)
    a0, b1 = t - pd.Timedelta(days=PAIR_WINDOW), t + pd.Timedelta(days=PAIR_WINDOW - 1)
    if t <= HOLE_END:                                   # the arrival is before the hole: stop at it
        b1 = min(b1, HOLE_START - day)
    else:                                               # after the hole: do not look back across it
        a0 = max(a0, HOLE_END + day)
    rb, ra = rate(x, a0, t - day), rate(x, t, b1)
    if min(rb["n_days"], ra["n_days"]) < PAIR_MIN_DAYS or not (np.isfinite(rb["endpoint"]) and np.isfinite(ra["endpoint"])):
        return None
    seg_b = x[(x.index >= a0) & (x.index < t)]
    seg_a = x[(x.index >= t) & (x.index <= b1)]
    return dict(before_start=seg_b.index[0], before_days=rb["n_days"], after_end=seg_a.index[-1], after_days=ra["n_days"],
                before=rb["endpoint"], after=ra["endpoint"], delta=ra["endpoint"] - rb["endpoint"])


def step1b(zones, avail, cut_seen) -> tuple[pd.DataFrame, dict]:
    """Every older-chip and successor pair the archive supports. A pair is an older chip and a later
    chip of the same line whose first spot price falls inside the archive, with at least
    PAIR_MIN_DAYS of the older chip's price on each side. The same before/after difference is then
    taken at every month-start that is not near an arrival, to show what a difference looks like
    when nothing arrived."""
    rows = []
    for line, chips in LINES.items():
        arrivals = {c: avail[c] for c in chips if c in avail}
        archive_start = min(z.index[0] for z in zones.values() if len(z))
        real = [t for c, t in arrivals.items() if t > archive_start + pd.Timedelta(days=30)]
        for j, new in enumerate(chips):
            t = arrivals.get(new)
            if t is None or t not in real:
                continue                                # the newer chip was already on spot when the archive starts
            for i, old in enumerate(chips[:j]):
                z = zones.get(old)
                if z is None or not len(z):
                    continue
                x, n_zones = pair_composite(z, t)
                r = before_after(x, t)
                if r is None:
                    continue
                pl = []
                for d in pd.date_range(x.index[0], x.index[-1], freq="MS"):
                    if min(abs((d - a).days) for a in real) < PLACEBO_GAP:
                        continue
                    q = before_after(x, d)
                    if q is not None:
                        pl.append(q["delta"])
                pl = np.array(pl)
                cut_in = "not a chip whose list price was cut"
                if old in LIST_CUT_CHIPS:
                    cut_in = ("in the window before the arrival" if r["before_start"] <= cut_seen < t else
                              "in the window after the arrival" if t <= cut_seen <= r["after_end"] else "outside both windows")
                rows.append(dict(line=line, older_chip=old, older_instance={**INST, **PAIR_INST}[old], successor=new,
                                 successor_first_spot_price=str(t.date()), generations_apart=j - i, n_zones=n_zones,
                                 before_days=r["before_days"], after_days=r["after_days"],
                                 pct_per_year_before=pct(r["before"]), pct_per_year_after=pct(r["after"]),
                                 delta_log_per_year=r["delta"], fell_faster_after=bool(r["delta"] < 0),
                                 list_cut_june_2025=cut_in, n_other_dates=len(pl),
                                 share_of_other_dates_with_a_lower_delta=float((pl < r["delta"]).mean()) if len(pl) else np.nan,
                                 other_dates_delta_q10=float(np.quantile(pl, 0.1)) if len(pl) else np.nan,
                                 other_dates_delta_median=float(np.median(pl)) if len(pl) else np.nan,
                                 other_dates_delta_q90=float(np.quantile(pl, 0.9)) if len(pl) else np.nan))
    pairs = pd.DataFrame(rows)
    clean = pairs[~pairs.list_cut_june_2025.str.startswith("in the window")]
    direct = pairs[pairs.generations_apart == 1]

    def block(d):
        if not len(d):
            return dict(n=0)
        return dict(n=int(len(d)), fell_faster_after=int(d.fell_faster_after.sum()),
                    median_delta_log_per_year=float(d.delta_log_per_year.median()),
                    delta_min=float(d.delta_log_per_year.min()), delta_max=float(d.delta_log_per_year.max()))

    summary = dict(all_pairs=block(pairs), pairs_with_no_list_cut_in_a_window=block(clean), next_generation_pairs=block(direct))
    summary.update({f"{line}_line": block(pairs[pairs.line == line]) for line in LINES})
    return pairs, summary


# -------------------------------------------------------------- simulator ----
# feedback: "iid"    = how often it steps + how big, each day independent of the last
#           "dir"    = the previous day's move (down / unchanged / up) picks which past days we draw from
#           "dirlvl" = previous day's move AND which side of its usual level the simulated price is on
# trend:    "history" = draws used as they happened;  "none" = re-centred so the average daily change is zero
VARIANTS = {
    "main": dict(feedback="dirlvl", trend="history", label="main: yesterday's move + usual range fed back"),
    "free": dict(feedback="dir", trend="history", label="no usual range (free to wander), history's trend"),
    "free_notrend": dict(feedback="dir", trend="none", label="no usual range, no trend"),
    "iid": dict(feedback="iid", trend="history", label="independent days (frequency + size only)"),
}


def fit_steps(x: pd.Series, origin, feedback: str, trend: str, succ_date=None, succ_shift: float = 0.0, learn=None):
    """Estimate the step behaviour from data up to `origin` only.
    For every pair of consecutive days inside one data segment (never across the hole) record
    (state of day t) -> (log change on day t+1).  A day's state is the direction of its own move
    and, for "dirlvl", whether the price is above the median level seen so far.  `learn` (optional
    boolean series) marks the days that may be learned from (young_zone_mask).  Returns None when
    fewer than MIN_TRAIN_DAYS usable day pairs exist."""
    h = x.loc[:origin]
    dates = h.index
    lp = np.log(h.to_numpy(dtype=float))
    use = np.ones(len(lp), dtype=bool) if learn is None else learn.reindex(dates).fillna(True).to_numpy(dtype=bool)
    chg = np.full(len(lp), np.nan)
    gap = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    d = np.diff(lp)
    d[gap != 1] = np.nan                      # first day after a gap has no "change"
    chg[1:] = d
    adj = chg.copy()
    if succ_date is not None and succ_shift != 0.0:
        # remove the successor shift from training days after the arrival, so it is added back
        # exactly once in the simulation (never double counted)
        adj = adj - (succ_shift / 365.0) * (dates >= succ_date)
    dirs = np.where(np.isnan(chg), -1, np.where(chg < -EPS, 0, np.where(chg > EPS, 2, 1)))
    ok = (dirs[:-1] >= 0) & (dirs[1:] >= 0) & use[:-1] & use[1:]
    if ok.sum() < MIN_TRAIN_DAYS:
        return None
    thr = float(np.median(lp[use]))           # the "usual level": median of the usable history
    lvl = (lp > thr).astype(int)
    if feedback == "iid":
        state, ns = np.zeros(len(lp), dtype=int), 1
    elif feedback == "dir":
        state, ns = np.where(dirs < 0, 1, dirs), 3
    elif feedback == "dirlvl":
        state, ns = np.where(dirs < 0, 1, dirs) + 3 * lvl, 6
    else:
        raise ValueError(feedback)
    prev_state, nxt, nxt_dir = state[:-1][ok], adj[1:][ok], dirs[1:][ok]
    pools = []
    for k in range(ns):
        m = prev_state == k
        if m.sum() < 5:                       # too few examples of this state: fall back to all days
            m = np.ones(len(nxt), dtype=bool)
        pools.append((nxt[m], nxt_dir[m]))
    L = np.array([len(p[0]) for p in pools])
    MC = np.zeros((ns, L.max()))
    MD = np.ones((ns, L.max()), dtype=np.int64)
    for k, (c, dd) in enumerate(pools):
        MC[k, :len(c)] = c
        MD[k, :len(c)] = dd
    raw = chg[1:][ok]
    return dict(MC=MC, MD=MD, L=L, st0=int(state[-1]), lp0=float(lp[-1]), thr=thr, feedback=feedback,
                offset=(-float(nxt.mean()) if trend == "none" else 0.0), n_pairs=int(len(nxt)),
                p_step=float((np.abs(raw) > EPS).mean()), mean_log_per_year=float(raw.mean() * 365.0),
                median_abs_step=float(np.median(np.abs(raw[np.abs(raw) > EPS]))) if (np.abs(raw) > EPS).any() else 0.0)


def simulate(fit: dict, origin, n_days: int, n_paths: int, rng, keep=(), daily_quantiles: bool = False,
             succ_date=None, succ_shift: float = 0.0, cap: float | None = None):
    """Roll forward one day at a time.  Each simulated day draws its log change from the past days
    that followed the same state, then that simulated day (its move and its new price) becomes the
    state for the next draw.  `cap` (USD per GPU-hour) is a ceiling no simulated day may pass: the
    on-demand list price, which no zone's spot price has exceeded since the June 2025 cut.  The
    random numbers drawn are the same with and without a cap."""
    cap_log = None if cap is None else math.log(cap)
    MC, MD, L, fb = fit["MC"], fit["MD"], fit["L"], fit["feedback"]
    st = np.full(n_paths, fit["st0"], dtype=np.int64)
    lp = np.full(n_paths, fit["lp0"])
    off = np.full(n_days, fit["offset"])
    if succ_date is not None and succ_shift != 0.0:
        days = pd.Timestamp(origin) + pd.to_timedelta(np.arange(1, n_days + 1), unit="D")
        off = off + (succ_shift / 365.0) * (days >= succ_date)
    keep = set(keep)
    out, dq = {}, (np.zeros((n_days, 3)) if daily_quantiles else None)
    for t in range(n_days):
        idx = (rng.random(n_paths) * L[st]).astype(np.int64)
        lp = lp + MC[st, idx] + off[t]
        if cap_log is not None:
            lp = np.minimum(lp, cap_log)
        if fb == "dir":
            st = MD[st, idx]
        elif fb == "dirlvl":
            st = MD[st, idx] + 3 * (lp > fit["thr"])
        if (t + 1) in keep:
            out[t + 1] = np.exp(lp)
        if daily_quantiles:
            dq[t] = np.exp(np.quantile(lp, [0.1, 0.5, 0.9]))
    return (out, dq) if daily_quantiles else out


def make_rng(*keys):
    return np.random.default_rng([SEED] + [zlib.crc32(str(k).encode()) for k in keys])


# ----------------------------------------------------------------- step 2 ----
def pinball(y, q, tau):
    return np.maximum(tau * (y - q), (tau - 1.0) * (y - q))


def backtest(name: str, x: pd.Series, variant: str, feedback: str, trend: str, succ_date=None, succ_shift: float = 0.0,
             learn=None):
    """Monthly origins; forecast made with data up to the origin only; truth read at origin + h days
    (skipped when that day is in the archive hole or past the end of the data)."""
    first = x.index[0] + pd.Timedelta(days=MIN_TRAIN_DAYS)
    origins = [d for d in pd.date_range(first, x.index[-1], freq="MS") if d in x.index]
    rows = []
    for o in origins:
        targets = {h: o + pd.Timedelta(days=h) for h in HORIZONS}
        hs = [h for h, t in targets.items() if t in x.index]
        if not hs:
            continue
        f = fit_steps(x, o, feedback, trend, succ_date, succ_shift, learn)
        if f is None:                          # not enough usable history yet
            continue
        sims = simulate(f, o, max(hs), N_PATHS_TEST, make_rng(name, o.date()), keep=hs,
                        succ_date=succ_date, succ_shift=succ_shift)   # same random numbers for every variant
        p0 = float(x[o])
        for h in hs:
            q10, q50, q90 = np.quantile(sims[h], [0.1, 0.5, 0.9])
            y = float(x[targets[h]])
            miss = max(math.log(q10 / y), math.log(y / q90))   # log distance outside the range (negative = inside)
            rows.append(dict(series=name, variant=variant, origin=o.date(), horizon_days=h, target=targets[h].date(),
                             crosses_hole=bool(o < HOLE_START and targets[h] > HOLE_END),
                             price_at_origin=p0, truth=y, q10=q10, q50=q50, q90=q90,
                             inside=bool(q10 <= y <= q90), below=bool(y < q10), above=bool(y > q90),
                             rel_width=(q90 - q10) / p0,
                             pin10=pinball(y, q10, 0.1), pin50=pinball(y, q50, 0.5), pin90=pinball(y, q90, 0.9),
                             lv10=pinball(y, p0, 0.1), lv50=pinball(y, p0, 0.5), lv90=pinball(y, p0, 0.9),
                             miss_log=miss))
    return pd.DataFrame(rows)


def wilson(p: float, n: float, z: float = 1.645) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - hw), min(1.0, c + hw)


def conformal_widening(scores: np.ndarray, level: float = 0.8) -> float:
    """How far (in logs) both ends of the simulated 10-90% range must be pushed out so that `level`
    of the test truths fall inside: the split-conformal quantile of the miss distances, as in
    conformalized quantile regression.  Negative means the range could have been narrower."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("nan")
    j = min(n, math.ceil((n + 1) * level))
    return float(s[j - 1])


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (series, variant, h), g in cells.groupby(["series", "variant", "horizon_days"], sort=False):
        o = pd.to_datetime(g.origin)
        n_indep = 1 + int((o.max() - o.min()).days // h)       # non-overlapping windows that fit
        cov = float(g.inside.mean())
        lo, hi = wilson(cov, n_indep)
        pin = (g.pin10.mean() + g.pin50.mean() + g.pin90.mean()) / 3
        lv = (g.lv10.mean() + g.lv50.mean() + g.lv90.mean()) / 3
        rows.append(dict(series=series, variant=variant, horizon_days=h, n_origins=len(g), n_independent_windows=n_indep,
                         first_origin=o.min().date(), last_origin=o.max().date(), n_cross_hole=int(g.crosses_hole.sum()),
                         coverage=cov, coverage_lo90=lo, coverage_hi90=hi, share_below=float(g.below.mean()),
                         share_above=float(g.above.mean()), median_rel_width=float(g.rel_width.median()),
                         pinball_vs_no_change=pin / lv, pin10_vs_no_change=g.pin10.mean() / g.lv10.mean(),
                         pin50_vs_no_change=g.pin50.mean() / g.lv50.mean(), pin90_vs_no_change=g.pin90.mean() / g.lv90.mean(),
                         pinball_usd=pin, no_change_pinball_usd=lv,
                         widen_each_end_pct_for_80pct=(math.exp(conformal_widening(g.miss_log.to_numpy())) - 1) * 100))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts ----
def style(ax, ylabel=None, title=None):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9c8c3")
        ax.spines[sp].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def gapped(x: pd.Series) -> pd.Series:
    """Daily index with NaN where there is no data, so lines break at the hole."""
    return x.reindex(pd.date_range(x.index[0], x.index[-1], freq="D"))


def shade_hole(ax, label=True):
    ax.axvspan(HOLE_START, HOLE_END + pd.Timedelta(days=1), color="#efeeea", zorder=0, lw=0)
    if label:
        ax.text(HOLE_START + (HOLE_END - HOLE_START) / 2, 0.02, "no data", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color=INK2, rotation=90)


def usd_axis(ax, ticks):
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"${t:g}" for t in ticks])
    ax.minorticks_off()


def chart_paths(pack, comps, avail, units):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.4), sharex=True, facecolor=SURFACE)
    lines = [("A100 (p4d), zone-median composite", pack["A100"], "A100", 2.0, "A100"),
             ("H100 (p5), 3 founding zones", comps["H100_founding"], "H100", 1.2, "H100"),
             ("H100 (p5), zone-median composite (28 zones)", pack["H100"], "H100", 2.2, "H100"),
             ("H200 (p5en), zone-median composite", pack["H200"], "H200", 1.6, "H200"),
             ("B200 (p6-b200), 5 founding zones", comps["B200_founding"], "B200", 1.6, "B200")]
    for ax, per_work in zip(axes, (False, True)):
        for lab, x, chip, lw, g in lines:
            y = gapped(x) / (units[g] if per_work else 1.0)
            ax.plot(y.index, y.values, color=COL[chip], lw=lw, label=lab, solid_capstyle="round", alpha=0.6 if lw < 1.5 else 1.0)
        shade_hole(ax)
        for chip, lab in (("H100", "H100 on AWS spot"), ("H200", "H200"), ("B200", "B200"), ("B300", "B300")):
            ax.axvline(avail[chip], color="#9b9a95", lw=0.8, zorder=1)
            if not per_work:
                ax.text(avail[chip], 1.01, f"{lab}\n{avail[chip]:%d %b %Y}", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7.5, color=INK2)
        if per_work:
            usd_axis(ax, [0.25, 0.5, 1, 2, 4])
            style(ax, "USD per A100-hour of work (log scale)")
            ax.set_title("The same prices per unit of work (one H100 = 2.46 A100s, H200 = 2.72, B200 = 7.91; inference throughput)",
                         color=INK, fontsize=11, loc="left", pad=8)
        else:
            usd_axis(ax, [0.5, 1, 2, 4, 8])
            ax.set_ylim(0.5, 14)
            style(ax, "USD per GPU-hour (log scale)")
            ax.set_title("AWS spot price per GPU-hour, Linux, end of day", color=INK, fontsize=11, loc="left", pad=30)
            ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2, ncol=1, borderaxespad=0.2)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.text(0.012, 0.008, "Speed ratios are the pack's capability units, a composite derived from published MLPerf Inference results. "
                           "Result not verified by MLCommons Association.", fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.savefig(os.path.join(OUT, "chart1_price_paths.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_trust(summ):
    show = [("A100_pack", "A100 composite (4 years of history)", COL["A100"], "-", 2.0, True),
            ("H100_founding", "H100, 3 founding zones (3 years)", COL["H100"], "-", 2.0, True),
            ("H100_pack", "H100 composite (short history)", COL["H100"], "--", 1.3, False)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), facecolor=SURFACE)
    xs = {h: i for i, h in enumerate(HORIZONS)}
    xl = ["30 d", "90 d", "180 d", "1 yr", "2 yr"]
    main = summ[summ.variant == "main"]
    for j, (key, lab, col, ls, lw, filled) in enumerate(show):
        g = main[(main.series == key) & (main.n_origins >= 5)].sort_values("horizon_days")   # fewer than 5 forecasts: not drawn
        x = np.array([xs[h] for h in g.horizon_days]) + (j - 1) * 0.07
        kw = dict(color=col, lw=lw, ls=ls, marker="o", ms=7, mfc=col if filled else SURFACE, mec=col, mew=1.6, label=lab)
        axes[0].errorbar(x, g.coverage * 100, yerr=[(g.coverage - g.coverage_lo90) * 100, (g.coverage_hi90 - g.coverage) * 100],
                         fmt="none", ecolor=col, elinewidth=0.9, alpha=0.45, capsize=0)
        axes[0].plot(x, g.coverage * 100, **kw)
        axes[1].plot(x, g.median_rel_width * 100, **kw)
        axes[2].plot(x, g.pinball_vs_no_change, **kw)

    def n_of(key, h):
        r = main[(main.series == key) & (main.horizon_days == h)]
        return int(r.n_independent_windows.iloc[0]) if len(r) else 0

    axes[0].axhline(80, color=INK, lw=0.9)
    axes[0].text(-0.3, 82, "target 80%", ha="left", fontsize=8, color=INK2)
    axes[0].set_ylim(0, 105)
    style(axes[0], "% of test forecasts", "How often the truth fell inside the 10-90% range")
    style(axes[1], "% of the price on the forecast day", "Width of the 10-90% range")
    axes[1].set_ylim(0, None)
    axes[2].axhline(1.0, color=INK, lw=0.9)
    axes[2].text(-0.3, 1.02, "same as \"price stays where it is\"", ha="left", fontsize=8, color=INK2)
    style(axes[2], "pinball loss / no-change loss (below 1 = better)", "Loss relative to a no-change forecast")
    axes[2].set_ylim(0, 1.25)
    for i, ax in enumerate(axes):
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels([f"{l}\n{n_of('A100_pack', h)} | {n_of('H100_founding', h)}" for l, h in zip(xl, HORIZONS)] if i == 0 else xl)
        ax.set_xlabel("how far ahead", color=INK2, fontsize=9.5)
        ax.set_xlim(-0.35, 4.35)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, frameon=False, labelcolor=INK2, bbox_to_anchor=(0.5, 0.055))
    fig.text(0.045, 0.02, "Under each horizon in the first panel: independent (non-overlapping) test windows, A100 | H100 founding zones. "
                          "Bars: 90% uncertainty given that count. Points with fewer than 5 test forecasts are not drawn.",
             fontsize=7.5, color=INK2)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.92, bottom=0.26, wspace=0.24)
    fig.savefig(os.path.join(OUT, "chart2_trust_horizon.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_outlook(pack, paths, origin, caps):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.3), facecolor=SURFACE)
    for ax, chip in zip(axes, ("H100", "A100")):
        col = COL[chip]
        hist = gapped(pack[chip].loc["2024-07-01":])
        ax.plot(hist.index, hist.values, color=col, lw=1.6, label="history (zone-median composite)")
        shade_hole(ax)
        p = paths[(paths.chip == chip) & (paths.variant == "main")]
        d = pd.to_datetime(p.date)
        ax.fill_between(d, p.q10_widened, p.q90_widened, color=col, alpha=0.12, lw=0,
                        label="10-90% range, widened until it passes this chip's own test (the range to use)")
        ax.fill_between(d, p.q10, p.q90, color=col, alpha=0.22, lw=0, label="10-90% range as simulated")
        ax.plot(d, p.q50, color=col, lw=2.0, ls=(0, (4, 2)), label="middle of the simulated paths")
        f = paths[(paths.chip == chip) & (paths.variant == "free_notrend")]
        fd = pd.to_datetime(f.date)
        ax.plot(fd, f.q10, color="#8a8984", lw=1.0, label="other reading (no pull back to the usual level): its 10% and 90%")
        ax.plot(fd, f.q90, color="#8a8984", lw=1.0)
        # the ceiling: AWS's on-demand list price today (median over the composite's zones)
        ax.plot([hist.index[0], d.iloc[-1]], [caps[chip]] * 2, color=INK, lw=0.9, ls=(0, (1, 2)),
                label=f"AWS on-demand list price today, \\${caps[chip]:.2f}: no path may pass it")
        # comparison only: a steady fall at a hardware-trend rate from the last price
        p0 = float(pack[chip].iloc[-1])
        yrs = np.arange(1, len(d) + 1) / 365.0
        ax.plot(d, p0 * 0.75 ** yrs, color=INK2, lw=1.1, ls=(0, (6, 2, 1, 2)),
                label="for comparison: a steady fall of 25% a year (hardware trend)")
        for i, h in enumerate(OUTLOOK_HORIZONS):
            r = p.iloc[h - 1]
            ax.plot([pd.Timestamp(r.date)], [r.q50], marker="o", ms=7, mfc=col, mec=SURFACE, mew=1.5, zorder=5)
            ax.text(pd.Timestamp(r.date), 0.975, f"+{i + 1} yr\n\\${r.q50:.2f}\n\\${r.q10_widened:.2f} to \\${r.q90_widened:.2f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=INK, linespacing=1.35)
        ax.axvline(origin, color="#9b9a95", lw=0.8)
        usd_axis(ax, [0.5, 1, 2, 4, 8])
        ax.set_ylim(0.4, 16)
        inst = INST[chip].split(".")[0]
        style(ax, "USD per GPU-hour (log scale)", f"{chip} ({inst}) AWS spot: outlook from {origin:%d %b %Y}")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(None, origin + pd.Timedelta(days=max(OUTLOOK_HORIZONS) + 170))
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.07), ncol=1, fontsize=8, frameon=False, labelcolor=INK2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.945, bottom=0.29, wspace=0.14)
    fig.savefig(os.path.join(OUT, "chart3_outlook.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------- main ----
def main() -> int:
    t0 = time.time()
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
    os.makedirs(OUT, exist_ok=True)
    key: dict = {"seed": SEED, "instrument": "AWS EC2 Linux/UNIX spot, USD per GPU-hour (instance price / 8), end-of-day, "
                                             "median over the pack's fixed availability-zone set (az_median)"}

    say("== data ==")
    build_pack_series()
    zones, pack, members = load_series()
    n_members = {k: len(v) for k, v in members.items()}
    ticks = first_ticks()
    rel = release_dates()
    ticks["chip_release_date_epoch"] = ticks.chip.map(lambda c: rel.get("H200" if c == "H200_p5e" else c, ""))
    ticks["months_release_to_aws_spot"] = [
        round((t - pd.Timestamp(r)).days / 30.44, 1) if r else np.nan for t, r in zip(ticks.first_tick_utc, ticks.chip_release_date_epoch)]
    ticks.to_csv(os.path.join(OUT, "availability.csv"), index=False)
    say(ticks[["chip", "instance_type", "first_tick_utc", "first_usd_per_gpu_hour", "n_zones_ever", "chip_release_date_epoch",
               "months_release_to_aws_spot"]].to_string(index=False))
    say(f"  next generation in the hardware file: VR200 (Vera Rubin) released {rel['VR200']}; no spot ticks in the pack yet")
    avail = {r.chip: pd.Timestamp(r.first_tick_utc).normalize() for r in ticks.itertuples()}
    key["first_spot_tick"] = {k: str(v.date()) for k, v in avail.items()}
    # The pack's daily series carry a zone's last price forward to the end of the archive segment. A machine
    # type whose ticks stop earlier (the K80's stop on 1 Jun 2025) has no real price after that day: cut it there.
    for r in ticks.itertuples():
        zones[r.chip] = zones[r.chip].loc[:pd.Timestamp(r.last_tick_utc).normalize()]
    key["last_spot_tick"] = {r.chip: str(pd.Timestamp(r.last_tick_utc).date()) for r in ticks.itertuples()}

    units = capability_units()
    comps = {}
    comps["A100_founding"], comps["A100_members"] = founding_composite(zones["A100"], "2022-06-01")
    comps["H100_founding"], comps["H100_members"] = founding_composite(zones["H100"], "2023-12-31")
    comps["B200_founding"], comps["B200_members"] = founding_composite(zones["B200"], "2025-07-31")
    b3 = zones["B300"]
    comps["B300_all_zones"] = b3.median(axis=1)[b3.notna().sum(axis=1) >= 3]
    say(f"  founding-zone composites: A100 {len(comps['A100_members'])} zones from {comps['A100_founding'].index[0].date()}, "
        f"H100 {len(comps['H100_members'])} zones from {comps['H100_founding'].index[0].date()}, "
        f"B200 {len(comps['B200_members'])} zones from {comps['B200_founding'].index[0].date()}")
    say(f"  pack composites: A100 {n_members['A100']} zones from {pack['A100'].index[0].date()}, "
        f"H100 {n_members['H100']} zones from {pack['H100'].index[0].date()}; last day {pack['H100'].index[-1].date()}")
    key["composite_zones"] = {k: int(v) for k, v in n_members.items()}
    arch0 = zones["A100"].index[0]
    learn = {chip: young_zone_mask(zones[chip], members[chip], pack[chip].index, arch0) for chip in ("A100", "H100")}
    key["young_zone_filter"] = {}
    for chip in ("A100", "H100"):
        bad = ~learn[chip]
        f_all = fit_steps(pack[chip], pack[chip].index[-1], "iid", "history")
        f_use = fit_steps(pack[chip], pack[chip].index[-1], "iid", "history", learn=learn[chip])
        info = dict(days_not_learned_from=int(bad.sum()), first=str(bad[bad].index.min().date()), last=str(bad[bad].index.max().date()),
                    history_trend_pct_per_year_all_days=pct(f_all["mean_log_per_year"]),
                    history_trend_pct_per_year_learned_days=pct(f_use["mean_log_per_year"]))
        key["young_zone_filter"][chip] = info
        say(f"  young-zone filter, {chip} composite: {info['days_not_learned_from']} days not learned from ({info['first']} to {info['last']}); "
            f"average change of the composite {info['history_trend_pct_per_year_all_days']:+.0f}%/yr on all days, "
            f"{info['history_trend_pct_per_year_learned_days']:+.0f}%/yr on the days kept")
    key["capability_units_a100_eq_1"] = units

    # tidy daily file of every series used
    allser = {"A100_pack": pack["A100"], "A100_founding": comps["A100_founding"], "A100_80GB_pack": pack["A100_80GB"],
              "H100_pack": pack["H100"], "H100_founding": comps["H100_founding"], "H200_pack": pack["H200"],
              "H200_p5e_pack": pack["H200_p5e"], "B200_founding": comps["B200_founding"], "B300_all_zones": comps["B300_all_zones"]}
    daily = pd.concat([pd.DataFrame({"series": k, "date": v.index.date, "usd_per_gpu_hour": v.to_numpy(),
                                     "segment": np.where(v.index < HOLE_START, "before_hole", "after_hole")})
                       for k, v in allser.items()], ignore_index=True)
    assert not ((pd.to_datetime(daily.date) >= HOLE_START) & (pd.to_datetime(daily.date) <= HOLE_END)).any(), "rows inside the hole"
    daily.to_csv(os.path.join(OUT, "series_daily.csv"), index=False, float_format="%.6f")
    key["levels"] = {k: {"2026-02-28": round(float(v.get(pd.Timestamp("2026-02-28"), np.nan)), 3),
                         "2026-07-01": round(float(v.get(pd.Timestamp("2026-07-01"), np.nan)), 3),
                         "2026-09-30": round(float(v.get(pd.Timestamp("2026-09-30"), np.nan)), 3),
                         "2026-10-31": round(float(v.iloc[-1]), 3)} for k, v in allser.items()}
    say("  levels (USD per GPU-hour) either side of the hole and at the end:")
    say(pd.DataFrame(key["levels"]).T.to_string())

    # ------------------------------------------------------------ step 0 ----
    say("\n== step 0: spot against the on-demand list price ==")
    last_day = pack["H100"].index[-1]
    lp = list_prices(last_day)
    cut = list_cut_check(last_day)
    lp.to_csv(os.path.join(OUT, "aws_list_prices.csv"), index=False, float_format="%.6f")
    s2l, list_info = spot_vs_list(zones, pack, members, lp, cut)
    s2l.to_csv(os.path.join(OUT, "spot_to_list.csv"), index=False, float_format="%.4f")
    cut_seen = pd.Timestamp(list_info["cut_reached_spot_ceilings_on"])
    list_info["cut_sizes_from_the_pack"] = cut.round(4).to_dict("records")
    key["list_price"] = list_info
    say("  size of the June 2025 list cut, from one site's earlier and current readings of AWS's on-demand price:")
    say(cut.round(4).to_string(index=False))
    say(f"  the cut reached the spot ceilings on {cut_seen.date()} (the day after the last day any H100 zone was above today's list price)")
    for chip in ("H100", "A100", "A100_80GB", "H200"):
        i = list_info[chip]
        say(f"  {chip}: list ${i['list_low']:.2f} to ${i['list_high']:.2f} across {i['regions_with_a_list_price']} regions "
            f"({i['regions_whose_price_changed_in_the_pack']} changed between {i['list_first_seen']} and {i['list_last_seen']}); "
            f"composite list ${i['composite_list_price']:.3f} ({i['composite_zones_with_a_list_price']} of {i['composite_zones']} zones); "
            f"spot ${i['spot_composite_last_day']:.3f} = {i['spot_share_of_list_last_day']:.2f} of list on {i['last_day']}; "
            f"highest zone since the cut {i['max_zone_share_of_list_since_cut']:.3f} of list "
            f"({i['zone_days_above_list_since_cut']} of {i['zone_days_since_cut']} zone-days above); "
            "median share of list: " + ", ".join(f"{k} {v:.2f}" for k, v in i["share_of_list_by_period"].items()))

    # ------------------------------------------------------------ step 1 ----
    say("\n== step 1: the older chip when the newer one arrived ==")
    rates, effect, per_work, pw = step1(zones, pack, comps, avail, units)
    say("(start_usd/end_usd = medians of the first/last 30 days; yearly rates only for windows of 180+ days)")
    say(rates[["series", "window", "start", "end", "start_usd", "end_usd", "pct_change_first_to_last_day", "pct_per_year_endpoint",
               "pct_per_year_trend", "zone_pct_q25", "zone_pct_median", "zone_pct_q75"]].round(2).to_string(index=False))
    delta = effect["delta_log_per_year"]
    say(f"  measured change in the A100's trend after the H100 arrived: {delta:+.3f} log points per year "
        f"({pct(delta):+.0f}%/yr); other ways of measuring it: "
        + "; ".join(f"{k} {v:+.2f}" for k, v in effect["alternatives"].items())
        + f"; across the {effect['n_zones']} founding zones q25/median/q75 = {effect['zone_q25']:+.2f}/{effect['zone_median']:+.2f}/{effect['zone_q75']:+.2f}")
    say(f"  same measure on the H100 (founding zones): at the H200's arrival {effect['h100_delta_at_h200']:+.2f}, at the B200's {effect['h100_delta_at_b200']:+.2f}")
    say(f"  per unit of work: {json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()}) for k, v in per_work.items()})}")
    key["step1"] = dict(effect=effect, per_work=per_work,
                        rates=rates[["series", "window", "start_usd", "end_usd", "pct_change_first_to_last_day", "pct_per_year_endpoint",
                                     "pct_per_year_trend", "zone_pct_q25", "zone_pct_q75"]].round(3).to_dict("records"))
    chart_paths(pack, comps, avail, units)

    # ----------------------------------------------------------- step 1b ----
    say("\n== step 1b: every older-chip and successor pair the archive supports ==")
    pairs, pair_summary = step1b(zones, avail, cut_seen)
    pairs.to_csv(os.path.join(OUT, "successor_pairs.csv"), index=False, float_format="%.4f")
    say(pairs[["line", "older_chip", "successor", "successor_first_spot_price", "generations_apart", "n_zones", "before_days", "after_days",
               "pct_per_year_before", "pct_per_year_after", "delta_log_per_year", "list_cut_june_2025", "n_other_dates",
               "share_of_other_dates_with_a_lower_delta", "other_dates_delta_q10", "other_dates_delta_q90"]].round(2).to_string(index=False))
    say("  " + json.dumps({k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} for k, v in pair_summary.items()}))
    key["step1b"] = dict(summary=pair_summary, pairs=pairs.round(4).to_dict("records"))

    # ------------------------------------------------------------ step 2 ----
    say("\n== step 2: roll-forward test ==")
    test_series = {"A100_pack": pack["A100"], "A100_founding": comps["A100_founding"],
                   "H100_pack": pack["H100"], "H100_founding": comps["H100_founding"]}
    test_learn = {"A100_pack": learn["A100"], "H100_pack": learn["H100"], "A100_founding": None, "H100_founding": None}
    cells = []
    for name, x in test_series.items():
        for v, spec in VARIANTS.items():
            cells.append(backtest(name, x, v, spec["feedback"], spec["trend"], learn=test_learn[name]))
    # step 3 variants (A100 only): add the successor effect at the H100's arrival; and the same with the sign flipped
    for name in ("A100_pack", "A100_founding"):
        for base in ("main", "free"):
            spec = VARIANTS[base]
            for tag, shift in (("successor", delta), ("successor_flipped", -delta)):
                cells.append(backtest(name, test_series[name], f"{base}+{tag}", spec["feedback"], spec["trend"],
                                      succ_date=avail["H100"], succ_shift=shift, learn=test_learn[name]))
    # the same adjustment out of sample: the A100's shift as it could be measured the day before the H200
    # arrived, applied to the H100 from the H200's arrival (a number from one chip, tested on another)
    one_day = pd.Timedelta(days=1)
    delta_known = (rate(pack["A100"], avail["H100"], avail["H200"] - one_day)["endpoint"]
                   - rate(pack["A100"], pack["A100"].index[0], avail["H100"] - one_day)["endpoint"])
    for base in ("main", "free"):
        spec = VARIANTS[base]
        for tag, shift in (("transfer", delta_known), ("transfer_flipped", -delta_known)):
            cells.append(backtest("H100_founding", test_series["H100_founding"], f"{base}+{tag}", spec["feedback"], spec["trend"],
                                  succ_date=avail["H200"], succ_shift=shift, learn=None))
    cells = pd.concat(cells, ignore_index=True)
    cells.to_csv(os.path.join(OUT, "backtest_cells.csv"), index=False, float_format="%.5f")
    summ = summarise(cells)
    summ.to_csv(os.path.join(OUT, "backtest_summary.csv"), index=False, float_format="%.4f")
    cols = ["series", "horizon_days", "n_origins", "n_independent_windows", "coverage", "share_below", "share_above",
            "median_rel_width", "pinball_vs_no_change", "pin10_vs_no_change", "pin50_vs_no_change", "pin90_vs_no_change",
            "widen_each_end_pct_for_80pct"]
    say("main simulator (yesterday's move + usual range fed back):")
    say(summ[summ.variant == "main"][cols].round(2).to_string(index=False))
    say("variants, pinball loss relative to no-change (rows) by horizon (columns):")
    piv = summ[~summ.variant.str.contains(r"\+")].pivot_table(index=["series", "variant"], columns="horizon_days", values="pinball_vs_no_change")
    say(piv.round(2).to_string())
    say("variants, coverage of the 10-90% range:")
    say(summ[~summ.variant.str.contains(r"\+")].pivot_table(index=["series", "variant"], columns="horizon_days", values="coverage").round(2).to_string())
    # widening: per chip, from that chip's own tests of the main simulator (the H100 has two test series)
    chip_tests = {"A100": ["A100_pack"], "H100": ["H100_pack", "H100_founding"]}
    widen = {}
    for chip, names in chip_tests.items():
        g0 = cells[(cells.variant == "main") & cells.series.isin(names)]
        run, widen[chip] = 0.0, {}
        for h, g in g0.groupby("horizon_days"):
            n_indep = int(summ[(summ.variant == "main") & summ.series.isin(names) & (summ.horizon_days == h)].n_independent_windows.max())
            e = conformal_widening(g.miss_log.to_numpy())
            if n_indep >= MIN_WINDOWS_TO_WIDEN:            # one window is an anecdote, not a calibration
                run = max(run, e)                          # never negative, never shrinking with horizon
            widen[chip][int(h)] = dict(n_forecasts=len(g), n_independent_windows=n_indep, coverage=float(g.inside.mean()),
                                       needed_each_end_pct=(math.exp(e) - 1) * 100, e_applied=run,
                                       applied_each_end_pct=(math.exp(run) - 1) * 100)
        say(f"widening of the main simulator's range, {chip} (own tests: {', '.join(names)}):")
        say(pd.DataFrame(widen[chip]).T.drop(columns="e_applied").round(2).to_string())
    key["step2"] = dict(main=summ[summ.variant == "main"][cols + ["coverage_lo90", "coverage_hi90", "first_origin", "last_origin", "n_cross_hole"]]
                        .round(3).astype({"first_origin": str, "last_origin": str}).to_dict("records"),
                        variants_pinball=piv.round(3).reset_index().to_dict("records"), widening_by_chip=widen)
    chart_trust(summ)

    # ------------------------------------------------------------ step 3 ----
    say("\n== step 3: does a 'new chip arrived' adjustment help? (A100 at the H100's arrival, in sample; "
        f"H100 at the H200's arrival with the A100's number {delta_known:+.3f}/yr, out of sample) ==")
    rows = []
    in_sample = {"successor": (delta, "measured"), "successor_flipped": (-delta, "sign flipped")}
    carried = {"transfer": (delta_known, "the A100's shift as known the day before; out of sample"),
               "transfer_flipped": (-delta_known, "the same, sign flipped")}
    tests = [("A100_pack", "H100", in_sample), ("A100_founding", "H100", in_sample), ("H100_founding", "H200", carried)]
    for name, newer, tags in tests:
        for base in ("main", "free"):
            b = cells[(cells.series == name) & (cells.variant == base)].set_index(["origin", "horizon_days"])
            for tag, (shift, what) in tags.items():
                e = cells[(cells.series == name) & (cells.variant == f"{base}+{tag}")].set_index(["origin", "horizon_days"])
                for h in HORIZONS:
                    if h not in b.index.get_level_values("horizon_days"):
                        continue                       # this series has no test forecast that far ahead
                    bb, ee = b.xs(h, level="horizon_days"), e.xs(h, level="horizon_days")
                    pb = (bb.pin10 + bb.pin50 + bb.pin90) / 3
                    pe = (ee.pin10 + ee.pin50 + ee.pin90) / 3
                    lv = (bb.lv10 + bb.lv50 + bb.lv90) / 3
                    after = pd.to_datetime(pb.index) + pd.Timedelta(days=h) >= avail[newer]
                    rows.append(dict(series=name, simulator=base, adjustment=f"{shift:+.2f}/yr ({what})",
                                     newer_chip=newer, newer_chip_first_spot_price=str(avail[newer].date()),
                                     horizon_days=h, n_origins=len(pb), n_origins_reaching_past_arrival=int(after.sum()),
                                     pinball_vs_no_change_without=pb.mean() / lv.mean(), pinball_vs_no_change_with=pe.mean() / lv.mean(),
                                     change_in_loss_pct=(pe.mean() / pb.mean() - 1) * 100,
                                     share_of_origins_improved=float((pe < pb - 1e-12).mean()),
                                     coverage_without=float(bb.inside.mean()), coverage_with=float(ee.inside.mean())))
    succ = pd.DataFrame(rows)
    succ.to_csv(os.path.join(OUT, "successor_test.csv"), index=False, float_format="%.4f")
    say(succ[succ.horizon_days >= 365][["series", "simulator", "adjustment", "horizon_days", "n_origins", "pinball_vs_no_change_without",
                                        "pinball_vs_no_change_with", "change_in_loss_pct", "share_of_origins_improved",
                                        "coverage_without", "coverage_with"]].round(3).to_string(index=False))
    key["step3"] = succ.round(4).to_dict("records")
    key["step3_shift_carried_to_h100_log_per_year"] = delta_known

    # ------------------------------------------------------------ step 4 ----
    say("\n== step 4: outlook from the last day of data ==")
    origin = pack["H100"].index[-1]
    succ_dates = {"A100": avail["H100"], "H100": avail["B200"]}     # the next GENERATION to reach AWS spot
    # Each composite is a median over zones and each zone has stayed at or below its region's list price,
    # so the composite cannot pass the median of its zones' list prices. "*_nocap" rows are the paths as
    # they were before this ceiling was added (same random numbers).
    caps = {chip: list_info[chip]["composite_list_price"] for chip in ("H100", "A100")}
    out_variants = [("main", "main", 0.0, True), ("main+successor", "main", delta, True), ("free_notrend", "free_notrend", 0.0, True),
                    ("free", "free", 0.0, True), ("main_nocap", "main", 0.0, False), ("free_notrend_nocap", "free_notrend", 0.0, False)]
    H = max(OUTLOOK_HORIZONS)
    eday_chip = {}
    for chip in widen:                                              # flat beyond the longest tested horizon
        kh = sorted(widen[chip])
        eday_chip[chip] = np.interp(np.arange(1, H + 1), kh, [widen[chip][h]["e_applied"] for h in kh])
    orow, prow, fits = [], [], {}
    for chip in ("H100", "A100"):
        x = pack[chip]
        eday = eday_chip[chip]
        for vname, base, shift, capped in out_variants:
            spec = VARIANTS[base]
            sd = succ_dates[chip] if shift != 0.0 else None
            top = caps[chip] if capped else float("inf")
            f = fit_steps(x, origin, spec["feedback"], spec["trend"], sd, shift, learn[chip])
            sims, dq = simulate(f, origin, H, N_PATHS_OUTLOOK, make_rng("outlook", chip), keep=OUTLOOK_HORIZONS,
                                daily_quantiles=True, succ_date=sd, succ_shift=shift, cap=caps[chip] if capped else None)
            fits[(chip, vname)] = f
            dates = origin + pd.to_timedelta(np.arange(1, H + 1), unit="D")
            is_main = base == "main"                              # the widening was measured on the main simulator only
            w10 = dq[:, 0] * np.exp(-eday) if is_main else np.full(H, np.nan)
            w90 = np.minimum(dq[:, 2] * np.exp(eday), top) if is_main else np.full(H, np.nan)   # widening stops at the ceiling too
            prow.append(pd.DataFrame({"chip": chip, "variant": vname, "date": dates.date, "q10": dq[:, 0], "q50": dq[:, 1],
                                      "q90": dq[:, 2], "q10_widened": w10, "q90_widened": w90}))
            for h in OUTLOOK_HORIZONS:
                q10, q50, q90 = np.quantile(sims[h], [0.1, 0.5, 0.9])
                e = float(eday[h - 1]) if is_main else float("nan")
                orow.append(dict(chip=chip, instance_type=INST[chip], series=f"aws_spot.{INST[chip]}.az_median", variant=vname,
                                 origin=origin.date(), price_at_origin=float(x.iloc[-1]), years_ahead=h // 365,
                                 target=(origin + pd.Timedelta(days=h)).date(), q10=q10, q50=q50, q90=q90,
                                 widen_each_end_pct=(math.exp(e) - 1) * 100, q10_widened=q10 * math.exp(-e),
                                 q90_widened=min(q90 * math.exp(e), top),
                                 share_of_paths_above_origin_price=float((sims[h] > x.iloc[-1]).mean()),
                                 list_price_cap=caps[chip] if capped else float("nan"),
                                 share_of_paths_at_cap=float((sims[h] >= caps[chip] * (1 - 1e-9)).mean()) if capped else float("nan")))
        # for comparison only: a steady yearly fall at a hardware-trend rate, from the last price (no range)
        for r in HARDWARE_TRENDS:
            for h in OUTLOOK_HORIZONS:
                v = float(x.iloc[-1]) * (1.0 + r) ** (h // 365)
                orow.append(dict(chip=chip, instance_type=INST[chip], series=f"aws_spot.{INST[chip]}.az_median",
                                 variant=f"hardware_trend_{int(round(-r * 100))}", origin=origin.date(), price_at_origin=float(x.iloc[-1]),
                                 years_ahead=h // 365, target=(origin + pd.Timedelta(days=h)).date(), q10=v, q50=v, q90=v,
                                 widen_each_end_pct=float("nan"), q10_widened=float("nan"), q90_widened=float("nan"),
                                 share_of_paths_above_origin_price=0.0, list_price_cap=float("nan"), share_of_paths_at_cap=float("nan")))
    outlook = pd.DataFrame(orow)
    outlook.to_csv(os.path.join(OUT, "outlook.csv"), index=False, float_format="%.4f")
    paths = pd.concat(prow, ignore_index=True)
    paths[paths.index % 7 == 6].to_csv(os.path.join(OUT, "outlook_paths_weekly.csv"), index=False, float_format="%.4f")
    say(outlook[["chip", "variant", "years_ahead", "target", "price_at_origin", "q10", "q50", "q90", "widen_each_end_pct",
                 "q10_widened", "q90_widened"]].round(2).to_string(index=False))
    for chip in ("H100", "A100"):
        f = fits[(chip, "main")]
        say(f"  {chip}: last price {math.exp(f['lp0']):.3f}; usual level (median of history) {math.exp(f['thr']):.3f}; "
            f"share of days with a step {f['p_step']:.2f}; median step {f['median_abs_step'] * 100:.2f}%; "
            f"history's average change {pct(f['mean_log_per_year']):+.0f}%/yr over the {f['n_pairs']} day pairs learned from")
    hw_fit = hardware_trend_fit()
    for name, v in hw_fit.items():
        say(f"  hardware trend, {name}: {v['n_gpus']} GPUs, {int(v['first_release_year'])} to {int(v['last_release_year'])}: cost of a unit "
            f"{v['cost_pct_per_year']:+.1f}%/yr ({v['cost_pct_per_year_slow']:+.1f} to {v['cost_pct_per_year_fast']:+.1f}), "
            f"halves every {v['halving_years']:.1f} years")
    key["step4"] = dict(origin=str(origin.date()), successor_dates={k: str(v.date()) for k, v in succ_dates.items()},
                        list_price_cap=caps, hardware_trend_rates=list(HARDWARE_TRENDS), hardware_trend_fit_on_pack=hw_fit,
                        successor_shift_log_per_year=delta, outlook=outlook.round(3).astype({"origin": str, "target": str}).to_dict("records"),
                        fit={chip: dict(last_price=math.exp(fits[(chip, 'main')]["lp0"]), usual_level=math.exp(fits[(chip, 'main')]["thr"]),
                                        p_step=fits[(chip, 'main')]["p_step"], median_abs_step=fits[(chip, 'main')]["median_abs_step"],
                                        history_trend_pct_per_year=pct(fits[(chip, 'main')]["mean_log_per_year"]))
                             for chip in ("H100", "A100")})
    chart_outlook(pack, paths, origin, caps)

    key["runtime_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "key_numbers.json"), "w") as fh:
        json.dump(key, fh, indent=1, default=str)
    try:
        os.rmdir(os.path.join(OUT, "_duck_tmp"))
    except OSError:
        pass
    say(f"\nwrote CSVs, 3 charts and key_numbers.json to {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
