#!/usr/bin/env python3
"""Resale outlook for used H100 and A100 data-centre GPUs, 1 to 3 years ahead.

One stated model ("value in use"):

    value at time t = sum, over each month m from t to the end of the GPU's economic life, of
                      max(0, rent(m) x utilisation - running cost) x 730 hours, discounted back to t

    running cost per GPU-hour = server power per GPU (kW) x all-in power, space and operations
                                cost (USD per kW-month) x 12 / 8760

The rent is the AWS spot rental outlook in out/outlook.csv (its 10% / 50% / 90% points at +1, +2 and
+3 years; held flat after year 3). Rental and resale are different instruments: the rental outlook is
an input to the model and is never joined to a resale price series.

What the script produces
  * today's quoted resale price of a bare GPU module: the midpoint of the used-price range in American
    Compute's GPU Prices CSV (two rows of data/resale_quotes.csv).
  * the model's value in use today (range over the assumptions), reported beside that quote. The two are
    different things: the model values the rent an installed GPU can still earn, the quote is what a bare
    module sells for. Nothing is fitted to the quotes and the gap between them is not a calibration check.
  * the model's value at +1, +2, +3 years as a share of its value today, and that share applied to
    today's quoted resale price (USD per bare GPU module).
  * one-at-a-time sensitivities, and the same calculation on one other rental benchmark.

Revised 15 Nov 2026, after reading the brief's references (not scored against the 7-13 Nov truth):
  * a floor: the low, middle and high resale figures cannot fall below 1%, 2% and 5% of the chip's
    launch price (the model alone lets a GPU that has stopped earning reach $0; old chips still sell);
  * the rental input is now the outlook capped at AWS's list price (longview.py, step 4);
  * the same calculation on three steady hardware-trend rent paths (-18%, -25%, -35% a year);
  * today's quote can be replaced on the command line without storing anyone's figure in the repo.
`--rent-variant main_nocap --no-floor` reproduces the figures as they were before this revision.

Inputs : longview/out/outlook.csv, longview/data/resale_quotes.csv, pack daily_statistics.csv and
         hardware_fundamentals/hardware_fundamentals.csv (launch prices)
Outputs: longview/out/resale_outlook.csv, resale_assumptions.csv, chart4_resale.png
No network. Fixed seed. Run from the repo root:  uv run python longview/resale.py
The pack is found at ./gpu-price-pack-v0.1.1-handover or at GPUVAL_PACK.
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True  # nothing is written next to the read-only data pack

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PACK = os.environ.get("GPUVAL_PACK", os.path.join(ROOT, "gpu-price-pack-v0.1.1-handover"))
OUT = os.path.join(HERE, "out")
QUOTES = os.path.join(HERE, "data", "resale_quotes.csv")
OUTLOOK = os.path.join(OUT, "outlook.csv")

SEED = 20261114
N_DRAWS = 20000
HOURS_PER_MONTH = 8760 / 12
Z90 = 1.2815515655446004            # standard normal 90% point
HORIZONS = (1, 2, 3)                # years ahead
MAX_MONTHS = 12 * 9                 # longest life considered, with room to spare
CHIPS = ("H100", "A100")

# --- today's quoted resale price: one publisher's CSV, used with the attribution its header asks for ----
AMC_CITE = "American Compute, GPU Prices, October 2026, https://www.amcompute.com/gpu-prices"
ANCHOR_URL = "https://www.amcompute.com/gpu-prices/data.csv"

# --- the floor: what a GPU still sells for once it has stopped earning its keep, as a share of launch price ----
# Low / middle / high. A judgement range; RESALE.md says what it leans on. Launch prices are the pack's
# (Epoch AI, hardware_fundamentals.csv): the H100 figure is a server price divided by eight and the
# A100 figure is the 40 GB part, the only A100 SXM row with a price.
FLOOR_SHARE = (0.01, 0.02, 0.05)
LAUNCH_ROW = {"H100": "NVIDIA H100 SXM5 80GB", "A100": "NVIDIA A100 SXM4 40 GB"}
SCENARIO_MAIN = "aws_spot_outlook"

# --- what each public figure is, read from the words in its `measure` column -------------------------
KIND_WORDS = (("middle of resale prices", "resale_middle"), ("executed secondary sale", "one_sale"),
              ("asking price", "ask"), ("appraised", "appraisal"))
KIND_STYLE = {  # kind: (chart marker, filled, plain name)
    "resale_middle": ("s", True, "middle of resale prices over 12 months"),
    "one_sale": ("o", True, "one individual sale"),
    "ask": ("o", False, "asking price"),
    "appraisal": ("D", False, "appraisal, one example"),
    "other": ("v", False, "other public figure"),
}
BASIS_NAME = {"server": "8-GPU server price / 8", "module": "bare GPU module"}

# --- assumptions: (low, central, high). Low and high are judgement unless a basis is given. --------
AMC_CALC = AMC_CITE + " (calculator default)"
ASSUME = {
    "H100": {
        "utilisation": (0.50, 0.70, 0.90),
        "server_kw_per_gpu": (1.00, 1.275, 1.60),
        "cost_per_kw_month": (150.0, 280.0, 450.0),
        "discount_rate": (0.08, 0.12, 0.20),
        "remaining_life_years": (2.0, 4.0, 6.0),
    },
    "A100": {
        "utilisation": (0.50, 0.70, 0.90),
        "server_kw_per_gpu": (0.65, 0.81, 1.00),
        "cost_per_kw_month": (150.0, 280.0, 450.0),
        "discount_rate": (0.08, 0.12, 0.20),
        "remaining_life_years": (1.0, 3.0, 5.0),
    },
}
META = {
    "utilisation": ("Utilisation", "share of hours billed",
                    "Judgement; no public figure for these fleets was found."),
    "server_kw_per_gpu": ("Server power per GPU", "kW",
                          "Central: an 8-GPU server's maximum draw divided by 8 (10.2 kW for an NVIDIA DGX H100, "
                          "6.5 kW for an NVIDIA DGX A100; NVIDIA's published DGX system specifications). "
                          "Low and high: judgement (little server overhead to added cooling overhead)."),
    "cost_per_kw_month": ("Power, space and operations cost", "USD per kW-month, all-in",
                          "Central: " + AMC_CALC + ". Low and high: judgement (cheap owned power to staffed premium colocation)."),
    "discount_rate": ("Discount rate", "per year",
                      "Central: " + AMC_CALC + ". Low and high: judgement (secured lender to equity return)."),
    "remaining_life_years": ("Remaining economic life from 31 Oct 2026", "years",
                             "Years the GPU still finds a renter at the benchmark price; worth nothing after. H100 central 4: "
                             "a six-year service life counted from a 2024 installation. Six years is the depreciation schedule "
                             "several large owners use (American Compute, GPU Depreciation Schedules vs Resale Value, 27 Oct 2026, "
                             "https://www.amcompute.com/blog/gpu-useful-life-depreciation-schedules); the installation year is "
                             "judgement. A100 central 3: CoreWeave signed an A100 contract that runs into 2029 (Tom's Hardware, "
                             "12 Aug 2026, https://www.tomshardware.com/tech-industry/coreweave-ceo-mike-intrator-says-it-has-"
                             "signed-an-a100-contract-running-into-2029). Low and high: judgement."),
}
ORDER = ["utilisation", "server_kw_per_gpu", "cost_per_kw_month", "discount_rate", "remaining_life_years"]

# one alternative rental benchmark: a different instrument (neocloud rate cards, on demand)
ALT = {"source_id": "fastgpu_daily", "instrument_class": "neocloud_rate_card", "pricing_tier": "on_demand",
       "gpu_model": {"H100": "H100 SXM 80GB", "A100": "A100 80GB"}}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COL = {"A100": "#2a78d6", "H100": "#eb6834"}
GREY = "#8a8984"


# ------------------------------------------------------------------ inputs ----
def load_outlook(variant="main"):
    """`variant` is "main" (paths capped at AWS's list price) or "main_nocap" (as before the cap)."""
    o = pd.read_csv(OUTLOOK)
    other = "free_notrend" + ("_nocap" if variant.endswith("_nocap") else "")
    out = {}
    for chip in CHIPS:
        m = o[(o.chip == chip) & (o.variant == variant)].sort_values("years_ahead")
        f = o[(o.chip == chip) & (o.variant == other)].sort_values("years_ahead")
        assert list(m.years_ahead) == [1, 2, 3] and list(f.years_ahead) == [1, 2, 3]
        p0 = float(m.price_at_origin.iloc[0])
        trend = {}
        for name, g in o[(o.chip == chip) & o.variant.str.startswith("hardware_trend_")].groupby("variant", sort=False):
            trend[name] = float(g.sort_values("years_ahead").q50.iloc[0]) / p0 - 1.0      # the steady yearly change
        out[chip] = {
            "origin": pd.Timestamp(m.origin.iloc[0]),
            "targets": [str(t) for t in m.target],
            "series": m.series.iloc[0],
            "p0": p0,
            "main": (m.q10_widened.to_numpy(float), m.q50.to_numpy(float), m.q90_widened.to_numpy(float)),
            "no_pullback_q50": f.q50.to_numpy(float),
            "variant": variant, "other_variant": other, "trend": trend,
        }
    return out


def launch_prices():
    """Launch price per GPU from the pack's hardware table (Epoch AI). Used only for the floor."""
    hw = pd.read_csv(os.path.join(PACK, "hardware_fundamentals", "hardware_fundamentals.csv"))
    out = {}
    for chip, name in LAUNCH_ROW.items():
        row = hw[hw.accelerator == name].iloc[0]
        out[chip] = {"usd": float(row.release_price_usd), "row": name, "release_date": str(row.release_date)}
    return out


def load_quotes():
    q = pd.read_csv(QUOTES, dtype=str).fillna("")
    lo, hi = [], []
    for v in q["value"]:
        parts = v.split("-")
        lo.append(float(parts[0]))
        hi.append(float(parts[-1]))
    q["lo"], q["hi"] = lo, hi
    q["mid"] = (q.lo + q.hi) / 2
    q["chip"] = np.where(q.gpu.str.startswith("H100"), "H100", np.where(q.gpu.str.startswith("A100"), "A100", ""))
    q["basis"] = np.where(q.unit.str.contains("8-GPU server"), "server",
                          np.where(q.unit.str.contains("bare module"), "module", "other"))
    per = np.where(q.basis == "server", 8.0, 1.0)
    q["lo_gpu"], q["hi_gpu"], q["mid_gpu"] = q.lo / per, q.hi / per, q.mid / per
    q["kind"] = [next((k for words, k in KIND_WORDS if words in s), "other") for s in q.measure]
    q["is_anchor"] = (q.url == ANCHOR_URL) & (q.basis == "module")
    return q


def anchors(q, chip):
    """Today's quoted resale price of a bare GPU module: the midpoint of the used-price range in the publisher's CSV."""
    rows = q[(q.chip == chip) & q.is_anchor]
    if len(rows) != 1:
        raise SystemExit(f"{QUOTES}: expected one bare-module row for {chip} from {ANCHOR_URL}, found {len(rows)}")
    row = rows.iloc[0]
    return {"module": float(row.mid), "lo": float(row.lo), "hi": float(row.hi), "gpu": row.gpu,
            "published": row.published_date,
            "source": f"midpoint of the used-price range for {row.gpu} ({row.lo:,.0f} to {row.hi:,.0f} USD) in {AMC_CITE}"}


def load_alt_benchmark(origin):
    ds = pd.read_csv(os.path.join(PACK, "daily_statistics.csv"), parse_dates=["date"])
    out = {}
    for chip in CHIPS:
        m = ds[(ds.source_id == ALT["source_id"]) & (ds.instrument_class == ALT["instrument_class"])
               & (ds.pricing_tier == ALT["pricing_tier"]) & (ds.gpu_model == ALT["gpu_model"][chip])
               & (ds.date <= origin)].sort_values("date")
        r = m.iloc[-1]
        out[chip] = {"level": float(r.median_usd_per_gpu_hour), "date": r.date, "n_providers": int(r.n_providers),
                     "label": f"{ALT['source_id']} {ALT['gpu_model'][chip]} {ALT['pricing_tier']} {ALT['instrument_class']} median"}
    return out


# ------------------------------------------------------------------- model ----
def knots_from_quantiles(p0, q10, q50, q90, z):
    """Log rent at years 0..3 for each draw. A draw sits at the same percentile of the range at every horizon."""
    z = np.atleast_1d(np.asarray(z, float))[:, None]
    up = np.log(q90 / q50)[None, :]
    dn = np.log(q50 / q10)[None, :]
    k = np.log(q50)[None, :] + np.where(z >= 0, z / Z90 * up, z / Z90 * dn)
    return np.concatenate([np.full((len(z), 1), np.log(p0)), k], axis=1)


def rent_from_knots(knots):
    """Monthly rent (USD per GPU-hour) at mid-month: straight line in logs between yearly points, flat after year 3."""
    t = np.minimum((np.arange(MAX_MONTHS) + 0.5) / 12, 3.0)
    j = np.minimum(np.floor(t).astype(int), 2)
    w = (t - j)[None, :]
    return np.exp(knots[:, j] * (1 - w) + knots[:, j + 1] * w)


def value_in_use(rent, util, kw, cost_kw_month, disc, life, start_year=0):
    """Discounted net rental income per GPU from `start_year` to the end of life. All parameters are arrays (n,)."""
    util, kw, cost_kw_month, disc, life = (np.atleast_1d(np.asarray(a, float))[:, None]
                                           for a in (util, kw, cost_kw_month, disc, life))
    m = np.arange(MAX_MONTHS)[None, :]
    running = kw * cost_kw_month * 12 / 8760                       # USD per GPU-hour, paid every hour
    net = np.maximum(0.0, rent * util - running) * HOURS_PER_MONTH  # idle rather than rent at a loss
    alive = np.clip(12 * life - m, 0.0, 1.0) * (m >= 12 * start_year)
    t = (m + 0.5) / 12 - start_year
    return (net * alive / (1 + disc) ** t).sum(axis=1)


def central(a):
    return {k: a[k][1] for k in ORDER}


def v(rent, p, start_year=0):
    return value_in_use(rent, p["utilisation"], p["server_kw_per_gpu"], p["cost_per_kw_month"],
                        p["discount_rate"], p["remaining_life_years"], start_year)


def q3(x):
    return np.quantile(x, [0.1, 0.5, 0.9])


def usd_with_floor(share, anchor_usd, floor_usd, years_ahead):
    """Resale figure in USD: the model's share of today's quote, never below the floor.
    The unfloored figure keeps its old rounding (nearest $100); a figure sitting on the floor is
    shown to the nearest $10 so that a floor of $150 does not read as $200."""
    raw = float(share) * anchor_usd
    if years_ahead == 0:
        return raw, raw
    raw = round(raw, -2)
    return max(raw, round(floor_usd, -1)), raw


def run_chip(chip, ol, alt, anchor, rng, launch_usd, use_floor=True):
    a = ASSUME[chip]
    floors = [f * launch_usd if use_floor else 0.0 for f in FLOOR_SHARE]
    q10, q50, q90 = ol["main"]
    p0 = ol["p0"]
    rent_c = rent_from_knots(knots_from_quantiles(p0, q10, q50, q90, [0.0]))
    pc = central(a)
    v0c = float(v(rent_c, pc)[0])

    # --- simulation: every assumption drawn from a triangle over low / central / high; rent path by percentile
    z = rng.standard_normal(N_DRAWS)
    draw = {k: rng.triangular(a[k][0], a[k][1], a[k][2], N_DRAWS) for k in ORDER}
    rent_z = rent_from_knots(knots_from_quantiles(p0, q10, q50, q90, z))
    v0 = v(np.repeat(rent_c, N_DRAWS, axis=0), draw)             # today: central rent path, own assumptions
    rows = [{"chip": chip, "years_ahead": 0, "target_date": ol["origin"].strftime("%Y-%m-%d"),
             "scenario": SCENARIO_MAIN, "rent_input": ol["series"],
             "rent_q10": p0, "rent_q50": p0, "rent_q90": p0,
             "model_value_in_use_usd": q3(v0), "share": np.array([1.0, 1.0, 1.0]),
             "share_of_draws_past_end_of_life": 0.0,
             "model_value_all_central_usd": v0c}]
    for h in HORIZONS:
        vh = v(rent_z, draw, start_year=h)
        rows.append({"chip": chip, "years_ahead": h, "target_date": ol["targets"][h - 1],
                     "scenario": SCENARIO_MAIN, "rent_input": ol["series"],
                     "rent_q10": q10[h - 1], "rent_q50": q50[h - 1], "rent_q90": q90[h - 1],
                     "model_value_in_use_usd": q3(vh), "share": q3(vh / v0),
                     "share_of_draws_past_end_of_life": float((draw["remaining_life_years"] <= h).mean()),
                     "model_value_all_central_usd": float(v(rent_c, pc, start_year=h)[0])})
    # --- the same calculation on steady hardware-trend rent paths (comparison, not a forecast): rent falls at
    #     one yearly rate from today's price for the whole of the GPU's life; the range is over the assumptions only
    t_mid = (np.arange(MAX_MONTHS) + 0.5) / 12
    for name, rate in ol["trend"].items():
        rent_s = (p0 * (1.0 + rate) ** t_mid)[None, :]
        v0s = v(rent_s, draw)
        v0sc = float(v(rent_s, pc)[0])
        label = f"steady {rate * 100:+.0f}% a year from ${p0:.2f} (hardware-trend comparison)"
        for h in (0,) + HORIZONS:
            vh = v(rent_s, draw, start_year=h)
            rh = p0 * (1.0 + rate) ** h
            rows.append({"chip": chip, "years_ahead": h, "scenario": name, "rent_input": label,
                         "target_date": ol["origin"].strftime("%Y-%m-%d") if h == 0 else ol["targets"][h - 1],
                         "rent_q10": rh, "rent_q50": rh, "rent_q90": rh,
                         "model_value_in_use_usd": q3(vh), "share": q3(vh / v0s) if h else np.array([1.0, 1.0, 1.0]),
                         "share_of_draws_past_end_of_life": float((draw["remaining_life_years"] <= h).mean()),
                         "model_value_all_central_usd": float(v(rent_s, pc, start_year=h)[0]) if h else v0sc})
    out = []
    for r in rows:
        mv, sh = r["model_value_in_use_usd"], r["share"]
        d = {"chip": chip, "years_ahead": r["years_ahead"], "target_date": r["target_date"],
             "scenario": r["scenario"], "rent_input_series": r["rent_input"],
             "rent_q10_widened": round(r["rent_q10"], 4), "rent_q50": round(r["rent_q50"], 4),
             "rent_q90_widened": round(r["rent_q90"], 4),
             "model_value_unit": "USD per GPU installed in a working server (the rent it can still earn; not a resale price)"}
        for name, val in zip(("low", "central", "high"), mv):
            d[f"model_value_in_use_usd_{name}"] = round(float(val), -2)
        d["model_value_all_assumptions_central_usd"] = round(r["model_value_all_central_usd"], -2)
        for name, val in zip(("low", "central", "high"), sh):
            d[f"share_of_today_{name}"] = round(float(val), 3)
        d["resale_unit"] = "USD per bare GPU module"
        d["quote_today_bare_module_usd"] = anchor["module"]
        d["quote_today_source"] = anchor["source"]
        unfloored = {}
        for name, val, floor in zip(("low", "central", "high"), sh, floors):
            d[f"resale_usd_bare_module_{name}"], unfloored[name] = usd_with_floor(val, anchor["module"], floor, r["years_ahead"])
        d["share_of_draws_past_end_of_life"] = round(r["share_of_draws_past_end_of_life"], 3)
        for name, floor in zip(("low", "central", "high"), floors):
            d[f"floor_usd_{name}"] = round(floor, -1)
            d[f"resale_usd_bare_module_{name}_no_floor"] = unfloored[name]
        d["launch_price_usd_for_floor"] = launch_usd
        d["rent_outlook_variant"] = ol["variant"] if r["scenario"] == SCENARIO_MAIN else ""
        out.append(d)

    # --- one-at-a-time sensitivities around the all-central case
    def share2(rent, p, v0_):
        return float(v(rent, p, start_year=2)[0]) / v0_

    sens = []
    rent_lo = rent_from_knots(knots_from_quantiles(p0, q10, q50, q90, [-Z90]))
    rent_hi = rent_from_knots(knots_from_quantiles(p0, q10, q50, q90, [Z90]))
    sens.append({"chip": chip, "kind": "assumption", "name": "Rental price path (AWS spot outlook, main, widened)",
                 "unit": "USD per GPU-hour at +1 / +2 / +3 years",
                 "low": " / ".join(f"{x:.2f}" for x in q10), "central": " / ".join(f"{x:.2f}" for x in q50),
                 "high": " / ".join(f"{x:.2f}" for x in q90),
                 "basis": f"longview/out/outlook.csv, {ol['series']}, variant {ol['variant']}; 10%, 50% and 90% points; "
                          f"${p0:.2f} at 31 Oct 2026; flat after year 3.",
                 "today_low": float(v(rent_lo, pc)[0]), "today_high": float(v(rent_hi, pc)[0]),
                 "share2_low": share2(rent_lo, pc, v0c), "share2_high": share2(rent_hi, pc, v0c)})
    for k in ORDER:
        lo_p, hi_p = dict(pc), dict(pc)
        lo_p[k], hi_p[k] = a[k][0], a[k][2]
        v_lo, v_hi = float(v(rent_c, lo_p)[0]), float(v(rent_c, hi_p)[0])
        sens.append({"chip": chip, "kind": "assumption", "name": META[k][0], "unit": META[k][1],
                     "low": a[k][0], "central": a[k][1], "high": a[k][2], "basis": META[k][2],
                     "today_low": v_lo, "today_high": v_hi,
                     "share2_low": share2(rent_c, lo_p, v_lo), "share2_high": share2(rent_c, hi_p, v_hi)})
    for s in sens:
        s["today_central"] = v0c
        s["share2_central"] = share2(rent_c, pc, v0c)
        s["swing_today_pct"] = abs(s["today_high"] - s["today_low"]) / v0c * 100
        s["swing_share2_points"] = abs(s["share2_high"] - s["share2_low"]) * 100
    for i, s in enumerate(sorted(sens, key=lambda s: -s["swing_today_pct"])):
        s["rank_by_swing_today"] = i + 1
    for i, s in enumerate(sorted(sens, key=lambda s: -s["swing_share2_points"])):
        s["rank_by_swing_share2"] = i + 1

    # --- the same calculation on other rental inputs (central assumptions throughout)
    bench = []

    def add(name, basis, level, rent):
        v0_ = float(v(rent, pc)[0])
        bench.append({"chip": chip, "kind": "rental_benchmark", "name": name, "unit": "USD per GPU-hour today",
                      "low": "", "central": round(level, 4), "high": "", "basis": basis,
                      "today_low": np.nan, "today_central": v0_, "today_high": np.nan,
                      "share2_low": np.nan, "share2_central": share2(rent, pc, v0_), "share2_high": np.nan,
                      "today_vs_main_pct": (v0_ / v0c - 1) * 100})

    add("AWS spot, main outlook, middle path (the base case)",
        "hyperscaler_spot; pulls back to the chip's usual level", p0, rent_c)
    nf = ol["no_pullback_q50"]
    add("AWS spot, other reading of the outlook: no pull-back, middle path",
        f"outlook.csv variant {ol['other_variant']}, 50% point", p0,
        np.exp(np.interp(np.minimum((np.arange(MAX_MONTHS) + 0.5) / 12, 3.0), [0, 1, 2, 3],
                         np.log(np.concatenate([[p0], nf]))))[None, :])
    lvl = alt[chip]["level"]
    basis = (f"{alt[chip]['label']}, {alt[chip]['date']:%d %b %Y}, {alt[chip]['n_providers']} providers; "
             "pack daily_statistics.csv (FastGPU, CC BY 4.0)")
    add("Neocloud rate-card median, held flat (no outlook exists for it)", basis, lvl,
        np.full((1, MAX_MONTHS), lvl))
    add("Neocloud rate-card median, moved in step with the AWS middle path (level effect only)",
        basis + "; a what-if, not a forecast of that instrument", lvl, rent_c * lvl / p0)
    for name, rate in ol["trend"].items():
        add(f"AWS spot today, then a steady {rate * 100:+.0f}% a year (hardware-trend comparison)",
            "outlook.csv variant " + name + "; the fall continues for the GPU's whole life; not a forecast", p0,
            (p0 * (1.0 + rate) ** t_mid)[None, :])

    # --- today's quote: every unfloored dollar figure scales with it
    quote = []
    c_rows = [r for r in rows if r["scenario"] == SCENARIO_MAIN and r["years_ahead"] > 0]
    for which, val in (("the low end of the published range", anchor["lo"]), ("the value used", anchor["module"]),
                       ("the high end of the published range", anchor["hi"])):
        mids = [usd_with_floor(r["share"][1], val, floors[1], r["years_ahead"])[0] for r in c_rows]
        quote.append({"chip": chip, "kind": "quote_sensitivity", "name": f"Today's quote at {which}", "unit": "USD per bare GPU module",
                      "low": "", "central": val, "high": "",
                      "basis": "resale middle at +1 / +2 / +3 years: " + " / ".join(f"{m:,.0f}" for m in mids)
                               + f"; each extra $1,000 on the quote adds {1000 / anchor['module'] * 100:.1f}% to every unfloored dollar figure"})

    # --- choices that are fixed, not ranged
    floor_txt = ("Not zero: the low, middle and high resale figures are floored at "
                 + ", ".join(f"{f:.0%}" for f in FLOOR_SHARE) + f" of the launch price (${launch_usd:,.0f}; pack hardware table, Epoch AI), "
                 + "that is $" + ", $".join(f"{round(x, -1):,.0f}" for x in floors) + ". A judgement range; see RESALE.md."
                 if use_floor else "Set to zero (run with --no-floor).")
    fixed = [("Value after the end of economic life (scrap, parts, reuse)", "USD", "floor" if use_floor else 0, floor_txt),
             ("Rent after year 3", "USD per GPU-hour", "flat at the +3 year level",
              "The rental outlook stops at +3 years."),
             ("Months when rent x utilisation is below running cost", "USD", 0,
              "Net income is floored at zero (the owner idles the GPU rather than rent at a loss)."),
             ("Costs left out", "", "staff, repairs, network, sales, tax, moving and re-installing",
              "Not modelled; each would lower the value in use."),
             ("How the ranges are built", "", f"{N_DRAWS} draws, seed {SEED}",
              "Each ranged assumption is drawn from a triangle over low / central / high, independently; the rent path is "
              "drawn by percentile of the outlook's range and stays at that percentile at every horizon. Low, central, high "
              "= 10%, 50%, 90% of the draws. Share of today's value = value at the horizon on the drawn rent path / value "
              "today on the middle rent path, same assumptions.")]
    fx = [{"chip": chip, "kind": "fixed_choice", "name": n, "unit": u, "low": "", "central": c, "high": "", "basis": b}
          for n, u, c, b in fixed]
    return out, sens + bench + quote + fx, v0, v0c


# ------------------------------------------------------------------- chart ----
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


def money(x):
    """Dollar label for chart text (the sign is escaped so matplotlib does not read it as maths)."""
    if x < 50:
        return "\\$0"
    if x < 1000:
        return f"\\${x:,.0f}"
    return f"\\${x / 1000:.1f}k" if x < 99950 else f"\\${x / 1000:.0f}k"


def usd(x):
    """Exact dollar label for chart text."""
    return f"\\${x:,.0f}"


def published_span(dates):
    """'Sep to Oct 2026' from the rows' published dates; empty if no row carries a full date."""
    d = pd.to_datetime(pd.Series(list(dates), dtype=str), format="%Y-%m-%d", errors="coerce").dropna()
    if d.empty:
        return ""
    a, b = d.min(), d.max()
    if (a.year, a.month) == (b.year, b.month):
        return f"{a:%b %Y}"
    return f"{a:%b} to {b:%b %Y}" if a.year == b.year else f"{a:%b %Y} to {b:%b %Y}"


def quote_marker(kind, is_anchor, **kw):
    """One marker per kind of public figure; the figure used as today's quote is drawn in full ink."""
    marker, filled, _ = KIND_STYLE[kind]
    c = INK if is_anchor else INK2
    return dict(marker=marker, mfc=c if filled else SURFACE, mec=c, mew=1.5, ls="none", **kw)


def chart(outlook_rows, quotes, anch, out_dir=OUT, use_floor=True):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.8), facecolor=SURFACE)
    X_QUOTE, X_MODEL, X_H = 0.0, 1.1, {1: 2.5, 2: 3.75, 3: 5.0}
    for ax, chip in zip(axes, CHIPS):
        col = COL[chip]
        rows = [r for r in outlook_rows if r["chip"] == chip and r["scenario"] == SCENARIO_MAIN]
        today = rows[0]
        a = anch[chip]
        qs = quotes[(quotes.chip == chip) & quotes.basis.isin(list(BASIS_NAME))].reset_index(drop=True)
        # public figures: today's quote (a bare module) first, then server prices divided by 8, low to high
        order = list(qs.sort_values(["basis", "mid_gpu"]).index)
        offs = np.linspace(-0.3, 0.3, len(order)) if len(order) > 1 else [0.0]
        x_anchor = X_QUOTE
        for off, i in zip(offs, order):
            r = qs.loc[i]
            if r["hi_gpu"] > r["lo_gpu"]:
                ax.plot([X_QUOTE + off] * 2, [r["lo_gpu"], r["hi_gpu"]], color=INK if r["is_anchor"] else INK2, lw=1.4,
                        solid_capstyle="round", zorder=3)
            ax.plot(X_QUOTE + off, r["mid_gpu"], zorder=4, **quote_marker(r["kind"], r["is_anchor"], ms=8.5))
            if r["is_anchor"]:
                x_anchor = X_QUOTE + off
        # today's quoted price of a bare module, carried across as the reference level
        ax.plot([x_anchor, X_H[3] + 1.2], [a["module"]] * 2, color=INK2, lw=0.9, zorder=1)
        ax.annotate(f"today's quote, bare module\n{usd(a['module'])}\n(range {usd(a['lo'])} to {usd(a['hi'])})",
                    (X_H[3] + 1.2, a["module"]), xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                    fontsize=8.5, color=INK2, linespacing=1.3)
        # the model's value in use today: the rent an installed GPU can still earn, a different thing from the quote
        lo, mid, hi = (today[f"model_value_in_use_usd_{k}"] for k in ("low", "central", "high"))
        ax.plot([X_MODEL] * 2, [lo, hi], color=GREY, lw=9, alpha=0.35, solid_capstyle="butt", zorder=2)
        ax.plot(X_MODEL, mid, marker="o", ms=9, mfc=GREY, mec=SURFACE, mew=2, ls="none", zorder=4)
        ax.text(X_MODEL, 0.975, f"model\n{money(mid)}\n{money(lo)} to {money(hi)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color=INK, linespacing=1.35)
        # outlook: today's quote for a bare module times the model's share of today's value
        for r in rows[1:]:
            x = X_H[r["years_ahead"]]
            lo, mid, hi = (r[f"resale_usd_bare_module_{k}"] for k in ("low", "central", "high"))
            ax.plot([x] * 2, [lo, hi], color=col, lw=9, alpha=0.3, solid_capstyle="butt", zorder=2)
            ax.plot(x, mid, marker="o", ms=9, mfc=col, mec=SURFACE, mew=2, ls="none", zorder=4)
            ax.text(x, 0.975, f"{mid / a['module']:.0%} of today\n{money(mid)}\n{money(lo)} to {money(hi)}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=INK, linespacing=1.35)
            # the same middle if rent instead falls a steady 25% a year (hardware-trend comparison)
            alt = [s for s in outlook_rows if s["chip"] == chip and s["scenario"] == "hardware_trend_25"
                   and s["years_ahead"] == r["years_ahead"]]
            if alt:
                ax.plot(x + 0.32, alt[0]["resale_usd_bare_module_central"], marker="D", ms=6.5, mfc=SURFACE, mec=col, mew=1.5,
                        ls="none", zorder=4)
        style(ax, "USD per GPU")
        ax.set_title(f"{chip}: resale price of a bare GPU module, today and in 1 to 3 years", color=INK, fontsize=11,
                     loc="left", pad=26)
        ax.set_xlim(-0.6, X_H[3] + 1.25)
        top = max([today["model_value_in_use_usd_high"]] + [r["resale_usd_bare_module_high"] for r in rows[1:]]
                  + list(qs.hi_gpu))
        ax.set_ylim(0, top * 1.36)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"\\${y / 1000:.0f}k"))
        ax.set_xticks([X_QUOTE, X_MODEL, X_H[1], X_H[2], X_H[3]])
        span = published_span(qs.published_date)
        ax.set_xticklabels(["public figures" + (f"\n{span}" if span else ""), "model\ntoday", "+1 yr", "+2 yr", "+3 yr"],
                           fontsize=9)
        ax.axvline(1.8, color="#c9c8c3", lw=0.8)
        ax.text(-0.55, 1.012, "today: two different things", transform=ax.get_xaxis_transform(), ha="left",
                va="bottom", fontsize=8.5, color=INK2)
        ax.text(1.95, 1.012, "outlook: bare module resale price, 10% to 90% range", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=8.5, color=INK2)
        # legend: every public figure is named by what it is
        h = [Line2D([], [], label="today's quote: bare GPU module, midpoint of the published range of resale prices",
                    **quote_marker("resale_middle", True, ms=8))]
        others = qs[~qs.is_anchor]
        for kind in KIND_STYLE:
            for basis in BASIS_NAME:
                if ((others["kind"] == kind) & (others["basis"] == basis)).any():
                    h.append(Line2D([], [], label=f"{KIND_STYLE[kind][2]}: {BASIS_NAME[basis]}",
                                    **quote_marker(kind, False, ms=8)))
        h.append(Line2D([], [], color=GREY, lw=7, alpha=0.35, marker="o", ms=8, mfc=GREY, mec=SURFACE, mew=2,
                        label="model, not a resale price: discounted net rent an installed GPU can still earn, 10% to 90%"))
        h.append(Line2D([], [], color=col, lw=7, alpha=0.3, marker="o", ms=8, mfc=col, mec=SURFACE, mew=2,
                        label="resale outlook, bare module: today's quote x the model's share of today's value"))
        if any(s["scenario"] == "hardware_trend_25" for s in outlook_rows):
            h.append(Line2D([], [], marker="D", ms=6.5, mfc=SURFACE, mec=col, mew=1.5, ls="none",
                            label="the same middle if rent instead falls a steady 25% a year (hardware-trend comparison)"))
        ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=1, fontsize=8, frameon=False, labelcolor=INK2)
    gpus = " and ".join(anch[c]["gpu"] for c in CHIPS)
    updated = ", ".join(sorted({anch[c]["published"] for c in CHIPS}))
    fig.text(0.012, 0.012,
             "The model values the rent an installed GPU can still earn. Today's quote is the resale price of a bare module. "
             "They are different things, shown side by side; nothing is fitted to the quotes.\n"
             f"Today's quote: midpoint of the used-price range for {gpus} in {AMC_CITE} "
             f"(CSV updated {updated}; free to use with attribution).\n"
             "Other public figures: data/resale_quotes.csv; server prices are divided by 8. Rental and resale are different "
             "instruments and are not joined: the AWS spot rental outlook only feeds the model.\n"
             "Tops of the ranges assume rents near the top of the rental outlook (itself capped at AWS's list price) and may be out of reach.\n"
             + ("Floor: the low, middle and high figures cannot fall below 1%, 2% and 5% of the launch price. "
                "A GPU that has stopped earning still sells for something."
                if use_floor else
                "A range that reaches \\$0 means the GPU has stopped earning in at least 10% of cases; the model gives it no scrap value."),
             fontsize=7.5, color=INK2, ha="left", va="bottom", linespacing=1.4)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.36, wspace=0.17)
    fig.savefig(os.path.join(out_dir, "chart4_resale.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------- main ----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rent-variant", default="main", choices=["main", "main_nocap"],
                    help="rental outlook to feed the model: capped at AWS's list price (default) or as before the cap")
    ap.add_argument("--no-floor", action="store_true", help="let a GPU that has stopped earning reach $0, as before the floor")
    ap.add_argument("--anchor-h100", type=float, help="today's quote for a bare H100 module, USD (default: midpoint in resale_quotes.csv)")
    ap.add_argument("--anchor-a100", type=float, help="today's quote for a bare A100 80GB module, USD")
    ap.add_argument("--out-dir", default=OUT, help="where to write the three output files")
    args = ap.parse_args(argv)
    out_dir = args.out_dir
    use_floor = not args.no_floor
    os.makedirs(out_dir, exist_ok=True)
    ol = load_outlook(args.rent_variant)
    quotes = load_quotes()
    alt = load_alt_benchmark(ol["H100"]["origin"])
    launch = launch_prices()
    rng = np.random.default_rng(SEED)
    outlook_rows, assume_rows, anch = [], [], {}
    given = {"H100": args.anchor_h100, "A100": args.anchor_a100}
    for chip in CHIPS:
        anch[chip] = anchors(quotes, chip)
        if given[chip]:
            anch[chip].update(module=float(given[chip]), source=f"${given[chip]:,.0f}, given on the command line (not from resale_quotes.csv)")
        o, s, v0, v0c = run_chip(chip, ol[chip], alt, anch[chip], rng, launch[chip]["usd"], use_floor)
        outlook_rows += o
        assume_rows += s

    pd.DataFrame(outlook_rows).to_csv(os.path.join(out_dir, "resale_outlook.csv"), index=False)
    a = pd.DataFrame(assume_rows)
    for c in ("today_low", "today_central", "today_high"):
        a[c] = a[c].round(-2)
    for c in ("share2_low", "share2_central", "share2_high"):
        a[c] = a[c].round(3)
    for c in ("swing_today_pct", "swing_share2_points", "today_vs_main_pct"):
        a[c] = a[c].round(1)
    for c in ("rank_by_swing_today", "rank_by_swing_share2"):
        a[c] = a[c].astype("Int64")
    a = a.rename(columns={"today_low": "value_today_usd_if_low", "today_central": "value_today_usd_central",
                          "today_high": "value_today_usd_if_high", "share2_low": "share_at_2y_if_low",
                          "share2_central": "share_at_2y_central", "share2_high": "share_at_2y_if_high"})
    cols = ["chip", "kind", "name", "unit", "low", "central", "high", "value_today_usd_if_low", "value_today_usd_central",
            "value_today_usd_if_high", "swing_today_pct", "rank_by_swing_today", "share_at_2y_if_low", "share_at_2y_central",
            "share_at_2y_if_high", "swing_share2_points", "rank_by_swing_share2", "today_vs_main_pct", "basis"]
    a[cols].to_csv(os.path.join(out_dir, "resale_assumptions.csv"), index=False)
    chart(outlook_rows, quotes, anch, out_dir, use_floor)

    # ---- console summary
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 70)
    df_all = pd.DataFrame(outlook_rows)
    df = df_all[df_all.scenario == SCENARIO_MAIN]
    print(f"rental outlook variant: {args.rent_variant}; floor: "
          + (", ".join(f"{f:.0%}" for f in FLOOR_SHARE) + " of launch price (low, middle, high)" if use_floor else "none"))
    for chip in CHIPS:
        d = df[df.chip == chip]
        t = d.iloc[0]
        print(f"\n=== {chip} ===  all resale figures below are USD per bare GPU module")
        print(f"today's quote, bare module: ${anch[chip]['module']:,.0f} ({anch[chip]['source']})")
        if use_floor:
            print(f"floor: launch price ${launch[chip]['usd']:,.0f} ({launch[chip]['row']}, released {launch[chip]['release_date']}; pack hardware table) "
                  f"-> ${t.floor_usd_low:,.0f} / ${t.floor_usd_central:,.0f} / ${t.floor_usd_high:,.0f}")
        print(f"model value in use today, installed GPU: ${t.model_value_in_use_usd_central:,.0f} "
              f"(${t.model_value_in_use_usd_low:,.0f} to ${t.model_value_in_use_usd_high:,.0f}); all-central "
              f"${t.model_value_all_assumptions_central_usd:,.0f}")
        print("   two different things, side by side: the model values the rent an installed GPU can still earn; "
              "the quote is the resale price of a bare module. Not a calibration check.")
        qs = quotes[(quotes.chip == chip) & quotes.basis.isin(list(BASIS_NAME))]
        for _, r in qs.iterrows():
            what = "today's quote" if r["is_anchor"] else KIND_STYLE[r["kind"]][2]
            print(f"   public figure {r.publisher[:34]:34s} {r.published_date:10s} {BASIS_NAME[r['basis']]:22s} "
                  f"${r.lo_gpu:>8,.0f} to ${r.hi_gpu:>8,.0f} per GPU   {what}")
        for _, r in d.iloc[1:].iterrows():
            print(f"  +{r.years_ahead}y share {r.share_of_today_central:.0%} ({r.share_of_today_low:.0%} to {r.share_of_today_high:.0%})"
                  f" | bare module ${r.resale_usd_bare_module_central:,.0f} (${r.resale_usd_bare_module_low:,.0f} to ${r.resale_usd_bare_module_high:,.0f})"
                  f" | model, installed GPU ${r.model_value_in_use_usd_central:,.0f} (${r.model_value_in_use_usd_low:,.0f} to ${r.model_value_in_use_usd_high:,.0f})"
                  f" | past end of life {r.share_of_draws_past_end_of_life:.0%}"
                  f" | before the floor ${r.resale_usd_bare_module_central_no_floor:,.0f} (${r.resale_usd_bare_module_low_no_floor:,.0f} to ${r.resale_usd_bare_module_high_no_floor:,.0f})")
        for name, g in df_all[(df_all.chip == chip) & (df_all.scenario != SCENARIO_MAIN)].groupby("scenario", sort=False):
            g = g[g.years_ahead > 0]
            print(f"  {name}: rent " + " / ".join(f"{x:.2f}" for x in g.rent_q50) + " | share of today "
                  + " / ".join(f"{x:.0%}" for x in g.share_of_today_central) + " | bare module middle "
                  + " / ".join(f"${x:,.0f}" for x in g.resale_usd_bare_module_central) + " (low "
                  + " / ".join(f"${x:,.0f}" for x in g.resale_usd_bare_module_low) + "; high "
                  + " / ".join(f"${x:,.0f}" for x in g.resale_usd_bare_module_high) + ")")
        print(a[a.chip == chip][["kind", "name", "low", "central", "high", "value_today_usd_if_low", "value_today_usd_central",
                                 "value_today_usd_if_high", "swing_today_pct", "share_at_2y_if_low", "share_at_2y_central",
                                 "share_at_2y_if_high", "today_vs_main_pct"]].to_string(index=False))
    print("\nwrote", os.path.join(out_dir, "resale_outlook.csv"), os.path.join(out_dir, "resale_assumptions.csv"),
          os.path.join(out_dir, "chart4_resale.png"))


if __name__ == "__main__":
    main()
