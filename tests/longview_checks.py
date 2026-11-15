"""Checks of the long view's ceiling and of its out-of-sample successor test. No pack, no network.

    uv run python tests/longview_checks.py

Each check guards an error that a code review found on 15 Nov 2026 (longview/LONGVIEW.md, sections 3 and 4;
longview/RESALE.md). They read only files committed under longview/out/.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "longview" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def check_state_at_the_ceiling(lv) -> None:
    """When the ceiling stops a rise, the next day is drawn from what the path did, not from what was drawn.
    Toy rules: after an "up" day the price always rises 0.5 (log), after a "flat" day it never moves, after a
    "down" day it always falls 0.5. From 0.2 under a ceiling at 1.0 the path must rise, stop, and then stay."""
    fit = dict(MC=np.array([[-0.5], [0.0], [0.5]]), MD=np.array([[0], [1], [2]], dtype=np.int64), L=np.array([1, 1, 1]),
               st0=2, lp0=0.2, thr=0.0, feedback="dir", offset=0.0)
    day = pd.Timestamp("2026-01-01")
    capped = lv.simulate(fit, day, 6, 4, np.random.default_rng(0), keep=range(1, 7), cap=float(np.exp(1.0)))
    assert np.allclose([np.log(capped[t][0]) for t in range(1, 7)], [0.7, 1.0, 1.0, 1.0, 1.0, 1.0])
    free = lv.simulate(fit, day, 3, 4, np.random.default_rng(0), keep=(1, 2, 3))                 # no ceiling: unchanged
    assert np.allclose([np.log(free[t][0]) for t in (1, 2, 3)], [0.7, 1.2, 1.7])
    falling = lv.simulate(dict(fit, st0=0, lp0=1.0), day, 2, 4, np.random.default_rng(0), keep=(1, 2), cap=float(np.exp(1.0)))
    assert np.allclose([np.log(falling[t][0]) for t in (1, 2)], [0.5, 0.0])                       # a fall is never altered


def check_resale_rent_stays_under_the_ceiling(rs) -> None:
    """No rent path drawn for the resale model passes AWS's list price; the uncapped variant has no ceiling."""
    ol = rs.load_outlook("main")
    z = np.random.default_rng(1).standard_normal(20000)
    for chip in ("H100", "A100"):
        q10, q50, q90 = ol[chip]["main"]
        cap = ol[chip]["cap"]
        assert cap is not None and q90.max() <= cap * (1 + 1e-9)
        rent = rs.rent_from_knots(rs.knots_from_quantiles(ol[chip]["p0"], q10, q50, q90, z, cap))
        assert rent.max() <= cap * (1 + 1e-12)
        loose = rs.knots_from_quantiles(ol[chip]["p0"], q10, q50, q90, z)
        assert (loose.max(axis=1) > np.log(cap)).mean() > 0.05      # the ceiling does bind on these draws
    assert rs.load_outlook("main_nocap")["H100"]["cap"] is None


def check_carried_shift_is_known_on_the_day() -> None:
    """In the out-of-sample successor test the A100's shift is measured afresh on every forecast day before
    the H200 arrived, and is the one final number on every day after."""
    k = json.loads((REPO / "longview" / "out" / "key_numbers.json").read_text())
    arrival = next(r["newer_chip_first_spot_price"] for r in k["step3"] if r["newer_chip"] == "H200")
    s = {d: v for d, v in k["step3_shift_carried_by_forecast_day"].items() if v is not None}
    before = [v for d, v in s.items() if d < arrival]
    after = {v for d, v in s.items() if d >= arrival}
    assert len(before) >= 5 and len(set(before)) == len(before)
    assert len(after) == 1 and abs(after.pop() - k["step3_shift_carried_to_h100_log_per_year"]) < 1e-3
    rows = [r for r in k["step3"] if r["newer_chip"] == "H200" and "on or after" in r["forecast_days"]]
    assert rows and all(r["n_origins"] > 0 for r in rows)


if __name__ == "__main__":
    lv, rs = load("longview"), load("resale")
    check_state_at_the_ceiling(lv)
    check_resale_rent_stays_under_the_ceiling(rs)
    check_carried_shift_is_known_on_the_day()
    print("LONGVIEW CHECKS OK")
