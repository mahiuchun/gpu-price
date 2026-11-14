"""Scoring primitives and the reference forecasts.

These mirror the starter kit (starter/_common.py, starter/baselines.py, starter/scorer.py)
so that numbers computed here match the organisers' scorer. tests/parity.py checks that.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

QS = (0.1, 0.5, 0.9)
HORIZONS = (1, 3, 7)
DRIFT_MIN_SAMPLES = 20  # starter default: fewer historical changes -> flat last value


def day_of(iso: str) -> int:
    return dt.date.fromisoformat(iso).toordinal()


def iso_of(day: int) -> str:
    return dt.date.fromordinal(int(day)).isoformat()


def pinball(y, f, tau):
    d = np.asarray(y, dtype=float) - np.asarray(f, dtype=float)
    return np.maximum(tau * d, (tau - 1.0) * d)


def cell_loss(y, q10, q50, q90):
    """Mean pinball loss over the three quantiles (one number per cell)."""
    return (pinball(y, q10, 0.1) + pinball(y, q50, 0.5) + pinball(y, q90, 0.9)) / 3.0


def h_step_log_changes(days: np.ndarray, vals: np.ndarray, h: int) -> np.ndarray:
    """All h-day log changes: pairs (t, t+h) where both calendar days exist."""
    if len(days) < 2:
        return np.empty(0)
    target = days + h
    idx = np.minimum(np.searchsorted(days, target), len(days) - 1)
    ok = days[idx] == target
    if not ok.any():
        return np.empty(0)
    return np.log(vals[idx[ok]] / vals[ok])


def drift_forecast(days: np.ndarray, vals: np.ndarray, h: int) -> tuple[float, float, float]:
    """The starter's `drift`: last value scaled by the empirical 0.1/0.5/0.9 quantiles of
    the series' own historical h-day log changes (all history up to the origin)."""
    last = float(vals[-1])
    ch = h_step_log_changes(days, vals, h)
    if len(ch) >= DRIFT_MIN_SAMPLES:
        q10, q50, q90 = (last * np.exp(np.quantile(ch, QS))).tolist()
    else:
        q10 = q50 = q90 = last
    return min(q10, q50), q50, max(q90, q50)


def wquantile(x: np.ndarray, w: np.ndarray | None, qs=QS) -> np.ndarray:
    """Quantiles of x with optional weights. Unweighted = numpy's default (linear)."""
    x = np.asarray(x, dtype=float)
    if w is None:
        return np.quantile(x, qs)
    order = np.argsort(x, kind="stable")
    x, w = x[order], np.asarray(w, dtype=float)[order]
    tot = w.sum()
    if tot <= 0:
        return np.quantile(x, qs)
    # weighted version of the linear rule: with equal weights this reproduces np.quantile
    n = len(x)
    if n == 1:
        return np.repeat(x[0], len(qs))
    cw = np.cumsum(w) - w[0]
    cw = cw / cw[-1] if cw[-1] > 0 else np.linspace(0, 1, n)
    return np.interp(qs, cw, x)
