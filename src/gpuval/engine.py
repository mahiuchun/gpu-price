"""The forecasting engine: replay each offer's own past, then recompute the middle of the list.

The reference forecast ("drift") replays how the *median* moved in the past. This engine looks
underneath: for a horizon of h days it takes every past pair of days (d, d+h), applies to today's
list what each individual offer did between d and d+h (reprice, stay, drop off the list, join it),
and recomputes the median. The spread of those replayed medians is the forecast range. Because it
starts from today's list, it knows which offer sits in the middle now and how far the neighbours
allow the middle to move.

Everything the AI scientist may change is a dial in SPEC_SCHEMA. No free-form code.
"""
from __future__ import annotations

import copy

import numpy as np

from .core import HORIZONS, QS, drift_forecast, wquantile

# name: (type, low, high, default, plain-language meaning)
SPEC_SCHEMA = {
    "lookback_days": (int, 14, 300, 60, "how many days back the replayed (d, d+h) pairs are taken from"),
    "halflife_days": (float, 0, 120, 0.0, "weight recent pairs more; a pair this many days old counts half. 0 = all equal"),
    "replay_exits": (bool, None, None, True, "replay offers dropping off the list"),
    "replay_entries": (bool, None, None, True, "replay offers (re)joining the list, at their last known price"),
    "momentum": (float, 0, 1.5, 0.0, "carry each actively repriced offer's recent trend forward (0 = off, 1 = full trend)"),
    "momentum_window": (int, 3, 30, 7, "days used to measure that recent trend"),
    "active_share": (float, 0.05, 1.0, 0.3, "an offer counts as actively repriced if it changed on at least this share of recent days"),
    "blend_drift": (float, 0, 1, 0.0, "weight given to the drift reference forecast (0 = engine only, 1 = drift only)"),
    "width_scale": (float, 0.3, 3.0, 1.0, "stretch (>1) or shrink (<1) the forecast range around its middle"),
    "center": (str, ("engine", "last"), None, "engine", "middle forecast: the engine's own median, or simply the last value"),
    "min_scenarios": (int, 3, 60, 10, "with fewer usable replay pairs than this, fall back to drift for that forecast"),
    "floor_up": (float, 0, 0.1, 0.0, "minimum distance of the high forecast above the middle, as a fraction of price (allows for a step up even when the replay shows none)"),
    "floor_down": (float, 0, 0.1, 0.0, "minimum distance of the low forecast below the middle, as a fraction of price (allows for a step down even when the replay shows none)"),
    "peer_pull": (float, 0, 1, 0.0, "shift the whole forecast toward the recent move of other series for the same GPU family (other sites or price types): 0 = ignore them, 1 = assume this series fully catches up with them"),
    "peer_window": (int, 1, 14, 3, "days over which that recent move of the other series is measured"),
}
OVERRIDE_KEYS = ("by_class", "by_horizon")
CLASSES = ("neocloud_rate_card", "neocloud_spot", "hyperscaler_spot", "marketplace_ask")


def default_spec() -> dict:
    spec = {k: v[3] for k, v in SPEC_SCHEMA.items()}
    spec["by_class"] = {}
    spec["by_horizon"] = {}
    return spec


def _coerce(key: str, val):
    typ, lo, hi, default, _ = SPEC_SCHEMA[key]
    if typ is bool:
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on")
        return bool(val)
    if typ is str:
        return val if val in lo else default
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    v = min(max(v, lo), hi)
    return int(round(v)) if typ is int else float(v)


def validate_spec(spec: dict) -> dict:
    """Clip every dial into its allowed range and drop anything unknown."""
    out = default_spec()
    for k in SPEC_SCHEMA:
        if k in spec:
            out[k] = _coerce(k, spec[k])
    for ok, allowed in (("by_class", CLASSES), ("by_horizon", tuple(str(h) for h in HORIZONS))):
        for name, sub in (spec.get(ok) or {}).items():
            name = str(name)
            if name in allowed and isinstance(sub, dict):
                clean = {k: _coerce(k, v) for k, v in sub.items() if k in SPEC_SCHEMA}
                if clean:
                    out[ok][name] = clean
    return out


def apply_patch(spec: dict, patch: dict) -> dict:
    """Return spec with a (possibly partial, possibly nested) patch applied, validated."""
    new = copy.deepcopy(spec)
    for k, v in (patch or {}).items():
        if k in OVERRIDE_KEYS and isinstance(v, dict):
            for name, sub in v.items():
                if sub is None:
                    new[k].pop(str(name), None)
                elif isinstance(sub, dict):
                    new[k].setdefault(str(name), {}).update(sub)
        elif k in SPEC_SCHEMA:
            new[k] = v
    return validate_spec(new)


def resolve(spec: dict, cls: str, h: int) -> dict:
    r = {k: spec[k] for k in SPEC_SCHEMA}
    r.update(spec.get("by_class", {}).get(cls, {}))
    r.update(spec.get("by_horizon", {}).get(str(h), {}))
    return r


def gpu_family(gpu_model: str) -> str:
    """A100 40GB and A100 SXM 80GB are one family; RTX cards are kept apart by full name."""
    g = str(gpu_model).strip()
    return g if g.upper().startswith(("RTX", "QUADRO")) else g.split()[0]


def build_engines(world: dict) -> dict:
    """One SeriesEngine per series, each knowing its same-GPU-family peers in this world."""
    engines = {sid: SeriesEngine(s) for sid, s in world["series"].items()}
    fam = {sid: gpu_family(s["meta"]["gpu_model"]) for sid, s in world["series"].items()}
    for sid, eng in engines.items():
        eng.peers = [(world["series"][p]["days"], world["series"][p]["value"])
                     for p in world["series"] if p != sid and fam[p] == fam[sid]]
    return engines


class SeriesEngine:
    """Per-series precomputation plus the replay forecast."""

    def __init__(self, s: dict):
        self.days = s["days"]
        self.value = s["value"]
        self.cls = s["meta"]["instrument_class"]
        with np.errstate(all="ignore"):
            self.logP = np.log(s["P"])
        # last known log price of every offer up to each day (forward fill)
        n, m = self.logP.shape
        filled = self.logP.copy()
        for i in range(1, n):
            row = filled[i]
            miss = np.isnan(row)
            row[miss] = filled[i - 1][miss]
        self.last_known = filled
        self.peers: list[tuple[np.ndarray, np.ndarray]] = []  # (days, value) of same-GPU-family series

    @staticmethod
    def _recent_move(days: np.ndarray, vals: np.ndarray, o: int, window: int, max_stale: int = 3):
        """Log change of a series between about (o - window) and o, using only days <= o."""
        k1 = int(np.searchsorted(days, o, side="right")) - 1
        k0 = int(np.searchsorted(days, o - window, side="right")) - 1
        if k1 < 0 or k0 < 0 or o - days[k1] > max_stale or (o - window) - days[k0] > max_stale:
            return None
        return float(np.log(vals[k1] / vals[k0]))

    def peer_signal(self, i0: int, window: int) -> float:
        """How much further the same-GPU series on other sites moved recently than this one did."""
        if not self.peers:
            return 0.0
        o = int(self.days[i0])
        own = self._recent_move(self.days, self.value, o, window)
        moves = [m for m in (self._recent_move(d, v, o, window) for d, v in self.peers) if m is not None]
        if own is None or not moves:
            return 0.0
        return float(np.clip(np.median(moves) - own, -0.3, 0.3))

    def _trend(self, i0: int, window: int, active_share: float) -> np.ndarray:
        """Per-offer recent daily log trend, zero for offers that are not actively repriced."""
        lo = max(0, i0 - window)
        seg = self.logP[lo:i0 + 1]
        m = seg.shape[1]
        if seg.shape[0] < 2:
            return np.zeros(m)
        d = np.diff(seg, axis=0)
        gaps = np.diff(self.days[lo:i0 + 1]).astype(float)[:, None]
        ok = ~np.isnan(d)
        n_ok = ok.sum(axis=0)
        changed = (np.abs(np.where(ok, d, 0.0)) > 1e-9).sum(axis=0)
        share = np.divide(changed, n_ok, out=np.zeros(m), where=n_ok > 0)
        tot = np.where(ok, d, 0.0).sum(axis=0)
        span = np.where(ok, gaps, 0.0).sum(axis=0)
        g = np.divide(tot, span, out=np.zeros(m), where=span > 0)
        return np.where((share >= active_share) & (n_ok >= 2), g, 0.0)

    def replay_changes(self, i0: int, h: int, r: dict) -> tuple[np.ndarray, np.ndarray]:
        """Replayed h-day log changes of the median, and the start-day ordinal of each replayed pair."""
        days = self.days
        o = int(days[i0])
        lo = int(np.searchsorted(days, o - int(r["lookback_days"]), side="left"))
        a = np.arange(lo, i0 + 1)
        tgt = days[a] + h
        b = np.minimum(np.searchsorted(days, tgt), i0)
        ok = (days[b] == tgt) & (b <= i0)
        a, b = a[ok], b[ok]
        if len(a) == 0:
            return np.empty(0), np.empty(0)
        x = self.logP[i0]
        today = ~np.isnan(x)
        A, B = self.logP[a], self.logP[b]
        pa, pb = ~np.isnan(A), ~np.isnan(B)
        new = np.broadcast_to(x, A.shape).copy()
        both = pa & pb
        move = np.where(both, B - A, 0.0)
        new = new + move                                  # offers listed on both days: replay their change
        if r["momentum"] > 0:
            new = new + r["momentum"] * h * self._trend(i0, int(r["momentum_window"]), float(r["active_share"]))
        if r["replay_exits"]:
            new[pa & ~pb & today] = np.nan               # offer dropped off the list between d and d+h
        if r["replay_entries"]:
            enter = ~pa & pb & ~today                     # offer joined between d and d+h and is absent today
            lk = np.broadcast_to(self.last_known[i0], A.shape)
            new = np.where(enter, lk, new)
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.exp(new), axis=1)
            base = np.nanmedian(np.exp(x))
        good = np.isfinite(med) & (med > 0)
        return np.log(med[good] / base), days[a][good].astype(float)

    def forecast(self, i0: int, spec: dict, horizons=HORIZONS) -> dict:
        """{h: (q10, q50, q90, source)} from data up to and including row i0."""
        last = float(self.value[i0])
        hist_days, hist_vals = self.days[:i0 + 1], self.value[:i0 + 1]
        out = {}
        for h in horizons:
            r = resolve(spec, self.cls, h)
            dq = np.asarray(drift_forecast(hist_days, hist_vals, h))
            d = np.log(dq / last)
            if r["blend_drift"] >= 1.0:
                out[h] = (*dq.tolist(), "drift")
                continue
            ch, start = self.replay_changes(i0, h, r)
            if len(ch) < r["min_scenarios"]:
                out[h] = (*dq.tolist(), "drift_fallback")
                continue
            w = None
            if r["halflife_days"] > 0:
                w = 0.5 ** ((float(self.days[i0]) - start) / float(r["halflife_days"]))
            c = wquantile(ch, w, QS).astype(float)
            if r["center"] == "last":
                c[1] = 0.0
            c[0] = c[1] + r["width_scale"] * (c[0] - c[1])
            c[2] = c[1] + r["width_scale"] * (c[2] - c[1])
            wgt = float(r["blend_drift"])
            c = (1.0 - wgt) * c + wgt * d
            if r["peer_pull"] > 0:
                c = c + float(r["peer_pull"]) * self.peer_signal(i0, int(r["peer_window"]))
            c[2] = max(c[2], c[1] + float(r["floor_up"]))
            c[0] = min(c[0], c[1] - float(r["floor_down"]))
            q10, q50, q90 = (last * np.exp(c)).tolist()
            out[h] = (min(q10, q50), q50, max(q90, q50), "engine")
        return out
