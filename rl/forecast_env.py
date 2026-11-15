"""Practice environment for training a small open model to forecast prices directly.

One episode = one prompt (the facts about one price on one day: its recent history and today's list
of offers) -> one answer (low, middle, high for 1, 3 and 7 days ahead) -> one reward.

Reward = how much smaller the answer's error is than drift's error on the same forecast, as a
percent of the price, averaged over the three horizons and clipped to +/-3. An unreadable answer
scores -3. So zero means "as good as drift" and positive means better.

Training uses AWS spot prices only (dates up to 30 Jun 2025). Later AWS dates are kept for checking,
and our own 37 prices are never used in training: they are the test.
"""
from __future__ import annotations

import json
import random

import numpy as np

from gpuval import agents, aws_test, data
from gpuval.core import HORIZONS, cell_loss, day_of, drift_forecast

TRAIN_END = "2025-06-30"
CHECK_FROM, CHECK_TO = "2025-07-04", "2026-02-20"
CLIP = 3.0
UNREADABLE = -3.0

SYSTEM = """You forecast a published GPU rental price. The published price is the MIDDLE (median) of a list of offers for one GPU, in USD per GPU-hour. Most offers never change price; a few change often; offers also join and leave the list.

For 1, 3 and 7 days ahead, say how far the price may move from TODAY's published price, in percent: low (10% chance the move is below it), mid (the most likely move) and high (10% chance the move is above it). Low is usually negative or zero, high is usually positive or zero. Moves are small: often a fraction of a percent over a day and a few percent over a week. A range wider than needed costs points; so does a range that misses.

Reply with JSON only, in exactly this shape (numbers are percent changes from today's price):
{"h1":[low,mid,high],"h3":[low,mid,high],"h7":[low,mid,high]}
Example: {"h1":[-0.3,0.0,0.4],"h3":[-0.8,0.0,1.0],"h7":[-1.5,0.2,2.5]}"""


def aws_world() -> dict:
    return data.load_world(aws_test.WORK / "world.pkl")


def make_item(world: dict, sid: str, i0: int) -> dict | None:
    """Facts, truth and drift for one price on one day; None if any of the three targets is missing."""
    s = world["series"][sid]
    days, vals = s["days"], s["value"]
    o, last = int(days[i0]), float(vals[i0])
    y, d = {}, {}
    for h in HORIZONS:
        k = int(np.searchsorted(days, o + h))
        if k >= len(days) or days[k] != o + h:
            return None
        y[h] = float(vals[k])
        d[h] = drift_forecast(days[:i0 + 1], vals[:i0 + 1], h)
    return {"id": f"{sid}|{o}", "series_id": sid, "origin": o, "last": last, "y": y, "drift": d,
            "prompt": agents.facts(s, i0, sid) + "\n\nGive your forecast as JSON."}


def sample_items(world: dict, n: int, lo: str, hi: str, rng: random.Random, min_history: int = 60) -> list[dict]:
    lo_d, hi_d = day_of(lo), day_of(hi)
    pool = [(sid, i) for sid, s in world["series"].items() for i in range(len(s["days"]))
            if lo_d <= s["days"][i] <= hi_d and s["days"][i] - s["days"][0] >= min_history]
    out = []
    while len(out) < n and pool:
        sid, i = pool.pop(rng.randrange(len(pool)))
        it = make_item(world, sid, i)
        if it:
            out.append(it)
    return out


def parse(text: str, last: float) -> dict | None:
    """Percent moves -> prices. None if the answer cannot be read or is absurd."""
    a = text.find("{")
    b = text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(text[a:b + 1])
        out = {}
        for h in HORIZONS:
            lo, mid, hi = (float(x) for x in obj[f"h{h}"][:3])
            if not np.isfinite([lo, mid, hi]).all() or max(abs(lo), abs(mid), abs(hi)) > 60:
                return None
            lo, hi = min(lo, mid), max(hi, mid)
            out[h] = (last * (1 + lo / 100), last * (1 + mid / 100), last * (1 + hi / 100))
        return out
    except Exception:
        return None


def loss_of(item: dict, q: dict) -> float:
    return float(np.mean([cell_loss(item["y"][h], *q[h]) for h in HORIZONS]))


def reward(item: dict, text: str) -> tuple[float, bool]:
    q = parse(text, item["last"])
    if q is None:
        return UNREADABLE, False
    base = float(np.mean([cell_loss(item["y"][h], *item["drift"][h]) for h in HORIZONS]))
    r = 100.0 * (base - loss_of(item, q)) / item["last"]
    return float(np.clip(r, -CLIP, CLIP)), True


def combine(qs: list[dict]) -> dict:
    """Average several answers (in log terms) into one forecast."""
    out = {}
    for h in HORIZONS:
        a = np.exp(np.mean(np.log(np.array([q[h] for q in qs])), axis=0))
        out[h] = (float(min(a[0], a[1])), float(a[1]), float(max(a[2], a[1])))
    return out
