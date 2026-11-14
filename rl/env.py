"""Practice environment for RL-training a small model to propose ideas (PROTOCOL.md, "RL arm").

One episode = one prompt (the scientist's instructions plus a practice world's diagnostics at
default settings) -> one idea as JSON -> one reward.

Reward = percentage fall in total error on the practice world's LATER period (the last 7 days
before its wall), which the prompt does not show; clipped to +/-5. No change scores 0.
Unreadable output scores -5.

Practice worlds are built ONLY from data dated up to 16 Oct 2026: the real data cut at several
earlier dates, and shuffled (placebo) copies with their own seeds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gpuval import data, harness, scientist
from gpuval.core import day_of
from gpuval.engine import SPEC_SCHEMA, apply_patch, build_engines, default_spec

DEV_WALL = "2026-10-16"   # nothing after this date is ever loaded here
PRACTICE_WALLS = ["2026-09-29", "2026-10-02", "2026-10-05", "2026-10-08", "2026-10-11", "2026-10-14", "2026-10-16"]
PLACEBO_SEEDS = list(range(101, 110))
REWARD_CLIP = 5.0
UNREADABLE = -5.0

FORMAT = (
    "\n\nOUTPUT FORMAT\nReply with ONE JSON object and nothing else:\n"
    '{"hypothesis": "<one sentence, at most 25 words>", "changes": [{"scope": "<all | class:NAME | horizon:N>", '
    '"dial": "<dial name>", "value": "<new value>"}]}\n'
    'Use at most 3 changes. Use an empty list for "changes" if you believe no change will help.'
)


def system_prompt() -> str:
    return scientist.SYSTEM + FORMAT


def user_prompt(wall: str, dev_df: pd.DataFrame) -> str:
    return (f"Today is {wall}; you can see data up to today. This is your only idea.\n\n"
            f"CURRENT KEPT SETTINGS\n{scientist.spec_text(default_spec())}\n\n"
            f"DIAGNOSTICS OF THE CURRENT SETTINGS ON THE DEVELOPMENT PERIOD\n{harness.diagnostics(dev_df)}\n\n"
            "LEDGER\n(empty: this is the first idea)\n\nGive your idea as JSON.")


@dataclass
class Practice:
    key: str
    kind: str            # "real" or "placebo"
    wall: str
    world: dict
    engines: dict
    later: pd.DataFrame  # cells that set the reward (not shown in the prompt)
    base_loss: float
    prompt: str


def prepare(world: dict, key: str, kind: str, score_cells: pd.DataFrame | None = None) -> Practice:
    """Diagnostics from targets up to wall-7; reward from targets in the last 7 days (or `score_cells`)."""
    wall = world["wall"]
    cells = harness.base_cells(world)
    engines = build_engines(world)
    dev, later = scientist.split_cells(cells, wall)
    dev_res = harness.run_spec(world, default_spec(), dev, engines)
    if score_cells is not None:
        later = score_cells
    base = harness.run_spec(world, default_spec(), later, engines)
    return Practice(key, kind, wall, world, engines, later, float(base.loss.mean()), user_prompt(wall, dev_res))


def build_practice(only_first: bool = False) -> dict[str, Practice]:
    full = data.build_world(DEV_WALL)
    data.assert_walled(full, DEV_WALL)
    out: dict[str, Practice] = {}
    for w in PRACTICE_WALLS:
        cut = data.cut_world(full, w)
        out[f"real@{w}"] = prepare(cut, f"real@{w}", "real")
        if only_first:
            return out
    for i, seed in enumerate(PLACEBO_SEEDS):
        w = PRACTICE_WALLS[::-1][i % len(PRACTICE_WALLS)]
        cut = data.cut_world(data.make_placebo(full, seed=seed), w)
        out[f"placebo{seed}@{w}"] = prepare(cut, f"placebo{seed}@{w}", "placebo")
    return out


def parse_idea(text: str):
    """(changes, readable). Takes the first JSON object in the text."""
    start = text.find("{")
    if start < 0:
        return [], False
    depth, end = 0, -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return [], False
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return [], False
    if not isinstance(obj, dict) or not isinstance(obj.get("changes"), list):
        return [], False
    changes = []
    for c in obj["changes"][:scientist.MAX_CHANGES]:
        if isinstance(c, dict) and c.get("dial") in SPEC_SCHEMA and str(c.get("scope", "all")) in scientist.SCOPES:
            changes.append({"scope": str(c.get("scope", "all")), "dial": c["dial"], "value": str(c.get("value", ""))})
    return changes, True


def canon(changes: list[dict]) -> str:
    return "; ".join(sorted(f"{c['scope']} {c['dial']}={c['value']}" for c in changes)) or "no change"


def reward_of(p: Practice, changes: list[dict]) -> float:
    spec = apply_patch(default_spec(), scientist.changes_to_patch(changes))
    if spec == default_spec():
        return 0.0
    res = harness.run_spec(p.world, spec, p.later, p.engines)
    r = 100.0 * (p.base_loss - float(res.loss.mean())) / p.base_loss
    return float(np.clip(r, -REWARD_CLIP, REWARD_CLIP))


# ---- multiprocessing helpers (worlds are built in the parent, workers are forked) ----
PRACTICE: dict[str, Practice] = {}


def score_text(key: str, text: str) -> dict:
    changes, ok = parse_idea(text)
    if not ok:
        return {"reward": UNREADABLE, "readable": False, "canon": "(unreadable)", "n_changes": 0}
    return {"reward": reward_of(PRACTICE[key], changes), "readable": True, "canon": canon(changes),
            "n_changes": len(changes)}
