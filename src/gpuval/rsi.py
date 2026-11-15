"""One self-improvement loop over the whole forecaster, with a learning vote.

    uv run python -m gpuval.rsi run                          # the loop: an AI proposes, the gate decides
    uv run python -m gpuval.rsi run --proposer random --seed 1 --name random_1   # control: random changes
    uv run python -m gpuval.rsi run --proposer replay        # repeat the recorded loop, no AI calls
    uv run python -m gpuval.rsi exam                         # every run on the fresh days, with attribution
    uv run python -m gpuval.rsi fresh                        # the follow-up on days the crowd had never answered (`fresh-ask` makes the AI calls)

Our other loops were separate tests, each started by hand. This one runs unattended: generation
after generation it proposes a change, tests it against the current team, keeps it or rolls
back, and archives everything. It has three parts and one rule.

  - A vote that learns (exponential weights). Each member's weight starts equal. When an outcome
    is published, the weight is multiplied by exp(-eta x its error / the error of "no change"),
    pulled a share `decay` back towards equal, and kept above `w_min`. Weights are per kind of
    price and horizon. A forecast only ever uses weights learned from outcomes published by its
    own day.
  - An accept test: a change replaces the current team only if it has less error with
    probability 0.9 or more when prices and days are resampled, and then passes a hidden check.
  - Members of different kinds in one team: the five GPT agents, the engine, and the two open
    models after RL training. Trained weights enter the loop as members the vote can learn to trust.

The rule: one kind of thing per generation. If one generation changed the vote, the ranges and
the members together, a gain could not be traced to its cause. So the kinds take turns (vote,
range, members), each generation is one edit, and the exam scores the final bundle with each
kept edit taken out again.

The starting bundle is the forecast we handed in (five agents, equal weights, the coach's range
stretch); `tests/rsi_checks.py` checks that it reproduces those numbers exactly. Everything is
rebuilt from recorded answers. The only AI calls are the proposer's, one per generation.

The rule (days, gate, kinds, bounds, how the exam is read) is in PROTOCOL.md, deviation 14,
committed before the first run.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import zlib

import numpy as np
import pandas as pd

from . import coach, data, range_memory, report
from .core import HORIZONS, cell_loss, day_of, iso_of

OUT = data.REPO / "runs" / "rsi"
ROLES = list(coach.base_config()["roles"])
EXTRA = {"engine": "the list-replay engine at default settings (no AI)",
         "qwen3_8b_rl": "the open model Qwen3-8B after RL training",
         "qwen3_5_9b_rl": "the open model Qwen3.5-9B after RL training"}
POOL = ROLES + list(EXTRA)
PRIOR = "base_rate_keeper_v0"     # the base-rate keeper before the coach rewrote it; only the audit uses it
RL_RUNS = {"qwen3_8b_rl": "fc1", "qwen3_5_9b_rl": "fc2"}

WALL = "2026-10-20"          # the loop sees no outcome published after this day
EXAM_FROM = "2026-10-21"     # forecasts made from this day on are the exam
KINDS = ("vote", "range", "members")
ACCEPT_P = 0.9               # share of resamples in which a change must have less error
CHECK_NOISE = 0.005          # the coach's hidden check: sd of the tolerance
PLATEAU = 6                  # stop after this many generations in a row with nothing kept
FLOOR_SHARE, LOSS_CAP = 0.002, 3.0          # the vote: "no change" error is at least 0.2% of the price; a scaled error is capped at 3
BY = ("all", "h", "class_h")
BOUNDS = {"eta": (0.0, 10.0), "decay": (0.0, 0.5), "w_min": (0.0, 0.3), "range_scale": (0.3, 3.0), "widen": (0.0, 2.0)}
MIN_MEMBERS, MAX_MEMBERS = 2, 8
BUDGET_USD = 5.0


def start_bundle() -> dict:
    """The forecast we handed in: the improved crowd, equal weights, the coach's range stretch."""
    cfg = json.loads((coach.OUT / "state.json").read_text())["config"]
    return {"members": list(ROLES), "vote": {"eta": 0.0, "decay": 0.05, "w_min": 0.05, "by": "class_h"},
            "range_scale": {str(h): float(cfg["range_scale"][str(h)]) for h in HORIZONS}, "widen": 0.0}


# ------------------------------------------------------------------ frame ----
def _agent_rows() -> pd.DataFrame:
    """Each agent of the improved crowd, on every day of our prices it has answered."""
    cfg = json.loads((coach.OUT / "state.json").read_text())["config"]
    assert list(cfg["roles"]) == ROLES
    base = coach.base_config()["roles"]
    changed = {n for n, r in cfg["roles"].items() if base.get(n) != r}
    ans = json.loads((coach.OUT / "answers.json").read_text())
    got: dict = {}
    for f in ("extra_answers", "test_answers"):     # answered before the coach existed; rewritten agents were asked again
        for j in json.loads((data.REPO / "runs" / "agents" / f"{f}.json").read_text()):
            got[(j["series_id"], day_of(j["origin"]))] = {
                n: (ans[coach.Answers.key(r, f"rec|ours|{j['series_id']}|{j['origin']}")] if n in changed else j["r1"][n])["q"]
                for n, r in cfg["roles"].items()} | {PRIOR: j["r1"]["base_rate_keeper"]["q"]}
    for iso in range_memory.COACH_OWN_DAYS + coach.DAYS["exam"]["ours"]:
        for sid in data.headline_ids():
            keys = {n: coach.Answers.key(r, f"ours|{sid}|{iso}") for n, r in cfg["roles"].items()}
            keys[PRIOR] = coach.Answers.key(base["base_rate_keeper"], f"ours|{sid}|{iso}")
            if all(k in ans for k in keys.values()):
                got[(sid, day_of(iso))] = {n: ans[k]["q"] for n, k in keys.items()}
    rows = []
    for (sid, o), by_role in got.items():
        for h in HORIZONS:
            row = {"series_id": sid, "origin": o, "h": h}
            for n, q in by_role.items():
                row[f"{n}_lo"], row[f"{n}_mid"], row[f"{n}_hi"] = (float(v) for v in q[str(h)])
            rows.append(row)
    return pd.DataFrame(rows)


def build_frame() -> pd.DataFrame:
    """Every scorable forecast of our prices on the crowd's days, with every member's answer."""
    full = data.build_world(data.DATA_END)
    cells = range_memory.truth_cells(full)
    f = cells.merge(_agent_rows(), on=["series_id", "origin", "h"], how="inner")
    f["engine_lo"], f["engine_mid"], f["engine_hi"] = f.e_lo, f.e_mid, f.e_hi
    for name, run in RL_RUNS.items():
        q = pd.read_parquet(data.REPO / "rl" / "runs" / run / "eval_cells.parquet")
        q = q[q.world == "ours"].assign(origin=lambda d: d.origin.map(day_of))[["series_id", "origin", "h", "trained_lo", "trained_mid", "trained_hi"]]
        q.columns = ["series_id", "origin", "h", f"{name}_lo", f"{name}_mid", f"{name}_hi"]
        f = f.merge(q, on=["series_id", "origin", "h"], how="inner")
    cols = [f"{m}_{c}" for m in POOL + [PRIOR] for c in ("lo", "mid", "hi")]
    assert f[cols].notna().all().all() and (f[cols] > 0).all().all()
    days = sorted(f.origin.unique())
    dev = [d for d in days if d < day_of(EXAM_FROM)]
    turn = {d: ("practice", "check")[i % 2] for i, d in enumerate(dev)}
    f["split"] = ""
    known = f.target <= day_of(WALL)
    f.loc[known, "split"] = f.loc[known, "origin"].map(turn)
    f.loc[f.origin >= day_of(EXAM_FROM), "split"] = "exam"
    f["moved"] = np.abs(f.y - f["last"]) > 1e-9
    return f.sort_values(["origin", "series_id", "h"]).reset_index(drop=True)


# ------------------------------------------------------------------- vote ----
def _context(f: pd.DataFrame, by: str) -> np.ndarray:
    if by == "all":
        return np.zeros(len(f), dtype=object)
    if by == "h":
        return f.h.astype(str).to_numpy(object)
    return (f.cls.astype(str) + "|" + f.h.astype(str)).to_numpy(object)


def walk(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """The team's forecast for every row, going through the forecast days in order. A row's weights
    come only from forecasts whose outcome was published on or before that row's day."""
    members, vote = bundle["members"], bundle["vote"]
    f = frame.sort_values(["origin", "series_id", "h"]).reset_index(drop=True).copy()
    n, k = len(f), len(members)
    Q = np.log(np.stack([f[[f"{m}_lo", f"{m}_mid", f"{m}_hi"]].to_numpy(float) for m in members], axis=1))    # row, member, quantile
    W = np.full((n, k), 1.0 / k)
    eta, decay, floor = float(vote["eta"]), float(vote["decay"]), min(float(vote["w_min"]), 1.0 / k)
    if eta > 0:
        y = f.y.to_numpy(float)
        base = np.maximum(np.maximum(f.loss_last.to_numpy(float), FLOOR_SHARE * f["last"].to_numpy(float)), 1e-12)
        scaled = np.minimum(LOSS_CAP, np.stack([cell_loss(y, *np.exp(Q[:, j, :]).T) for j in range(k)], axis=1) / base[:, None])
        ctx, origin, target = _context(f, vote["by"]), f.origin.to_numpy(), f.target.to_numpy()
        order = np.lexsort((f.h.to_numpy(), f.series_id.to_numpy(), origin, target))     # the order outcomes are published in
        state: dict = {}
        ptr = 0
        for d in np.unique(origin):
            while ptr < n and target[order[ptr]] <= d:
                i = order[ptr]
                w = state.get(ctx[i], np.full(k, 1.0 / k)) * np.exp(-eta * scaled[i])
                w = (1 - decay) * w / w.sum() + decay / k
                state[ctx[i]] = floor + (1 - k * floor) * w
                ptr += 1
            for i in np.where(origin == d)[0]:
                if ctx[i] in state:
                    W[i] = state[ctx[i]]
    lq = np.einsum("nm,nmq->nq", W, Q)
    mid = np.exp(lq[:, 1])
    lo, hi = np.minimum(np.exp(lq[:, 0]), mid), np.maximum(np.exp(lq[:, 2]), mid)
    s = f.h.map({h: float(bundle["range_scale"][str(h)]) for h in HORIZONS}).to_numpy(float)
    lo, hi = mid * (lo / mid) ** s, mid * (hi / mid) ** s
    if bundle["widen"] > 0:      # wider where the members' middles disagree
        spread = (W * np.abs(Q[:, :, 1] - lq[:, 1][:, None])).sum(axis=1)
        lo, hi = lo * np.exp(-bundle["widen"] * spread), hi * np.exp(bundle["widen"] * spread)
    f["lo"], f["mid"], f["hi"] = lo, mid, hi
    f["loss"] = cell_loss(f.y, lo, mid, hi)
    f["inside"] = (f.y >= lo - 1e-12) & (f.y <= hi + 1e-12)
    f["below"], f["above"] = f.y < lo - 1e-12, f.y > hi + 1e-12
    for j, m in enumerate(members):
        f[f"w_{m}"] = W[:, j]
    return f


# ------------------------------------------------------------------- gate ----
def paired(g: pd.DataFrame, new: str, old: str, seed: int = 0) -> dict:
    """How much less error `new` has than `old`, and how often it has less when prices and target
    days are resampled (2,000 resamples, as everywhere in this repo)."""
    M = report._mats(g, [new, old])
    A, B = M[new], M[old]
    rng = np.random.default_rng(seed)
    ns, nd = A.shape
    ws = rng.multinomial(ns, np.ones(ns) / ns, size=report.N_BOOT).astype(float)
    wd = rng.multinomial(nd, np.ones(nd) / nd, size=report.N_BOOT).astype(float)
    num, den = np.einsum("bs,sd,bd->b", ws, A, wd), np.einsum("bs,sd,bd->b", ws, B, wd)
    lo, hi = np.nanquantile(1 - num / np.where(den > 0, den, np.nan), [0.05, 0.95])
    return {"gain": float(1 - A.sum() / B.sum()), "p_better": float(np.mean(num < den)), "lo90": float(lo), "hi90": float(hi), "n": int(len(g))}


def _join(new: pd.DataFrame, old: pd.DataFrame, split: str) -> pd.DataFrame:
    k = ["series_id", "origin", "h"]
    return new[new.split == split][k + ["target", "loss"]].merge(old[old.split == split][k + ["loss"]], on=k, suffixes=("_new", "_old"))


def gate(new: pd.DataFrame, old: pd.DataFrame, rng: np.random.Generator) -> dict:
    p = paired(_join(new, old, "practice"), "loss_new", "loss_old")
    out = {"practice_gain": p["gain"], "p_better": p["p_better"], "check": "not run", "kept": False}
    if p["gain"] > 0 and p["p_better"] >= ACCEPT_P:
        c = _join(new, old, "check")
        ok = bool(c.loss_new.sum() < c.loss_old.sum() * (1 + rng.normal(0, CHECK_NOISE)))
        out["_check_gain"] = float(1 - c.loss_new.sum() / c.loss_old.sum())      # for the audit; never shown to the proposer
        out["check"], out["kept"] = ("PASS" if ok else "FAIL"), ok
    return out


# ------------------------------------------------------------------ edits ----
def _clip(v, key: str) -> float:
    lo, hi = BOUNDS[key]
    v = float(v)
    return float(min(max(v if np.isfinite(v) else lo, lo), hi))


def apply_edit(bundle: dict, edit: dict) -> dict | None:
    """One bounded edit of one kind, on a copy. None if it changes nothing or is not allowed."""
    b = copy.deepcopy(bundle)
    kind = edit.get("kind")
    if kind == "vote":
        b["vote"] = {"eta": _clip(edit["eta"], "eta"), "decay": _clip(edit["decay"], "decay"), "w_min": _clip(edit["w_min"], "w_min"),
                     "by": edit["by"] if edit.get("by") in BY else b["vote"]["by"]}
    elif kind == "range":
        b["range_scale"] = {str(h): _clip(edit["range_scale"][str(h)], "range_scale") for h in HORIZONS}
        b["widen"] = _clip(edit["widen"], "widen")
    elif kind == "members":
        m, act = edit.get("member"), edit.get("action")
        if act == "add" and m in POOL and m not in b["members"] and len(b["members"]) < MAX_MEMBERS:
            b["members"].append(m)
        elif act == "drop" and m in b["members"] and len(b["members"]) > MIN_MEMBERS:
            b["members"].remove(m)
    return None if b == bundle else b


def rebuild(edits: list[dict]) -> dict:
    b = start_bundle()
    for e in edits:
        b = apply_edit(b, e) or b
    return b


def edit_text(e: dict) -> str:
    if e["kind"] == "vote":
        return f"vote: eta {e['eta']:g}, decay {e['decay']:g}, w_min {e['w_min']:g}, weights per {e['by']}"
    if e["kind"] == "range":
        return "range: stretch " + " / ".join(f"{e['range_scale'][str(h)]:g}" for h in HORIZONS) + f", widen {e['widen']:g}"
    return f"members: {e.get('action')} {e.get('member')}"


# -------------------------------------------------------------- proposers ----
SYSTEM = f"""You improve a system that forecasts published GPU rental prices. The system is a small team of forecasters ("members"). For each price each member gives low / mid / high for 1, 3 and 7 days ahead. The team's forecast is a weighted average of the members' answers, and then its range is stretched.

All you can change is how the team is put together:
- members: which forecasters vote. Available: {', '.join(ROLES)} (five AI agents with different jobs, each reading the price's list of offers); {'; '.join(f'{k} ({v})' for k, v in EXTRA.items())}. Between {MIN_MEMBERS} and {MAX_MEMBERS} members.
- vote: how the weights are learned. Weights start equal. Each time an outcome is published, each member's weight is multiplied by exp(-eta x its error / the error of forecasting no change), then pulled a share "decay" back towards equal weights, and no weight may fall below "w_min". eta = 0 keeps the weights equal for ever. "by" says how many sets of weights there are: "all" (one), "h" (one per horizon) or "class_h" (one per kind of price and horizon). Bounds: eta {BOUNDS['eta'][0]:g} to {BOUNDS['eta'][1]:g}, decay {BOUNDS['decay'][0]:g} to {BOUNDS['decay'][1]:g}, w_min {BOUNDS['w_min'][0]:g} to {BOUNDS['w_min'][1]:g}.
- range: the stretch at 1, 3 and 7 days (1.0 = as the weighted average gave it, 1.3 = 30% wider, bounds {BOUNDS['range_scale'][0]:g} to {BOUNDS['range_scale'][1]:g}), and "widen", which adds widen x (how far the members' middles disagree) to each side of the range (bounds {BOUNDS['widen'][0]:g} to {BOUNDS['widen'][1]:g}).

Scores are error against the simple reference forecast called drift: +0.10 means 10% less error than drift. Error is pinball loss: a range that is too wide costs points when nothing moves, and a range that misses costs more when the price jumps. The truth should fall inside the range about 80% of the time, below it about 10% and above it about 10%.

Each generation you may change ONE kind of thing, and the message tells you which. Your change is tested against the current team on the same forecasts. It is KEPT only if (a) on the practice days it has less error with at least {ACCEPT_P:.0%} confidence when prices and days are resampled, and (b) it passes a hidden check on other days, for which you are told only PASS or FAIL. A kept change becomes the new baseline. Changes that fit a few practice cases tend to fail, so prefer a change with a general reason. If you think nothing of this kind will help, repeat the current values: no change is a valid answer.

Fill in every field; only the fields of the kind you were asked for are used."""

SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "description": "What the scores show, in two or three plain sentences."},
        "eta": {"type": "number"}, "decay": {"type": "number"}, "w_min": {"type": "number"}, "by": {"type": "string", "enum": list(BY)},
        "range_scale_1d": {"type": "number"}, "range_scale_3d": {"type": "number"}, "range_scale_7d": {"type": "number"}, "widen": {"type": "number"},
        "member_action": {"type": "string", "enum": ["add", "drop", "none"]}, "member": {"type": "string", "enum": POOL},
        "expectation": {"type": "string", "description": "What you expect the change to do."},
    },
    "required": ["diagnosis", "eta", "decay", "w_min", "by", "range_scale_1d", "range_scale_3d", "range_scale_7d", "widen", "member_action", "member", "expectation"],
    "additionalProperties": False,
}


def _to_edit(kind: str, a: dict) -> dict:
    if kind == "vote":
        return {"kind": kind, "eta": a["eta"], "decay": a["decay"], "w_min": a["w_min"], "by": a["by"]}
    if kind == "range":
        return {"kind": kind, "range_scale": {"1": a["range_scale_1d"], "3": a["range_scale_3d"], "7": a["range_scale_7d"]}, "widen": a["widen"]}
    return {"kind": kind, "action": a["member_action"], "member": a["member"]}


class GPTProposer:
    kind = "gpt"

    def __init__(self):
        from dotenv import load_dotenv
        from openai import OpenAI

        from . import agents
        from .llm import Meter

        load_dotenv(data.REPO / ".env")
        self.client, self.model, self.meter = OpenAI(), agents.MODEL, Meter(BUDGET_USD)

    def propose(self, kind: str, bundle: dict, view: str, rng: random.Random) -> tuple[dict, dict]:
        from .llm import ask_json

        a = ask_json(self.client, self.model, "medium", SYSTEM, view + f"\n\nThis generation you may change: {kind.upper()}. Propose one change.",
                     SCHEMA, "change", self.meter, flex=False)
        return _to_edit(kind, a), {"diagnosis": a["diagnosis"], "expectation": a["expectation"]}

    def usage(self) -> dict:
        return self.meter.summary()


class RandomProposer:
    """Control: a random bounded edit of the asked kind, with no reading of the scores."""
    kind = "random"

    def propose(self, kind: str, bundle: dict, view: str, rng: random.Random) -> tuple[dict, dict]:
        if kind == "vote":
            e = {"kind": kind, "eta": rng.choice([0.25, 0.5, 1, 2, 4, 8]), "decay": rng.choice([0.0, 0.02, 0.05, 0.1, 0.2]),
                 "w_min": rng.choice([0.0, 0.02, 0.05, 0.1]), "by": rng.choice(BY)}
        elif kind == "range":
            e = {"kind": kind, "range_scale": {str(h): round(float(bundle["range_scale"][str(h)]) * rng.choice([0.8, 0.9, 1.0, 1.1, 1.25]), 3) for h in HORIZONS},
                 "widen": rng.choice([0.0, 0.25, 0.5, 1.0])}
        else:
            can_add = [m for m in POOL if m not in bundle["members"]]
            act = rng.choice(["add", "drop"]) if can_add else "drop"
            e = {"kind": kind, "action": act, "member": rng.choice(can_add if act == "add" else bundle["members"])}
        return e, {"diagnosis": "(random control: no diagnosis)", "expectation": ""}

    def usage(self) -> dict:
        return {}


class ReplayProposer:
    """Replays the edits recorded in an earlier run (no AI calls)."""
    kind = "replay"

    def __init__(self, ledger: list[dict]):
        self.queue = list(ledger)

    def propose(self, kind: str, bundle: dict, view: str, rng: random.Random) -> tuple[dict, dict]:
        e = self.queue.pop(0)
        assert e["kind"] == kind
        return e["edit"], {"diagnosis": e["diagnosis"], "expectation": e["expectation"]}

    def usage(self) -> dict:
        return {}


# ------------------------------------------------------------ what it sees ----
def _vs_drift(g: pd.DataFrame, col: str = "loss") -> float:
    return float(1 - g[col].sum() / g.loss_drift.sum())


def view(bundle: dict, w: pd.DataFrame, frame: pd.DataFrame, ledger: list[dict], left: int) -> str:
    """What the proposer is shown: the current team and its scores on the practice days only."""
    p = w[w.split == "practice"]
    v = bundle["vote"]
    lines = ["CURRENT TEAM", "Members: " + ", ".join(bundle["members"]),
             f"Vote: eta {v['eta']:g}, decay {v['decay']:g}, w_min {v['w_min']:g}, weights per {v['by']}",
             "Range stretch: " + ", ".join(f"{h} day {bundle['range_scale'][str(h)]:g}" for h in HORIZONS) + f"; widen {bundle['widen']:g}",
             f"\nPRACTICE SCORES ({len(p)} forecasts on {p.origin.nunique()} days): team {_vs_drift(p):+.3f} against drift; truth below/inside/above the range: "
             f"{p.below.mean():.2f}/{p.inside.mean():.2f}/{p.above.mean():.2f}",
             "By kind of price and horizon (score against drift, forecasts, below/inside/above):"]
    for (cls, h), g in p.groupby(["cls", "h"]):
        lines.append(f"  {cls} {h}d: {_vs_drift(g):+.3f}, {len(g)}, {g.below.mean():.2f}/{g.inside.mean():.2f}/{g.above.mean():.2f}")
    fp = frame[frame.split == "practice"]
    lines.append("Each available forecaster alone on the practice days, against drift (all / 1d / 3d / 7d), ranges as it gave them:")
    for m in POOL:
        loss = cell_loss(fp.y, fp[f"{m}_lo"], fp[f"{m}_mid"], fp[f"{m}_hi"])
        by_h = " / ".join(f"{1 - loss[fp.h == h].sum() / fp.loss_drift[fp.h == h].sum():+.3f}" for h in HORIZONS)
        lines.append(f"  {m}{'' if m in bundle['members'] else ' (not a member)'}: {1 - loss.sum() / fp.loss_drift.sum():+.3f} / {by_h}")
    last = p[p.origin == p.origin.max()]
    lines.append(f"Weights on the last practice day ({iso_of(last.origin.max())}), averaged over prices, by horizon:")
    for h, g in last.groupby("h"):
        lines.append(f"  {h}d: " + ", ".join(f"{m} {g[f'w_{m}'].mean():.2f}" for m in bundle["members"]))
    hist = "\n".join(f"#{e['n']} {edit_text(e['edit'])}: practice {e['practice_gain']:+.4f} against the team before it, confidence {e['p_better']:.2f}; "
                     f"hidden check {e['check']}; {'KEPT' if e['kept'] else 'REJECTED'}" for e in ledger) or "(none yet)"
    lines.append("\nCHANGES TRIED SO FAR\n" + hist + f"\n\nGenerations left including this one: {left}.")
    return "\n".join(lines)


# -------------------------------------------------------------------- run ----
def run(proposer, generations: int, seed: int, name: str, frame: pd.DataFrame | None = None, save: bool = True, root=None, log=print) -> dict:
    frame = build_frame() if frame is None else frame
    seen = frame[frame.target <= day_of(WALL)]            # the loop is handed no outcome published after the wall
    rng_np = np.random.default_rng(zlib.crc32(f"rsi|{name}|{seed}".encode()))
    rng_py = random.Random(f"rsi|{name}|{seed}")
    bundle, edits, ledger, quiet, stop = start_bundle(), [], [], 0, "generations"
    best = walk(seen, bundle)
    start = {s: _vs_drift(best[best.split == s]) for s in ("practice", "check")}
    log(f"[{name}] start: practice {start['practice']:+.4f}, check {start['check']:+.4f} against drift "
        f"({(best.split == 'practice').sum()} and {(best.split == 'check').sum()} forecasts)")
    for g in range(1, generations + 1):
        kind = KINDS[(g - 1) % len(KINDS)]
        edit, why = proposer.propose(kind, bundle, view(bundle, best, seen, ledger, generations - g + 1), rng_py)
        cand = apply_edit(bundle, edit)
        entry = {"n": g, "kind": kind, "edit": edit, **why, "practice_before": _vs_drift(best[best.split == "practice"]),
                 "practice_gain": 0.0, "p_better": 0.0, "check": "not run (no change)", "kept": False}
        if cand is not None:
            cw = walk(seen, cand)
            entry.update(gate(cw, best, rng_np))
            if entry["kept"]:
                bundle, best = cand, cw
                edits.append(edit)
        quiet = 0 if entry["kept"] else quiet + 1
        entry["bundle_after"] = copy.deepcopy(bundle)
        ledger.append(entry)
        log(f"[{name}] gen {g} {edit_text(edit)} | practice {entry['practice_gain']:+.4f}, confidence {entry['p_better']:.2f} | check {entry['check']} | "
            f"{'KEPT' if entry['kept'] else 'rejected'}")
        if quiet >= PLATEAU:
            stop = "plateau"
            break
    state = {"name": name, "proposer": proposer.kind, "seed": seed, "generations": generations, "stop": stop, "start": start,
             "end": {s: _vs_drift(best[best.split == s]) for s in ("practice", "check")}, "bundle": bundle, "kept_edits": edits,
             "ledger": ledger, "usage": proposer.usage()}
    if save:
        root = OUT if root is None else root
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "state.json").write_text(json.dumps(state, indent=1))
    log(f"[{name}] stopped ({stop}): kept {len(edits)} of {len(ledger)}; practice {start['practice']:+.4f} -> {state['end']['practice']:+.4f}, "
        f"check {start['check']:+.4f} -> {state['end']['check']:+.4f} | usage {state['usage']}")
    return state


# ------------------------------------------------------------------- exam ----
def _exam_block(g: pd.DataFrame, cols: dict) -> dict:
    ci = report.skill_ci
    return {"n": int(len(g)), "n_days": int(g.origin.nunique()),
            "vs_drift": {k: ci(g, c, "loss_drift") for k, c in cols.items()},
            "vs_start": {k: ci(g, c, cols["start"]) for k, c in cols.items() if k != "start"}}


def exam(frame: pd.DataFrame | None = None, root=None, save: bool = True, log=print) -> dict:
    """Every saved run on the fresh days. The whole sequence of days is walked, so the weights on an
    exam day have learned from every outcome published by then; the settings are frozen."""
    frame = build_frame() if frame is None else frame
    fm = report.fmt_ci
    ex = lambda w: w[w.split == "exam"].reset_index(drop=True)
    s = ex(walk(frame, start_bundle()))
    out = {"rule": {"wall": WALL, "exam_from": EXAM_FROM, "accept_p": ACCEPT_P, "check_noise": CHECK_NOISE, "kinds": list(KINDS), "plateau": PLATEAU,
                    "bounds": BOUNDS, "pool": POOL},
           "days": {k: sorted({iso_of(d) for d in frame[frame.split == k].origin}) for k in ("practice", "check", "exam")},
           "n": {k: int((frame.split == k).sum()) for k in ("practice", "check", "exam")},
           "start_bundle": start_bundle(), "runs": {}}
    alone = {m: report.skill_ci(s.assign(x=cell_loss(s.y, s[f"{m}_lo"], s[f"{m}_mid"], s[f"{m}_hi"])), "x", "loss_drift") for m in POOL}
    out["members_alone_exam_vs_drift"] = alone
    out["start_on_exam"] = {"vs_drift": report.skill_ci(s, "loss", "loss_drift"), "vs_engine": report.skill_ci(s, "loss", "loss_engine"),
                            "inside": float(s.inside.mean()), "inside_when_moved": float(s[s.moved].inside.mean())}
    keep = ["series_id", "origin", "h", "target", "y", "last", "moved", "loss_drift", "loss_engine"]
    cells = s[keep].assign(lo_start=s.lo, mid_start=s.mid, hi_start=s.hi, loss_start=s.loss)
    root = OUT if root is None else root
    for path in sorted(root.glob("*/state.json")):
        st = json.loads(path.read_text())
        name, edits = st["name"], st["kept_edits"]
        arms = {"start": s, "final": ex(walk(frame, rebuild(edits)))}
        for i, e in enumerate(edits):          # attribution: each kept edit alone, and the final bundle without it
            arms[f"only_{i + 1}"] = ex(walk(frame, rebuild([e])))
            arms[f"without_{i + 1}"] = ex(walk(frame, rebuild(edits[:i] + edits[i + 1:])))
        g = s[keep].copy()
        for k, w in arms.items():
            g[f"loss_{k}"] = w.loss.to_numpy()
        cols = {k: f"loss_{k}" for k in arms}
        b = _exam_block(g, cols)
        b["by_h"] = {int(h): {"final_vs_start": report.skill_ci(gh, "loss_final", "loss_start")} for h, gh in g.groupby("h")}
        b["inside"] = {k: float(arms[k].inside.mean()) for k in ("start", "final")}
        b["inside_when_moved"] = {k: float(arms[k][arms[k].moved].inside.mean()) for k in ("start", "final")}
        b["final_vs_engine"] = report.skill_ci(g, "loss_final", "loss_engine")
        fin = arms["final"]
        b["weights_last_exam_day"] = {int(h): {m: float(gh[f"w_{m}"].mean()) for m in rebuild(edits)["members"]}
                                      for h, gh in fin[fin.origin == fin.origin.max()].groupby("h")}
        # the incumbent after each generation, scored on the exam days only now, after the loop has ended
        curve, kept_so_far, last_exam = [], [], _vs_drift(s)
        for e in st["ledger"]:
            if e["kept"]:
                kept_so_far.append(e["edit"])
                last_exam = _vs_drift(ex(walk(frame, rebuild(kept_so_far))))
            curve.append({"n": e["n"], "kind": e["kind"], "kept": e["kept"], "exam_vs_drift": last_exam})
        b.update({"proposer": st["proposer"], "seed": st["seed"], "stop": st["stop"], "generations_run": len(st["ledger"]),
                  "kept": [edit_text(e) for e in edits], "final_bundle": st["bundle"], "practice": [st["start"]["practice"], st["end"]["practice"]],
                  "check": [st["start"]["check"], st["end"]["check"]], "curve": curve, "usage": st["usage"],
                  "established": bool(b["vs_start"]["final"].get("excludes_zero") and b["vs_start"]["final"]["skill"] > 0) if edits else False})
        out["runs"][name] = b
        if st["proposer"] != "random":
            cells[f"lo_{name}"], cells[f"mid_{name}"], cells[f"hi_{name}"], cells[f"loss_{name}"] = fin.lo.to_numpy(), fin.mid.to_numpy(), fin.hi.to_numpy(), fin.loss.to_numpy()
        log(f"== {name} ({st['proposer']}, {len(st['ledger'])} generations, kept {len(edits)}): exam {b['n']} forecasts on {b['n_days']} days")
        log(f"   start {fm(b['vs_drift']['start'])} | final {fm(b['vs_drift']['final'])} against drift" + (f" | final against start {fm(b['vs_start']['final'])}" if edits else ""))
        for i, e in enumerate(edits):
            log(f"   kept {i + 1}: {edit_text(e)} | alone against start {fm(b['vs_start'][f'only_{i + 1}'])} | final without it against start {fm(b['vs_start'][f'without_{i + 1}'])}")
        if edits:
            log(f"   inside {b['inside']['start']:.3f} -> {b['inside']['final']:.3f}; when the price moved {b['inside_when_moved']['start']:.3f} -> {b['inside_when_moved']['final']:.3f}")
    rnd = [b["vs_start"]["final"]["skill"] if b["kept"] else 0.0 for b in out["runs"].values() if b["proposer"] == "random"]
    if rnd:
        out["random_control"] = {"runs": len(rnd), "kept_any": int(sum(bool(b["kept"]) for b in out["runs"].values() if b["proposer"] == "random")),
                                 "final_vs_start": {"min": float(min(rnd)), "median": float(np.median(rnd)), "max": float(max(rnd))}}
        log(f"random control, {len(rnd)} runs: final against start on the exam days {min(rnd):+.4f} to {max(rnd):+.4f} (median {np.median(rnd):+.4f})")
    out["after_the_result"] = audit(frame, [json.loads(p.read_text()) for p in sorted(root.glob("*/state.json"))], log)
    if save:
        (data.REPO / "findings" / "data" / "rsi_loop.json").write_text(json.dumps(out, indent=1, default=str))
        cells.to_parquet(OUT / "exam_cells.parquet", index=False)
    return out


# ------------------------------------------------------------------ audit ----
def first_crowd_bundle() -> dict:
    """The crowd as first written: the base-rate keeper before its rewrite, no range stretch."""
    b = start_bundle()
    return {**b, "members": [PRIOR if m == "base_rate_keeper" else m for m in ROLES], "range_scale": {str(h): 1.0 for h in HORIZONS}}


def _judge(frame: pd.DataFrame, new: dict, old: dict) -> dict:
    """One change against the bundle before it: the gate's numbers on the loop's days, and the exam."""
    wn, wo = walk(frame, new), walk(frame, old)
    p, c, x = (_join(wn, wo, k) for k in ("practice", "check", "exam"))
    own = {day_of(d) for d in range_memory.COACH_OWN_DAYS}
    return {"practice": paired(p, "loss_new", "loss_old"), "practice_without_coach_days": paired(p[~p.origin.isin(own)], "loss_new", "loss_old"),
            "check_gain": float(1 - c.loss_new.sum() / c.loss_old.sum()), "exam": report.skill_ci(x, "loss_new", "loss_old")}


def audit(frame: pd.DataFrame, states: list[dict], log=print) -> dict:
    """Readings made after the result; none of this is part of the test in PROTOCOL.md.
    (a) Every edit any run proposed, scored on the exam days: what the gate turned away.
    (b) Would this gate have kept the coach's two changes, which held up on fresh days?"""
    fm, start = report.fmt_ci, start_bundle()
    seen: dict = {}
    for st in states:
        b = start_bundle()
        for e in st["ledger"]:
            if b == start:          # only edits proposed against the starting bundle are comparable
                seen.setdefault(json.dumps(e["edit"], sort_keys=True), {"edit": e["edit"], "proposers": set()})["proposers"].add(st["proposer"])
            b = e["bundle_after"]
    turned_away = []
    for v in seen.values():
        cand = apply_edit(start, v["edit"])
        if cand is not None:
            turned_away.append({"edit": edit_text(v["edit"]), "kind": v["edit"]["kind"], "proposers": sorted(v["proposers"]), **_judge(frame, cand, start)})
    turned_away.sort(key=lambda r: -r["practice"]["p_better"])
    log("\nAFTER THE RESULT (a): every proposed edit, practice confidence and what it did on the exam days")
    for r in turned_away:
        log(f"   {r['practice']['p_better']:.2f} | practice {r['practice']['gain']:+.4f} | check {r['check_gain']:+.4f} | exam {fm(r['exam'])} | {r['edit']} ({'/'.join(r['proposers'])})")
    b0 = first_crowd_bundle()
    b1 = {**b0, "range_scale": start["range_scale"]}
    control = {"wider ranges at 3 and 7 days": _judge(frame, b1, b0), "the rewritten agent, on top of the wider ranges": _judge(frame, start, b1),
               "both changes": _judge(frame, start, b0)}
    log("AFTER THE RESULT (b): the coach's two kept changes put to this gate")
    for k, r in control.items():
        p, q = r["practice"], r["practice_without_coach_days"]
        log(f"   {k}: practice {p['gain']:+.4f} [{p['lo90']:+.3f}, {p['hi90']:+.3f}], confidence {p['p_better']:.2f} "
            f"(without the coach's own days {q['gain']:+.4f}, {q['p_better']:.2f}) | check {r['check_gain']:+.4f} | exam {fm(r['exam'])}")
    # one more setting of the vote, which no loop happened to propose
    v = {"eta": 2.0, "decay": 0.05, "w_min": 0.05, "by": "class_h"}
    extra = {"vote only": _judge(frame, {**start, "vote": v}, start), "vote and widen 0.5": _judge(frame, {**start, "vote": v, "widen": 0.5}, start)}
    log("AFTER THE RESULT (c): one more setting of the vote, which no loop proposed (eta 2, decay 0.05, w_min 0.05, per kind of price and horizon)")
    for k, r in extra.items():
        log(f"   {k}: practice {r['practice']['gain']:+.4f}, confidence {r['practice']['p_better']:.2f} | check {r['check_gain']:+.4f} | exam {fm(r['exam'])}")

    def count(rows: list[dict]) -> dict:
        return {"n": len(rows), "better_on_exam": int(sum(r["exam"]["skill"] > 0 for r in rows)),
                "clearly_better_on_exam": int(sum(r["exam"]["lo90"] > 0 for r in rows)), "clearly_worse_on_exam": int(sum(r["exam"]["hi90"] < 0 for r in rows))}

    out = {"note": "readings made after the result; not part of the pre-stated test", "turned_away": turned_away, "coach_changes_through_this_gate": control,
           "one_more_vote_setting": extra, "all_edits": count(turned_away), "by_kind": {k: count([r for r in turned_away if r["kind"] == k]) for k in KINDS},
           "edits_with_confidence_0.7_or_more": count([r for r in turned_away if r["practice"]["p_better"] >= 0.7])}
    log(f"   all edits: {out['all_edits']} | by kind: {out['by_kind']} | at confidence 0.7 or more: {out['edits_with_confidence_0.7_or_more']}")
    return out


# ------------------------------------------------------------------ fresh ----
FRESH_EDIT = {"kind": "members", "action": "add", "member": "qwen3_8b_rl"}     # the one edit that was ahead three times
FRESH_BUDGET_USD = 18.0


def fresh_days() -> list[str]:
    """Days of our prices on which the open models were scored and no agent was ever asked."""
    q = pd.read_parquet(data.REPO / "rl" / "runs" / "fc1" / "eval_cells.parquet")
    return sorted(set(q[q.world == "ours"].origin) - set(range_memory.SCORE_DAYS) - set(range_memory.COACH_OWN_DAYS))


def fresh_ask(workers: int = 48) -> None:
    """Ask the improved crowd, once, about the fresh days. Same model, effort, instructions and facts as its other answers."""
    from concurrent.futures import ThreadPoolExecutor

    from dotenv import load_dotenv
    from openai import OpenAI

    from . import agents
    from .llm import Meter, ask_json

    load_dotenv(data.REPO / ".env")
    client, meter = OpenAI(), Meter(FRESH_BUDGET_USD)
    path = OUT / "fresh_answers.json"
    saved = json.loads(path.read_text()) if path.exists() else {"answers": {}, "usage": None}
    roles = json.loads((coach.OUT / "state.json").read_text())["config"]["roles"]
    items = coach.build_items({"fresh": {"ours": fresh_days()}})
    todo = [(n, it) for it in items.values() for n in roles if f"{n}|{it['series_id']}|{it['day']}" not in saved["answers"]]
    print(f"{len(items)} price-days on {len(fresh_days())} days; {len(todo)} calls to make", flush=True)

    def one(job):
        n, it = job
        user = it["facts"] + ("\n" + agents.refs_text(it["refs"]) if roles[n]["refs"] else "") + "\n\nGive your forecast."
        ans = ask_json(client, agents.MODEL, agents.EFFORT, coach.SYSTEM.format(role=roles[n]["instruction"]), user, agents.SCHEMA, "forecast", meter, flex=False)
        return f"{n}|{it['series_id']}|{it['day']}", {"q": {str(h): list(v) for h, v in agents.clean(ans, it["last"]).items()}, "reason": ans.get("reason", "")}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (key, ans) in enumerate(ex.map(one, todo), 1):
            saved["answers"][key] = ans
            if i % 500 == 0 or i == len(todo):
                saved["usage"] = meter.summary()
                path.write_text(json.dumps(saved))
                print(f"  {i} of {len(todo)} | {meter.summary()}", flush=True)


def fresh_frame() -> pd.DataFrame:
    store = json.loads((OUT / "fresh_answers.json").read_text())["answers"]
    by: dict = {}
    for key, a in store.items():
        n, sid, iso = key.split("|")
        by.setdefault((sid, day_of(iso)), {})[n] = a["q"]
    rows = []
    for (sid, o), q in by.items():
        if set(q) == set(ROLES):
            for h in HORIZONS:
                row = {"series_id": sid, "origin": o, "h": h}
                for n in ROLES:
                    row[f"{n}_lo"], row[f"{n}_mid"], row[f"{n}_hi"] = (float(v) for v in q[n][str(h)])
                rows.append(row)
    f = range_memory.truth_cells(data.build_world(data.DATA_END)).merge(pd.DataFrame(rows), on=["series_id", "origin", "h"], how="inner")
    f["engine_lo"], f["engine_mid"], f["engine_hi"] = f.e_lo, f.e_mid, f.e_hi
    for name, run_name in RL_RUNS.items():
        q = pd.read_parquet(data.REPO / "rl" / "runs" / run_name / "eval_cells.parquet")
        q = q[q.world == "ours"].assign(origin=lambda d: d.origin.map(day_of))[["series_id", "origin", "h", "trained_lo", "trained_mid", "trained_hi"]]
        q.columns = ["series_id", "origin", "h", f"{name}_lo", f"{name}_mid", f"{name}_hi"]
        f = f.merge(q, on=["series_id", "origin", "h"], how="inner")
    f["split"], f["moved"] = "fresh", np.abs(f.y - f["last"]) > 1e-9
    return f.sort_values(["origin", "series_id", "h"]).reset_index(drop=True)


def fresh_table(log=print) -> dict:
    """The handed-in team against the same team with a trained open model given a vote, on the fresh days."""
    f, fm, ci = fresh_frame(), report.fmt_ci, report.skill_ci
    add = lambda *ms: rebuild([{"kind": "members", "action": "add", "member": m} for m in ms])
    arms = {"five": start_bundle(), "plus_qwen3_8b_rl": rebuild([FRESH_EDIT]), "plus_qwen3_5_9b_rl": add("qwen3_5_9b_rl"), "plus_both": add("qwen3_8b_rl", "qwen3_5_9b_rl")}
    w = {k: walk(f, b) for k, b in arms.items()}
    g = f[["series_id", "origin", "h", "target", "loss_drift", "loss_engine"]].copy()
    for k in arms:
        g[f"loss_{k}"] = w[k].loss.to_numpy()
    for m in RL_RUNS:
        g[f"loss_{m}"] = np.asarray(cell_loss(f.y, f[f"{m}_lo"], f[f"{m}_mid"], f[f"{m}_hi"]))
    out = {"rule": {"edit": edit_text(FRESH_EDIT), "days": fresh_days()}, "n": int(len(g)), "n_days": int(g.origin.nunique()), "n_prices": int(g.series_id.nunique()),
           "vs_drift": {k: ci(g, f"loss_{k}", "loss_drift") for k in [*arms, *RL_RUNS, "engine"]},
           "vs_five": {k: ci(g, f"loss_{k}", "loss_five") for k in arms if k != "five"},
           "five_vs_engine": ci(g, "loss_five", "loss_engine"),
           "by_h": {int(h): ci(gh, "loss_plus_qwen3_8b_rl", "loss_five") for h, gh in g.groupby("h")},
           "inside": {k: float(w[k].inside.mean()) for k in arms}, "inside_when_moved": {k: float(w[k][w[k].moved].inside.mean()) for k in arms},
           "usage": json.loads((OUT / "fresh_answers.json").read_text())["usage"]}
    p = out["vs_five"]["plus_qwen3_8b_rl"]
    out["helps_by_the_rule"] = bool(p["skill"] > 0 and p["excludes_zero"])
    log(f"FRESH DAYS: {out['n']} forecasts, {out['n_prices']} prices, {out['n_days']} days the crowd had never answered")
    for k, v in out["vs_drift"].items():
        log(f"   {k:20s} against drift {fm(v)}" + (f" | against the five {fm(out['vs_five'][k])} | inside {out['inside'][k]:.3f}" if k in out["vs_five"] else ""))
    log(f"   by horizon, six against five: " + "; ".join(f"{h}d {fm(v)}" for h, v in out["by_h"].items()) + f" | helps by the rule: {out['helps_by_the_rule']} | usage {out['usage']}")
    (data.REPO / "findings" / "data" / "rsi_fresh.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "exam", "fresh-ask", "fresh"])
    ap.add_argument("--proposer", choices=["gpt", "random", "replay"], default="gpt")
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="main")
    a = ap.parse_args(argv)
    if a.cmd in ("exam", "fresh-ask", "fresh"):
        {"exam": exam, "fresh-ask": fresh_ask, "fresh": fresh_table}[a.cmd]()
        return 0
    if a.proposer == "replay":
        rec = json.loads((OUT / a.name / "state.json").read_text())
        st = run(ReplayProposer(rec["ledger"]), len(rec["ledger"]), rec["seed"], a.name, save=False)
        same = [e["kept"] for e in st["ledger"]] == [e["kept"] for e in rec["ledger"]] and st["bundle"] == rec["bundle"]
        print("replay matches the recorded run:", same)
        return 0 if same else 1
    run(GPTProposer() if a.proposer == "gpt" else RandomProposer(), a.generations, a.seed, a.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
