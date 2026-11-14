"""The AI scientist: propose a hypothesis as a change to the engine's dials, test it, keep what survives.

One research step = `n_ideas` iterations at one wall date T, on a world that contains nothing after T:
  1. the proposer (an LLM, or a control) reads the ledger and the diagnostics and returns one idea;
  2. the fixed harness scores the idea on the development period (targets up to T-7) and on a
     hidden check (targets in the last 7 days before T);
  3. the gate keeps it only if it helps on development AND passes the hidden check;
  4. the outcome is appended to the ledger, which the proposer reads next time.
The proposer never sees the hidden-check numbers, only pass/fail.
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
import zlib

import numpy as np

from . import harness
from .core import HORIZONS, day_of, iso_of
from .engine import CLASSES, SPEC_SCHEMA, apply_patch, default_spec, validate_spec

MODEL = os.environ.get("GPUVAL_MODEL", "gpt-6.1-sol")
REASONING = os.environ.get("GPUVAL_REASONING", "medium")
DEV_MIN_GAIN = 0.002      # total error must fall by at least 0.2% on development
AVG_TOLERANCE = 0.002     # per-series average skill may not fall by more than this
HIDDEN_NOISE = 0.005      # sd of the noise on the hidden-check threshold (limits what leaks through pass/fail)
MAX_CHANGES = 3

SCOPES = ["all"] + [f"class:{c}" for c in CLASSES] + [f"horizon:{h}" for h in HORIZONS]

IDEA_SCHEMA = {
    "type": "object",
    "properties": {
        "lesson": {"type": "string", "description": "What the previous result taught you, in one or two plain sentences. Empty string if there is no previous result."},
        "hypothesis": {"type": "string", "description": "One sentence: a claim about what moves these prices or what causes the current forecast errors."},
        "test": {"type": "string", "description": "How the change below tests the hypothesis, and what result would count against it."},
        "prediction": {"type": "string", "description": "What you expect to happen to total skill and where (which class or horizon)."},
        "changes": {
            "type": "array",
            "description": f"At most {MAX_CHANGES} dial changes. Empty list means: change nothing.",
            "items": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": SCOPES},
                    "dial": {"type": "string", "enum": list(SPEC_SCHEMA)},
                    "value": {"type": "string", "description": "New value as text, e.g. '45', '0.5', 'true', 'last'."},
                },
                "required": ["scope", "dial", "value"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lesson", "hypothesis", "test", "prediction", "changes"],
    "additionalProperties": False,
}


def dials_text() -> str:
    lines = []
    for k, (typ, lo, hi, default, doc) in SPEC_SCHEMA.items():
        rng = "true/false" if typ is bool else ("one of " + "/".join(lo) if typ is str else f"{lo} to {hi}")
        lines.append(f"- {k} ({rng}; default {default}): {doc}")
    return "\n".join(lines)


SYSTEM = f"""You are a research scientist studying what moves posted cloud GPU rental prices. You improve a forecasting engine by running experiments, one small experiment per turn.

WHAT IS BEING FORECAST
Each series is the daily MEDIAN of a price list published by one price-tracking site for one GPU model and price type (USD per GPU-hour). A list has between 3 and a few thousand offers. Forecasts are made 1, 3 and 7 days ahead as a low / middle / high number (the 10%, 50% and 90% points). The truth should land inside low..high about 80% of the time, below about 10% and above about 10%.
Series classes: neocloud_rate_card (provider list prices), neocloud_spot, hyperscaler_spot (AWS/Azure/GCP spot across regions), marketplace_ask.

THE REFERENCE FORECAST ("drift")
Drift takes the last value and applies the 10/50/90% points of how that median itself moved over the same number of days in the past. It knows nothing about the list behind the median.

THE ENGINE
The engine looks underneath the median. For a horizon of h days it takes every past pair of days (d, d+h) in a lookback window, applies to TODAY's list what each individual offer did between d and d+h (repriced, stayed, dropped off the list, joined it), and recomputes the median. The spread of those replayed medians gives low / middle / high. Because it starts from today's list it knows which offer is in the middle now and how far its neighbours let the middle move.

A STARTING OBSERVATION FROM A HUMAN (not yet tested by you; treat it as a hint, not a fact)
On the lists a human inspected, most providers changed their price at most twice in ten weeks, while one or two providers repriced almost daily, sometimes in a steady direction. Much of the movement in a median came from those one or two offers and from offers joining or leaving the list.

SCORING
Error is pinball loss in dollars. "Total skill" = 1 - (our total error / total error of a flat last-value forecast); expensive GPUs weigh more. "Per-series average skill" scores each series separately and then averages. Higher is better. Drift's own total skill is shown for comparison.

HOW AN IDEA IS JUDGED
An idea is KEPT only if (a) on the development period total error falls by at least {DEV_MIN_GAIN:.1%} and per-series average skill does not fall, and (b) it passes a hidden check on the most recent week, for which you are told only PASS or FAIL. Ideas that help by chance on development tend to fail the hidden check. A kept idea becomes the new baseline for later ideas.

THE DIALS YOU MAY CHANGE (nothing else can be changed)
{dials_text()}
A change has a scope: "all", one class ("class:neocloud_rate_card" ...), or one horizon ("horizon:1", "horizon:3", "horizon:7"). Horizon settings override class settings, which override "all".

HOW TO WORK
- State ONE hypothesis about the market or about why the current forecasts miss, then the smallest change (at most {MAX_CHANGES} dial changes) that tests it, and predict the outcome before seeing it.
- Read the ledger first. Do not repeat a failed idea unless you explain what is different. Use the diagnostics to aim at where error is concentrated.
- If the evidence says nothing further will help, return an empty list of changes and say so. Restraint is a valid result: a false improvement costs more than no change.
- Write plainly. No jargon."""


def changes_to_patch(changes: list[dict]) -> dict:
    patch: dict = {}
    for c in (changes or [])[:MAX_CHANGES]:
        scope, dial, value = str(c.get("scope", "all")), c.get("dial"), c.get("value")
        if dial not in SPEC_SCHEMA:
            continue
        if scope == "all":
            patch[dial] = value
        elif scope.startswith("class:"):
            patch.setdefault("by_class", {}).setdefault(scope[6:], {})[dial] = value
        elif scope.startswith("horizon:"):
            patch.setdefault("by_horizon", {}).setdefault(scope[8:], {})[dial] = value
    return patch


def spec_text(spec: dict) -> str:
    base = {k: spec[k] for k in SPEC_SCHEMA}
    s = "all: " + json.dumps(base)
    for k in ("by_class", "by_horizon"):
        for name, sub in spec.get(k, {}).items():
            s += f"\n{k[3:]}:{name}: " + json.dumps(sub)
    return s


def ledger_text(ledger: list[dict], limit: int = 40) -> str:
    if not ledger:
        return "(empty: this is the first idea)"
    out = []
    for e in ledger[-limit:]:
        ch = "; ".join(f"{c['scope']} {c['dial']}={c['value']}" for c in e["changes"]) or "no change"
        d = e["dev"]
        out.append(
            f"#{e['n']} [{e['wall']} idea {e['idea']}] H: {e['hypothesis']} | change: {ch} | development: total skill "
            f"{d['before_total']:+.3f} -> {d['after_total']:+.3f}, per-series average {d['before_avg']:+.3f} -> {d['after_avg']:+.3f} | "
            f"hidden check: {e['hidden']} | {'KEPT' if e['kept'] else 'REJECTED'}"
            + (f" | lesson: {e['lesson']}" if e.get("lesson") else ""))
    return "\n".join(out)


# ------------------------------------------------------------------ proposers ----
class OpenAIProposer:
    """An OpenAI reasoning model with a strict JSON schema for the idea."""
    kind = "gpt"

    def __init__(self, model: str = MODEL, effort: str = REASONING):
        from openai import OpenAI

        self.client = OpenAI()
        self.model, self.effort = model, effort
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def propose(self, user_msg: str, rng: random.Random) -> dict:
        last = None
        for attempt in range(4):
            try:
                resp = self.client.responses.create(
                    model=self.model, reasoning={"effort": self.effort},
                    input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}],
                    text={"format": {"type": "json_schema", "name": "idea", "schema": IDEA_SCHEMA, "strict": True}},
                )
                self.usage["calls"] += 1
                u = getattr(resp, "usage", None)
                if u is not None:
                    self.usage["input_tokens"] += int(getattr(u, "input_tokens", 0) or 0)
                    self.usage["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)
                return json.loads(resp.output_text)
            except Exception as e:  # network / rate limit / parse: back off and retry
                last = e
                time.sleep(2 + 4 * attempt)
        raise RuntimeError(f"proposer failed after retries: {last}")


class RandomProposer:
    """Control arm: random dial changes, no reading of evidence. Shows what search alone buys."""
    kind = "random"

    def __init__(self):
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def propose(self, user_msg: str, rng: random.Random) -> dict:
        self.usage["calls"] += 1
        changes = []
        for _ in range(rng.choice([1, 1, 2])):
            dial = rng.choice(list(SPEC_SCHEMA))
            typ, lo, hi, default, _ = SPEC_SCHEMA[dial]
            if typ is bool:
                val = rng.choice(["true", "false"])
            elif typ is str:
                val = rng.choice(list(lo))
            elif typ is int:
                val = str(rng.randint(int(lo), int(hi)))
            else:
                val = f"{rng.uniform(lo, hi):.3f}"
            changes.append({"scope": rng.choice(SCOPES), "dial": dial, "value": val})
        return {"lesson": "", "hypothesis": "(random control: no hypothesis)", "test": "random dial change",
                "prediction": "none", "changes": changes}


class ReplayProposer:
    """Replays the ideas recorded in an earlier ledger (reproduction without any AI calls)."""
    kind = "replay"

    def __init__(self, ledger: list[dict]):
        self.queue = [e for e in ledger]
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def propose(self, user_msg: str, rng: random.Random) -> dict:
        e = self.queue.pop(0)
        return {"lesson": "", "hypothesis": e["hypothesis"], "test": e.get("test", ""),
                "prediction": e.get("prediction", ""), "changes": e["changes"]}


def make_proposer(kind: str, **kw):
    if kind == "gpt":
        return OpenAIProposer(**kw)
    if kind == "random":
        return RandomProposer()
    raise ValueError(f"unknown proposer {kind!r}")


# ------------------------------------------------------------------- one step ----
def new_state() -> dict:
    return {"spec": default_spec(), "fallback": [], "ledger": [], "versions": []}


def split_cells(cells, wall: str):
    wd = day_of(wall)
    dev = cells[cells.target <= wd - 7]
    hidden = cells[(cells.target > wd - 7) & (cells.target <= wd)]
    return dev, hidden


def evaluate(world: dict, spec: dict, cells, engines: dict) -> dict:
    wall = world["wall"]
    res = harness.run_spec(world, spec, cells, engines)
    dev, hidden = split_cells(res, wall)
    return {"res": res, "dev": harness.summarize(dev), "hidden": harness.summarize(hidden),
            "dev_df": dev, "hidden_df": hidden}


def gate(best: dict, cand: dict, rng: np.random.Generator) -> tuple[bool, str, bool]:
    """(kept, hidden verdict, passed development)."""
    b, c = best["dev"], cand["dev"]
    dev_ok = (c["loss"] < b["loss"] * (1 - DEV_MIN_GAIN)
              and (c["skill_avg_series"] or 0) >= (b["skill_avg_series"] or 0) - AVG_TOLERANCE)
    bh, chh = best["hidden"], cand["hidden"]
    if not bh.get("n_cells"):
        hidden_ok = True
    else:
        hidden_ok = chh["loss"] < bh["loss"] * (1 + rng.normal(0, HIDDEN_NOISE))
    if not dev_ok:
        return False, "not run (failed development)", False
    return bool(hidden_ok), "PASS" if hidden_ok else "FAIL", True


def series_fallback(res) -> list[str]:
    """Series to forecast with drift instead of the engine. The rehearsal (PROTOCOL.md) showed that
    switching a series to drift because the engine had done worse on it so far LOWERED blind skill
    (the comparison is too noisy per series), so no series is switched. Hedging toward drift is left
    to the `blend_drift` dial, which has to pass the gate like any other idea."""
    return []


def series_switch_alt(res) -> list[str]:
    """The dropped rule, kept only so the examiner can report it side by side: series where the
    engine did not beat drift on everything visible."""
    per = res.groupby("series_id").agg(l=("loss", "sum"), d=("loss_drift", "sum"))
    return sorted(per.index[per.l >= per.d].tolist())


def run_step(world: dict, state: dict, proposer, n_ideas: int = 8, seed: int = 0, log=print) -> dict:
    """One research step on a world cut at its wall. Returns the updated state (new ledger entries,
    possibly a new kept spec, and a frozen version)."""
    wall = world["wall"]
    state = copy.deepcopy(state)
    rng_py = random.Random(f"{world['name']}|{wall}|{seed}")
    rng_np = np.random.default_rng(zlib.crc32(f"{world['name']}|{wall}|{seed}".encode()))
    cells = harness.base_cells(world)
    engines: dict = {}
    best_spec = validate_spec(state["spec"])
    best = evaluate(world, best_spec, cells, engines)
    log(f"[{world['name']} {wall}] start: dev total {best['dev']['skill_total']:+.3f} avg {best['dev']['skill_avg_series']:+.3f} "
        f"(drift {best['dev']['drift_skill_total']:+.3f}), {best['dev']['n_cells']} dev cells, {best['hidden'].get('n_cells', 0)} hidden")
    prev = None
    for k in range(1, n_ideas + 1):
        user = (f"Today is {wall}; you can see data up to today. Ideas left in this step including this one: {n_ideas - k + 1}.\n\n"
                f"CURRENT KEPT SETTINGS\n{spec_text(best_spec)}\n\n"
                f"DIAGNOSTICS OF THE CURRENT SETTINGS ON THE DEVELOPMENT PERIOD\n{harness.diagnostics(best['dev_df'])}\n\n"
                f"LEDGER (all ideas so far, oldest first)\n{ledger_text(state['ledger'])}\n\n"
                + (f"RESULT OF YOUR LAST IDEA\n{prev}\n\n" if prev else "")
                + "Give your next idea.")
        idea = proposer.propose(user, rng_py)
        if state["ledger"] and idea.get("lesson") and state["ledger"][-1].get("lesson") in (None, ""):
            state["ledger"][-1]["lesson"] = idea["lesson"]
        changes = [c for c in idea.get("changes", []) if c.get("dial") in SPEC_SCHEMA][:MAX_CHANGES]
        cand_spec = apply_patch(best_spec, changes_to_patch(changes))
        no_change = cand_spec == best_spec
        if no_change:
            cand, kept, hidden_verdict, dev_ok = best, False, "not run (no change)", False
        else:
            cand = evaluate(world, cand_spec, cells, engines)
            kept, hidden_verdict, dev_ok = gate(best, cand, rng_np)
        entry = {
            "n": len(state["ledger"]) + 1, "world": world["name"], "wall": wall, "idea": k, "proposer": proposer.kind,
            "hypothesis": idea.get("hypothesis", ""), "test": idea.get("test", ""), "prediction": idea.get("prediction", ""),
            "changes": changes, "no_change": bool(no_change),
            "dev": {"before_total": best["dev"]["skill_total"], "after_total": cand["dev"]["skill_total"],
                    "before_avg": best["dev"]["skill_avg_series"], "after_avg": cand["dev"]["skill_avg_series"],
                    "before_loss": best["dev"]["loss"], "after_loss": cand["dev"]["loss"],
                    "inside_after": cand["dev"]["inside"]},
            "dev_ok": bool(dev_ok), "hidden": hidden_verdict, "kept": bool(kept), "lesson": "",
            "spec_after": cand_spec if kept else best_spec,
            # recorded for the examiner's audit only; never shown to the proposer
            "_hidden_loss_before": best["hidden"].get("loss"), "_hidden_loss_after": cand["hidden"].get("loss"),
        }
        state["ledger"].append(entry)
        d = entry["dev"]
        prev = (f"Development: total skill {d['before_total']:+.3f} -> {d['after_total']:+.3f}; per-series average "
                f"{d['before_avg']:+.3f} -> {d['after_avg']:+.3f}; truth inside range {d['inside_after']:.2f}. "
                f"Hidden check: {hidden_verdict}. Verdict: {'KEPT' if kept else 'REJECTED'}.")
        log(f"[{world['name']} {wall}] idea {k}: {'KEPT' if kept else 'rejected'} | dev {d['before_total']:+.3f}->{d['after_total']:+.3f} "
            f"| hidden {hidden_verdict} | {'; '.join(c['scope'] + ' ' + c['dial'] + '=' + str(c['value']) for c in changes) or 'no change'}")
        if kept:
            best_spec, best = cand_spec, cand
    state["spec"] = best_spec
    state["fallback"] = series_fallback(best["res"])
    state["versions"].append({"wall": wall, "spec": best_spec, "fallback": state["fallback"],
                              "switch_alt": series_switch_alt(best["res"]),
                              "dev": {k: best["dev"].get(k) for k in ("skill_total", "skill_avg_series", "drift_skill_total", "inside", "n_cells")}})
    u = getattr(proposer, "usage", {})
    tot = state.get("usage", {})
    state["usage"] = {k: int(tot.get(k, 0)) + int(u.get(k, 0)) for k in ("calls", "input_tokens", "output_tokens")}
    return state
