"""An AI analyst that writes its own test, in a sealed container, judged on data it never sees.

    uv run python -m gpuval.analyst run        # 12 ideas: about 12 AI calls, a few minutes of Modal CPU
    uv run python -m gpuval.analyst table      # re-print the saved ledger

The AI scientist in this repo could only turn 15 dials of the engine. This analyst writes code: for
each idea, one Python function `effect(world) -> float` that measures a pattern in the offer lists.

How an idea is judged
- The function is run on the real data and on 39 shuffled copies, in which every offer keeps its
  listing pattern and the same set of price changes but the ORDER of those changes in time is
  scrambled (`data.make_placebo`). A pattern in timing survives only in the real data.
- Discovery (data to 7 Oct): the idea passes if the real number lies outside all 39 shuffled ones.
  By luck that happens one time in twenty. The analyst is told the result and may build on it.
- Confirmation (8 Oct to 6 Nov, never shown to the analyst and never run until it has finished):
  a passed idea is confirmed only if the real number again lies outside all 39 shuffled ones, on
  the same side. By luck, a dead idea gets through both about once in 800.

Where things run
- The AI is called from this machine. Its code runs in a Modal container that is given the data
  and nothing else: no keys, no network (`block_network=True`), no access to our Modal account
  (`restrict_modal_access=True`), in a separate process with a time limit.
- The shuffles, the pass rule and the confirmation data stay outside the analyst's reach.

The rule is in PROTOCOL.md, deviation 12, committed before the first run.
"""
from __future__ import annotations

import json
import pickle
import sys

import modal
import numpy as np

from . import agents, data
from .core import day_of
from .llm import Meter, ask_json

OUT = data.REPO / "runs" / "analyst"
SPLIT = "2026-10-07"          # discovery uses data up to this day; confirmation uses the days after it
N_IDEAS = 12
N_SHUFFLES = 39
EFFORT = "medium"
BUDGET_USD = 5.0

app = modal.App("gpuval-analyst")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy>=1.24", "pandas>=2.0")
         .add_local_python_source("gpuval"))

RUNNER = r'''
import importlib.util, json, pickle, sys
worlds = pickle.load(open(sys.argv[1], "rb"))
spec = importlib.util.spec_from_file_location("idea", sys.argv[2])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
out = []
for w in worlds:
    out.append(float(mod.effect(w)))
json.dump(out, open(sys.argv[3], "w"))
'''


@app.function(image=image, block_network=True, restrict_modal_access=True, timeout=900, cpu=2.0, memory=8192)
def run_idea(code: str, world_blob: bytes, seeds: list[int], limit_s: int = 300) -> dict:
    """Run `effect` on the real world and on its shuffled copies. No secrets, no network."""
    import os
    import subprocess
    import tempfile

    from gpuval import data as d

    real = pickle.loads(world_blob)
    worlds = [real] + [d.make_placebo(real, s) for s in seeds]
    with tempfile.TemporaryDirectory() as td:
        wp, cp, rp, op = (os.path.join(td, n) for n in ("worlds.pkl", "idea.py", "runner.py", "out.json"))
        pickle.dump(worlds, open(wp, "wb"), protocol=4)
        open(cp, "w").write(code)
        open(rp, "w").write(RUNNER)
        try:
            p = subprocess.run([sys.executable, "-I", rp, wp, cp, op], capture_output=True, text=True, timeout=limit_s, cwd=td,
                               env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {limit_s} seconds"}
        if p.returncode != 0 or not os.path.exists(op):
            return {"ok": False, "error": (p.stderr or p.stdout)[-1500:]}
        vals = json.load(open(op))
    if not all(np.isfinite(vals)):
        return {"ok": False, "error": "effect() returned a value that is not a finite number"}
    return {"ok": True, "real": vals[0], "shuffled": vals[1:]}


# ------------------------------------------------------------------ worlds ----
def tail_world(world: dict, after: str) -> dict:
    """The part of a world dated after `after` (the confirmation data)."""
    ad = day_of(after)
    out = {"wall": world["wall"], "name": world["name"], "series": {}}
    for sid, s in world["series"].items():
        k = int(np.searchsorted(s["days"], ad, side="right"))
        if len(s["days"]) - k < 10:
            continue
        P = s["P"][k:]
        keep = ~np.all(np.isnan(P), axis=0)
        out["series"][sid] = {"days": s["days"][k:].copy(), "value": s["value"][k:].copy(), "P": P[:, keep].copy(),
                              "slots": [x for x, kk in zip(s["slots"], keep) if kk], "meta": dict(s["meta"])}
    return out


def verdict(real: float, shuffled: list[float]) -> str:
    if real > max(shuffled):
        return "above all"
    if real < min(shuffled):
        return "below all"
    return "inside"


# ------------------------------------------------------------------ analyst ----
SYSTEM = """You are an analyst looking for patterns in GPU rental price lists. You write code; a fixed test decides what counts.

THE DATA. `world["series"]` maps a series id to one published price series. Each published price is the MEDIAN of one tracking site's list of offers for one GPU, in USD per GPU-hour. For a series `s`:
- s["days"]   int64 array, one entry per observed day: proleptic Gregorian ordinals (datetime.date.fromordinal(int(d)))
- s["value"]  float array, the published median on each of those days
- s["P"]      float array [n_days, n_offers]: each offer's price on each day, NaN where the offer was not listed
- s["slots"]  list of offer names, one per column of P, as "provider|sku|region|s<snapshot>|d<dup>"
- s["meta"]   dict with source_id (the tracking site), instrument_class, gpu_model, pricing_tier
There are a few dozen series, most with a few dozen days. Known already, so not worth re-finding: most offers never change price, a few change almost daily, and offers join and leave lists.

YOUR JOB. Each turn, propose ONE idea about WHEN or IN WHAT ORDER prices or lists change, and write a Python function

    def effect(world) -> float

that returns a single number measuring it. You may import numpy, pandas, math, datetime, collections, itertools. No files, no network, no randomness. It must run in under a minute and return a finite float even if a series is short or empty.

THE TEST. Your function is run on the real data and on 39 shuffled copies. In a shuffled copy every offer keeps its listing pattern and exactly the same set of day-to-day price changes, but the order of those changes in time is scrambled (one scramble per series, shared by all its offers). So anything about how often or how far prices move is identical in the copies; only timing differs. Your idea PASSES if the real number is outside all 39 shuffled numbers (above all or below all). An idea that is really about list structure, not timing, cannot pass.

Passed ideas are later checked once on newer data you never see. Prefer ideas that would matter to someone forecasting these prices a few days ahead. Do not repeat an idea that already failed with a small variation. Reply with JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "idea": {"type": "string", "description": "The hypothesis in one or two plain sentences."},
        "why_it_matters": {"type": "string", "description": "One sentence: what a forecaster would do differently if it is true."},
        "code": {"type": "string", "description": "Python source defining effect(world) -> float. Imports inside the source."},
    },
    "required": ["idea", "why_it_matters", "code"],
    "additionalProperties": False,
}


def ledger_text(ledger: list[dict]) -> str:
    if not ledger:
        return "(nothing tried yet)"
    lines = []
    for e in ledger:
        if e["discovery"]["ok"]:
            d = e["discovery"]
            lines.append(f"#{e['n']} {e['idea']}\n    real {d['real']:.5g}; shuffled copies from {min(d['shuffled']):.5g} to {max(d['shuffled']):.5g} "
                         f"(middle {float(np.median(d['shuffled'])):.5g}): {'PASSED' if d['verdict'] != 'inside' else 'failed'} ({d['verdict']})")
        else:
            lines.append(f"#{e['n']} {e['idea']}\n    the code did not run: {e['discovery']['error'][-300:]}")
    return "\n".join(lines)


def run() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(data.REPO / ".env")
    client = OpenAI()
    meter = Meter(BUDGET_USD)
    full = data.build_world(data.DATA_END)
    disc, conf = data.cut_world(full, SPLIT), tail_world(full, SPLIT)
    data.assert_walled(disc, SPLIT)
    disc_blob, conf_blob = pickle.dumps(disc, protocol=4), pickle.dumps(conf, protocol=4)
    seeds_d, seeds_c = list(range(1, N_SHUFFLES + 1)), list(range(101, 101 + N_SHUFFLES))
    OUT.mkdir(parents=True, exist_ok=True)
    ledger: list[dict] = []
    with modal.enable_output(), app.run():
        for n in range(1, N_IDEAS + 1):
            user = f"IDEAS TRIED SO FAR\n{ledger_text(ledger)}\n\nIdeas left including this one: {N_IDEAS - n + 1}. Propose one idea and its code."
            ans = ask_json(client, agents.MODEL, EFFORT, SYSTEM, user, SCHEMA, "idea", meter)
            res = run_idea.remote(ans["code"], disc_blob, seeds_d)
            if res["ok"]:
                res["verdict"] = verdict(res["real"], res["shuffled"])
            ledger.append({"n": n, **ans, "discovery": res})
            (OUT / "ledger.json").write_text(json.dumps({"split": SPLIT, "ledger": ledger, "usage": meter.summary()}, indent=1))
            print(f"idea {n}: {ans['idea'][:110]} | " + (f"real {res['real']:.4g} vs shuffled {min(res['shuffled']):.4g}..{max(res['shuffled']):.4g} -> {res['verdict']}" if res["ok"] else "did not run"), flush=True)
        # the analyst is finished; only now is the confirmation data touched, once per passed idea
        for e in ledger:
            d = e["discovery"]
            if d["ok"] and d["verdict"] != "inside":
                res = run_idea.remote(e["code"], conf_blob, seeds_c)
                if res["ok"]:
                    res["verdict"] = verdict(res["real"], res["shuffled"])
                    res["confirmed"] = res["verdict"] == d["verdict"]
                e["confirmation"] = res
    (OUT / "ledger.json").write_text(json.dumps({"split": SPLIT, "ledger": ledger, "usage": meter.summary()}, indent=1))
    table()


def drop_provider(world: dict, provider: str) -> dict:
    """The same world without one provider's offers; the published value is the median of what is left."""
    out = {"wall": world["wall"], "name": world["name"], "series": {}}
    for sid, s in world["series"].items():
        keep = np.array([not name.lower().startswith(provider.lower() + "|") for name in s["slots"]], dtype=bool)
        P = s["P"][:, keep]
        with np.errstate(all="ignore"):
            val = np.round(np.nanmedian(P, axis=1), 4) if P.shape[1] else np.full(len(P), np.nan)
        ok = np.isfinite(val)
        if ok.sum() < 10:
            continue
        out["series"][sid] = {"days": s["days"][ok].copy(), "value": val[ok], "P": P[ok].copy(),
                              "slots": [x for x, k in zip(s["slots"], keep) if k], "meta": dict(s["meta"])}
    return out


def without(provider: str) -> None:
    """A reading made after the result was in, not part of the test: re-run each confirmed idea's code,
    in the same sealed container, on the data with one provider's offers taken out.

        uv run python -m gpuval.analyst without scaleway
    """
    saved = json.loads((OUT / "ledger.json").read_text())
    full = data.build_world(data.DATA_END)
    worlds = {"discovery": (drop_provider(data.cut_world(full, SPLIT), provider), list(range(1, N_SHUFFLES + 1))),
              "confirmation": (drop_provider(tail_world(full, SPLIT), provider), list(range(101, 101 + N_SHUFFLES)))}
    blobs = {k: (pickle.dumps(w, protocol=4), seeds) for k, (w, seeds) in worlds.items()}
    with modal.enable_output(), app.run():
        for e in saved["ledger"]:
            if not e.get("confirmation", {}).get("confirmed"):
                continue
            e.setdefault("without", {})[provider] = {}
            for part, (blob, seeds) in blobs.items():
                res = run_idea.remote(e["code"], blob, seeds)
                if res["ok"]:
                    res["verdict"] = verdict(res["real"], res["shuffled"])
                e["without"][provider][part] = res
                print(f"idea {e['n']} without {provider}, {part}: " + (f"real {res['real']:.4g} vs {min(res['shuffled']):.4g}..{max(res['shuffled']):.4g} -> {res['verdict']}" if res["ok"] else "did not run"), flush=True)
    (OUT / "ledger.json").write_text(json.dumps(saved, indent=1))
    table()


def table() -> dict:
    saved = json.loads((OUT / "ledger.json").read_text())
    led = saved["ledger"]
    out = {"split": saved["split"], "ideas": len(led), "ran": sum(e["discovery"]["ok"] for e in led),
           "passed_discovery": sum(e["discovery"]["ok"] and e["discovery"]["verdict"] != "inside" for e in led),
           "confirmed": sum(bool(e.get("confirmation", {}).get("confirmed")) for e in led), "usage": saved["usage"], "rows": []}
    for e in led:
        d, c = e["discovery"], e.get("confirmation")
        row = {"n": e["n"], "idea": e["idea"], "why_it_matters": e["why_it_matters"], "ran": d["ok"]}
        if d["ok"]:
            row.update({"real": d["real"], "shuffled_min": min(d["shuffled"]), "shuffled_max": max(d["shuffled"]), "discovery": d["verdict"]})
        if c:
            row["confirmation"] = ({"real": c["real"], "shuffled_min": min(c["shuffled"]), "shuffled_max": max(c["shuffled"]), "verdict": c["verdict"],
                                    "confirmed": c["confirmed"]} if c["ok"] else {"error": c["error"][-300:]})
        for prov, parts in e.get("without", {}).items():
            row.setdefault("without", {})[prov] = {part: ({"real": r["real"], "shuffled_min": min(r["shuffled"]), "shuffled_max": max(r["shuffled"]), "verdict": r["verdict"]}
                                                          if r["ok"] else {"error": r["error"][-300:]}) for part, r in parts.items()}
        out["rows"].append(row)
        tail = ""
        if c:
            tail = (f" | newer data: real {c['real']:.4g} vs {min(c['shuffled']):.4g}..{max(c['shuffled']):.4g} -> {'CONFIRMED' if c['confirmed'] else 'not confirmed'}"
                    if c["ok"] else " | newer data: did not run")
        print(f"#{e['n']:2d} {'PASS' if d['ok'] and d['verdict'] != 'inside' else 'fail' if d['ok'] else 'ERR '} {e['idea'][:150]}{tail}")
    print(f"{out['ideas']} ideas, {out['ran']} ran, {out['passed_discovery']} passed discovery, {out['confirmed']} confirmed on newer data | usage {out['usage']}")
    (data.REPO / "findings" / "data" / "analyst.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    without(sys.argv[2]) if sys.argv[1] == "without" else {"run": run, "table": table}[sys.argv[1]]()
