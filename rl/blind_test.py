"""Blind test of the RL-trained proposer (PROTOCOL.md, "RL arm").

    cd rl && uv run python blind_test.py --checkpoint tinker://.../sampler_weights/<name> --name blind

At each time-machine step (16, 23, 30 Oct) in the real world, each proposer gives 16 single ideas
from the diagnostics it can see at that step; each idea is scored on the following blind week
(origins on or after the step date, targets within 7 days), as the percentage fall in total error
relative to the default settings. Proposers: the small model before training, after training, and
the random control. The trained weights are fixed before this script is run.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics
from pathlib import Path

import env
from gpuval import data, harness, scientist
from gpuval.core import day_of
from gpuval.engine import build_engines, default_spec
from gpuval.timemachine import RECORD_STEPS, plus_days
from policy_gradient import Sampler

HERE = Path(__file__).resolve().parent
N_IDEAS = 16


def build_tests() -> dict[str, env.Practice]:
    tests = {}
    for T in RECORD_STEPS:
        end = plus_days(T, 7)
        full = data.build_world(end)
        cut = data.cut_world(full, T)
        data.assert_walled(cut, T)
        shown = env.prepare(cut, f"shown@{T}", "real")            # what the proposer sees
        cells = harness.base_cells(full)
        blind = cells[(cells.origin >= day_of(T)) & (cells.target <= day_of(end))]
        engines = build_engines(full)
        base = harness.run_spec(full, default_spec(), blind, engines)
        tests[T] = env.Practice(T, "real", T, full, engines, blind, float(base.loss.mean()), shown.prompt)
    return tests


def summarize(rewards: list[float]) -> dict:
    return {"n": len(rewards), "share_helpful": round(sum(r > 0 for r in rewards) / len(rewards), 3),
            "share_harmful": round(sum(r < 0 for r in rewards) / len(rewards), 3),
            "share_no_effect": round(sum(r == 0 for r in rewards) / len(rewards), 3),
            "mean_change_pct": round(statistics.fmean(rewards), 3), "median_change_pct": round(statistics.median(rewards), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", default="blind")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    env.PRACTICE.update(build_tests())
    pool = mp.get_context("fork").Pool(a.workers)
    steps = list(RECORD_STEPS)
    out = {"checkpoint": a.checkpoint, "base_model": a.base_model, "steps": steps, "arms": {}, "ideas": []}
    samplers = {"small model, untrained": Sampler(base_model=a.base_model, system_prompt=env.system_prompt()),
                "small model, RL-trained": Sampler(checkpoint=a.checkpoint, system_prompt=env.system_prompt())}
    for label, sampler in samplers.items():
        groups = sampler.sample([env.PRACTICE[T].prompt for T in steps], N_IDEAS, temperature=1.0, max_tokens=260)
        flat = [(T, c.text) for T, g in zip(steps, groups) for c in g]
        scored = pool.starmap(env.score_text, flat)
        out["arms"][label] = {"all": summarize([s["reward"] for s in scored]),
                              "by_step": {T: summarize([s["reward"] for (t, _), s in zip(flat, scored) if t == T]) for T in steps},
                              "readable": round(sum(s["readable"] for s in scored) / len(scored), 3)}
        out["ideas"] += [{"arm": label, "step": t, "idea": s["canon"], "change_pct": s["reward"]} for (t, _), s in zip(flat, scored)]
    rng = random.Random(0)
    rp = scientist.RandomProposer()
    flat = [(T, json.dumps({"hypothesis": "random", "changes": rp.propose("", rng)["changes"]})) for T in steps for _ in range(N_IDEAS)]
    scored = pool.starmap(env.score_text, flat)
    out["arms"]["random control"] = {"all": summarize([s["reward"] for s in scored]),
                                     "by_step": {T: summarize([s["reward"] for (t, _), s in zip(flat, scored) if t == T]) for T in steps},
                                     "readable": 1.0}
    out["ideas"] += [{"arm": "random control", "step": t, "idea": s["canon"], "change_pct": s["reward"]} for (t, _), s in zip(flat, scored)]
    dest = HERE / "runs" / f"{a.name}.json"
    dest.write_text(json.dumps(out, indent=1))
    for label, r in out["arms"].items():
        print(f"{label:26s} {r['all']}")
    print(f"wrote {dest}")
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
