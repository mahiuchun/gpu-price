"""RL-train a small open model (on Tinker) to forecast prices directly.

    cd rl && uv run python train_forecaster.py --name fc1 --rounds 60

Each round: 16 practice prices, 8 answers each from the current weights; every answer is scored
against what actually happened; one policy-gradient step. Practice is AWS spot prices up to
30 Jun 2025. Every 10 rounds the current weights are checked on a fixed set of later AWS dates.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

import forecast_env as env
from policy_gradient import Trainer

HERE = Path(__file__).resolve().parent


def score_group(item: dict, completions) -> list[tuple[float, bool]]:
    return [env.reward(item, c.text) for c in completions]


def check(sampler, items: list[dict], n: int = 2) -> dict:
    groups = sampler.sample([it["prompt"] for it in items], n, temperature=1.0, max_tokens=90)
    rs, ok = [], []
    for it, g in zip(items, groups):
        for r, good in score_group(it, g):
            rs.append(r); ok.append(good)
    return {"mean_reward": round(statistics.fmean(rs), 4), "readable": round(sum(ok) / len(ok), 3), "n": len(rs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--adv-std", type=float, default=0.02, help="advantages are rescaled to this spread each round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B", help="any model on Tinker, e.g. Qwen/Qwen3.5-9B")
    a = ap.parse_args()
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    world = env.aws_world()
    check_items = env.sample_items(world, 64, env.CHECK_FROM, env.CHECK_TO, random.Random(123))
    trainer = Trainer(a.base_model, learning_rate=a.lr, system_prompt=env.SYSTEM, seed=a.seed, session_name=f"gpuval-forecaster-{a.name}")
    log = {"args": vars(a), "base_model": trainer.base_model, "rounds": [], "checks": []}
    t0 = time.time()
    c0 = check(trainer, check_items)
    log["checks"].append({"round": 0, **c0})
    print(f"check before training: mean reward {c0['mean_reward']:+.3f}, readable {c0['readable']:.2f}", flush=True)
    for r in range(1, a.rounds + 1):
        items = env.sample_items(world, a.prompts, "2022-08-05", env.TRAIN_END, rng)
        groups = trainer.sample([it["prompt"] for it in items], a.samples, temperature=1.0, max_tokens=90)
        triples, rs, ok, spread = [], [], [], []
        for it, g in zip(items, groups):
            sc = score_group(it, g)
            grp = [x[0] for x in sc]
            spread += [x - statistics.fmean(grp) for x in grp]
            for c, (rew, good) in zip(g, sc):
                triples.append((it["prompt"], c, rew)); rs.append(rew); ok.append(good)
        sd = float(np.std(spread)) or 1.0
        up = trainer.update([(p, c, rew * a.adv_std / sd) for p, c, rew in triples])
        row = {"round": r, "mean_reward": round(statistics.fmean(rs), 4), "readable": round(sum(ok) / len(ok), 3),
               "within_group_sd": round(sd, 4), "skipped": up.get("skipped")}
        log["rounds"].append(row)
        msg = f"round {r:3d}: mean reward {row['mean_reward']:+.3f} | readable {row['readable']:.2f} | spread {sd:.3f}"
        if r % 10 == 0:
            c = check(trainer, check_items)
            log["checks"].append({"round": r, **c})
            msg += f" || check on later dates: {c['mean_reward']:+.3f} (readable {c['readable']:.2f})"
        print(msg, flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))
    ckpt = trainer.save(f"{a.name}-final")
    log.update(checkpoint=ckpt, usage=trainer.usage, cost_estimate_usd=trainer.cost(), minutes=round((time.time() - t0) / 60, 1))
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"checkpoint {ckpt}\nusage {trainer.usage} cost about ${trainer.cost()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
