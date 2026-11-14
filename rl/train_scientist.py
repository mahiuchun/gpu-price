"""RL-train a small open model (on Tinker) to propose better ideas for the forecasting engine.

    cd rl && uv run python train_scientist.py --name trial --trial            # ~200 episodes, one world
    cd rl && uv run python train_scientist.py --name full --rounds 31         # ~2,000 episodes, all practice worlds

Each round: sample ideas from the current weights for several practice worlds, score every idea
with the fixed harness (the reward), and take one policy-gradient step. Everything here reads data
dated up to 16 Oct 2026 only (see env.py).
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import random
import statistics
import time
from pathlib import Path

import env
from policy_gradient import Trainer

HERE = Path(__file__).resolve().parent


def round_stats(scored: list[dict], kinds: list[str]) -> dict:
    rewards = [s["reward"] for s in scored]
    canon = collections.Counter(s["canon"] for s in scored)
    by_kind = {}
    for k in sorted(set(kinds)):
        rs = [s["reward"] for s, kk in zip(scored, kinds) if kk == k]
        by_kind[k] = round(statistics.fmean(rs), 3)
    return {
        "mean_reward": round(statistics.fmean(rewards), 4),
        "median_reward": round(statistics.median(rewards), 4),
        "share_helpful": round(sum(r > 0 for r in rewards) / len(rewards), 3),
        "share_harmful": round(sum(r < 0 for r in rewards) / len(rewards), 3),
        "readable": round(sum(s["readable"] for s in scored) / len(scored), 3),
        "no_change": round(sum(s["canon"] == "no change" for s in scored) / len(scored), 3),
        "distinct_share": round(len(canon) / len(scored), 3),
        "top_answer_share": round(canon.most_common(1)[0][1] / len(scored), 3),
        "top_answers": canon.most_common(4),
        "by_kind": by_kind,
    }


def collect(sampler, keys: list[str], n: int, temperature: float, pool) -> tuple[list, list[dict], list[str]]:
    """Sample n ideas per practice world and score them. Returns (triples, scored, kinds)."""
    prompts = [env.PRACTICE[k].prompt for k in keys]
    groups = sampler.sample(prompts, n, temperature=temperature, max_tokens=260)
    flat = [(k, c) for k, g in zip(keys, groups) for c in g]
    scored = pool.starmap(env.score_text, [(k, c.text) for k, c in flat])
    triples = [(env.PRACTICE[k].prompt, c, s["reward"]) for (k, c), s in zip(flat, scored)]
    kinds = [env.PRACTICE[k].kind for k, _ in flat]
    return triples, scored, kinds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--trial", action="store_true", help="one practice world, 3 rounds of 64 episodes")
    ap.add_argument("--rounds", type=int, default=31)
    ap.add_argument("--trial-rounds", type=int, default=3)
    ap.add_argument("--worlds-per-round", type=int, default=8)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reward-scale", type=float, default=0.01, help="reward (in %) is multiplied by this before the update")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env.PRACTICE.update(env.build_practice(only_first=a.trial))
    keys_all = sorted(env.PRACTICE)
    print(f"practice worlds: {len(keys_all)} ({sum(p.kind == 'real' for p in env.PRACTICE.values())} real cuts, "
          f"{sum(p.kind == 'placebo' for p in env.PRACTICE.values())} shuffled) built in {time.time() - t0:.0f}s", flush=True)
    rng = random.Random(a.seed)
    pool = mp.get_context("fork").Pool(a.workers)
    trainer = Trainer(learning_rate=a.lr, system_prompt=env.system_prompt(), seed=a.seed,
                      session_name=f"gpuval-rl-{a.name}")
    rounds = a.trial_rounds if a.trial else a.rounds
    log = {"args": vars(a), "base_model": trainer.base_model, "rounds": []}
    for r in range(rounds):
        if a.trial:
            keys, n = keys_all[:1], a.worlds_per_round * a.samples
        else:
            keys, n = rng.sample(keys_all, min(a.worlds_per_round, len(keys_all))), a.samples
        triples, scored, kinds = collect(trainer, keys, n, a.temperature, pool)
        st = round_stats(scored, kinds)
        up = trainer.update([(p, c, rew * a.reward_scale) for p, c, rew in triples])
        st.update(round=r, episodes=len(scored), skipped=up.get("skipped"), update_seconds=round(up.get("seconds", 0), 1))
        log["rounds"].append(st)
        print(f"round {r:2d}: mean reward {st['mean_reward']:+.3f}% | helpful {st['share_helpful']:.2f} harmful {st['share_harmful']:.2f} "
              f"| readable {st['readable']:.2f} | no change {st['no_change']:.2f} | distinct {st['distinct_share']:.2f} "
              f"| top answer {st['top_answer_share']:.2f} | {st['by_kind']}", flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))
    # control on fresh samples from every practice world: untrained base against the trained weights
    ckpt = trainer.save(f"{a.name}-final")
    base = trainer.base_sampler()
    ctl = {}
    for label, sampler in (("base", base), ("trained", trainer)):
        _, scored, kinds = collect(sampler, keys_all, 8 if not a.trial else 64, 1.0, pool)
        ctl[label] = round_stats(scored, kinds)
        print(f"control {label:7s}: mean reward {ctl[label]['mean_reward']:+.3f}% | helpful {ctl[label]['share_helpful']:.2f} "
              f"| no change {ctl[label]['no_change']:.2f} | distinct {ctl[label]['distinct_share']:.2f} | {ctl[label]['by_kind']}", flush=True)
    log.update(checkpoint=ckpt, control=ctl, usage=trainer.usage, cost_estimate_usd=trainer.cost(),
               minutes=round((time.time() - t0) / 60, 1))
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"checkpoint {ckpt}\nusage {trainer.usage} cost about ${trainer.cost()}\nwrote {out / 'log.json'}")
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
