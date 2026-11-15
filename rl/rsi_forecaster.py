"""A self-improvement loop with RL training inside it (on Tinker).

    cd rl && uv run python rsi_forecaster.py --name rsi1 --base-model Qwen/Qwen3.5-9B --until 13:28
    cd rl && uv run python eval_forecaster.py --name rsi1 --base-from fc2     # the kept weights on our prices

Each generation:
  1. TRAIN     the kept weights practise for `--rounds` rounds (16 prices x 8 answers a round, one
               policy-gradient step each) at this generation's learning rate. Practice is AWS spot
               prices up to 30 Jun 2025, as in train_forecaster.py.
  2. VALIDATE  the new weights answer a fixed watch set of later AWS dates that training never sees.
  3. ACCEPT    the new weights replace the kept ones only if they score higher on the watch set and
               still do in at least 90% of resamples of its price-days. Otherwise they are thrown away.
  4. CHANGE    after a rejection the next generation restarts from the kept weights with the learning
               rate halved or doubled, within bounds. One kind of thing changes in a generation: the weights.
  5. ARCHIVE   every generation goes into runs/<name>/log.json.
It stops at --until (local time), after --generations, or after --plateau rejections in a row. Our own
37 prices are never used here: they are the test, run once on the kept weights after the loop ends.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

import forecast_env as env
from policy_gradient import Trainer, estimate_cost

HERE = Path(__file__).resolve().parent
WATCH_N, WATCH_ANSWERS = 128, 4
ACCEPT_P = 0.9
LR_BOUNDS = (1e-5, 1.6e-4)


def watch_scores(sampler, items: list[dict]) -> tuple[np.ndarray, float]:
    """Mean reward per watch price-day (percent less error than drift), and the share of readable answers."""
    groups = sampler.sample([it["prompt"] for it in items], WATCH_ANSWERS, temperature=1.0, max_tokens=90)
    per, ok = [], []
    for it, g in zip(items, groups):
        sc = [env.reward(it, c.text) for c in g]
        per.append(statistics.fmean(x[0] for x in sc)); ok += [x[1] for x in sc]
    return np.array(per), sum(ok) / len(ok)


def p_better(new: np.ndarray, old: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    """Share of resamples of the watch price-days in which the new weights score higher."""
    d = new - old
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    return float((d[idx].mean(axis=1) > 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=8, help="training rounds in one generation")
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--adv-std", type=float, default=0.02)
    ap.add_argument("--plateau", type=int, default=4)
    ap.add_argument("--until", default=None, help="local HH:MM; no generation starts after it")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    until = dt.datetime.combine(dt.date.today(), dt.time.fromisoformat(a.until)) if a.until else None
    rng, rng_lr = random.Random(a.seed), random.Random(f"lr|{a.seed}")
    world = env.aws_world()
    watch = env.sample_items(world, WATCH_N, env.CHECK_FROM, env.CHECK_TO, random.Random(123))
    session = f"gpuval-rsi-{a.name}"
    trainer = Trainer(a.base_model, learning_rate=a.lr, system_prompt=env.SYSTEM, seed=a.seed, session_name=session)
    usage = {k: 0 for k in trainer.usage}
    t0 = time.time()

    def spent(tr) -> dict:
        return {k: usage[k] + tr.usage[k] for k in usage}

    scores, readable = watch_scores(trainer, watch)
    kept = {"generation": 0, "watch": float(scores.mean()), "scores": scores, "sampler": trainer.save(f"{a.name}-g0", with_state=True),
            "state": trainer.last_state_path}
    log = {"args": vars(a), "base_model": trainer.base_model, "watch": {"n": WATCH_N, "answers": WATCH_ANSWERS, "from": env.CHECK_FROM, "to": env.CHECK_TO},
           "accept_p": ACCEPT_P, "start": {"watch": kept["watch"], "readable": readable}, "generations": [], "stop": "generations"}
    print(f"untrained: watch {kept['watch']:+.4f} (readable {readable:.2f}) on {WATCH_N} price-days x {WATCH_ANSWERS} answers", flush=True)
    lr, quiet = a.lr, 0
    for g in range(1, a.generations + 1):
        if until and dt.datetime.now() >= until:
            log["stop"] = "time"
            break
        tg, rounds = time.time(), []
        for _ in range(a.rounds):                                           # 1. TRAIN
            items = env.sample_items(world, a.prompts, "2022-08-05", env.TRAIN_END, rng)
            groups = trainer.sample([it["prompt"] for it in items], a.samples, temperature=1.0, max_tokens=90)
            triples, rs, spread = [], [], []
            for it, grp in zip(items, groups):
                sc = [env.reward(it, c.text)[0] for c in grp]
                spread += [x - statistics.fmean(sc) for x in sc]
                for c, rew in zip(grp, sc):
                    triples.append((it["prompt"], c, rew)); rs.append(rew)
            sd = float(np.std(spread)) or 1.0
            trainer.update([(p, c, rew * a.adv_std / sd) for p, c, rew in triples], learning_rate=lr)
            rounds.append(round(statistics.fmean(rs), 4))
        scores, readable = watch_scores(trainer, watch)                     # 2. VALIDATE
        gain, p = float(scores.mean() - kept["watch"]), p_better(scores, kept["scores"])
        accepted = gain > 0 and p >= ACCEPT_P                               # 3. ACCEPT
        row = {"generation": g, "from_generation": kept["generation"], "learning_rate": lr, "rounds": rounds, "train_reward": round(statistics.fmean(rounds), 4),
               "watch": float(scores.mean()), "watch_kept_before": kept["watch"], "gain": gain, "p_better": p, "readable": readable, "accepted": accepted}
        if accepted:
            sampler = trainer.save(f"{a.name}-g{g}", with_state=True)
            kept = {"generation": g, "watch": float(scores.mean()), "scores": scores, "sampler": sampler, "state": trainer.last_state_path}
            quiet = 0
        else:                                                               # 4. CHANGE: back to the kept weights, another learning rate
            quiet += 1
            lr = float(min(LR_BOUNDS[1], max(LR_BOUNDS[0], lr * rng_lr.choice((0.5, 2.0)))))
            usage = spent(trainer)
            trainer = Trainer(a.base_model, learning_rate=lr, system_prompt=env.SYSTEM, resume_from=kept["state"], session_name=session)
        row.update(kept_generation=kept["generation"], next_learning_rate=lr, seconds=round(time.time() - tg))
        log["generations"].append(row)                                      # 5. ARCHIVE
        log.update(checkpoint=kept["sampler"], kept_generation=kept["generation"], kept_watch=kept["watch"], usage=spent(trainer),
                   cost_estimate_usd=estimate_cost(spent(trainer), trainer.base_model), minutes=round((time.time() - t0) / 60, 1))
        (out / "log.json").write_text(json.dumps(log, indent=1))
        print(f"gen {g}: lr {row['learning_rate']:.1e} | train reward {row['train_reward']:+.3f} | watch {row['watch']:+.4f} against kept {row['watch_kept_before']:+.4f} "
              f"(gain {gain:+.4f}, {p:.2f} of resamples) | {'ACCEPTED' if accepted else 'rolled back'} | {row['seconds']} s | about ${log['cost_estimate_usd']:.2f} so far", flush=True)
        if quiet >= a.plateau:
            log["stop"] = "plateau"
            break
    log.update(checkpoint=kept["sampler"], kept_generation=kept["generation"], kept_watch=kept["watch"], usage=spent(trainer),
               cost_estimate_usd=estimate_cost(spent(trainer), trainer.base_model), minutes=round((time.time() - t0) / 60, 1))
    (out / "log.json").write_text(json.dumps(log, indent=1))
    n_acc = sum(r["accepted"] for r in log["generations"])
    print(f"stopped ({log['stop']}): {n_acc} of {len(log['generations'])} generations accepted; kept generation {kept['generation']}, watch "
          f"{log['start']['watch']:+.4f} -> {kept['watch']:+.4f}; checkpoint {kept['sampler']}; about ${log['cost_estimate_usd']:.2f}, {log['minutes']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
