"""Smoke test: toy single-step RL on Tinker using policy_gradient.Trainer (real API calls).

Task: the model must emit {"x": <number in [0, 1]>}. Reward = -(x - 0.7)^2, or -1 if the output
is not such a JSON object. The prompts never mention 0.7, so the only way to improve is RL.

    uv run python smoke_rl.py            # ~2 minutes, about 1 cent
    uv run python smoke_rl.py --lr 3e-4 --rounds 6 --n-per-prompt 4
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time

from policy_gradient import DEFAULT_BASE_MODEL, Completion, Sampler, Trainer, estimate_cost

TARGET = 0.7
INVALID_REWARD = -1.0

# Worded to make the base model spread its answers: asked plainly for "a number between 0 and 1"
# it answers {"x": 0.5} every single time at temperature 1, and a group whose rewards are all
# equal has zero advantage, i.e. no gradient.
PROMPTS = [
    "You are searching for the best value of a hidden setting x between 0 and 1. Make a guess; "
    "guesses should vary from attempt to attempt, so do not just pick 0.5. Use two decimal places. "
    'Reply with only a JSON object {"x": <number>} and nothing else.',
    "Pick a random number between 0 and 1 with two decimal places. Reply with only a JSON object "
    'of the form {"x": <number>} and nothing else.',
    "We are tuning a setting x in [0, 1] by trial and error; the best value is unknown. Propose the "
    'next value to try (two decimals, be exploratory). Output only JSON: {"x": <number>}.',
    "Roll a 100-sided die in your head, divide the result by 100, and report it. "
    'Output only JSON: {"x": <number>}.',
]


def parse_x(text: str) -> float | None:
    """The x of a {"x": number} object with 0 <= x <= 1, else None."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else text)["x"]
    except (ValueError, KeyError, TypeError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        return None
    return float(value)


def reward_fn(text: str) -> float:
    x = parse_x(text)
    return INVALID_REWARD if x is None else -((x - TARGET) ** 2)


def summarise(groups: list[list[Completion]]) -> dict[str, float]:
    xs = [parse_x(c.text) for group in groups for c in group]
    valid = [x for x in xs if x is not None]
    return {
        "mean_reward": statistics.fmean(reward_fn(c.text) for group in groups for c in group),
        "parse_rate": len(valid) / len(xs),
        "mean_x": statistics.fmean(valid) if valid else float("nan"),
        "distinct": len({c.text for group in groups for c in group}),  # 1 per prompt = policy has collapsed
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--rounds", type=int, default=8, help="number of policy-gradient updates")
    parser.add_argument("--n-per-prompt", type=int, default=8, help="samples per prompt per round")
    parser.add_argument("--eval-n", type=int, default=16, help="samples per prompt in the final control comparison")
    parser.add_argument("--temperature", type=float, default=1.0, help="training-sample temperature (controls use 1.0)")
    parser.add_argument("--loss", default="importance_sampling")
    parser.add_argument("--normalize-advantages", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    trainer = Trainer(
        args.model,
        lora_rank=args.rank,
        learning_rate=args.lr,
        loss_fn=args.loss,
        normalize_advantages=args.normalize_advantages,
        session_name="rl-smoke",
    )
    print(
        f"model={args.model} rank={args.rank} lr={args.lr} loss={args.loss} temperature={args.temperature} "
        f"samples/round={len(PROMPTS)}x{args.n_per_prompt} setup={time.time() - t0:.1f}s"
    )

    nan = float("nan")
    history, sample_secs, train_secs, refresh_secs = [], [], [], []
    for round_idx in range(args.rounds + 1):  # the last pass only measures the final policy
        t_sample = time.time()
        groups = trainer.sample(PROMPTS, args.n_per_prompt, temperature=args.temperature, max_tokens=32)
        sample_secs.append(time.time() - t_sample)
        summary = summarise(groups)
        history.append(summary["mean_reward"])
        line = (
            f"round {round_idx}: mean_reward={summary['mean_reward']:+.4f} "
            f"parse_rate={summary['parse_rate']:.2f} mean_x={summary['mean_x']:.3f} "
            f"distinct={summary['distinct']} sample={sample_secs[-1]:.1f}s"
        )
        if round_idx < args.rounds:
            batch = [
                (prompt, completion, reward_fn(completion.text))
                for prompt, group in zip(PROMPTS, groups)
                for completion in group
            ]
            stats = trainer.update(batch)
            if stats["skipped"]:
                line += " update=skipped (every group has identical rewards: no signal)"
            else:
                train_secs.append(stats["fwd_bwd_optim_seconds"])
                refresh_secs.append(stats["refresh_sampler_seconds"])
                line += (
                    f" update={stats['seconds']:.1f}s datums={stats['n_datums']:.0f} "
                    f"logprob_gap={stats['sampler_learner_logprob_gap']:+.4f}"
                )
        print(line, flush=True)

    checkpoint = trainer.save(f"smoke-{int(time.time())}", ttl_seconds=24 * 3600)
    print(f"saved {checkpoint} (expires in 24h)")

    # Control: untrained base model vs the checkpoint re-loaded from its path, same prompts.
    base = summarise(trainer.base_sampler().sample(PROMPTS, args.eval_n, max_tokens=32))
    reloaded = Sampler(checkpoint=checkpoint)
    tuned = summarise(reloaded.sample(PROMPTS, args.eval_n, max_tokens=32))
    for name, s in (("base   ", base), ("trained", tuned)):
        print(f"control {name}: mean_reward={s['mean_reward']:+.4f} parse_rate={s['parse_rate']:.2f} mean_x={s['mean_x']:.3f}")

    usage = {k: trainer.usage[k] + reloaded.usage[k] for k in trainer.usage}
    cost = estimate_cost(usage, args.model)
    print("rewards by round: " + " ".join(f"{r:+.4f}" for r in history))
    print(
        f"timing: sample batch median {statistics.median(sample_secs):.1f}s; update median "
        f"{statistics.median(train_secs or [nan]):.1f}s fwd_bwd+optim + "
        f"{statistics.median(refresh_secs or [nan]):.1f}s sampler refresh; total {time.time() - t0:.0f}s"
    )
    print(f"tokens: {usage}  estimated cost: " + (f"${cost:.4f}" if cost is not None else "n/a (no price entry)"))
    improved = history[-1] > history[0] and tuned["mean_reward"] > base["mean_reward"]
    print("RESULT: " + ("improved" if improved else "NOT improved") + f" (round 0 {history[0]:+.4f} -> final {history[-1]:+.4f})")


if __name__ == "__main__":
    main()
