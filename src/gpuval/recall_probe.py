"""Does the agents' model already remember the AWS prices it was tested on?

    uv run python -m gpuval.recall_probe run       # about 280 AI calls (OPENAI_API_KEY)
    uv run python -m gpuval.recall_probe table     # re-score the saved answers, no AI calls

The agents forecast AWS spot prices on 12 sampled days, 11 of them before April 2026, where their
model's knowledge ends. If the model can recite those prices, that test was not a fair one.

The probe shows no evidence at all. It names a machine type and a date and asks what the price was
(median across zones, USD per GPU-hour) and whether it was higher, lower or the same a week later.
"I don't know" is allowed and a best guess is still asked for. Dates after April 2026, which the
model cannot have seen, are asked in exactly the same way and act as the control.

The rule for reading the answers is in PROTOCOL.md, deviation 12, committed before the first run.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import agents, aws_test, data
from .core import day_of
from .llm import Meter, ask_json

OUT = data.REPO / "runs" / "recall"
KNOWLEDGE_END = "2026-04-30"
BEFORE = aws_test.AGENT_DAYS[:11]                                       # the agents' AWS days inside the model's knowledge
AFTER = ["2026-07-24", "2026-08-21", "2026-09-04", "2026-10-02"]        # the four AWS days after it
MOVE = 0.01                    # a week-on-week change smaller than 1% counts as "same"
N_PERM = 5000

SYSTEM = """You are asked what you REMEMBER, not to forecast. No data is shown.

The question is about the AWS EC2 spot price of one GPU machine type (Linux/UNIX) on one date: the median across availability zones, in USD per GPU-hour (the instance's hourly spot price divided by its number of GPUs).

If you do not actually remember this price for this date, set remembers to false. Give your best guess of the price either way. For the week that followed, say whether the price was higher, lower or about the same (within 1%) seven days later, or dont_know. Reply with JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "remembers": {"type": "boolean", "description": "true only if you recall this price for this date, not a general level"},
        "price_usd_per_gpu_hour": {"type": "number"},
        "week_later": {"type": "string", "enum": ["higher", "lower", "same", "dont_know"]},
        "basis": {"type": "string", "description": "One short sentence: what the number is based on."},
    },
    "required": ["remembers", "price_usd_per_gpu_hour", "week_later", "basis"],
    "additionalProperties": False,
}


def items() -> list[dict]:
    world = data.load_world(aws_test.WORK / "world.pkl")
    out = []
    for iso in BEFORE + AFTER:
        d = day_of(iso)
        for sid, s in world["series"].items():
            i = int(np.searchsorted(s["days"], d))
            k = int(np.searchsorted(s["days"], d + 7))
            if i >= len(s["days"]) or s["days"][i] != d or k >= len(s["days"]) or s["days"][k] != d + 7:
                continue
            out.append({"id": f"{sid}|{iso}", "series_id": sid, "day": iso, "machine": s["meta"]["gpu_model"],
                        "after_knowledge": iso > KNOWLEDGE_END, "price": float(s["value"][i]), "price_week_later": float(s["value"][k])})
    return out


def run(budget: float = 3.0) -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(data.REPO / ".env")
    client = OpenAI()
    meter = Meter(budget)
    its = items()

    def one(it):
        user = f"Machine type: {it['machine']}\nDate: {it['day']}\n\nWhat was the median spot price across zones that day, in USD per GPU-hour, and where was it a week later?"
        return {**it, "answer": ask_json(client, agents.MODEL, agents.EFFORT, SYSTEM, user, SCHEMA, "recall", meter)}

    with ThreadPoolExecutor(max_workers=24) as ex:
        rows = list(ex.map(one, its))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answers.json").write_text(json.dumps({"model": agents.MODEL, "effort": agents.EFFORT, "usage": meter.summary(), "rows": rows}, indent=1))
    print(f"{len(rows)} questions asked | usage {meter.summary()}")
    table()


def _date_specific(g: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Are the answers closer to the truth of their OWN date than to the truth of other dates for the
    same machine type? Shuffle the dates within each machine type and compare."""
    err = float(np.median(np.abs(np.log(g.answer / g.price))))
    groups = [(grp.answer.to_numpy(), grp.price.to_numpy()) for _, grp in g.groupby("series_id") if len(grp) >= 2]
    sims = np.empty(N_PERM)
    for b in range(N_PERM):
        sims[b] = np.median(np.concatenate([np.abs(np.log(a / rng.permutation(p))) for a, p in groups]))
    return {"median_abs_log_error": err, "same_after_shuffling_dates": float(np.median(sims)),
            "p_closer_to_own_date": float((1 + np.sum(sims <= err)) / (N_PERM + 1))}


def table() -> dict:
    saved = json.loads((OUT / "answers.json").read_text())
    df = pd.DataFrame([{**{k: r[k] for k in ("series_id", "day", "machine", "after_knowledge", "price", "price_week_later")},
                        "remembers": r["answer"]["remembers"], "answer": float(r["answer"]["price_usd_per_gpu_hour"]),
                        "week_later": r["answer"]["week_later"]} for r in saved["rows"]])
    df = df[df.answer > 0]
    chg = df.price_week_later / df.price - 1
    df["truth_dir"] = np.where(chg > MOVE, "higher", np.where(chg < -MOVE, "lower", "same"))
    rng = np.random.default_rng(0)
    out = {"model": saved["model"], "usage": saved["usage"], "blocks": {}}
    for name, g in (("before April 2026 (11 days)", df[~df.after_knowledge]), ("after April 2026 (4 days)", df[df.after_knowledge])):
        said = g[g.week_later != "dont_know"]
        moved = said[said.truth_dir != "same"]
        hits = int((moved.week_later == moved.truth_dir).sum())
        n = len(moved)
        lo, hi = (float("nan"), float("nan"))
        if n:
            draws = rng.binomial(n, hits / n, size=4000) / n
            lo, hi = (float(x) for x in np.quantile(draws, [0.05, 0.95]))
        rel = np.abs(g.answer / g.price - 1)
        out["blocks"][name] = {
            "questions": int(len(g)), "says_it_remembers": float(g.remembers.mean()),
            "price_within_10pct": float((rel <= 0.10).mean()), "price_within_25pct": float((rel <= 0.25).mean()),
            "median_abs_error_pct": float(100 * rel.median()), **_date_specific(g, rng),
            "direction_dont_know": float((g.week_later == "dont_know").mean()),
            "direction_answered_where_price_moved": n, "direction_right": hits,
            "direction_right_share": hits / n if n else None, "direction_right_90pct": [lo, hi],
            "share_of_weeks_that_moved": float((g.truth_dir != "same").mean())}
    (data.REPO / "findings" / "data" / "recall_probe.json").write_text(json.dumps(out, indent=1))
    for name, b in out["blocks"].items():
        print(f"{name}: {b['questions']} questions | says it remembers {b['says_it_remembers']:.2f} | price within 10% {b['price_within_10pct']:.2f}, within 25% {b['price_within_25pct']:.2f}, "
              f"median error {b['median_abs_error_pct']:.0f}%")
        print(f"    own date vs shuffled dates: error {b['median_abs_log_error']:.3f} vs {b['same_after_shuffling_dates']:.3f} (p = {b['p_closer_to_own_date']:.3f})")
        print(f"    week later: 'don't know' {b['direction_dont_know']:.2f}; where it answered and the price moved: right {b['direction_right']} of {b['direction_answered_where_price_moved']}"
              + (f" ({b['direction_right_share']:.2f}, 90% {b['direction_right_90pct'][0]:.2f} to {b['direction_right_90pct'][1]:.2f})" if b["direction_answered_where_price_moved"] else ""))
    return out


if __name__ == "__main__":
    {"run": run, "table": table}[sys.argv[1]]()
