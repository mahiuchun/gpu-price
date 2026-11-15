"""Test the open model before and after RL training, on days it never practised on.

    cd rl && uv run python eval_forecaster.py --name fc1            # checkpoint read from runs/fc1/log.json
    cd rl && uv run python eval_forecaster.py --name fc1 --table    # re-print the table from saved answers

Test days
- our own 37 prices, every day from 24 Sep to 5 Nov 2026 (the model never trained on any of them)
- AWS spot prices, Fridays from 27 Feb to 23 Oct 2026 (after the training dates, which end 30 Jun 2025,
  and after the dates used to watch training, which end 20 Feb 2026)

Each model gives 8 answers per price-day and the forecast is their average. Scored against drift and
the engine everywhere, and against the GPT crowd on the days where the crowd answered the same question.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_env as env
from gpuval import coach, data, report
from gpuval.core import HORIZONS, cell_loss, day_of
from policy_gradient import Sampler, estimate_cost

HERE = Path(__file__).resolve().parent
N = 8
CHUNK = 150


def span(lo: str, hi: str, step: int = 1) -> list[str]:
    a, b = dt.date.fromisoformat(lo), dt.date.fromisoformat(hi)
    return [(a + dt.timedelta(days=i)).isoformat() for i in range(0, (b - a).days + 1, step)]


DAYS = {"wide": {"ours": span("2026-09-24", "2026-11-05"), "aws": span("2026-02-27", "2026-10-23", 7)}}


def answers_for(sampler: Sampler, items: list[dict], label: str) -> dict:
    out = {}
    for k in range(0, len(items), CHUNK):
        part = items[k:k + CHUNK]
        groups = sampler.sample([it["facts"] + "\n\nGive your forecast as JSON." for it in part], N, temperature=1.0, max_tokens=90)
        for it, g in zip(part, groups):
            qs = [env.parse(c.text, it["last"]) for c in g]
            out[it["id"]] = [{str(h): list(q[h]) for h in HORIZONS} if q else None for q in qs]
        print(f"{label}: {min(k + CHUNK, len(items))}/{len(items)}", flush=True)
    return out


def gpt_crowd() -> dict:
    """The GPT crowd's saved round-one answers, by (world, series, day)."""
    out = {}
    for world, files in (("ours", ("test_answers", "extra_answers")), ("aws", ("aws_answers",))):
        for f in files:
            for e in json.loads((data.REPO / "runs" / "agents" / f"{f}.json").read_text()):
                out[f"{world}|{e['series_id']}|{e['origin']}"] = e["crowd1"]
    return out


def cells(items: list[dict], ans: dict) -> pd.DataFrame:
    crowd = gpt_crowd()
    rows = []
    for it in items:
        per = {}
        for label in ("base", "trained"):
            qs = [{h: tuple(q[str(h)]) for h in HORIZONS} for q in ans[label][it["id"]] if q]
            per[label] = (env.combine(qs) if qs else None, qs, len(ans[label][it["id"]]))
        for h, y in it["y"].items():
            h = int(h)
            d = cell_loss(y, *it["refs"]["drift"][h])
            row = {"id": it["id"], "world": it["world"], "series_id": it["series_id"], "origin": it["day"], "h": h, "target": day_of(it["day"]) + h,
                   "last": it["last"], "y": y, "drift": d, "engine": cell_loss(y, *it["refs"]["engine"][h])}
            for label, (avg, qs, n) in per.items():
                row[label] = cell_loss(y, *avg[h]) if avg else d
                row[label + "_single"] = float(np.mean([cell_loss(y, *q[h]) for q in qs] + [d] * (n - len(qs))))
                row[label + "_inside"] = bool(avg[h][0] <= y <= avg[h][2]) if avg else True
                row[label + "_lo"], row[label + "_mid"], row[label + "_hi"] = avg[h] if avg else it["refs"]["drift"][h]
                row[label + "_readable"] = len(qs) / n
            g = crowd.get(it["id"])
            row["gpt"] = cell_loss(y, *g[str(h)]) if g else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("drift", "engine", "base", "trained", "base_single", "trained_single", "gpt"):
        df["rel_" + c] = df[c] / df["last"]
    return df


def table(df: pd.DataFrame) -> dict:
    out = {}
    for name, g, pre in (("ours", df[df.world == "ours"], ""), ("aws", df[df.world == "aws"], ""), ("all", df, "rel_")):
        ci = lambda a, b, frame=g: report.skill_ci(frame, pre + a, pre + b)
        has = g[g.gpt.notna()]
        out[name] = {"n": int(len(g)), "n_days": int(g.origin.nunique()),
                     "base_vs_drift": ci("base", "drift"), "trained_vs_drift": ci("trained", "drift"), "trained_vs_base": ci("trained", "base"),
                     "one_answer_base_vs_drift": ci("base_single", "drift"), "one_answer_trained_vs_drift": ci("trained_single", "drift"),
                     "engine_vs_drift": ci("engine", "drift"), "trained_vs_engine": ci("trained", "engine"),
                     "inside": {"base": float(g.base_inside.mean()), "trained": float(g.trained_inside.mean())},
                     "readable": {"base": float(g.base_readable.mean()), "trained": float(g.trained_readable.mean())},
                     "on_gpt_days": {"n": int(len(has)), "n_days": int(has.origin.nunique()),
                                     "gpt_vs_drift": ci("gpt", "drift", has), "trained_vs_drift": ci("trained", "drift", has),
                                     "trained_vs_gpt": ci("trained", "gpt", has), "engine_vs_drift": ci("engine", "drift", has)} if len(has) else None,
                     "by_h": {int(h): {"trained_vs_drift": ci("trained", "drift", gh), "base_vs_drift": ci("base", "drift", gh)} for h, gh in g.groupby("h")}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--base-from", default=None, help="reuse the untrained model's saved answers from this run (same base model)")
    a = ap.parse_args()
    out = HERE / "runs" / a.name
    log = json.loads((out / "log.json").read_text())
    items = list(coach.build_items(DAYS).values())
    print(f"{len(items)} price-days: ours {sum(i['world'] == 'ours' for i in items)}, aws {sum(i['world'] == 'aws' for i in items)}", flush=True)
    path = out / "eval_answers.json"
    if a.table:
        ans = json.loads(path.read_text())
    else:
        ckpt = a.checkpoint or log["checkpoint"]
        trained = Sampler(checkpoint=ckpt, system_prompt=env.SYSTEM)
        if a.base_from:      # the untrained model's answers were saved by an earlier run of the same base model
            prev = json.loads((HERE / "runs" / a.base_from / "log.json").read_text())
            assert prev["base_model"] == log["base_model"]
            ans = {"checkpoint": ckpt, "base": json.loads((HERE / "runs" / a.base_from / "eval_answers.json").read_text())["base"],
                   "trained": answers_for(trained, items, "after training")}
            usage = dict(trained.usage)
        else:
            base = Sampler(base_model=log["base_model"], system_prompt=env.SYSTEM)
            ans = {"checkpoint": ckpt, "base": answers_for(base, items, "before training"), "trained": answers_for(trained, items, "after training")}
            usage = {k: base.usage[k] + trained.usage[k] for k in base.usage}
        ans["usage"], ans["cost_estimate_usd"] = usage, estimate_cost(usage, log["base_model"])
        path.write_text(json.dumps(ans))
    df = cells(items, ans)
    df.to_parquet(out / "eval_cells.parquet", index=False)
    t = table(df)
    (out / "eval.json").write_text(json.dumps({"checkpoint": ans["checkpoint"], "n_answers_per_price_day": N, "usage": ans.get("usage"),
                                                "cost_estimate_usd": ans.get("cost_estimate_usd"), "table": t}, indent=1))
    f = report.fmt_ci
    for name, b in t.items():
        print(f"\n== {name}: {b['n']} forecasts on {b['n_days']} days")
        print(f"  before training {f(b['base_vs_drift'])} | after training {f(b['trained_vs_drift'])} | after vs before {f(b['trained_vs_base'])}")
        print(f"  one answer only: before {f(b['one_answer_base_vs_drift'])} | after {f(b['one_answer_trained_vs_drift'])}")
        print(f"  engine {f(b['engine_vs_drift'])} | after training vs engine {f(b['trained_vs_engine'])}")
        print(f"  truth inside range: before {b['inside']['base']:.2f}, after {b['inside']['trained']:.2f} | readable: before {b['readable']['base']:.3f}, after {b['readable']['trained']:.3f}")
        g = b["on_gpt_days"]
        if g:
            print(f"  on the {g['n_days']} days the GPT crowd answered ({g['n']}): GPT crowd {f(g['gpt_vs_drift'])} | after training {f(g['trained_vs_drift'])} "
                  f"| after training vs GPT crowd {f(g['trained_vs_gpt'])} | engine {f(g['engine_vs_drift'])}")
        print("  by horizon: " + "; ".join(f"{h}d before {f(v['base_vs_drift'])}, after {f(v['trained_vs_drift'])}" for h, v in b["by_h"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
