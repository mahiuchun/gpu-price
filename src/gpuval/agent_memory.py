"""Do the agents forecast better when each one can see its own past forecasts and how they came out?

    uv run python -m gpuval.agent_memory run       # about 10,600 AI calls (OPENAI_API_KEY); resumes if stopped
    uv run python -m gpuval.agent_memory table     # re-score the saved answers, no AI calls

Three arms, the same five agents, model, instructions and facts in each. The 19 forecast days are
lived through in date order, so a memory only ever holds what was known on the day.

    none      the agent as first written (asked again now, so that all three arms are the same age)
    record    plus its own earlier forecasts for this price whose outcomes are known, each marked
              inside / below / above its range
    lesson    the record, plus a lesson of at most three sentences that the agent wrote for itself
              from its whole scored record (rewritten before each forecast day)

Each arm's memory holds that arm's own forecasts. The recorded answers of 14 Nov are a fourth column:
"none" against them shows how much the numbers move when the same agents are simply asked again.

The rule for reading the result is in PROTOCOL.md, deviation 12, committed before the first run.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import agents, coach, data, harness, range_memory, report
from .core import HORIZONS, cell_loss, day_of, iso_of
from .engine import default_spec
from .llm import Meter, ask_json

OUT = data.REPO / "runs" / "agent_memory"
DAYS = sorted(range_memory.SCORE_DAYS)
ARMS = ("none", "record", "lesson")
MAX_LINES = 12            # most recent resolved forecasts shown for a price
LESSON_MIN = 30           # an agent writes a lesson only once it has this many resolved forecasts
BUDGET_USD = 45.0

LESSON_SYSTEM = """You forecast published GPU rental prices: for each price you give low / mid / high for 1, 3 and 7 days ahead. The truth should fall inside your range about 80% of the time, below it about 10% and above it about 10%. A range that is too wide costs points when nothing moves; a range that misses costs more when the price jumps.

{role}

Below is your own record so far. Write the lesson you should carry into your next forecasts: at most three sentences. Draw it from the whole record, not from a single miss. Keep it general: do not name providers, GPUs, dates or prices. Reply with JSON only."""

LESSON_SCHEMA = {"type": "object", "properties": {"lesson": {"type": "string"}}, "required": ["lesson"], "additionalProperties": False}


# ------------------------------------------------------------ memory text ----
def resolved(store: dict, role: str, sid: str, today: int, truth: dict) -> list[dict]:
    """This agent's earlier forecasts for this price whose target day is today or earlier."""
    out = []
    for (r, s, o), ans in store.items():
        if r != role or s != sid or o >= today:
            continue
        for h in HORIZONS:
            y = truth.get((sid, o + h))
            if o + h <= today and y is not None:
                lo, mid, hi = ans["q"][str(h)]
                out.append({"origin": o, "h": h, "lo": lo, "mid": mid, "hi": hi, "y": y, "last": ans["last"],
                            "where": "inside" if lo - 1e-12 <= y <= hi + 1e-12 else ("below" if y < lo else "above")})
    return sorted(out, key=lambda x: (x["origin"] + x["h"], x["h"]))


def record_text(rows: list[dict]) -> str:
    if not rows:
        return "YOUR OWN EARLIER FORECASTS FOR THIS PRICE: none has an outcome yet."
    lines = ["YOUR OWN EARLIER FORECASTS FOR THIS PRICE (only those whose outcome is known today; latest last):"]
    for x in rows[-MAX_LINES:]:
        mark = {"inside": "inside your range", "below": "BELOW your range", "above": "ABOVE your range"}[x["where"]]
        lines.append(f"  made {iso_of(x['origin'])[5:]} for {iso_of(x['origin'] + x['h'])[5:]} ({x['h']}d): low {x['lo']:.4f} / mid {x['mid']:.4f} / high {x['hi']:.4f} "
                     f"-> came in at {x['y']:.4f}: {mark}")
    n = len(rows)
    c = {k: sum(x["where"] == k for x in rows) for k in ("inside", "below", "above")}
    moved = sum(abs(x["y"] - x["last"]) > 1e-9 for x in rows)
    lines.append(f"All {n} of your resolved forecasts for this price: inside {c['inside']}, below {c['below']}, above {c['above']}. "
                 f"The price had moved from the day you forecast in {moved} of {n}.")
    return "\n".join(lines)


def scorecard(store: dict, role: str, today: int, truth: dict, meta: dict) -> tuple[str, int]:
    """One agent's whole record across all prices, as text for writing a lesson."""
    rows = []
    for sid in {s for (r, s, _) in store if r == role}:
        for x in resolved(store, role, sid, today, truth):
            rows.append({**x, "sid": sid})
    if len(rows) < LESSON_MIN:
        return "", len(rows)
    df = pd.DataFrame(rows)
    df["width"] = 100 * (df.hi - df.lo) / df["last"]
    df["move"] = 100 * (df.y - df["last"]).abs() / df["last"]
    df["miss"] = 100 * np.maximum(df.lo - df.y, np.maximum(df.y - df.hi, 0)) / df["last"]
    lines = [f"YOUR RECORD SO FAR: {len(df)} forecasts with known outcomes, on {df.sid.nunique()} prices."]
    for h, g in df.groupby("h"):
        lines.append(f"  {h} day{'s' if h > 1 else ''} ahead ({len(g)}): truth inside your range {np.mean(g['where'] == 'inside'):.2f}, below {np.mean(g['where'] == 'below'):.2f}, "
                     f"above {np.mean(g['where'] == 'above'):.2f} (target 0.80 / 0.10 / 0.10). The price moved in {np.mean(g.move > 0):.2f} of cases; "
                     f"your middle differed from that day's price in {np.mean((g.mid - g['last']).abs() > 1e-9):.2f}. "
                     f"Your typical range was {g.width.median():.2f}% of the price wide; when the price moved, the typical move was {g.move[g.move > 0].median() if (g.move > 0).any() else 0:.2f}%.")
    lines.append("Your five largest misses (how far outside your range, as % of the price):")
    for x in df.sort_values("miss", ascending=False).head(5).itertuples():
        m = meta[x.sid]
        lines.append(f"  a {m['instrument_class']} price, {x.h}d ahead: your range was {100 * (x.lo / x.last - 1):+.2f}% to {100 * (x.hi / x.last - 1):+.2f}% around that day's price; "
                     f"it moved {100 * (x.y / x.last - 1):+.2f}% ({x.where} your range by {x.miss:.2f}%)")
    return "\n".join(lines), len(df)


# -------------------------------------------------------------------- run ----
def _load() -> dict:
    p = OUT / "answers.json"
    return json.loads(p.read_text()) if p.exists() else {"days_done": [], "answers": {a: {} for a in ARMS}, "lessons": {}, "usage": None}


def run() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(data.REPO / ".env")
    client = OpenAI()
    OUT.mkdir(parents=True, exist_ok=True)
    saved = _load()
    meter = Meter(BUDGET_USD)
    if saved["usage"]:
        meter.use.update(saved["usage"])
    items = coach.build_items({"memory": {"ours": DAYS}})
    full = data.build_world(data.DATA_END)
    truth = {(sid, int(d)): float(v) for sid, s in full["series"].items() for d, v in zip(s["days"], s["value"])}
    meta = {sid: s["meta"] for sid, s in full["series"].items()}
    roles = coach.base_config()["roles"]
    store = {a: {tuple(json.loads(k)): v for k, v in saved["answers"][a].items()} for a in ARMS}   # (role, sid, origin ordinal) -> answer
    t0 = time.time()
    for iso in DAYS:
        if iso in saved["days_done"]:
            continue
        d = day_of(iso)
        todays = [it for it in items.values() if it["day"] == iso]
        # 1. lessons for the lesson arm, one per agent, from its record up to today
        lessons = {}

        def write_lesson(name):
            card, n = scorecard(store["lesson"], name, d, truth, meta)
            if not card:
                return name, None
            prev = saved["lessons"].get(name, [])
            user = card + (f"\n\nThe lesson you wrote last time: {prev[-1]['lesson']}" if prev else "") + "\n\nWrite your lesson."
            ans = ask_json(client, agents.MODEL, agents.EFFORT, LESSON_SYSTEM.format(role=roles[name]["instruction"]), user, LESSON_SCHEMA, "lesson", meter)
            return name, {"day": iso, "n_resolved": n, "lesson": ans["lesson"].strip()}

        with ThreadPoolExecutor(max_workers=5) as ex:
            for name, les in ex.map(write_lesson, list(roles)):
                if les:
                    lessons[name] = les["lesson"]
                    saved["lessons"].setdefault(name, []).append(les)

        # 2. every agent, every price, three arms
        def one(job):
            arm, name, it = job
            role = roles[name]
            user = it["facts"] + ("\n" + agents.refs_text(it["refs"]) if role["refs"] else "")
            if arm in ("record", "lesson"):
                user += "\n\n" + record_text(resolved(store[arm], name, it["series_id"], d, truth))
            if arm == "lesson" and name in lessons:
                user += "\n\nA LESSON YOU WROTE FOR YOURSELF FROM YOUR WHOLE RECORD SO FAR: " + lessons[name]
            ans = ask_json(client, agents.MODEL, agents.EFFORT, agents.SYSTEM.format(role=role["instruction"]), user + "\n\nGive your forecast.",
                           agents.SCHEMA, "forecast", meter)
            q = agents.clean(ans, it["last"])
            return arm, name, it["series_id"], {"q": {str(h): list(q[h]) for h in HORIZONS}, "reason": ans.get("reason", ""), "last": it["last"]}

        jobs = [(arm, name, it) for arm in ARMS for name in roles for it in todays]
        with ThreadPoolExecutor(max_workers=48) as ex:
            got = list(ex.map(one, jobs))
        for arm, name, sid, ans in got:          # today's answers enter the memory only after the whole day is answered
            store[arm][(name, sid, d)] = ans
        saved["answers"] = {a: {json.dumps(list(k)): v for k, v in store[a].items()} for a in ARMS}
        saved["days_done"].append(iso)
        saved["usage"] = meter.summary()
        (OUT / "answers.json").write_text(json.dumps(saved))
        print(f"{iso}: {len(todays)} prices x {len(roles)} agents x {len(ARMS)} arms, {len(lessons)} lessons | {(time.time() - t0) / 60:.1f} min | {meter.summary()}", flush=True)
    table()


# ------------------------------------------------------------------ score ----
def table() -> dict:
    saved = _load()
    full = data.build_world(data.DATA_END)
    cells = harness.run_spec(full, default_spec(), harness.base_cells(full)).rename(columns={"loss": "loss_engine"})
    roles = list(coach.base_config()["roles"])
    rows = []
    by_arm = {a: {tuple(json.loads(k)): v for k, v in saved["answers"][a].items()} for a in ARMS}
    for arm in ARMS:
        keys = sorted({(sid, o) for (_, sid, o) in by_arm[arm]})
        for sid, o in keys:
            if not all((r, sid, o) in by_arm[arm] for r in roles):
                continue
            for h in HORIZONS:
                qs = np.array([by_arm[arm][(r, sid, o)]["q"][str(h)] for r in roles], dtype=float)
                m = np.exp(np.mean(np.log(qs), axis=0))
                row = {"arm": arm, "series_id": sid, "origin": o, "h": h, "lo": min(m[0], m[1]), "mid": m[1], "hi": max(m[2], m[1])}
                for r, q in zip(roles, qs):
                    row[f"{r}_lo"], row[f"{r}_mid"], row[f"{r}_hi"] = q
                rows.append(row)
    f = pd.DataFrame(rows)
    wide = None
    for arm in ARMS:
        g = cells.merge(f[f.arm == arm].drop(columns="arm"), on=["series_id", "origin", "h"], how="inner")
        g[f"loss_{arm}"] = cell_loss(g.y, g.lo, g.mid, g.hi)
        g[f"in_{arm}"] = (g.y >= g.lo - 1e-12) & (g.y <= g.hi + 1e-12)
        g[f"mid_moved_{arm}"] = (g.mid - g["last"]).abs() > 1e-9
        g[f"width_{arm}"] = (g.hi - g.lo) / g["last"]
        for r in roles:
            g[f"loss_{arm}_{r}"] = cell_loss(g.y, g[f"{r}_lo"], g[f"{r}_mid"], g[f"{r}_hi"])
        keep = ["series_id", "origin", "h", "target", "y", "last", "loss_drift", "loss_engine"] + [c for c in g.columns if c.endswith(f"_{arm}") or f"_{arm}_" in c]
        g = g[keep]
        wide = g if wide is None else wide.merge(g.drop(columns=["target", "y", "last", "loss_drift", "loss_engine"]), on=["series_id", "origin", "h"], how="inner")
    rec = range_memory.crowd_frame(cells)[["series_id", "origin", "h", "lo", "mid", "hi", "y"]]
    rec["loss_recorded"] = cell_loss(rec.y, rec.lo, rec.mid, rec.hi)
    wide = wide.merge(rec[["series_id", "origin", "h", "loss_recorded"]], on=["series_id", "origin", "h"], how="inner")
    wide["moved"] = (wide.y - wide["last"]).abs() > 1e-9
    wide.to_parquet(OUT / "cells.parquet", index=False)

    ci = report.skill_ci
    days = sorted(wide.origin.unique())
    late = wide[wide.origin >= days[len(days) // 2]] if len(days) > 1 else wide
    out = {"days_done": saved["days_done"], "usage": saved["usage"], "n": int(len(wide)), "n_days": len(days), "blocks": {}}
    for name, g in (("all days", wide), ("later half of the days", late)):
        b = {"n": int(len(g)), "n_days": int(g.origin.nunique()),
             "vs_drift": {a: ci(g, f"loss_{a}", "loss_drift") for a in (*ARMS, "recorded")},
             "none_vs_recorded": ci(g, "loss_none", "loss_recorded"),
             "record_vs_none": ci(g, "loss_record", "loss_none"), "lesson_vs_none": ci(g, "loss_lesson", "loss_none"),
             "lesson_vs_record": ci(g, "loss_lesson", "loss_record"),
             "inside": {a: float(g[f"in_{a}"].mean()) for a in ARMS},
             "inside_when_moved": {a: float(g[g.moved][f"in_{a}"].mean()) for a in ARMS},
             "typical_width_pct": {a: float(100 * g[f"width_{a}"].median()) for a in ARMS},
             "middle_moved_share": {a: float(g[f"mid_moved_{a}"].mean()) for a in ARMS},
             "by_h": {int(h): {"n": int(len(gh)), "record_vs_none": ci(gh, "loss_record", "loss_none"), "lesson_vs_none": ci(gh, "loss_lesson", "loss_none"),
                               "inside": {a: float(gh[f"in_{a}"].mean()) for a in ARMS}} for h, gh in g.groupby("h")},
             "by_agent": {r: {"record_vs_none": ci(g, f"loss_record_{r}", f"loss_none_{r}"), "lesson_vs_none": ci(g, f"loss_lesson_{r}", f"loss_none_{r}")} for r in roles}}
        out["blocks"][name] = b
    out["lessons"] = saved["lessons"]
    (data.REPO / "findings" / "data" / "agent_memory.json").write_text(json.dumps(out, indent=1, default=str))
    fm = report.fmt_ci
    for name, b in out["blocks"].items():
        print(f"\n== {name}: {b['n']} forecasts on {b['n_days']} days")
        print("  against drift: " + " | ".join(f"{a} {fm(b['vs_drift'][a])}" for a in (*ARMS, "recorded")))
        print(f"  asked again vs recorded {fm(b['none_vs_recorded'])} | record vs none {fm(b['record_vs_none'])} | lesson vs none {fm(b['lesson_vs_none'])} | lesson vs record {fm(b['lesson_vs_record'])}")
        print(f"  inside: {b['inside']} | when moved: {b['inside_when_moved']} | typical width %: {b['typical_width_pct']}")
        for h, v in b["by_h"].items():
            print(f"    {h}d (n {v['n']}): record vs none {fm(v['record_vs_none'])} | lesson vs none {fm(v['lesson_vs_none'])} | inside {v['inside']}")
        for r, v in b["by_agent"].items():
            print(f"    {r:18s} record vs none {fm(v['record_vs_none'])} | lesson vs none {fm(v['lesson_vs_none'])}")
    print("usage:", out["usage"])
    return out


if __name__ == "__main__":
    {"run": run, "table": table}[sys.argv[1]]()
