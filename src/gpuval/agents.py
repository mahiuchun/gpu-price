"""A crowd of AI agents that forecast each price directly.

    uv run python -m gpuval.agents --origins 2026-10-16,2026-10-22,2026-10-28,2026-11-03   # test on past days
    uv run python -m gpuval.agents --origins 2026-11-06 --final                               # the real forecast

Five agents with different jobs each read the same facts about one price (its recent history, today's
list of offers, who sits in the middle, who reprices) and give a low / middle / high for 1, 3 and 7
days ahead. Round 1: each works alone. Round 2: each sees the others' answers and may revise.
The crowd's forecast is the average of the agents' answers. Every answer is saved, so the result can
be re-scored without calling the AI again.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, harness
from .core import HORIZONS, cell_loss, day_of, drift_forecast, iso_of
from .engine import build_engines, default_spec

_TAIL = re.compile(r"\|s\d+\|d\d+$")


def offer_name(slot: str) -> str:
    return _TAIL.sub("", slot).rstrip("|")


MODEL = "gpt-6.1-sol"
EFFORT = "low"
OUT = data.REPO / "runs" / "agents"

ROLES = {
    "list_reader": "Your job: read the list. Work out which offer is in the middle, how far the middle can move given the offers next to it, and whether an offer joining or leaving would shift it.",
    "repricer_watcher": "Your job: watch the offers that change price often. Decide whether their recent direction is likely to continue, and whether that would move the middle of the list.",
    "base_rate_keeper": "Your job: keep to base rates. On most days a published price does not change. Move away from today's value only with clear evidence, and size the range from how often and how far this price has moved before.",
    "combiner": "Your job: weigh the three reference forecasts you are given (drift, the list-replay engine, and a pretrained time-series model) against the facts. Decide where they disagree and which to trust for this price.",
    "risk_spotter": "Your job: look for what could make this price jump or snap back: a one-day jump that may be temporary, an offer that just appeared or vanished, a price that looks out of line with the rest of the list.",
}

SYSTEM = """You forecast a published GPU rental price. The published price is the MIDDLE (median) of one tracking site's list of offers for one GPU, in USD per GPU-hour. Most offers never change price; a few change almost daily; offers also join and leave the list.

{role}

Give, for 1, 3 and 7 days ahead, three numbers: low (10% chance the price is below it), mid (the most likely middle value), high (10% chance the price is above it). If you expect no change, low, mid and high can all equal today's price. Do not make the range wide just to be safe: a wide range costs points when nothing moves. Reply with JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        **{f"h{h}": {"type": "object", "properties": {k: {"type": "number"} for k in ("low", "mid", "high")},
                     "required": ["low", "mid", "high"], "additionalProperties": False} for h in HORIZONS},
        "reason": {"type": "string", "description": "One or two plain sentences."},
    },
    "required": [f"h{h}" for h in HORIZONS] + ["reason"],
    "additionalProperties": False,
}


def facts(s: dict, i0: int, sid: str) -> str:
    """What every agent is shown about one price on one day. Uses rows up to i0 only."""
    days, val, P, slots = s["days"], s["value"], s["P"], s["slots"]
    m = s["meta"]
    today = P[i0]
    here = np.where(~np.isnan(today))[0]
    order = here[np.argsort(today[here], kind="stable")]
    n = len(order)
    lo30 = max(0, int(np.searchsorted(days, days[i0] - 30)))
    lines = [f"PRICE: site {m['source_id']}, {m['gpu_model']}, {m['pricing_tier']} ({m['instrument_class']})",
             f"TODAY: {iso_of(days[i0])}. Published price today: {val[i0]:.4f}",
             "PUBLISHED PRICE, last observed days: " + ", ".join(f"{iso_of(d)[5:]}={v:.3f}" for d, v in zip(days[max(0, i0 - 20):i0 + 1], val[max(0, i0 - 20):i0 + 1])),
             f"LIST TODAY: {n} offers, sorted by price. '<<' marks the middle."]
    mid_pos = {(n - 1) // 2, n // 2}
    show = range(n) if n <= 14 else range(max(0, (n - 1) // 2 - 5), min(n, n // 2 + 6))
    if n > 14:
        lines.append(f"  (showing the offers around the middle; {min(show)} cheaper ones from {today[order[0]]:.3f} and {n - 1 - max(show)} dearer ones up to {today[order[-1]]:.3f} are not shown)")
    for pos in show:
        j = order[pos]
        col = P[lo30:i0 + 1, j]
        ok = ~np.isnan(col)
        d = np.diff(col[ok])
        nchg = int((np.abs(d) > 1e-9).sum())
        last = ""
        if nchg:
            idx = np.where(ok)[0]
            k = np.where(np.abs(d) > 1e-9)[0][-1]
            last = f"; last change {iso_of(days[lo30 + idx[k + 1]])[5:]}: {col[idx[k]]:.3f} -> {col[idx[k + 1]]:.3f}"
        lines.append(f"  {offer_name(slots[j])[:46]:46s} {today[j]:9.4f}{' <<' if pos in mid_pos else '   '}  listed {int(ok.sum())} of last {len(col)} days; price changes in that time: {nchg}{last}")
    lo14 = max(0, int(np.searchsorted(days, days[i0] - 14)))
    was = ~np.isnan(P[lo14])
    joined = [offer_name(slots[j]) for j in here if not was[j]][:6]
    left = [offer_name(slots[j]) for j in np.where(was & np.isnan(today))[0]][:6]
    lines.append(f"LIST CHANGES since {iso_of(days[lo14])[5:]}: joined: {', '.join(joined) or 'none'}; left: {', '.join(left) or 'none'}")
    moved = int((np.abs(np.diff(val[lo30:i0 + 1])) > 1e-9).sum())
    lines.append(f"The published price changed on {moved} of the last {i0 - lo30} observed days.")
    return "\n".join(lines)


def refs_text(refs: dict) -> str:
    out = ["REFERENCE FORECASTS (low / mid / high):"]
    for name, by_h in refs.items():
        out.append(f"  {name}: " + "; ".join(f"{h}d {by_h[h][0]:.4f} / {by_h[h][1]:.4f} / {by_h[h][2]:.4f}" for h in HORIZONS if h in by_h))
    return "\n".join(out)


def clean(ans: dict, last: float) -> dict:
    out = {}
    for h in HORIZONS:
        a = ans.get(f"h{h}", {})
        mid = float(a.get("mid", last) or last)
        mid = mid if np.isfinite(mid) and mid > 0 else last
        lo = float(a.get("low", mid) or mid)
        hi = float(a.get("high", mid) or mid)
        out[h] = (max(min(lo, mid), 1e-6), mid, max(hi, mid))
    return out


def ask(client, role: str, user: str) -> tuple[dict, dict]:
    last_err = None
    for attempt in range(4):
        try:
            r = client.responses.create(model=MODEL, reasoning={"effort": EFFORT},
                                        input=[{"role": "system", "content": SYSTEM.format(role=ROLES[role])}, {"role": "user", "content": user}],
                                        text={"format": {"type": "json_schema", "name": "forecast", "schema": SCHEMA, "strict": True}})
            u = getattr(r, "usage", None)
            return json.loads(r.output_text), {"in": int(getattr(u, "input_tokens", 0) or 0), "out": int(getattr(u, "output_tokens", 0) or 0)}
        except Exception as e:
            last_err = e
            time.sleep(2 + 4 * attempt)
    raise RuntimeError(f"agent call failed: {last_err}")


def combine(answers: list[dict]) -> dict:
    """The crowd's forecast: the average (in log terms) of each number across the agents."""
    out = {}
    for h in HORIZONS:
        q = np.exp(np.mean(np.log(np.array([a[h] for a in answers])), axis=0))
        out[h] = (float(min(q[0], q[1])), float(q[1]), float(max(q[2], q[1])))
    return out


def run_origin(full: dict, origin: str, chronos: pd.DataFrame | None, client, workers: int, notes: str = "", rounds: int = 2) -> list[dict]:
    world = data.cut_world(full, origin)
    data.assert_walled(world, origin)
    engines = build_engines(world)
    od = day_of(origin)
    jobs = []
    for sid, s in world["series"].items():
        if int(s["days"][-1]) != od:
            continue  # the price was not observed on this day
        i0 = len(s["days"]) - 1
        last = float(s["value"][i0])
        f = engines[sid].forecast(i0, default_spec())
        refs = {"drift": {h: drift_forecast(s["days"], s["value"], h) for h in HORIZONS}, "engine": {h: f[h][:3] for h in HORIZONS}}
        if chronos is not None:
            c = chronos[(chronos.series_id == sid) & (chronos.origin == od)]
            if len(c):
                refs["pretrained model"] = {int(r.h): (r.c10, r.c50, r.c90) for r in c.itertuples()}
        jobs.append({"series_id": sid, "origin": origin, "last": last, "facts": facts(s, i0, sid) + (("\nNOTES FROM NEWS AND PROVIDER PAGES:\n" + notes) if notes else ""), "refs": refs})

    def round1(job):
        res = {}
        for role in ROLES:
            user = job["facts"] + ("\n" + refs_text(job["refs"]) if role == "combiner" else "") + "\n\nGive your forecast."
            ans, use = ask(client, role, user)
            res[role] = {"q": clean(ans, job["last"]), "reason": ans.get("reason", ""), "usage": use}
        return res

    def round2(job):
        board = "\n".join(f"  Analyst {k + 1}: " + "; ".join(f"{h}d {a['q'][h][0]:.4f}/{a['q'][h][1]:.4f}/{a['q'][h][2]:.4f}" for h in HORIZONS) + f" | {a['reason'][:220]}"
                          for k, a in enumerate(job["r1"].values()))
        res = {}
        for role in ROLES:
            user = (job["facts"] + ("\n" + refs_text(job["refs"]) if role == "combiner" else "")
                    + "\n\nFIVE ANALYSTS' FIRST ANSWERS (low/mid/high), yours among them:\n" + board
                    + "\n\nGive your final forecast. Change your answer only if another analyst saw something you missed.")
            ans, use = ask(client, role, user)
            res[role] = {"q": clean(ans, job["last"]), "reason": ans.get("reason", ""), "usage": use}
        return res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for job, r in zip(jobs, ex.map(round1, jobs)):
            job["r1"] = r
        if rounds >= 2:
            for job, r in zip(jobs, ex.map(round2, jobs)):
                job["r2"] = r
    for job in jobs:
        job["crowd1"] = combine([a["q"] for a in job["r1"].values()])
        if rounds >= 2:
            job["crowd2"] = combine([a["q"] for a in job["r2"].values()])
    return jobs


def score(jobs: list[dict], cells: pd.DataFrame) -> pd.DataFrame:
    """Join agent forecasts to the blind cells (which carry the truth and the other forecasters' losses)."""
    rows = []
    for job in jobs:
        od = day_of(job["origin"])
        for h in HORIZONS:
            row = {"series_id": job["series_id"], "origin": od, "h": h}
            for name in ("crowd1", "crowd2"):
                row[f"{name}_q10"], row[f"{name}_q50"], row[f"{name}_q90"] = job[name][h]
            for role in ROLES:
                row[f"{role}_q10"], row[f"{role}_q50"], row[f"{role}_q90"] = job["r1"][role]["q"][h]
            rows.append(row)
    m = cells.merge(pd.DataFrame(rows), on=["series_id", "origin", "h"], how="inner")
    for name in ["crowd1", "crowd2", *ROLES]:
        m[f"loss_{name}"] = cell_loss(m.y, m[f"{name}_q10"], m[f"{name}_q50"], m[f"{name}_q90"])
        m[f"in_{name}"] = (m.y >= m[f"{name}_q10"]) & (m.y <= m[f"{name}_q90"])
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origins", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--final", action="store_true", help="no scoring: write forecasts for the origin")
    ap.add_argument("--notes-file", default=None)
    ap.add_argument("--name", default="test")
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(data.REPO / ".env")
    client = OpenAI()
    OUT.mkdir(parents=True, exist_ok=True)
    origins = [o.strip() for o in a.origins.split(",")]
    full = data.build_world(data.DATA_END)
    cells = pd.read_parquet(data.REPO / "findings" / "data" / "ml_challenger_cells.parquet")
    notes = Path(a.notes_file).read_text() if a.notes_file else ""
    all_jobs = []
    t0 = time.time()
    for o in origins:
        chron = cells[["series_id", "origin", "h", "c10", "c50", "c90"]] if not a.final else None
        if a.final:
            cq = pd.read_csv(data.REPO / "versions" / f"chronos2_quantiles_{o}.csv")
            cq["origin"] = day_of(o)
            chron = cq
        jobs = run_origin(full, o, chron, client, a.workers, notes)
        all_jobs += jobs
        print(f"{o}: {len(jobs)} prices x {len(ROLES)} agents x 2 rounds done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        (OUT / f"{a.name}_answers.json").write_text(json.dumps(all_jobs, indent=1, default=str))
    use = {"in": 0, "out": 0, "calls": 0}
    for j in all_jobs:
        for rd in ("r1", "r2"):
            for x in j[rd].values():
                use["in"] += x["usage"]["in"]; use["out"] += x["usage"]["out"]; use["calls"] += 1
    print("usage:", use)
    if a.final:
        return 0
    m = score(all_jobs, cells)
    m.to_parquet(OUT / f"{a.name}_cells.parquet", index=False)
    base, drift = m.loss_last.sum(), m.loss_drift.sum()
    print(f"\n{len(m)} forecasts scored, {m.series_id.nunique()} prices, origins {origins}")
    print(f"{'forecast':34s} {'less error than drift':>22s} {'skill':>7s} {'per-price avg':>14s} {'inside':>7s}")
    def line(name, col, incol=None):
        per = m.groupby("series_id").agg(l=(col, "mean"), b=("loss_last", "mean")); sk = (1 - per.l / per.b)[per.b > 0].mean()
        ins = f"{m[incol].mean():.2f}" if incol else "   -"
        print(f"{name:34s} {100 * (1 - m[col].sum() / drift):21.1f}% {1 - m[col].sum() / base:7.3f} {sk:14.3f} {ins:>7s}")
    line("drift", "loss_drift"); line("engine", "loss_v0"); line("pretrained model (Chronos-2)", "loss_ml")
    for role in ROLES:
        line(f"agent: {role}", f"loss_{role}", f"in_{role}")
    line("AGENT CROWD, round 1 (alone)", "loss_crowd1", "in_crowd1")
    line("AGENT CROWD, round 2 (after seeing each other)", "loss_crowd2", "in_crowd2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
