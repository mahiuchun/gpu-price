"""The time machine: the scientist lives through the weeks one at a time; a separate examiner
scores every frozen version on the weeks that followed it.

    uv run python -m gpuval.timemachine --name rehearsal --full-wall 2026-10-16 \
        --steps 2026-10-02,2026-10-09 --placebos 1 --ideas 4            # rehearsal (data up to 16 Oct)
    uv run python -m gpuval.timemachine --name record --modal            # the for-the-record run

The driver (this process) is the only place that holds data after a step's wall. Each research step
receives a copy cut at its wall, either in-process (--local, default) or in its own Modal container.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from . import data, harness, scientist
from .core import day_of, iso_of
from .engine import default_spec

RUNS = data.REPO / "runs"
RECORD_STEPS = ["2026-10-16", "2026-10-23", "2026-10-30"]


def plus_days(iso: str, n: int) -> str:
    return (dt.date.fromisoformat(iso) + dt.timedelta(days=n)).isoformat()


def pack_world(world: dict) -> bytes:
    return zlib.compress(pickle.dumps(world, protocol=4), 3)


def unpack_world(blob: bytes) -> dict:
    return pickle.loads(zlib.decompress(blob))


def research_local(blob: bytes, state: dict, kind: str, n_ideas: int, seed: int) -> dict:
    """One research step. Receives only a world cut at its wall."""
    world = unpack_world(blob)
    data.assert_walled(world, world["wall"])
    logs: list[str] = []

    def log(msg: str):
        print(msg, flush=True)
        logs.append(msg)

    state = scientist.run_step(world, state, scientist.make_proposer(kind), n_ideas=n_ideas, seed=seed, log=log)
    state.setdefault("logs", []).extend(logs)
    return state


def examine(full: dict, versions: list[dict], steps: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every version on every later blind week. Version 0 = default settings, no research.
    Returns (summary rows, per-cell losses)."""
    all_versions = [{"wall": None, "spec": default_spec(), "fallback": []}] + versions
    rows, cell_frames = [], []
    for j, T in enumerate(steps, start=1):
        end = plus_days(T, 7)
        if end > full["wall"]:
            continue
        world = data.cut_world(full, end)
        cells = harness.base_cells(world)
        cells = cells[(cells.origin >= day_of(T)) & (cells.target <= day_of(end))]
        if cells.empty:
            continue
        engines: dict = {}
        keep = cells[["series_id", "cls", "origin", "h", "target", "y", "last", "loss_last", "loss_drift"]].copy()
        keep["week"] = j
        d = cells.assign(loss=cells.loss_drift, q10=cells.d10, q50=cells.d50, q90=cells.d90)
        rows.append({"week": j, "step": T, "version": "drift", **harness.summarize(d)})
        for k, v in enumerate(all_versions):
            if k > j:
                continue  # a version is only examined on weeks after it was frozen
            res = harness.run_spec(world, v["spec"], cells, engines, fallback=set(v.get("fallback", [])))
            rows.append({"week": j, "step": T, "version": f"v{k}", **harness.summarize(res)})
            keep[f"loss_v{k}"] = res.loss.to_numpy()
            keep[f"in_v{k}"] = ((res.y >= res.q10) & (res.y <= res.q90)).to_numpy()
            if k == j and v.get("switch_alt"):
                # side report only: the dropped "switch a series to drift" rule applied to the live version
                alt = harness.run_spec(world, v["spec"], cells, engines, fallback=set(v["switch_alt"]))
                rows.append({"week": j, "step": T, "version": f"v{k}+switch", **harness.summarize(alt)})
        cell_frames.append(keep)
    cells_out = pd.concat(cell_frames, ignore_index=True) if cell_frames else pd.DataFrame()
    return pd.DataFrame(rows), cells_out


def run_arm(full: dict, kind: str, steps: list[str], n_ideas: int, seed: int, research, out: Path) -> dict:
    """One world x one arm through all steps, then the examiner."""
    out.mkdir(parents=True, exist_ok=True)
    state = scientist.new_state()
    t0 = time.time()
    for T in steps:
        cut = data.cut_world(full, T)
        data.assert_walled(cut, T)
        blob = pack_world(cut)
        for attempt in range(3):  # a failed container produced no state, so retrying the step is clean
            try:
                state = research(blob, state, kind, n_ideas, seed)
                break
            except Exception as e:
                print(f"[{full['name']} {kind} {T}] attempt {attempt + 1} failed: {type(e).__name__}: {e}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(20)
        (out / "state.json").write_text(json.dumps(state, indent=1, default=str))
    summary, cells = examine(full, state["versions"], steps)
    summary.insert(0, "arm", kind)
    summary.insert(0, "world", full["name"])
    summary.to_csv(out / "exam.csv", index=False)
    if len(cells):
        cells.insert(0, "arm", kind)
        cells.insert(0, "world", full["name"])
        cells.to_parquet(out / "cells.parquet", index=False)
    (out / "state.json").write_text(json.dumps(state, indent=1, default=str))
    return {"world": full["name"], "arm": kind, "minutes": round((time.time() - t0) / 60, 1),
            "kept": sum(e["kept"] for e in state["ledger"]), "ideas": len(state["ledger"]),
            "usage": state.get("usage", {})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name; outputs go to runs/<name>/")
    ap.add_argument("--full-wall", default=data.DATA_END, help="last date the driver itself may read")
    ap.add_argument("--steps", default=",".join(RECORD_STEPS), help="comma list of step dates")
    ap.add_argument("--arms", default="gpt,random")
    ap.add_argument("--placebos", type=int, default=5)
    ap.add_argument("--ideas", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modal", action="store_true", help="run each research step in its own Modal container")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(data.REPO / ".env")
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    if plus_days(steps[-1], 7) > a.full_wall and steps[-1] != a.full_wall:
        raise SystemExit("the last step needs 7 days of data after it for the examiner")
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)

    real = data.build_world(a.full_wall)
    data.assert_walled(real, a.full_wall)
    worlds = [real] + [data.make_placebo(real, seed=s) for s in range(1, a.placebos + 1)]
    jobs = [(w, arm) for w in worlds for arm in arms]
    meta = {"name": a.name, "full_wall": a.full_wall, "steps": steps, "arms": arms, "placebos": a.placebos,
            "ideas": a.ideas, "seed": a.seed, "modal": a.modal, "model": scientist.MODEL,
            "reasoning": scientist.REASONING, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"run {a.name}: {len(worlds)} worlds x {len(arms)} arms, steps {steps}, {a.ideas} ideas per step, "
          f"{'Modal' if a.modal else 'local'}", flush=True)

    def go(research):
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run_arm, w, arm, steps, a.ideas, a.seed, research, out / f"{w['name']}_{arm}")
                    for w, arm in jobs]
            return [f.result() for f in futs]

    if a.modal:
        import modal

        from .modal_app import app, research_step

        with modal.enable_output(), app.run():
            results = go(lambda blob, state, kind, n, seed: research_step.remote(blob, state, kind, n, seed))
    else:
        results = go(research_local)

    meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["results"] = results
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    exam = pd.concat([pd.read_csv(out / f"{w['name']}_{arm}" / "exam.csv") for w, arm in jobs], ignore_index=True)
    exam.to_csv(out / "exam_all.csv", index=False)
    print(json.dumps(results, indent=1))
    print(f"done: runs/{a.name}/exam_all.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
