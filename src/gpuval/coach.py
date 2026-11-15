"""A self-improving agent crowd: a coach AI rewrites the crowd, and tests decide what is kept.

    uv run python -m gpuval.coach run --generations 8     # baseline, then the improvement loop
    uv run python -m gpuval.coach exam                    # original crowd vs improved crowd on fresh days

The crowd is described by a small config: each agent's instructions, whether it is shown the
reference forecasts, and how much the crowd's ranges are stretched at each horizon. Each generation
the coach sees the crowd's scores and worst misses on PRACTICE days and proposes one change. The
change is kept only if the crowd improves on the practice days and is not worse on separate CHECK
days, for which the coach is told only pass or fail. The EXAM days are never shown to the coach.

Practice and check days: AWS spot prices 2024-2025 and our own prices before 16 Oct.
Exam days: our own prices in the three test weeks on days no agent had been scored on, plus AWS
days after April 2026, which the model cannot have seen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import agents, aws_test, data
from .core import HORIZONS, cell_loss, day_of, drift_forecast, iso_of
from .engine import build_engines, default_spec

OUT = data.REPO / "runs" / "coach"
DAYS = {
    "practice": {"aws": ["2024-01-12", "2024-06-14", "2024-10-04", "2025-02-07"], "ours": ["2026-09-26", "2026-10-08"]},
    "check": {"aws": ["2024-04-12", "2024-12-06", "2025-05-09"], "ours": ["2026-10-02", "2026-10-14"]},
    "exam": {"aws": ["2026-07-24", "2026-08-21", "2026-10-02"], "ours": ["2026-10-18", "2026-10-21", "2026-10-26", "2026-10-29", "2026-11-02"]},
}
MIN_GAIN = 0.005
CHECK_NOISE = 0.005

SYSTEM = agents.SYSTEM  # the same frame the original crowd used; only the role text varies


def base_config() -> dict:
    return {"roles": {k: {"instruction": v, "refs": k == "combiner"} for k, v in agents.ROLES.items()},
            "range_scale": {"1": 1.0, "3": 1.0, "7": 1.0}}


# ------------------------------------------------------------------ items ----
def build_items(days: dict | None = None) -> dict:
    """Every (price, day) the crowd is asked about, with the facts, the truth and the reference forecasts."""
    ours = data.build_world(data.DATA_END)
    aws = data.load_world(aws_test.WORK / "world.pkl")
    pc = pd.read_parquet(data.REPO / "findings" / "data" / "periods_cells.parquet")[["series_id", "origin", "h", "c10", "c50", "c90"]]
    ac = pd.read_parquet(data.REPO / "findings" / "data" / "aws_cells.parquet")
    ac = ac[ac.c50.notna()][["series_id", "origin", "h", "c10", "c50", "c90"]]
    chron = {(r.series_id, int(r.origin), int(r.h)): (r.c10, r.c50, r.c90) for r in pd.concat([pc, ac]).itertuples()}
    worlds = {"ours": (ours, build_engines(ours)), "aws": (aws, build_engines(aws))}
    items = {}
    for split, by_world in (days or DAYS).items():
        for wname, days in by_world.items():
            world, engines = worlds[wname]
            for iso in days:
                od = day_of(iso)
                for sid, s in world["series"].items():
                    i0 = int(np.searchsorted(s["days"], od))
                    if i0 >= len(s["days"]) or s["days"][i0] != od or od - s["days"][0] < 14:
                        continue
                    y = {}
                    for h in HORIZONS:
                        k = int(np.searchsorted(s["days"], od + h))
                        if k < len(s["days"]) and s["days"][k] == od + h:
                            y[h] = float(s["value"][k])
                    if not y:
                        continue
                    f = engines[sid].forecast(i0, default_spec())
                    refs = {"drift": {h: drift_forecast(s["days"][:i0 + 1], s["value"][:i0 + 1], h) for h in HORIZONS},
                            "engine": {h: f[h][:3] for h in HORIZONS}}
                    cr = {h: chron[(sid, od, h)] for h in HORIZONS if (sid, od, h) in chron}
                    if cr:
                        refs["pretrained model"] = cr
                    items[f"{wname}|{sid}|{iso}"] = {"id": f"{wname}|{sid}|{iso}", "split": split, "world": wname, "series_id": sid, "day": iso,
                                                    "last": float(s["value"][i0]), "y": y, "refs": refs, "facts": agents.facts(s, i0, sid)}
    return items


# ---------------------------------------------------------------- answers ----
class Answers:
    """Every answer an agent gave, keyed by its exact instructions and the price-day. Saved to disk."""

    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / "answers.json"
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.usage = {"calls": 0, "in": 0, "out": 0}

    @staticmethod
    def key(role: dict, item_id: str) -> str:
        return hashlib.md5((role["instruction"] + "|" + str(bool(role["refs"]))).encode()).hexdigest()[:12] + "|" + item_id

    def fill(self, client, role: dict, items: list[dict], workers: int = 32) -> None:
        todo = [it for it in items if self.key(role, it["id"]) not in self.d]

        def one(it):
            user = it["facts"] + ("\n" + agents.refs_text(it["refs"]) if role["refs"] else "") + "\n\nGive your forecast."
            last_err = None
            for attempt in range(4):
                try:
                    r = client.responses.create(model=agents.MODEL, reasoning={"effort": agents.EFFORT},
                                                input=[{"role": "system", "content": SYSTEM.format(role=role["instruction"])}, {"role": "user", "content": user}],
                                                text={"format": {"type": "json_schema", "name": "forecast", "schema": agents.SCHEMA, "strict": True}})
                    u = getattr(r, "usage", None)
                    ans = json.loads(r.output_text)
                    return it["id"], {"q": {str(h): list(v) for h, v in agents.clean(ans, it["last"]).items()}, "reason": ans.get("reason", "")}, \
                        int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0)
                except Exception as e:
                    last_err = e
                    time.sleep(2 + 4 * attempt)
            raise RuntimeError(f"agent call failed: {last_err}")

        if todo:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for iid, ans, tin, tout in ex.map(one, todo):
                    self.d[self.key(role, iid)] = ans
                    self.usage["calls"] += 1; self.usage["in"] += tin; self.usage["out"] += tout
            self.path.write_text(json.dumps(self.d))

    def get(self, role: dict, item_id: str) -> dict:
        return self.d[self.key(role, item_id)]


# ---------------------------------------------------------------- scoring ----
def crowd_forecast(config: dict, answers: Answers, item: dict) -> dict:
    qs = np.array([[answers.get(r, item["id"])["q"][str(h)] for h in HORIZONS] for r in config["roles"].values()], dtype=float)
    m = np.exp(np.mean(np.log(qs), axis=0))
    out = {}
    for j, h in enumerate(HORIZONS):
        lo, mid, hi = m[j]
        s = float(config["range_scale"][str(h)])
        out[h] = (float(mid * (min(lo, mid) / mid) ** s), float(mid), float(mid * (max(hi, mid) / mid) ** s))
    return out


def score(config: dict, answers: Answers, items: list[dict], solo: bool = False) -> dict:
    """Error relative to the price, pooled, against drift. 1 - ours / drift; higher is better."""
    rows = []
    for it in items:
        f = crowd_forecast(config, answers, it)
        for h, y in it["y"].items():
            h = int(h)
            row = {"world": it["world"], "h": h, "last": it["last"], "y": y, "id": it["id"],
                   "loss": cell_loss(y, *f[h]) / it["last"], "drift": cell_loss(y, *it["refs"]["drift"][h]) / it["last"],
                   "engine": cell_loss(y, *it["refs"]["engine"][h]) / it["last"],
                   "usd_loss": cell_loss(y, *f[h]), "usd_drift": cell_loss(y, *it["refs"]["drift"][h]), "usd_engine": cell_loss(y, *it["refs"]["engine"][h]),
                   "inside": f[h][0] <= y <= f[h][2], "below": y < f[h][0], "above": y > f[h][2], "lo": f[h][0], "mid": f[h][1], "hi": f[h][2]}
            if solo:
                for name, r in config["roles"].items():
                    row["solo_" + name] = cell_loss(y, *answers.get(r, it["id"])["q"][str(h)]) / it["last"]
            rows.append(row)
    df = pd.DataFrame(rows)
    out = {"n": int(len(df)), "loss": float(df.loss.sum()), "vs_drift": float(1 - df.loss.sum() / df.drift.sum()),
           "vs_engine": float(1 - df.loss.sum() / df.engine.sum()), "inside": float(df.inside.mean()),
           "below": float(df.below.mean()), "above": float(df.above.mean()),
           "by_h": {int(h): {"vs_drift": float(1 - g.loss.sum() / g.drift.sum()), "inside": float(g.inside.mean())} for h, g in df.groupby("h")},
           "by_world": {w: float(1 - g.loss.sum() / g.drift.sum()) for w, g in df.groupby("world")}, "df": df}
    if solo:
        out["solo"] = {name: float(1 - df["solo_" + name].sum() / df.drift.sum()) for name in config["roles"]}
    return out


def miss_cards(config: dict, answers: Answers, items: list[dict], sc: dict, n: int = 6) -> str:
    df = sc["df"]
    worst = (df.assign(gap=df.loss - df.drift).groupby("id").gap.sum().sort_values(ascending=False).head(n).index.tolist())
    by_id = {it["id"]: it for it in items}
    cards = []
    for iid in worst:
        it = by_id[iid]
        f = crowd_forecast(config, answers, it)
        lines = it["facts"].splitlines()
        head = "\n".join(lines[:3]) + "\n" + "\n".join(l for l in lines[3:] if "<<" in l or l.startswith("LIST") or l.startswith("The published"))
        truth = "; ".join(f"{h}d truth {it['y'][h]:.4f} vs crowd {f[h][0]:.4f}/{f[h][1]:.4f}/{f[h][2]:.4f}" for h in it["y"])
        said = "; ".join(f"{name}: {answers.get(r, iid)['reason'][:110]}" for name, r in config["roles"].items())
        cards.append(f"--- MISS\n{head}\nOUTCOME: {truth}\nWHAT THE AGENTS SAID: {said}")
    return "\n".join(cards)


# ------------------------------------------------------------------ coach ----
COACH_SYSTEM = """You coach a crowd of AI agents that forecast published GPU rental prices. Each agent reads the same facts about one price (its recent history and today's list of offers) and gives low / mid / high for 1, 3 and 7 days ahead. The crowd's forecast is the average of the agents' answers, and its range can be stretched or shrunk at each horizon.

Scores are error against the simple reference forecast called drift: +0.10 means 10% less error than drift. Error is pinball loss: a range that is too wide costs points when nothing moves, and a range that misses costs more when the price jumps. The truth should fall inside the range about 80% of the time.

Each turn you propose ONE change to the crowd:
- rewrite_role: replace one agent's instructions (give role_name and the full new instruction)
- add_role: add a new agent (give a new role_name, its instruction, and whether it is shown the reference forecasts)
- drop_role: remove an agent (give role_name)
- set_range_scale: set how much the crowd's range is stretched at each horizon (1.0 = as the agents gave it; 1.3 = 30% wider; 0.8 = narrower)

Your change is tested. It is KEPT only if the crowd improves on the practice days and also passes a hidden check on other days, for which you are told only PASS or FAIL. Changes that fit a few practice cases tend to fail the hidden check, so prefer general lessons over fixes for single cases.

An agent's instruction is one short paragraph that starts with "Your job:". Keep instructions plain and general. Do not mention specific providers, GPUs, dates or prices in them."""

COACH_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "description": "What the scores and misses show, in two or three plain sentences."},
        "change_type": {"type": "string", "enum": ["rewrite_role", "add_role", "drop_role", "set_range_scale"]},
        "role_name": {"type": "string", "description": "snake_case name; empty for set_range_scale."},
        "instruction": {"type": "string", "description": "Full new instruction for rewrite_role or add_role; empty otherwise."},
        "uses_reference_forecasts": {"type": "boolean"},
        "range_scale_1d": {"type": "number"}, "range_scale_3d": {"type": "number"}, "range_scale_7d": {"type": "number"},
        "expectation": {"type": "string", "description": "What you expect the change to do."},
    },
    "required": ["diagnosis", "change_type", "role_name", "instruction", "uses_reference_forecasts", "range_scale_1d", "range_scale_3d", "range_scale_7d", "expectation"],
    "additionalProperties": False,
}


def describe(config: dict, sc: dict) -> str:
    lines = ["CURRENT CROWD"]
    for name, r in config["roles"].items():
        lines.append(f"- {name} (shown reference forecasts: {'yes' if r['refs'] else 'no'}): {r['instruction']}")
    lines.append(f"Range stretch: 1 day {config['range_scale']['1']}, 3 days {config['range_scale']['3']}, 7 days {config['range_scale']['7']}")
    lines.append(f"\nPRACTICE SCORES ({sc['n']} forecasts): crowd {sc['vs_drift']:+.3f} against drift; truth below/inside/above the range: "
                 f"{sc['below']:.2f}/{sc['inside']:.2f}/{sc['above']:.2f}")
    lines.append("By horizon: " + "; ".join(f"{h}d {v['vs_drift']:+.3f} (inside {v['inside']:.2f})" for h, v in sc["by_h"].items()))
    lines.append("By kind of price: " + "; ".join(f"{'provider lists' if w == 'ours' else 'cloud spot across zones'} {v:+.3f}" for w, v in sc["by_world"].items()))
    if "solo" in sc:
        lines.append("Each agent alone: " + "; ".join(f"{k} {v:+.3f}" for k, v in sc["solo"].items()))
    return "\n".join(lines)


def apply_change(config: dict, ch: dict) -> dict | None:
    c = copy.deepcopy(config)
    t, name = ch["change_type"], ch["role_name"].strip().lower().replace(" ", "_")
    if t == "set_range_scale":
        c["range_scale"] = {"1": float(np.clip(ch["range_scale_1d"], 0.3, 3)), "3": float(np.clip(ch["range_scale_3d"], 0.3, 3)),
                            "7": float(np.clip(ch["range_scale_7d"], 0.3, 3))}
    elif t == "drop_role":
        if name not in c["roles"] or len(c["roles"]) <= 2:
            return None
        del c["roles"][name]
    elif t in ("rewrite_role", "add_role"):
        if not ch["instruction"].strip() or (t == "rewrite_role" and name not in c["roles"]) or (t == "add_role" and (name in c["roles"] or len(c["roles"]) >= 8)):
            return None
        c["roles"][name] = {"instruction": ch["instruction"].strip(), "refs": bool(ch["uses_reference_forecasts"])}
    return None if c == config else c


def run(generations: int, seed: int = 0) -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(data.REPO / ".env")
    client = OpenAI()
    items = build_items()
    practice = [it for it in items.values() if it["split"] == "practice"]
    check = [it for it in items.values() if it["split"] == "check"]
    print(f"practice {len(practice)} price-days, check {len(check)}, exam {sum(it['split'] == 'exam' for it in items.values())}", flush=True)
    answers = Answers()
    config = base_config()
    for r in config["roles"].values():
        answers.fill(client, r, practice + check)
    best_p, best_c = score(config, answers, practice, solo=True), score(config, answers, check)
    print(f"start: practice {best_p['vs_drift']:+.3f}, check {best_c['vs_drift']:+.3f} | usage {answers.usage}", flush=True)
    rng = np.random.default_rng(zlib.crc32(f"coach|{seed}".encode()))
    ledger = []
    state = {"start": {"practice": best_p["vs_drift"], "check": best_c["vs_drift"]}, "ledger": ledger, "config": config, "base": base_config()}
    for g in range(1, generations + 1):
        hist = "\n".join(f"#{e['n']} {e['change_type']} {e['role_name']}: practice {e['practice_before']:+.3f} -> {e['practice_after']:+.3f}; hidden check {e['check']}; "
                         f"{'KEPT' if e['kept'] else 'REJECTED'}. Idea: {e['diagnosis'][:200]}" for e in ledger) or "(none yet)"
        user = (describe(config, best_p) + "\n\nWORST MISSES ON PRACTICE DAYS (crowd low/mid/high)\n" + miss_cards(config, answers, practice, best_p)
                + "\n\nCHANGES TRIED SO FAR\n" + hist + f"\n\nGenerations left including this one: {generations - g + 1}. Propose one change.")
        resp = client.responses.create(model=agents.MODEL, reasoning={"effort": "medium"},
                                       input=[{"role": "system", "content": COACH_SYSTEM}, {"role": "user", "content": user}],
                                       text={"format": {"type": "json_schema", "name": "change", "schema": COACH_SCHEMA, "strict": True}})
        ch = json.loads(resp.output_text)
        cand = apply_change(config, ch)
        entry = {"n": g, **{k: ch[k] for k in ("change_type", "role_name", "diagnosis", "instruction", "expectation", "uses_reference_forecasts")},
                 "range_scale": [ch["range_scale_1d"], ch["range_scale_3d"], ch["range_scale_7d"]] if ch["change_type"] == "set_range_scale" else None,
                 "practice_before": best_p["vs_drift"], "practice_after": best_p["vs_drift"], "check": "not run", "kept": False}
        if cand is not None:
            for r in cand["roles"].values():
                answers.fill(client, r, practice + check)
            cp, cc = score(cand, answers, practice, solo=True), score(cand, answers, check)
            entry["practice_after"] = cp["vs_drift"]
            entry["_check_before"], entry["_check_after"] = best_c["vs_drift"], cc["vs_drift"]
            if cp["loss"] < best_p["loss"] * (1 - MIN_GAIN):
                ok = cc["loss"] < best_c["loss"] * (1 + rng.normal(0, CHECK_NOISE))
                entry["check"] = "PASS" if ok else "FAIL"
                if ok:
                    config, best_p, best_c, entry["kept"] = cand, cp, cc, True
        ledger.append(entry)
        state["config"] = config
        (OUT / "state.json").write_text(json.dumps(state, indent=1))
        print(f"gen {g}: {entry['change_type']} {entry['role_name']} | practice {entry['practice_before']:+.3f} -> {entry['practice_after']:+.3f} | check {entry['check']} | "
              f"{'KEPT' if entry['kept'] else 'rejected'} | {entry['diagnosis'][:150]}", flush=True)
    state["end"] = {"practice": best_p["vs_drift"], "check": best_c["vs_drift"]}
    state["usage"] = answers.usage
    (OUT / "state.json").write_text(json.dumps(state, indent=1))
    print(f"end: practice {best_p['vs_drift']:+.3f}, check {best_c['vs_drift']:+.3f} | kept {sum(e['kept'] for e in ledger)} of {len(ledger)} | usage {answers.usage}")


def exam() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    from . import report

    load_dotenv(data.REPO / ".env")
    client = OpenAI()
    items = [it for it in build_items().values() if it["split"] == "exam"]
    state = json.loads((OUT / "state.json").read_text())
    answers = Answers()
    out = {"n_price_days": len(items)}
    frames = {}
    stretch_only = {**base_config(), "range_scale": state["config"]["range_scale"]}
    for label, cfg in (("original", base_config()), ("improved", state["config"]), ("stretch_only", stretch_only)):
        for r in cfg["roles"].values():
            answers.fill(client, r, items)
        frames[label] = score(cfg, answers, items)
    o, i = frames["original"]["df"], frames["improved"]["df"]
    m = o[["id", "world", "h", "usd_drift", "usd_engine", "drift", "engine"]].copy()
    m["series_id"] = m.id.str.split("|").str[1]
    m["target"] = m.id.str.split("|").str[2].map(day_of) + m.h
    m["usd_orig"], m["usd_impr"], m["rel_orig"], m["rel_impr"] = o.usd_loss.values, i.usd_loss.values, o.loss.values, i.loss.values
    m["y"], m["last"], m["n_agents"] = o.y.values, o["last"].values, len(state["config"]["roles"])
    m["usd_stretch"], m["rel_stretch"] = frames["stretch_only"]["df"].usd_loss.values, frames["stretch_only"]["df"].loss.values
    for tag, fr in (("o", o), ("i", i)):
        for c in ("lo", "mid", "hi"):
            m[f"{tag}_{c}"] = fr[c].values
    for wname, g in list(m.groupby("world")) + [("all", m)]:
        col = ("usd_orig", "usd_impr", "usd_drift", "usd_engine") if wname != "all" else ("rel_orig", "rel_impr", "drift", "engine")
        st = "usd_stretch" if wname != "all" else "rel_stretch"
        out[wname] = {"n": int(len(g)), "stretch_only_vs_original": report.skill_ci(g, st, col[0]), "improved_vs_stretch_only": report.skill_ci(g, col[1], st),
                      "by_h": {int(h): {"original_vs_drift": report.skill_ci(gh, col[0], col[2]), "improved_vs_drift": report.skill_ci(gh, col[1], col[2]),
                                        "improved_vs_original": report.skill_ci(gh, col[1], col[0])} for h, gh in g.groupby("h")},
                      "original_vs_drift": report.skill_ci(g, col[0], col[2]), "improved_vs_drift": report.skill_ci(g, col[1], col[2]),
                      "improved_vs_original": report.skill_ci(g, col[1], col[0]), "engine_vs_drift": report.skill_ci(g, col[3], col[2]),
                      "improved_vs_engine": report.skill_ci(g, col[1], col[3])}
    out["inside"] = {"original": frames["original"]["inside"], "improved": frames["improved"]["inside"]}
    out["inside_by_h"] = {k: {int(h): v["inside"] for h, v in frames[k]["by_h"].items()} for k in ("original", "improved")}
    out["usage"] = answers.usage
    (OUT / "exam.json").write_text(json.dumps(out, indent=1, default=str))
    m.to_parquet(OUT / "exam_cells.parquet", index=False)
    f = report.fmt_ci
    for wname in ("ours", "aws", "all"):
        b = out[wname]
        print(f"{wname:5s} ({b['n']} forecasts): original {f(b['original_vs_drift'])} | improved {f(b['improved_vs_drift'])} | improved vs original {f(b['improved_vs_original'])} "
              f"| engine {f(b['engine_vs_drift'])} | improved vs engine {f(b['improved_vs_engine'])}")
        print(f"        wider ranges alone vs original {f(b['stretch_only_vs_original'])} | rewritten agent on top {f(b['improved_vs_stretch_only'])} | by horizon improved vs original: "
              + "; ".join(f"{h}d {f(v['improved_vs_original'])}" for h, v in b["by_h"].items()))
    print("inside by horizon:", out["inside_by_h"])
    print("truth inside range: original", round(out["inside"]["original"], 3), "| improved", round(out["inside"]["improved"], 3), "| usage", answers.usage)


OPEN_MODELS = {"qwen3_8b": "fc1", "qwen3_5_9b": "fc2"}  # RL runs in rl/runs/


def compare() -> None:
    """One table on the exam days: every forecaster on exactly the same forecasts."""
    from . import report

    m = pd.read_parquet(OUT / "exam_cells.parquet")
    names = {"engine": ("usd_engine", "engine"), "gpt_crowd_original": ("usd_orig", "rel_orig"), "gpt_crowd_improved": ("usd_impr", "rel_impr")}
    w = 1.0 / (int(m.n_agents.iloc[0]) + 1)  # a trained open model joins the crowd as one more agent
    for tag, run_name in OPEN_MODELS.items():
        path = data.REPO / "rl" / "runs" / run_name / "eval_cells.parquet"
        if not path.exists():
            continue
        q = pd.read_parquet(path)[["id", "h", "base", "trained", "trained_lo", "trained_mid", "trained_hi"]]
        m = m.merge(q.rename(columns={c: f"{tag}_{c}" for c in q.columns if c not in ("id", "h")}), on=["id", "h"], how="inner")
        mix = {c: np.exp((1 - w) * np.log(m[f"i_{c}"]) + w * np.log(m[f"{tag}_trained_{c}"])) for c in ("lo", "mid", "hi")}
        m[f"{tag}_plus"] = [cell_loss(y, min(a, b), b, max(c, b)) for y, a, b, c in zip(m.y, mix["lo"], mix["mid"], mix["hi"])]
        for c in ("base", "trained", "plus"):
            m[f"rel_{tag}_{c}"] = m[f"{tag}_{c}"] / m["last"]
        names.update({f"{tag}_before_rl": (f"{tag}_base", f"rel_{tag}_base"), f"{tag}_after_rl": (f"{tag}_trained", f"rel_{tag}_trained"),
                      f"gpt_improved_plus_{tag}": (f"{tag}_plus", f"rel_{tag}_plus")})
    out = {}
    for wname, g in (("ours", m[m.world == "ours"]), ("aws", m[m.world == "aws"]), ("all", m)):
        k = 0 if wname != "all" else 1
        ref = ("usd_drift", "drift")[k]
        ci = lambda a, b: report.skill_ci(g, names[a][k], names[b][k])
        out[wname] = {"n": int(len(g)), "n_days": int(g.id.str.split("|").str[2].nunique()),
                      "vs_drift": {name: report.skill_ci(g, cols[k], ref) for name, cols in names.items()},
                      "improved_vs_original": ci("gpt_crowd_improved", "gpt_crowd_original"), "gpt_improved_vs_engine": ci("gpt_crowd_improved", "engine"),
                      "open_models": {tag: {"after_vs_before": ci(f"{tag}_after_rl", f"{tag}_before_rl"), "after_vs_gpt_improved": ci(f"{tag}_after_rl", "gpt_crowd_improved"),
                                            "after_vs_engine": ci(f"{tag}_after_rl", "engine"), "as_sixth_agent_vs_improved": ci(f"gpt_improved_plus_{tag}", "gpt_crowd_improved")}
                                      for tag in OPEN_MODELS if f"{tag}_after_rl" in names}}
    (data.REPO / "findings" / "data" / "fresh_days.json").write_text(json.dumps(out, indent=1))
    f = report.fmt_ci
    for wname, b in out.items():
        print(f"\n== {wname}: {b['n']} forecasts on {b['n_days']} fresh days, percent less error than drift")
        for name, v in b["vs_drift"].items():
            print(f"  {name:32s} {f(v)}")
        print(f"  improved crowd vs original crowd {f(b['improved_vs_original'])} | improved crowd vs engine {f(b['gpt_improved_vs_engine'])}")
        for tag, v in b["open_models"].items():
            print(f"  {tag}: after vs before {f(v['after_vs_before'])} | vs improved crowd {f(v['after_vs_gpt_improved'])} | vs engine {f(v['after_vs_engine'])} "
                  f"| as a sixth agent, vs improved crowd {f(v['as_sixth_agent_vs_improved'])}")


def _recorded_jobs() -> list[dict]:
    jobs = []
    for world, files in (("ours", ("extra_answers", "test_answers")), ("aws", ("aws_answers",))):
        for f in files:
            for j in json.loads((data.REPO / "runs" / "agents" / f"{f}.json").read_text()):
                jobs.append({**j, "world": world, "set": {"extra_answers": "7 earlier days", "test_answers": "7 test days", "aws_answers": "AWS, 12 days"}[f],
                             "id": f"rec|{world}|{j['series_id']}|{j['origin']}",
                             "refs": {name: {int(h): v for h, v in by_h.items()} for name, by_h in j["refs"].items()}})
    return jobs


def _improved_from_recorded(cfg: dict, answers: Answers, client, jobs: list[dict]) -> None:
    """Fill job["impr"] (the improved crowd) and job["stretch"] (original agents, new range stretch).
    Agents the coach left alone keep their recorded answers; only rewritten or new agents are asked."""
    base = base_config()
    changed = [name for name, r in cfg["roles"].items() if base["roles"].get(name) != r]
    for name in changed:
        answers.fill(client, cfg["roles"][name], jobs)

    def crowd(qs_by_h: dict, scale: dict) -> dict:
        out = {}
        for h in HORIZONS:
            lo, mid, hi = np.exp(np.mean(np.log(np.array(qs_by_h[h], dtype=float)), axis=0))
            sc = float(scale[str(h)])
            out[h] = (float(mid * (min(lo, mid) / mid) ** sc), float(mid), float(mid * (max(hi, mid) / mid) ** sc))
        return out

    for j in jobs:
        rec = {name: {h: a["q"][str(h)] for h in HORIZONS} for name, a in j["r1"].items()}
        new = {name: ({h: answers.get(cfg["roles"][name], j["id"])["q"][str(h)] for h in HORIZONS} if name in changed else rec[name]) for name in cfg["roles"]}
        j["orig"] = crowd({h: [q[h] for q in rec.values()] for h in HORIZONS}, base["range_scale"])
        j["stretch"] = crowd({h: [q[h] for q in rec.values()] for h in HORIZONS}, cfg["range_scale"])
        j["impr"] = crowd({h: [q[h] for q in new.values()] for h in HORIZONS}, cfg["range_scale"])
        j["new_answers"] = {name: answers.get(cfg["roles"][name], j["id"]) for name in changed}


def confirm() -> None:
    """A second, larger check of the kept changes, on days the original crowd had answered before
    the coach existed (none of them practice or check days)."""
    from dotenv import load_dotenv
    from openai import OpenAI

    from . import report

    load_dotenv(data.REPO / ".env")
    jobs = _recorded_jobs()
    cfg = json.loads((OUT / "state.json").read_text())["config"]
    used = {(w, d) for split in ("practice", "check") for w, ds in DAYS[split].items() for d in ds}
    assert not any((j["world"], j["origin"]) in used for j in jobs)
    answers = Answers()
    _improved_from_recorded(cfg, answers, OpenAI(), jobs)
    worlds = {"ours": data.build_world(data.DATA_END), "aws": data.load_world(aws_test.WORK / "world.pkl")}
    rows = []
    for j in jobs:
        s = worlds[j["world"]]["series"][j["series_id"]]
        od = day_of(j["origin"])
        for h in HORIZONS:
            k = int(np.searchsorted(s["days"], od + h))
            if k >= len(s["days"]) or s["days"][k] != od + h:
                continue
            y = float(s["value"][k])
            row = {"set": j["set"], "world": j["world"], "series_id": j["series_id"], "origin": j["origin"], "h": h, "target": od + h, "last": j["last"],
                   "drift": cell_loss(y, *j["refs"]["drift"][h]), "engine": cell_loss(y, *j["refs"]["engine"][h])}
            for k2 in ("orig", "stretch", "impr"):
                row[k2] = cell_loss(y, *j[k2][h])
                row[k2 + "_inside"] = bool(j[k2][h][0] <= y <= j[k2][h][2])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "confirm_cells.parquet", index=False)
    out, f = {}, report.fmt_ci
    ours = df[df.world == "ours"]
    for name, g in list(df.groupby("set", sort=False)) + [("our prices, all 14 days", ours)]:
        out[name] = {"n": int(len(g)), "n_days": int(g.origin.nunique()),
                     "original_vs_drift": report.skill_ci(g, "orig", "drift"), "improved_vs_drift": report.skill_ci(g, "impr", "drift"),
                     "improved_vs_original": report.skill_ci(g, "impr", "orig"), "stretch_only_vs_original": report.skill_ci(g, "stretch", "orig"),
                     "improved_vs_stretch_only": report.skill_ci(g, "impr", "stretch"), "improved_vs_engine": report.skill_ci(g, "impr", "engine"),
                     "inside": {"original": float(g.orig_inside.mean()), "improved": float(g.impr_inside.mean())},
                     "inside_by_h": {int(h): {"original": float(gh.orig_inside.mean()), "improved": float(gh.impr_inside.mean())} for h, gh in g.groupby("h")}}
        b = out[name]
        print(f"{name} ({b['n']} forecasts): original {f(b['original_vs_drift'])} | improved {f(b['improved_vs_drift'])} | improved vs original {f(b['improved_vs_original'])}\n"
              f"     wider ranges alone {f(b['stretch_only_vs_original'])} | rewritten agent on top {f(b['improved_vs_stretch_only'])} | improved vs engine {f(b['improved_vs_engine'])} "
              f"| inside {b['inside']['original']:.2f} -> {b['inside']['improved']:.2f}")
    out["usage"] = answers.usage
    (OUT / "confirm.json").write_text(json.dumps(out, indent=1))
    print("usage", answers.usage)


def final() -> None:
    """The improved crowd's 6 Nov forecast. Agents the coach left alone keep the answers recorded
    when the original crowd made its forecast; only rewritten or new agents are asked, from the same facts."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(data.REPO / ".env")
    jobs = json.loads((data.REPO / "runs" / "agents" / "final_answers.json").read_text())
    for jb in jobs:
        jb["id"] = f"final|{jb['series_id']}|{jb['origin']}"
        jb["refs_recorded"] = jb["refs"]
        jb["refs"] = {name: {int(h): v for h, v in by_h.items()} for name, by_h in jb["refs"].items()}
    cfg = json.loads((OUT / "state.json").read_text())["config"]
    answers = Answers()
    _improved_from_recorded(cfg, answers, OpenAI(), jobs)
    out = [{"series_id": jb["series_id"], "origin": jb["origin"], "last": jb["last"], "facts": jb["facts"], "refs": jb["refs_recorded"],
            "r1": {name: (jb["new_answers"][name] if name in jb["new_answers"] else {"q": jb["r1"][name]["q"], "reason": jb["r1"][name]["reason"]}) for name in cfg["roles"]},
            "asked_again": sorted(jb["new_answers"]), "range_scale": cfg["range_scale"], "crowd": {str(h): list(jb["impr"][h]) for h in HORIZONS}}
           for jb in jobs]
    (OUT / "final_answers.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / 'final_answers.json'}: {len(out)} prices, {len(cfg['roles'])} agents, asked again: {out[0]['asked_again']}, usage {answers.usage}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "exam", "compare", "confirm", "final"])
    ap.add_argument("--generations", type=int, default=8)
    a = ap.parse_args()
    {"run": lambda: run(a.generations), "exam": exam, "compare": compare, "confirm": confirm, "final": final}[a.cmd]()
