"""The agent crowd answered by another model, on the coach's 479 exam forecasts of our prices.

    uv run python runs/other_models/run_crowd.py --env .env --dry
    uv run python runs/other_models/run_crowd.py --env .env --model claude-opus-5-5 --cap 15
    uv run python runs/other_models/table.py          # score every model's answers; no AI calls

Needs the `anthropic` package for Claude models and `tinker` for open models (neither is a dependency of the repo).

The crowd is gpuval.coach's: five agents with different jobs each answer once about every price-day and the
answers are averaged. Six sets of instructions are asked: the five as first written, plus the one agent the coach
rewrote (base_rate_keeper). That gives two crowds per model: the ORIGINAL crowd, and the IMPROVED crowd (the
rewritten agent and the wider ranges, both chosen by the coach while it was working with gpt-6.1-sol).
Prompts, answer format and the averaging are this repo's own code (coach.build_items, agents.SCHEMA,
coach.crowd_forecast); only the model call changes. Reasoning effort is low, as in the recorded gpt-6.1-sol run.

--dry asks nothing: it rebuilds the prompts, checks them against the recorded run (the same price-days, the same
outcomes, and the recorded gpt-6.1-sol answers scored through this file give the recorded numbers) and prints sizes.

Saving. Answers and the spend ledger are written as a set of instructions goes along and again when it ends,
including when it ends on an error or the spending limit, so a stopped run continues where it left off and its
spend still counts. A folder holds one model at one reasoning effort: a run with other settings is refused.

Spending limit. Each call is priced from the token counts the API returns. Before a call is sent, the most it
could cost is set aside: every byte of the prompt counted as a token, an allowance for what the API adds, the
largest answer allowed (--max-out), and no flex discount. If the money spent, plus what is set aside for calls
still running, plus this call's share would pass --cap, the call is not sent and the run stops. So --cap needs
room above the expected spend for as many calls as run at once (the run prints the amount). A call that fails
without returning token counts is not metered.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from gpuval import agents, coach, data, report

HERE = Path(__file__).resolve().parent
# USD per million tokens (input, cached input, cache write, output), read 15 Nov 2026. OpenAI: standard tier, short
# context, https://developers.openai.com/api/docs/pricing (flex is half of every number). Anthropic: standard rates.
# Tinker: https://tinker-docs.thinkingmachines.ai/tinker/models.json.
PRICE = {
    "gpt-6.1-sol": {"prefill": 2.00, "cached": 0.10, "cache_write": 2.50, "sample": 10.00},
    "gpt-6-luna": {"prefill": 0.10, "cached": 0.01, "cache_write": 0.125, "sample": 0.50},
    "gpt-6-astra": {"prefill": 10.00, "cached": 1.00, "cache_write": 12.50, "sample": 50.00},
    "claude-sonnet-5-5": {"prefill": 2.00, "cached": 0.10, "cache_write": 2.50, "sample": 10.00},
    "claude-opus-5-5": {"prefill": 4.00, "cached": 0.20, "cache_write": 5.00, "sample": 20.00},
    "claude-fable-5-2": {"prefill": 8.00, "cached": 0.20, "cache_write": 10.00, "sample": 40.00},
    "Qwen/Qwen3.5-9B": {"prefill": 0.30, "cached": 0.06, "cache_write": 0.30, "sample": 0.807},  # Tinker, base model, no training
}
MAX_OUT = 8000  # largest answer allowed, hidden reasoning included; answers are about 250 tokens (--max-out)
INPUT_ALLOWANCE = 2000  # tokens set aside for what the API adds to a prompt (answer format, chat template)
SAVE_EVERY = 25  # answers between saves within one set of instructions
# Tinker cannot force the answer format, so an open model is told the format in words. An answer that still cannot be
# read is asked for once more; if that fails too the agent's answer is "no change from today's price".
OPEN_FORMAT = ('\nReply with one JSON object and nothing else, in exactly this form: {"h1": {"low": 0.0, "mid": 0.0, "high": 0.0}, '
               '"h3": {"low": 0.0, "mid": 0.0, "high": 0.0}, "h7": {"low": 0.0, "mid": 0.0, "high": 0.0}, "reason": "one or two plain sentences"}')
OPEN_TEMPERATURE, OPEN_MAX_TOKENS, OPEN_TRIES = 0.7, 500, 2  # one answer per agent, sampled at temperature 0.7


class BudgetExceeded(RuntimeError):
    pass


def read_open_answer(raw: str) -> dict:
    """The JSON object in an open model's reply. Untrained Qwen3.5-9B stops one character early, before the final
    brace, in almost half its answers (the numbers and the reason are all there), so that brace is put back."""
    lo = raw.find("{")
    body = (raw[lo:] if lo >= 0 else raw).rstrip()
    for cand in (body[: body.rfind("}") + 1], body + "}", body + '"}'):
        try:
            ans = json.loads(cand)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(ans, dict) and all(isinstance(ans.get(f"h{h}"), dict) for h in agents.HORIZONS):
            return ans
    raise ValueError("no JSON object with all three horizons")


class Store:
    """Answers keyed exactly as gpuval.coach keys them (instructions + price-day)."""

    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text()) if path.exists() else {}

    def get(self, role: dict, item_id: str) -> dict:
        return self.d[coach.Answers.key(role, item_id)]


def prompt(role: dict, it: dict) -> tuple[str, str]:
    user = it["facts"] + ("\n" + agents.refs_text(it["refs"]) if role["refs"] else "") + "\n\nGive your forecast."
    return coach.SYSTEM.format(role=role["instruction"]), user


def frames(configs: dict, store: Store, items: list[dict]) -> pd.DataFrame:
    out = None
    for tag, cfg in configs.items():
        df = coach.score(cfg, store, items)["df"]
        if out is None:
            out = df[["id", "h", "y", "last", "usd_drift", "usd_engine"]].copy()
        for c in ("usd_loss", "lo", "mid", "hi"):
            out[f"{tag}_{c}"] = df[c].values
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--model")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", help="folder for answers and spend (default: <model> next to this file, with _<effort> added unless low)")
    ap.add_argument("--cap", type=float, default=1.0, help="spending limit in USD for this folder, earlier runs included")
    ap.add_argument("--max-out", type=int, default=MAX_OUT, help="largest answer allowed, in tokens; the spending limit sets this much aside per call")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tier", default="flex", choices=["flex", "standard"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="only the first N price-days (a test)")
    ap.add_argument("--redo-unreadable", action="store_true", help="ask again for the answers that were recorded as unreadable")
    a = ap.parse_args()

    items = sorted((it for it in coach.build_items().values() if it["split"] == "exam" and it["world"] == "ours"), key=lambda it: it["id"])
    state = json.loads((coach.OUT / "state.json").read_text())
    configs = {"orig": coach.base_config(), "impr": state["config"]}
    roles = {}
    for cfg in configs.values():
        for name, r in cfg["roles"].items():
            roles.setdefault(coach.Answers.key(r, ""), (name, r))
    jobs = [(r, it) for _, r in roles.values() for it in items]
    chars = sum(len(s) + len(u) for s, u in (prompt(r, it) for r, it in jobs))
    print(f"{len(items)} price-days, {len(roles)} sets of instructions, {len(jobs)} calls, {chars} prompt characters", flush=True)

    # the same forecasts as the recorded exam?
    cells = pd.read_parquet(coach.OUT / "exam_cells.parquet")
    cells = cells[cells.world == "ours"]
    mine = {(it["id"], int(h)): y for it in items for h, y in it["y"].items()}
    rec = {(r.id, int(r.h)): r.y for r in cells.itertuples()}
    print(f"forecasts: {len(mine)} here, {len(rec)} recorded, same set: {set(mine) == set(rec)}, "
          f"largest difference in outcome: {max(abs(mine[k] - rec[k]) for k in mine if k in rec):.2e}", flush=True)

    if a.dry:
        recorded = Store(coach.OUT / "answers.json")
        df = frames(configs, recorded, items).merge(cells[["id", "h", "usd_orig", "usd_impr"]], on=["id", "h"])
        print(f"recorded gpt-6.1-sol answers scored here: original {1 - df.orig_usd_loss.sum() / df.usd_drift.sum():+.4f} "
              f"(recorded {1 - df.usd_orig.sum() / df.usd_drift.sum():+.4f}), improved {1 - df.impr_usd_loss.sum() / df.usd_drift.sum():+.4f} "
              f"(recorded {1 - df.usd_impr.sum() / df.usd_drift.sum():+.4f}); largest difference per forecast "
              f"{max((df.orig_usd_loss - df.usd_orig).abs().max(), (df.impr_usd_loss - df.usd_impr).abs().max()):.2e} USD")
        reasons = [len(json.dumps({"h1": {"low": 1.2345, "mid": 1.2345, "high": 1.2345}, "h3": {"low": 1.2345, "mid": 1.2345, "high": 1.2345},
                                   "h7": {"low": 1.2345, "mid": 1.2345, "high": 1.2345}, "reason": recorded.get(r, it["id"])["reason"]})) for r, it in jobs]
        print(f"recorded answers: {sum(reasons)} characters in all, {sum(reasons) / len(reasons):.0f} per answer")
        for line in Path(a.env).read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic

            client = anthropic.Anthropic(max_retries=6)
            sample = jobs[:: max(1, len(jobs) // 90)]

            def count(job):
                s, u = prompt(*job)
                return client.messages.count_tokens(model="claude-sonnet-5-5", system=s, messages=[{"role": "user", "content": u}]).input_tokens, len(s) + len(u)

            with ThreadPoolExecutor(max_workers=4) as ex:
                got = list(ex.map(count, sample))
            tok, ch = sum(t for t, _ in got), sum(c for _, c in got)
            print(f"Claude token count (free) on {len(sample)} prompts: {tok} tokens for {ch} characters = {ch / tok:.3f} characters per token "
                  f"-> about {chars * tok / ch / 1e6:.3f}M input tokens for all {len(jobs)} calls")
        return

    out = Path(a.out) if a.out else HERE / (a.model.split("/")[-1] + ("" if a.effort == "low" else f"_{a.effort}"))
    out.mkdir(parents=True, exist_ok=True)
    store = Store(out / "answers.json")
    ledger_path = out / "spend.json"
    # One folder holds one model at one effort. Saved answers are keyed by instructions and price-day only, so
    # answers from other settings would be reused without a word.
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if (ledger.get("model"), ledger.get("effort")) != (a.model, a.effort):
            raise SystemExit(f"{out} holds answers from {ledger.get('model')} at effort {ledger.get('effort')}; this run asks for "
                             f"{a.model} at effort {a.effort}. Give another folder with --out.")
    elif store.d:
        raise SystemExit(f"{out} holds answers but no spend.json, so the model and effort they came from are unknown. Give another folder with --out.")
    else:
        ledger = {"model": a.model, "effort": a.effort, "usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0,
                  "thinking_tokens": 0, "flex_calls": 0, "retries": 0, "bad_answers": 0}
    if a.redo_unreadable:
        drop = [k for k, v in store.d.items() if v.get("unreadable")]
        for k in drop:
            del store.d[k]
        print(f"asking again for {len(drop)} answers that were recorded as unreadable", flush=True)
    price = PRICE[a.model]
    claude, open_model = a.model.startswith("claude-"), "/" in a.model
    wanted = "ANTHROPIC_API_KEY" if claude else "TINKER_API_KEY" if open_model else "OPENAI_API_KEY"
    for line in Path(a.env).read_text().splitlines():
        if line.startswith(wanted + "="):
            os.environ[wanted] = line.split("=", 1)[1].strip().strip('"').strip("'")
    if claude:
        import anthropic as sdk

        client = sdk.Anthropic(max_retries=0, timeout=900)
    elif open_model:
        import tinker as sdk

        fut = sdk.ServiceClient().create_sampling_client(base_model=a.model)
        client = fut.result() if hasattr(fut, "result") else fut
        tok = client.get_tokenizer()
    else:
        import openai as sdk

        client = sdk.OpenAI(max_retries=0, timeout=900)
    retryable = (sdk.RateLimitError, sdk.APITimeoutError, sdk.APIConnectionError, sdk.InternalServerError)
    lock = threading.Lock()
    room_freed = threading.Condition(lock)
    state_ = {"set_aside": 0.0, "running": 0}  # USD set aside for the calls that are running, and how many there are

    def most(system: str, user: str) -> float:
        """The most one call can cost. A token is at least one byte, so the prompt's bytes bound its tokens; the
        input is priced at the dearest input rate and the answer at its largest allowed size, with no flex discount."""
        n_in = len((system + user + (OPEN_FORMAT if open_model else "")).encode()) + INPUT_ALLOWANCE
        n_out = OPEN_MAX_TOKENS if open_model else a.max_out
        return (n_in * max(price["prefill"], price["cache_write"]) + n_out * price["sample"]) / 1e6

    def call(system: str, user: str, attempt: int) -> tuple[str, dict]:
        if open_model:
            text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user + OPEN_FORMAT}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(text, add_special_tokens=False)
            r = client.sample(prompt=sdk.types.ModelInput.from_ints(ids), num_samples=1,
                              sampling_params=sdk.types.SamplingParams(max_tokens=OPEN_MAX_TOKENS, temperature=OPEN_TEMPERATURE)).result()
            got = list(r.sequences[0].tokens)
            return tok.decode(got, skip_special_tokens=True), {"in": len(ids), "out": len(got), "flex": 0, "think": 0,
                                                              "usd": (len(ids) * price["prefill"] + len(got) * price["sample"]) / 1e6}
        if claude:
            # no fallback model on a refusal: another model's answer would not be this model's score
            r = client.messages.create(model=a.model, max_tokens=a.max_out, system=system, messages=[{"role": "user", "content": user}],
                                       output_config={"effort": a.effort, "format": {"type": "json_schema", "schema": agents.SCHEMA}})
            u = r.usage
            cached, write = int(u.cache_read_input_tokens or 0), int(u.cache_creation_input_tokens or 0)
            usd = (u.input_tokens * price["prefill"] + cached * price["cached"] + write * price["cache_write"] + u.output_tokens * price["sample"]) / 1e6
            return "".join(b.text for b in r.content if b.type == "text"), {
                "in": u.input_tokens + cached + write, "out": u.output_tokens, "usd": usd, "flex": 0,
                "think": int(getattr(getattr(u, "output_tokens_details", None), "thinking_tokens", 0) or 0)}
        kw = {"model": a.model, "reasoning": {"effort": a.effort}, "max_output_tokens": a.max_out, "prompt_cache_options": {"mode": "explicit"},
              "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
              "text": {"format": {"type": "json_schema", "name": "forecast", "schema": agents.SCHEMA, "strict": True}}}
        if a.tier == "flex" and attempt < 4:
            kw["service_tier"] = "flex"
        r = client.responses.create(**kw)
        u = r.usage
        din = getattr(u, "input_tokens_details", None)
        cached, write = int(getattr(din, "cached_tokens", 0) or 0), int(getattr(din, "cache_write_tokens", 0) or 0)
        flex = getattr(r, "service_tier", None) == "flex"
        usd = ((u.input_tokens - cached - write) * price["prefill"] + cached * price["cached"] + write * price["cache_write"]
               + u.output_tokens * price["sample"]) / 1e6 * (0.5 if flex else 1.0)
        return r.output_text or "", {"in": u.input_tokens, "out": u.output_tokens, "usd": usd, "flex": int(flex),
                                     "think": int(getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0)}

    def one(job):
        role, it = job
        system, user = prompt(role, it)
        last, unread, w = None, 0, most(system, user)
        for attempt in range(12):
            with room_freed:
                while True:
                    if ledger["usd"] + w > a.cap:  # it would not fit even with nothing else running
                        raise BudgetExceeded(f"spent ${ledger['usd']:.4f} and the next call could cost ${w:.4f}; cap ${a.cap}")
                    if ledger["usd"] + (state_["set_aside"] if state_["running"] else 0.0) + w <= a.cap:
                        break
                    room_freed.wait()  # a running call will cost less than was set aside for it; wait for it, then look again
                state_["set_aside"] += w
                state_["running"] += 1
            try:
                text, c = call(system, user, attempt)
            except BaseException as e:
                with room_freed:
                    state_["set_aside"] -= w
                    state_["running"] -= 1
                    room_freed.notify_all()
                if not isinstance(e, retryable):
                    raise
                last = e
                try:
                    asked = float(e.response.headers.get("retry-after", 0))
                except Exception:  # noqa: BLE001
                    asked = 0.0
                time.sleep(min(90.0, max(asked, 3 + 5 * attempt)))
                with lock:
                    ledger["retries"] += 1
                continue
            with room_freed:  # what was set aside becomes what was spent, in one step
                state_["set_aside"] -= w
                state_["running"] -= 1
                room_freed.notify_all()
                ledger["usd"] += c["usd"]
                ledger["calls"] += 1
                ledger["input_tokens"] += c["in"]
                ledger["output_tokens"] += c["out"]
                ledger["thinking_tokens"] += c["think"]
                ledger["flex_calls"] += c["flex"]
            try:
                ans = read_open_answer(text) if open_model else json.loads(text)
                return it["id"], {"q": {str(h): list(v) for h, v in agents.clean(ans, it["last"]).items()}, "reason": str(ans.get("reason", ""))}
            except Exception as e:  # noqa: BLE001  (a cut-off or refused answer: paid for, asked again)
                last, unread = e, unread + 1
                with lock:
                    ledger["bad_answers"] += 1
                if open_model and unread >= OPEN_TRIES:  # an open model that cannot give the format: no change from today's price
                    with lock:
                        ledger["flat_answers"] = ledger.get("flat_answers", 0) + 1
                    return it["id"], {"q": {str(h): list(v) for h, v in agents.clean({}, it["last"]).items()}, "reason": "", "unreadable": True}
        raise RuntimeError(f"no usable answer after 12 tries: {last}")

    todo_items = items[: a.limit] if a.limit else items
    room = a.workers * max(most(*prompt(r, it)) for r, it in jobs)
    print(f"{a.model}, effort {a.effort}, {a.workers} parallel calls, cap ${a.cap}, already spent ${ledger['usd']:.4f}; "
          f"up to ${room:.2f} is set aside for calls that are running, and calls wait their turn when the cap is that close", flush=True)

    def save() -> None:
        with lock:
            store.path.write_text(json.dumps(store.d))
            ledger_path.write_text(json.dumps(ledger, indent=1))

    for name, role in roles.values():
        todo = [(role, it) for it in todo_items if coach.Answers.key(role, it["id"]) not in store.d]
        t0, before = time.time(), ledger["usd"]
        if todo:
            futures, kept, stopped = [], set(), None

            def keep(fut) -> None:
                if fut not in kept and fut.done() and not fut.cancelled() and fut.exception() is None:
                    iid, ans = fut.result()
                    store.d[coach.Answers.key(role, iid)] = ans
                    kept.add(fut)

            try:
                with ThreadPoolExecutor(max_workers=a.workers) as ex:
                    futures = [ex.submit(one, job) for job in todo]
                    try:
                        for n, fut in enumerate(as_completed(futures), 1):
                            fut.result()  # the first failure stops the run here
                            keep(fut)
                            if n % SAVE_EVERY == 0:
                                save()
                    except BaseException:
                        for fut in futures:
                            fut.cancel()  # send nothing new; calls already running finish as the pool closes
                        raise
            except BudgetExceeded as e:
                stopped = str(e)
            finally:  # every answer that was paid for is kept, and the spend with it, however the set ended
                for fut in futures:
                    keep(fut)
                save()
            if stopped:
                raise SystemExit(f"stopped at the spending limit while asking {name}: {stopped}. {len(store.d)} answers and the spend are saved "
                                 f"in {out}; run again with a higher --cap to continue.")
        print(f"  {name:18s} {len(todo):4d} calls, ${ledger['usd'] - before:.4f}, {time.time() - t0:.0f} s | total ${ledger['usd']:.4f} "
              f"({ledger['retries']} retries, {ledger['bad_answers']} bad answers)", flush=True)

    df = frames(configs, store, todo_items)
    df.to_parquet(out / "cells.parquet", index=False)
    f = lambda v: f"{100 * v['skill']:+.1f}% ({100 * v['lo90']:+.1f} to {100 * v['hi90']:+.1f})"
    df["series_id"], df["target"] = df.id.str.split("|").str[1], df.id.str.split("|").str[2].map(coach.day_of) + df.h
    print(f"{len(df)} forecasts. original crowd {f(report.skill_ci(df, 'orig_usd_loss', 'usd_drift'))} | improved crowd "
          f"{f(report.skill_ci(df, 'impr_usd_loss', 'usd_drift'))} less error than drift")
    print(f"done. spent ${ledger['usd']:.4f} on {ledger['calls']} calls ({ledger['input_tokens']} input + {ledger['output_tokens']} output tokens, "
          f"{ledger['thinking_tokens']} of them thinking; {ledger['flex_calls']} at flex)")


if __name__ == "__main__":
    main()
