"""OpenAI calls for the experiments added on 15 Nov: one helper, with a spending meter.

Calls go to the cheaper "flex" tier first (same model, slower) and fall back to the standard tier
when flex has no capacity. The meter adds up tokens and an estimated cost and stops the run if a
budget is passed. Prices are USD per million tokens for gpt-6.1-sol as read from OpenAI's pricing
page on 15 Nov 2026; they are only used for the estimate.
"""
from __future__ import annotations

import json
import threading
import time

PRICE = {"flex": (1.00, 5.00), "standard": (2.00, 10.00)}     # (input, output) per million tokens


class BudgetExceeded(RuntimeError):
    pass


class Meter:
    def __init__(self, budget_usd: float):
        self.budget = float(budget_usd)
        self.lock = threading.Lock()
        self.use = {"calls": 0, "in": 0, "out": 0, "flex_calls": 0, "standard_calls": 0, "usd": 0.0}

    def add(self, tier: str, tin: int, tout: int) -> None:
        pin, pout = PRICE[tier]
        with self.lock:
            u = self.use
            u["calls"] += 1; u["in"] += tin; u["out"] += tout; u[f"{tier}_calls"] += 1
            u["usd"] += (tin * pin + tout * pout) / 1e6

    def check(self) -> None:
        if self.use["usd"] > self.budget:
            raise BudgetExceeded(f"estimated spend ${self.use['usd']:.2f} is over the budget of ${self.budget:.2f}")

    def summary(self) -> dict:
        with self.lock:
            return {**self.use, "usd": round(self.use["usd"], 2)}


def ask_json(client, model: str, effort: str, system: str, user: str, schema: dict, name: str, meter: Meter,
             flex: bool = True) -> dict:
    """One structured answer. Flex first (three tries), then the standard tier (three tries)."""
    meter.check()
    last = None
    plan = (["flex"] * 3 if flex else []) + ["standard"] * 3
    for attempt, tier in enumerate(plan):
        try:
            kw = {"service_tier": "flex"} if tier == "flex" else {}
            r = client.with_options(timeout=900.0).responses.create(
                model=model, reasoning={"effort": effort},
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}, **kw)
            u = getattr(r, "usage", None)
            meter.add(tier, int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0))
            return json.loads(r.output_text)
        except Exception as e:      # no capacity on flex, a timeout, a rate limit: wait and try the next step of the plan
            last = e
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"call failed after {len(plan)} tries: {last}")
