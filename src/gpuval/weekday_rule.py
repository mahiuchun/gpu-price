"""Does the weekday pattern the analyst found make forecasts better? A rule with no AI in it.

    uv run python -m gpuval.weekday_rule

The analyst (gpuval.analyst) found that prices are repriced more on some weekdays than others, and
the pattern held on newer data. Some published prices turn out to move on certain weekdays only.

The rule, for 1-day forecasts only: look at this price's own history up to the forecast day. If the
published price has never moved into the weekday being forecast, in at least 8 past cases, forecast
"no change" with no range. Otherwise leave the forecast as it was. It uses nothing after the
forecast day and has no setting fitted to results.

It is applied to the engine and to drift on every forecast day from 21 Aug to 5 Nov, and to the
first crowd's recorded answers on its 19 days. Written in PROTOCOL.md, deviation 12, and committed
before the first run.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data, harness, range_memory, report
from .core import cell_loss, day_of
from .engine import default_spec

MIN_CASES = 8
FROM, TO = "2026-08-21", "2026-11-05"
OUT = data.REPO / "findings" / "data"


def quiet_weekday(full: dict, cells: pd.DataFrame) -> pd.Series:
    """True for a 1-day forecast whose target weekday has never seen this price move, in MIN_CASES or more
    earlier consecutive-day pairs dated on or before the forecast day."""
    out = np.zeros(len(cells), dtype=bool)
    cases = np.zeros(len(cells), dtype=int)
    pos = {idx: n for n, idx in enumerate(cells.index)}
    for sid, g in cells.groupby("series_id", sort=False):
        s = full["series"][sid]
        days, val = s["days"], s["value"]
        pair_ok = np.diff(days) == 1                      # pair k is (days[k], days[k+1])
        moved = np.abs(np.diff(val)) > 1e-9
        wd = days[1:] % 7                                 # weekday of the later day of each pair
        for idx, o, h in zip(g.index, g.origin, g.h):
            if h != 1:
                continue
            past = pair_ok & (days[1:] <= o) & (wd == (o + 1) % 7)
            n = int(past.sum())
            cases[pos[idx]] = n
            out[pos[idx]] = n >= MIN_CASES and not moved[past].any()
    return pd.Series(out, index=cells.index), pd.Series(cases, index=cells.index)


def main() -> int:
    f = report.fmt_ci
    full = data.build_world(data.DATA_END)
    cells = harness.run_spec(full, default_spec(), harness.base_cells(full)).rename(columns={"loss": "loss_engine"})
    cells = cells[(cells.origin >= day_of(FROM)) & (cells.origin <= day_of(TO))].copy()
    cells["quiet"], cells["cases"] = quiet_weekday(full, cells)
    cells["loss_engine_rule"] = np.where(cells.quiet, cells.loss_last, cells.loss_engine)
    cells["loss_drift_rule"] = np.where(cells.quiet, cells.loss_last, cells.loss_drift)
    cells["moved"] = (cells.y - cells["last"]).abs() > 1e-9
    one = cells[cells.h == 1]
    q = one[one.quiet]
    out = {"rule": {"min_cases": MIN_CASES, "horizon": 1, "from": FROM, "to": TO},
           "one_day_forecasts": int(len(one)), "rule_fired": int(len(q)), "prices_where_it_fired": int(q.series_id.nunique()),
           "price_moved_when_it_fired": int(q.moved.sum()), "price_moved_otherwise": [int(one[~one.quiet].moved.sum()), int((~one.quiet).sum())],
           "engine": {"one_day": {"rule_vs_plain": report.skill_ci(one, "loss_engine_rule", "loss_engine"),
                                  "plain_vs_drift": report.skill_ci(one, "loss_engine", "loss_drift"), "rule_vs_drift": report.skill_ci(one, "loss_engine_rule", "loss_drift")},
                      "all_horizons": {"rule_vs_plain": report.skill_ci(cells, "loss_engine_rule", "loss_engine"),
                                       "plain_vs_drift": report.skill_ci(cells, "loss_engine", "loss_drift"), "rule_vs_drift": report.skill_ci(cells, "loss_engine_rule", "loss_drift")}},
           "drift": {"one_day": {"rule_vs_plain": report.skill_ci(one, "loss_drift_rule", "loss_drift")}},
           "by_price": {sid.replace(".median", ""): {"fired": int(len(g)), "moved": int(g.moved.sum()),
                                                      "engine_error_removed_usd": float((g.loss_engine - g.loss_last).sum())}
                        for sid, g in q.groupby("series_id")}}
    # the first crowd's recorded answers on its 19 days
    crowd = range_memory.crowd_frame(cells.drop(columns=["quiet", "cases"]).rename(columns={"loss_engine": "loss_engine_"}))
    crowd = crowd[crowd.origin.isin({day_of(d) for d in range_memory.SCORE_DAYS})].copy()
    crowd["loss_crowd"] = cell_loss(crowd.y, crowd.lo, crowd.mid, crowd.hi)
    crowd["quiet"], _ = quiet_weekday(full, crowd)
    crowd["loss_crowd_rule"] = np.where(crowd.quiet, crowd.loss_last, crowd.loss_crowd)
    c1 = crowd[crowd.h == 1]
    out["first_crowd"] = {"one_day_forecasts": int(len(c1)), "rule_fired": int(c1.quiet.sum()),
                          "price_moved_when_it_fired": int(((c1.y - c1["last"]).abs() > 1e-9)[c1.quiet].sum()),
                          "one_day": {"rule_vs_plain": report.skill_ci(c1, "loss_crowd_rule", "loss_crowd")},
                          "all_horizons": {"rule_vs_plain": report.skill_ci(crowd, "loss_crowd_rule", "loss_crowd"),
                                           "plain_vs_drift": report.skill_ci(crowd, "loss_crowd", "loss_drift"), "rule_vs_drift": report.skill_ci(crowd, "loss_crowd_rule", "loss_drift")}}
    (OUT / "weekday_rule.json").write_text(json.dumps(out, indent=1))
    print(f"1-day forecasts {out['one_day_forecasts']}; the rule fired on {out['rule_fired']} of them, on {out['prices_where_it_fired']} prices; "
          f"the price moved in {out['price_moved_when_it_fired']} of those (elsewhere: {out['price_moved_otherwise'][0]} of {out['price_moved_otherwise'][1]})")
    e = out["engine"]
    print(f"ENGINE, 1-day: rule vs plain {f(e['one_day']['rule_vs_plain'])} | plain vs drift {f(e['one_day']['plain_vs_drift'])} | with rule vs drift {f(e['one_day']['rule_vs_drift'])}")
    print(f"ENGINE, all horizons: rule vs plain {f(e['all_horizons']['rule_vs_plain'])} | plain vs drift {f(e['all_horizons']['plain_vs_drift'])} | with rule vs drift {f(e['all_horizons']['rule_vs_drift'])}")
    print(f"DRIFT, 1-day: rule vs plain {f(out['drift']['one_day']['rule_vs_plain'])}")
    c = out["first_crowd"]
    print(f"FIRST CROWD ({c['one_day_forecasts']} 1-day forecasts, rule fired on {c['rule_fired']}, price moved in {c['price_moved_when_it_fired']}): "
          f"1-day rule vs plain {f(c['one_day']['rule_vs_plain'])} | all horizons {f(c['all_horizons']['rule_vs_plain'])}")
    for sid, v in sorted(out["by_price"].items(), key=lambda kv: -kv[1]["fired"])[:10]:
        print(f"   {sid:58s} fired {v['fired']:3d}, price moved {v['moved']}, engine error removed ${v['engine_error_removed_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
