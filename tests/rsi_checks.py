"""Checks on the self-improvement loop (gpuval.rsi). No AI calls.

    uv run python tests/rsi_checks.py

1. The starting bundle is the forecast we handed in: on every day the improved crowd was scored
   before, it gives the same numbers as the coach's own files.
2. The learning vote never uses an outcome published after the forecast day.
3. Weights stay on the simplex and above the floor; edits are clipped to their bounds; a run's
   kept edits rebuild its final bundle.
"""
import numpy as np
import pandas as pd

from gpuval import coach, data, rsi
from gpuval.core import cell_loss, day_of

f = rsi.build_frame()
start = rsi.start_bundle()
w = rsi.walk(f, start)
k = ["series_id", "origin", "h"]

# 1a. the coach's exam days: low / middle / high as recorded
e = pd.read_parquet(coach.OUT / "exam_cells.parquet")
e = e[e.world == "ours"].assign(origin=lambda d: d.target - d.h)
m = w.merge(e[k + ["i_lo", "i_mid", "i_hi", "usd_impr"]], on=k, how="inner")
assert len(m) == len(e) == 479, (len(m), len(e))
for a, b in (("lo", "i_lo"), ("mid", "i_mid"), ("hi", "i_hi"), ("loss", "usd_impr")):
    assert np.allclose(m[a], m[b], rtol=0, atol=1e-9), a
# 1b. the 14 days of the second check: the same error, forecast by forecast
c = pd.read_parquet(coach.OUT / "confirm_cells.parquet")
c = c[c.world == "ours"].assign(origin=lambda d: d.origin.map(day_of))
m = w.merge(c[k + ["impr"]], on=k, how="inner")
assert len(m) == len(c) == 1415, (len(m), len(c))
assert np.allclose(m.loss, m.impr, rtol=0, atol=1e-9)
print(f"1. starting bundle = the improved crowd on {len(e)} exam-day and {len(c)} second-check forecasts")

# 2. no outcome from after the forecast day reaches a forecast
learn = {**start, "members": list(rsi.POOL), "vote": {"eta": 2.0, "decay": 0.05, "w_min": 0.02, "by": "class_h"}, "widen": 0.5}
full = rsi.walk(f, learn)
days = sorted(f.origin.unique())
for d in (days[5], days[12], days[-3]):
    g = f.copy()
    late = g.target > d
    g.loc[late, "y"] = g.loc[late, "y"] * 3.0                      # change every outcome published after day d
    g["loss_last"] = cell_loss(g.y, g["last"], g["last"], g["last"])
    alt = rsi.walk(g, learn)
    upto = full.origin <= d
    for col in ("lo", "mid", "hi"):
        assert np.array_equal(full.loc[upto, col].to_numpy(), alt.loc[upto.to_numpy(), col].to_numpy()), (d, col)
    assert not np.allclose(full.loc[~upto, "lo"], alt.loc[~upto.to_numpy(), "lo"])          # and later forecasts do react
print("2. forecasts up to a day do not change when later outcomes change")

# 3. weights, bounds, rebuild
W = full[[f"w_{m}" for m in rsi.POOL]].to_numpy()
assert np.allclose(W.sum(axis=1), 1.0) and W.min() >= 0.02 - 1e-12 and W.std() > 0
assert np.allclose(w[[f"w_{m}" for m in rsi.ROLES]].to_numpy(), 0.2)
b = rsi.apply_edit(start, {"kind": "vote", "eta": 99, "decay": -1, "w_min": 5, "by": "nonsense"})
assert b["vote"] == {"eta": 10.0, "decay": 0.0, "w_min": 0.3, "by": "class_h"}
assert rsi.apply_edit(start, {"kind": "range", "range_scale": start["range_scale"], "widen": 0.0}) is None
assert rsi.apply_edit(start, {"kind": "members", "action": "add", "member": "list_reader"}) is None
assert rsi.apply_edit(start, {"kind": "members", "action": "add", "member": "engine"})["members"][-1] == "engine"
edits = [{"kind": "members", "action": "add", "member": "engine"}, {"kind": "vote", "eta": 1, "decay": 0.1, "w_min": 0.0, "by": "h"}]
assert rsi.rebuild(edits)["members"] == rsi.ROLES + ["engine"] and rsi.rebuild(edits)["vote"]["by"] == "h" and rsi.rebuild([]) == start
print("3. weights sum to one and respect the floor; edits are clipped; kept edits rebuild the bundle")
print({s: int((f.split == s).sum()) for s in ("practice", "check", "exam", "")}, "forecasts;",
      {s: int(f[f.split == s].origin.nunique()) for s in ("practice", "check", "exam")}, "days")
print("ok")
