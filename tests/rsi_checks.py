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
# 1c. the audit's "first crowd" bundle is the crowd as first written
w0 = rsi.walk(f, rsi.first_crowd_bundle())
m0 = w0.merge(c[k + ["orig"]], on=k, how="inner")
assert len(m0) == 1415 and np.allclose(m0.loss, m0.orig, rtol=0, atol=1e-9)
m0 = w0.merge(e[k + ["usd_orig"]], on=k, how="inner")
assert len(m0) == 479 and np.allclose(m0.loss, m0.usd_orig, rtol=0, atol=1e-9)
print(f"1. starting bundle = the improved crowd on {len(e)} exam-day and {len(c)} second-check forecasts; first-crowd bundle = the first crowd")

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
# 4. the path a kept edit takes, which the recorded run never took: with a lower bar in a scratch folder, an edit is
#    kept, the run replays to the same bundle, and the exam takes the kept edit out again
import tempfile
from pathlib import Path


class Scripted:
    kind = "scripted"

    def __init__(self, edits):
        self.edits = list(edits)

    def propose(self, kind, bundle, view, rng):
        return self.edits.pop(0), {"diagnosis": "", "expectation": ""}

    def usage(self):
        return {}


script = [{"kind": "vote", "eta": 4, "decay": 0.05, "w_min": 0.0, "by": "h"},
          {"kind": "range", "range_scale": {"1": 1.0, "3": 1.25, "7": 1.65}, "widen": 0.0},
          {"kind": "members", "action": "add", "member": "qwen3_8b_rl"}]
bar, quiet = rsi.ACCEPT_P, lambda *a, **k: None
with tempfile.TemporaryDirectory() as td:
    rsi.ACCEPT_P = 0.6
    try:
        st = rsi.run(Scripted(script), 3, 0, "scratch", frame=f, root=Path(td), log=quiet)
        again = rsi.run(rsi.ReplayProposer(st["ledger"]), 3, 0, "scratch", frame=f, save=False, log=quiet)
        ex = rsi.exam(frame=f, root=Path(td), save=False, log=quiet)["runs"]["scratch"]
    finally:
        rsi.ACCEPT_P = bar
assert [e["kept"] for e in st["ledger"]] == [False, False, True] and st["ledger"][1]["check"] == "not run (no change)"
assert st["bundle"] == rsi.rebuild(st["kept_edits"]) == again["bundle"] and st["bundle"]["members"][-1] == "qwen3_8b_rl"
assert abs(ex["vs_start"]["only_1"]["skill"] - ex["vs_start"]["final"]["skill"]) < 1e-12 and abs(ex["vs_start"]["without_1"]["skill"]) < 1e-12
assert ex["curve"][-1]["kept"] and ex["curve"][0]["exam_vs_drift"] != ex["curve"][-1]["exam_vs_drift"]
print("4. with a lower bar an edit is kept, the run replays to the same bundle, and the exam takes the kept edit out again")
print({s: int((f.split == s).sum()) for s in ("practice", "check", "exam", "")}, "forecasts;",
      {s: int(f[f.split == s].origin.nunique()) for s in ("practice", "check", "exam")}, "days")
print("ok")
