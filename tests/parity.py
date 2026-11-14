"""Check that this repo's scoring and drift reference match the organisers' starter kit.

Usage: uv run python tests/parity.py [wall]
Forecasts a block of cells with (a) our drift and (b) the engine at default settings, writes them
in the starter's forecast format, scores them with starter/scorer.py in backtest mode, and
compares the starter's numbers with ours.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from gpuval import data, harness
from gpuval.core import day_of
from gpuval.engine import default_spec


def main(wall: str = "2026-10-16") -> int:
    world = data.build_world(wall)
    data.assert_walled(world, wall)
    cells = harness.base_cells(world)
    lo = day_of(wall) - 21
    cells = cells[cells.origin >= lo]
    res = harness.run_spec(world, default_spec(), cells)
    series_csv = data.CACHE / f"starter_{wall}" / "series.csv"
    ids = sorted(cells.series_id.unique())
    ok = True
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "ids.txt").write_text("\n".join(ids) + "\n")
        for name, df in (("drift", cells.assign(q10=cells.d10, q50=cells.d50, q90=cells.d90, loss=cells.loss_drift)),
                         ("engine", res)):
            fc = td / f"{name}.csv"
            sub = harness.to_submission(df, method_note=name)
            # full precision for the comparison
            sub["q10"], sub["q50"], sub["q90"] = df.q10.to_numpy(), df.q50.to_numpy(), df.q90.to_numpy()
            sub.to_csv(fc, index=False)
            rep = td / f"{name}.json"
            cmd = [sys.executable, "-B", str(data.PACK / "starter" / "scorer.py"), "--forecast", str(fc),
                   "--truth", str(series_csv), "--history", str(series_csv), "--horizons", "1,3,7",
                   "--series-ids", f"@{td / 'ids.txt'}", "--out", str(rep)]
            subprocess.run(cmd, check=True, capture_output=True)
            theirs = json.loads(rep.read_text())["overall"]
            ours = harness.summarize(df)
            pairs = [("n_cells", theirs["n_cells"], ours["n_cells"]),
                     ("pinball", theirs["pinball"], ours["loss"]),
                     ("baseline_pinball", theirs["baseline_pinball"], ours["loss_last"]),
                     ("skill_pooled", theirs["skill_pooled"], ours["skill_total"]),
                     ("skill_mean_of_series", theirs["skill_mean_of_series"], ours["skill_avg_series"]),
                     ("within_q10_q90", theirs["coverage"]["within_q10_q90"], ours["inside"])]
            for k, a, b in pairs:
                same = bool(np.isclose(a, b, atol=2e-6))
                ok &= same
                print(f"{name:7s} {k:22s} starter={a!s:12} ours={round(b, 6)!s:12} {'OK' if same else 'MISMATCH'}")
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
