"""Write the forecast file from a frozen version. No AI calls, no Modal, no network.

    uv run python -m gpuval.submit --version versions/final.json --origin 2026-11-06 \
        --out submission/forecast_2026-11-06.csv

Reads only the pack (rows dated on or before the origin) and the frozen settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from . import data
from .core import HORIZONS, day_of, iso_of
from .engine import build_engines, validate_spec

import numpy as np


def forecast_origin(world: dict, spec: dict, origin: str, horizons=HORIZONS, team_id: str = "team") -> pd.DataFrame:
    spec = validate_spec(spec)
    engines = build_engines(world)
    od = day_of(origin)
    rows = []
    for sid in sorted(world["series"]):
        s = world["series"][sid]
        i0 = int(np.searchsorted(s["days"], od, side="right")) - 1
        if i0 < 0:
            continue
        f = engines[sid].forecast(i0, spec, horizons)
        stale = od - int(s["days"][i0])
        for h in horizons:
            q10, q50, q90, src = f[h]
            note = f"{src};last_value_date={iso_of(s['days'][i0])}" + (f";stale_days={stale}" if stale else "")
            rows.append({"series_id": sid, "origin_date": origin, "horizon_days": h,
                         "q10": round(q10, 6), "q50": round(q50, 6), "q90": round(q90, 6),
                         "target_date": iso_of(od + h), "team_id": team_id, "method_note": note})
    df = pd.DataFrame(rows)
    # the file format requires q10 <= q50 <= q90 and q10 > 0 after rounding
    df["q10"] = df[["q10", "q50"]].min(axis=1)
    df["q90"] = df[["q90", "q50"]].max(axis=1)
    assert (df.q10 > 0).all() and (df.q10 <= df.q50).all() and (df.q50 <= df.q90).all()
    return df


def from_agents(answers_path: Path, origin: str, team_id: str = "team") -> pd.DataFrame:
    """The five-agent crowd's forecast, rebuilt from the agents' recorded answers (no AI calls).
    Each number is the average, in log terms, of the five agents' first-round answers."""
    jobs = json.loads(Path(answers_path).read_text())
    od = day_of(origin)
    rows = []
    for j in sorted(jobs, key=lambda x: x["series_id"]):
        if j["origin"] != origin:
            continue
        for h in HORIZONS:
            q = np.exp(np.mean(np.log(np.array([a["q"][str(h)] for a in j["r1"].values()], dtype=float)), axis=0))
            mid = float(q[1])
            lo, hi = min(float(q[0]), mid), max(float(q[2]), mid)
            note = "average of five AI agents forecasting directly (gpt-6.1-sol; answers recorded 14 Nov 2026 from data up to the origin)"
            if "range_scale" in j:  # a crowd rewritten by the coach (gpuval.coach): its own agents and range stretch
                sc = float(j["range_scale"][str(h)])
                lo, hi = mid * (lo / mid) ** sc, mid * (hi / mid) ** sc
                rs = j["range_scale"]
                note = (f"average of {len(j['r1'])} AI agents forecasting directly; with the changes a coach AI proposed and tests kept: "
                        f"{len(j.get('asked_again', []))} agent rewritten; ranges stretched x{rs['1']}/x{rs['3']}/x{rs['7']} at 1/3/7 days "
                        "(gpt-6.1-sol; answers recorded 14 Nov 2026 from data up to the origin)")
            rows.append({"series_id": j["series_id"], "origin_date": origin, "horizon_days": h,
                         "q10": round(lo, 6), "q50": round(mid, 6), "q90": round(hi, 6),
                         "target_date": iso_of(od + h), "team_id": team_id, "method_note": note})
    df = pd.DataFrame(rows)
    df["q10"] = df[["q10", "q50"]].min(axis=1)
    df["q90"] = df[["q90", "q50"]].max(axis=1)
    assert len(df) and (df.q10 > 0).all() and (df.q10 <= df.q50).all() and (df.q50 <= df.q90).all()
    return df


def starter_check(forecast_csv: Path, origin: str) -> str:
    """Run the organisers' scorer in frozen mode on the file. With no truth yet every cell is
    'missing truth', but the file is validated and the honesty checks run. Returns the status."""
    series_csv = data.CACHE / f"starter_{origin}" / "series.csv"
    with tempfile.TemporaryDirectory() as td:
        rep = Path(td) / "report.json"
        cmd = [sys.executable, "-B", str(data.PACK / "starter" / "scorer.py"), "--forecast", str(forecast_csv),
               "--truth", str(series_csv), "--history", str(series_csv), "--frozen-cutoff", origin,
               "--horizons", "1,3,7", "--out", str(rep)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        r = json.loads(rep.read_text()) if rep.exists() else {}
        val = r.get("forecast_validation", {})
        return (f"scorer exit code {p.returncode}; status {r.get('status')}; "
                f"rows read {val.get('n_rows_read', val.get('n_rows'))}; rows dropped {val.get('n_rows_dropped', len(val.get('problems', [])))}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=str(data.REPO / "versions" / "final.json"))
    ap.add_argument("--origin", default=data.DATA_END)
    ap.add_argument("--out", default=str(data.REPO / "submission" / f"forecast_{data.DATA_END}.csv"))
    ap.add_argument("--team-id", default="team")
    ap.add_argument("--from-agents", default=None, help="rebuild the agent crowd's forecast from its recorded answers")
    a = ap.parse_args(argv)
    world = data.build_world(a.origin)   # also builds the history file the organisers' check needs
    data.assert_walled(world, a.origin)
    if a.from_agents:
        df = from_agents(Path(a.from_agents), a.origin, team_id=a.team_id)
        assert set(df.series_id) == set(world["series"]), "the recorded answers do not cover every price"
    else:
        version = json.loads(Path(a.version).read_text())
        df = forecast_origin(world, version["spec"], a.origin, team_id=a.team_id)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} ({len(df)} rows, {df.series_id.nunique()} series) sha256 {sha}")
    if not a.from_agents:
        print("engine forecasts:", int(df.method_note.str.startswith("engine").sum()), "| drift fallbacks:",
              int((~df.method_note.str.startswith("engine")).sum()))
    print("starter check:", starter_check(out, a.origin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
