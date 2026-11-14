#!/usr/bin/env bash
# One command: rebuild the forecast file from the data pack and the frozen settings.
# No AI calls, no Modal, no network (after `uv sync` has installed the dependencies).
#
#   bash reproduce.sh                 # forecasts only (about 15 seconds)
#   bash reproduce.sh --with-backtest # also re-run the blind test by replaying the recorded ideas (about 4 minutes on a many-core machine)
#
# Needs: uv (https://docs.astral.sh/uv/) and the unchanged pack in ./gpu-price-pack-v0.1.1-handover
# (or GPUVAL_PACK=/path/to/pack).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
uv sync -q
uv run python tests/parity.py 2026-11-06 | tail -1
uv run python -m gpuval.submit --version versions/final.json --origin 2026-11-06 \
    --out submission/forecast_2026-11-06.csv
if [[ "${1:-}" == "--with-backtest" ]]; then
  uv run python -m gpuval.timemachine --name replay --replay runs/record
  uv run python -m gpuval.report --name replay | tail -40
fi
