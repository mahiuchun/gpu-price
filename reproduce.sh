#!/usr/bin/env bash
# One command: rebuild the forecast files from the data pack. No AI calls, no Modal, no network
# (after `uv sync` has installed the dependencies).
#
#   bash reproduce.sh                 # both forecast files (about 20 seconds)
#   bash reproduce.sh --with-backtest # also replay the week-by-week blind test of the AI scientist (about 4 minutes on a many-core machine)
#
# Needs: uv (https://docs.astral.sh/uv/) and the unchanged pack in ./gpu-price-pack-v0.1.1-handover
# (or GPUVAL_PACK=/path/to/pack).
#
# The main forecast comes from five AI agents. Their answers were recorded when they ran
# (runs/agents/final_answers.json), so this script rebuilds the file from those answers exactly.
# To ask the agents again (needs OPENAI_API_KEY; wording and numbers will differ slightly):
#   uv run python -m gpuval.agents --origins 2026-11-06 --final --name rerun
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
uv sync -q
uv run python tests/parity.py 2026-11-06 | tail -1
# the engine's forecast, computed from the pack
uv run python -m gpuval.submit --version versions/final.json --origin 2026-11-06 \
    --out submission/alt_engine_2026-11-06.csv
# the main forecast: the five agents' recorded answers, averaged
uv run python -m gpuval.submit --from-agents runs/agents/final_answers.json --origin 2026-11-06 \
    --out submission/forecast_2026-11-06.csv
if [[ "${1:-}" == "--with-backtest" ]]; then
  uv run python -m gpuval.timemachine --name replay --replay runs/record
  uv run python -m gpuval.report --name replay | tail -40
fi
