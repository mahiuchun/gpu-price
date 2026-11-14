# An AI scientist for GPU rental prices

Entry for "Recursive Self-Improvement by Pacific: An AI Scientist for GPU Value" (Direction A).
It forecasts 37 posted GPU rental price series 1, 3 and 7 days ahead, and tests whether an AI
scientist can improve those forecasts on weeks it has not seen.

- **Findings:** [FINDINGS.md](FINDINGS.md)
- **Forecast file:** [submission/forecast_2026-11-06.csv](submission/forecast_2026-11-06.csv)
- **Test protocol, fixed before the test:** [PROTOCOL.md](PROTOCOL.md)

## Reproduce the forecasts

```bash
bash reproduce.sh                  # about 15 seconds; no AI calls, no Modal, no network
bash reproduce.sh --with-backtest  # also replays the recorded ideas through the blind test
```

Needs [uv](https://docs.astral.sh/uv/) and the unchanged data pack in
`./gpu-price-pack-v0.1.1-handover` (or `GPUVAL_PACK=/path/to/pack`). The script checks that this
repo's scoring matches the organisers' `starter/scorer.py`, then writes the forecast file and prints
its SHA-256.

## How it works

| Piece | File | What it does |
|---|---|---|
| Engine | `src/gpuval/engine.py` | Each series is the median of a site's price list. The engine replays what every individual offer did over past pairs of days, applies that to today's list, and recomputes the median. |
| Test harness | `src/gpuval/harness.py`, `core.py` | Builds forecast cells, scores them exactly as the starter kit does. Fixed code; the AI cannot change it. |
| AI scientist | `src/gpuval/scientist.py` | An OpenAI model proposes one hypothesis at a time as a change to the engine's dials. A gate keeps an idea only if it helps on past data and passes a hidden check. A ledger records every idea and what it taught. |
| Time machine | `src/gpuval/timemachine.py` | The scientist lives through October one week at a time. A separate examiner scores each frozen version on the weeks after it. |
| Shuffled worlds | `src/gpuval/data.py` (`make_placebo`) | Copies of the data with the timing of price changes shuffled, so there is nothing about timing to learn. They show how much of any gain is real. |
| Modal | `src/gpuval/modal_app.py` | Every research step runs in its own container that is sent only data up to that step's date. Real and shuffled worlds run side by side. |
| RL | `rl/` | A small open model is trained with reinforcement learning on Tinker to propose better ideas, then tested blind. |
| What moves a median | `src/gpuval/insight.py` | Splits each day's change in a median into list changes, frequent repricers and rare repricers; rolls the engine forward to see how far its ranges hold. |
| Long view | `longview/` | AWS spot rent for H100 and A100 one to three years out, and a resale view. |

## Re-run the research (needs keys in `.env`)

```bash
uv run python -m gpuval.timemachine --name record --modal \
    --steps 2026-10-16,2026-10-23,2026-10-30,2026-11-06     # OPENAI_API_KEY + Modal token
uv run python -m gpuval.report --name record                 # tables with 90% intervals
uv run python -m gpuval.insight                              # what moves the median; roll-forward test
uv run python -m gpuval.figures                              # figures in findings/figures/
uv run python longview/longview.py && uv run python longview/resale.py
cd rl && uv run python train_scientist.py --name full --rounds 31   # TINKER_API_KEY
```

A fresh run of the scientist will not repeat the recorded ideas word for word (the model is not
deterministic). `reproduce.sh --with-backtest` replays the recorded ideas instead, and reproduces
the blind-test table exactly.

## What is in `runs/`

- `runs/record/`: the for-the-record run. One folder per world and arm with the ledger
  (`state.json`), the blind scores (`exam.csv`) and every blind cell (`cells.parquet`).
- `runs/rehearsal_*/`: rehearsals on early-October steps, data up to 16 Oct only.

## Data and attribution

Pacific Machines (2026). gpu-price-pack v0.1.1-handover, data cutoff 6 November 2026. CC BY 4.0 on
the compilation. The pack is not redistributed here. Forecast targets and offer rows come from:

- Hubbard, A. (2026). GPU Price Tracker dataset. https://github.com/alex-hubbard/gpu_price_tracker (CC BY 4.0).
- GPU rental price data by gpurentalprices.com (https://gpurentalprices.com), licensed under CC BY 4.0.
- Data: FastGPU GPU cloud pricing dataset (https://fastgpu.co/dataset), CC BY 4.0.
- "AWS Spot Price History", Eric Pauley (University of Wisconsin-Madison), Zenodo, https://doi.org/10.5281/zenodo.14198917, CC BY 4.0 (long view only).
- MLCommons MLPerf Inference and Training results repositories, Apache License 2.0, and Epoch AI,
  'Data on machine learning hardware', https://epoch.ai/data/machine-learning-hardware (long view only).

Scoring code mirrors the pack's starter kit (`starter/_common.py`, `baselines.py`, `scorer.py`).
