# An AI scientist for GPU rental prices

Entry for "Recursive Self-Improvement by Pacific: An AI Scientist for GPU Value" (Direction A).
It forecasts 37 posted GPU rental prices 1, 3 and 7 days ahead, and tests which kinds of AI help
make those forecasts better on data the system has not seen.

- **Findings:** [FINDINGS.md](FINDINGS.md)
- **Forecast file:** [submission/forecast_2026-11-06.csv](submission/forecast_2026-11-06.csv), the average of five AI agents
  after two changes that a coach AI proposed and tests kept.
  The engine alone is [submission/alt_engine_2026-11-06.csv](submission/alt_engine_2026-11-06.csv).
- **Test protocol, fixed before the test:** [PROTOCOL.md](PROTOCOL.md)

## Reproduce the forecasts

```bash
bash reproduce.sh                  # the forecast files, about 15 seconds; no AI calls, no Modal, no network
bash reproduce.sh --with-backtest  # also replays the recorded ideas through the blind test (about 4 minutes)
```

Needs [uv](https://docs.astral.sh/uv/) and the unchanged data pack in
`./gpu-price-pack-v0.1.1-handover` (or `GPUVAL_PACK=/path/to/pack`). The script checks that this
repo's scoring matches the organisers' `starter/scorer.py`, then writes the forecast files and prints
their SHA-256. The main file is rebuilt from the agents' recorded answers in
`runs/coach/final_answers.json` (the crowd before the coach's changes: `runs/agents/final_answers.json`).

## How it works

| Piece | File | What it does |
|---|---|---|
| Engine | `src/gpuval/engine.py` | Each series is the median of a site's price list. The engine replays what every individual offer did over past pairs of days, applies that to today's list, and recomputes the median. |
| Test harness | `src/gpuval/harness.py`, `core.py` | Builds forecast cells, scores them exactly as the starter kit does. Fixed code; the AI cannot change it. |
| AI scientist | `src/gpuval/scientist.py` | An OpenAI model proposes one hypothesis at a time as a change to the engine's dials. A gate keeps an idea only if it helps on past data and passes a hidden check. A ledger records every idea and what it taught. |
| Time machine | `src/gpuval/timemachine.py` | The scientist lives through October one week at a time. A separate examiner scores each frozen version on the weeks after it. |
| Shuffled worlds | `src/gpuval/data.py` (`make_placebo`) | Copies of the data with the timing of price changes shuffled, so there is nothing about timing to learn. They show how much of any gain is real. |
| Modal | `src/gpuval/modal_app.py` | Every research step runs in its own container that is sent only data up to that step's date. Real and shuffled worlds run side by side. |
| RL, proposing ideas | `rl/train_scientist.py`, `blind_test.py` | A small open model is trained with reinforcement learning on Tinker to propose better engine settings, then tested blind. |
| RL, forecasting | `rl/forecast_env.py`, `train_forecaster.py`, `eval_forecaster.py` | Qwen3-8B and Qwen3.5-9B are trained on AWS prices to forecast directly (reward: less error than drift), then tested on our prices, which they never practised on. |
| RL on Modal | `rl/modal_forecaster.py` | The same training with our own sampling and policy-gradient code on Modal H100s. Each run is a container, so learning rates and random seeds run side by side; the rate is chosen on a watch set, and test prompts are sent without outcomes. |
| Five AI agents | `src/gpuval/agents.py` | Five agents with different jobs each read a price's list and forecast it directly. |
| Coach | `src/gpuval/coach.py` | A coach AI proposes one change at a time to the agent crowd. A change is kept only if it helps on practice days and passes a hidden check; fresh exam days decide whether the result is used. The improved crowd is the forecast we hand in. |
| ML model | `src/gpuval/ml_challenger.py` | Chronos-2, a pretrained time-series model, run as it comes on a Modal GPU for comparison. One agent (the one that weighs reference forecasts) is shown its numbers; `ml_challenger final` writes them for the forecast day. |
| More test periods | `src/gpuval/periods.py`, `aws_test.py` | The same forecasters on eleven weekly periods of our prices and on four years of AWS spot prices. |
| How much room there is | `src/gpuval/headroom.py` | What a perfect forecast, and the best settings chosen with hindsight, would have scored. |
| What moves a median | `src/gpuval/insight.py` | Splits each day's change in a median into list changes, frequent repricers and rare repricers; rolls the engine forward to see how far its ranges hold. |
| Range memory | `src/gpuval/range_memory.py` | A forecaster sets its own range width, day by day, from its earlier forecasts whose outcomes are known. No AI calls. Also the control for the RL result: untrained model plus memory against trained model. |
| Checks on our own claims | `src/gpuval/checks.py` | How often ranges hold when the price moved; how few prices and days carry the pooled figures; each agent alone; the AWS test before and after the model's knowledge ends; the price that follows the euro; the engine on every other price in the pack. No AI calls. |
| Agent memory | `src/gpuval/agent_memory.py` | The five agents lived through 19 days in order, in three arms: no memory, each agent's own scored record for the price, and the record plus a lesson it writes for itself. |
| Recall probe | `src/gpuval/recall_probe.py` | Asks the agents' model, with no data shown, what the AWS prices were on its test days. Days after its knowledge ends are the control. |
| Analyst | `src/gpuval/analyst.py` | An AI writes one analysis function per idea. It runs in a Modal container with no keys and no network, against 39 shuffled copies of the data, and a passed idea must pass again on newer data the analyst never sees. |
| Weekday rule | `src/gpuval/weekday_rule.py` | Tests whether the weekday pattern the analyst found makes 1-day forecasts better. No AI calls. It does not. |
| AI calls with a meter | `src/gpuval/llm.py` | One helper for the three modules above: cheaper flex tier first, token and cost totals, a budget that stops the run. |
| Long view | `longview/` | AWS spot rent for H100 and A100 one to three years out (capped at AWS's list price, with hardware-trend paths beside it), and a resale view with a floor. |

## Re-run the research (needs keys in `.env`)

```bash
uv run python -m gpuval.timemachine --name record --modal \
    --steps 2026-10-16,2026-10-23,2026-10-30,2026-11-06     # OPENAI_API_KEY + Modal token
uv run python -m gpuval.report --name record                 # tables with 90% intervals
uv run python -m gpuval.insight                              # what moves the median; roll-forward test
uv run python -m gpuval.figures                              # figures in findings/figures/
uv run python -m gpuval.ml_challenger                        # pretrained model on a Modal GPU
uv run python -m gpuval.agents --origins 2026-10-16,2026-10-19,2026-10-22,2026-10-25,2026-10-28,2026-10-31,2026-11-03
uv run python -m gpuval.periods build && uv run python -m gpuval.periods agents && uv run python -m gpuval.periods table
uv run python -m gpuval.aws_test build && uv run python -m gpuval.aws_test agents && uv run python -m gpuval.aws_test table
uv run python longview/longview.py && uv run python longview/resale.py
uv run python -m gpuval.coach run --generations 10 && uv run python -m gpuval.coach exam   # the coach loop, then its exam
uv run python -m gpuval.coach confirm && uv run python -m gpuval.coach final                # second check; the 6 Nov forecast
cd rl && uv run python train_scientist.py --name full --rounds 31   # TINKER_API_KEY
cd rl && uv run python train_forecaster.py --name fc2 --rounds 60 --base-model Qwen/Qwen3.5-9B && uv run python eval_forecaster.py --name fc2
uv run python -m gpuval.coach compare                               # every forecaster on the same fresh days
uv run python -m gpuval.checks && uv run python -m gpuval.checks wide   # checks on the write-up's claims; no keys needed
uv run python -m gpuval.range_memory                                # range memory and the RL control; no keys needed
uv run python -m gpuval.ml_challenger final                         # the ML model's numbers for 6 Nov (Modal token)
uv run python -m gpuval.ml_challenger crossrun                      # the ML model with cross-learning and list covariates (Modal token)
uv run python -m gpuval.recall_probe run                            # about 210 AI calls; `table` re-scores saved answers
uv run python -m gpuval.agent_memory run                            # about 10,600 AI calls on the flex tier; `table` re-scores
uv run python -m gpuval.analyst run                                 # 12 AI calls and a sealed Modal container; `table` re-prints
uv run python -m gpuval.weekday_rule                                # the rule that follows from the analyst's finding; no keys needed
cd rl && uv run modal deploy modal_forecaster.py && uv run python modal_forecaster.py launch --plan sweep1   # 8 training runs on Modal GPUs
cd rl && uv run python modal_forecaster.py collect --plan sweep1 && uv run python modal_forecaster.py summarize
```

A fresh run of the scientist or the agents will not repeat the recorded answers word for word (the
model is not deterministic). Every answer is saved: `runs/record/` for the scientist,
`runs/agents/` for the agents and `runs/coach/` for the coach loop. `reproduce.sh --with-backtest` replays the scientist's recorded ideas
and reproduces the blind-test table exactly.

## What is in `runs/`

- `runs/record/`: the for-the-record run. One folder per world and arm with the ledger
  (`state.json`), the blind scores (`exam.csv`) and every blind cell (`cells.parquet`).
- `runs/rehearsal_*/`: rehearsals on early-October steps, data up to 16 Oct only.
- `runs/agents/`: every answer the five agents gave, in the tests and for the 6 Nov forecast.
- `runs/coach/`: the coach's ledger of ten proposed changes (`state.json`), every agent answer it
  caused (`answers.json`), the exam and second-check scores, and the improved crowd's 6 Nov answers.
- `runs/agent_memory/`, `runs/recall/`, `runs/analyst/`: every answer from the three tests added on 15 Nov
  (the analyst's ledger holds each idea's code and its numbers on the real and shuffled data).
- `rl/runs/fc1/`, `rl/runs/fc2/`: training logs and test answers of the two RL-trained forecasters.
- `rl/runs/modal/`: every Modal training run (log, test answers, scored cells); `findings/data/modal_rl.json` puts them next to the Tinker runs.
- `docs/pages/`: three explainer pages (how it works, the test periods, the Exa lookups).

## Data and attribution

Pacific Machines (2026). gpu-price-pack v0.1.1-handover, data cutoff 6 November 2026. CC BY 4.0 on
the compilation. The pack is not redistributed here. Forecast targets and offer rows come from:

- Hubbard, A. (2026). GPU Price Tracker dataset. https://github.com/alex-hubbard/gpu_price_tracker (CC BY 4.0).
- GPU rental price data by gpurentalprices.com (https://gpurentalprices.com), licensed under CC BY 4.0.
- Data: FastGPU GPU cloud pricing dataset (https://fastgpu.co/dataset), CC BY 4.0.
- "AWS Spot Price History", Eric Pauley (University of Wisconsin-Madison), Zenodo, https://doi.org/10.5281/zenodo.14198917, CC BY 4.0. Used for the long view, as the second test bed (`aws_test.py`), as the coach's practice and check days, and as the RL forecaster's training prices. No AWS row is an input to a forecast of the 37 prices, but the coach's two kept changes, which are in the forecast we hand in, were chosen partly on AWS days.
- MLCommons MLPerf Inference and Training results repositories, Apache License 2.0, and Epoch AI,
  'Data on machine learning hardware', https://epoch.ai/data/machine-learning-hardware (long view only).

Scoring code mirrors the pack's starter kit (`starter/_common.py`, `baselines.py`, `scorer.py`).
