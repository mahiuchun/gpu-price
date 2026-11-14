# Test protocol (fixed before the experiments)

This file is committed before the for-the-record run. The commit hash is quoted in the findings.
Anything done differently afterwards is listed under "Deviations" at the bottom, with the reason.

## What is forecast

- **Series:** the 37 series in `data/headline_series.txt`. They are the starter kit's "headline" tier
  for posted rental prices at 6 Nov 2026 (enough history, at least three providers, prices that move).
  Each series is one site's daily median price for one GPU model and price type, in USD per GPU-hour.
  Caveat: this list was chosen with hindsight to 6 Nov, so results describe the scored set, not a random series.
- **Horizons:** 1, 3 and 7 days. **Forecast:** the 10%, 50% and 90% points.
- **Score:** pinball loss, exactly as `starter/scorer.py` computes it (`tests/parity.py` checks this).
  "Total skill" is 1 − our total error ÷ the error of a flat last-value forecast. "Per-series average"
  scores each series separately, then averages.
- **Reference to beat:** the starter's `drift` forecast.
- A cell is one (series, origin day, horizon). A forecast for a cell uses only rows dated on or before
  its origin. Origins are days on which the series was observed.

## What had been seen before this file was written

The builder (a human with an AI coding assistant) had looked at data dated up to **16 Oct 2026** only:
baseline scores, how often prices move, and the offer lists behind two series. From that came one
hand-made observation, given to the AI scientist as a labelled hint: most providers rarely change
price, and one or two that reprice almost daily drive most of the movement in a median.
Nothing dated 17 Oct or later had been looked at.

## The time machine

The AI scientist lives through October one week at a time.

| Step | It can see data up to | It then forecasts blind |
|---|---|---|
| 1 | 16 Oct | origins 16–22 Oct, targets up to 23 Oct |
| 2 | 23 Oct | origins 23–29 Oct, targets up to 30 Oct |
| 3 | 30 Oct | origins 30 Oct–5 Nov, targets up to 6 Nov |
| 4 | 6 Nov | origin 6 Nov: the submission (targets 7, 9, 13 Nov) |

- **The wall.** At each step the scientist's process receives a copy of the data that contains nothing
  after that step's date (`data.cut_world`, checked by `data.assert_walled`). On Modal each step runs
  in its own container that is sent only that copy.
- **Research at a step.** 8 ideas. Each idea is one hypothesis plus at most 3 changes to the engine's
  dials (`engine.SPEC_SCHEMA`); the scientist cannot write code.
- **Keeping an idea.** Kept only if, on targets up to 7 days before the step date, total error falls by
  at least 0.2% and per-series average skill does not fall by more than 0.002, **and** it does not do
  worse on a hidden check (targets in the last 7 days before the step date). The scientist is told only
  pass or fail for the hidden check, and the pass mark carries a little random noise (sd 0.5%), so the
  check cannot be learned by repeated asking.
- **Freezing a version.** After the 8 ideas the settings are frozen as that step's version.
- **No per-series switch to drift.** The first plan was to forecast a series with drift whenever the
  engine had not beaten drift on it so far. The rehearsal showed that rule lowers blind skill (see
  "Rehearsal"), so it was dropped before the for-the-record run. The scientist can still lean on
  drift through the `blend_drift` dial, which has to pass the gate like any other idea. When the
  engine has fewer than `min_scenarios` usable replay pairs for a forecast, that forecast uses drift.
- **Examining.** A separate examiner, which the scientist never talks to, scores every frozen version
  on each later week (origins on or after that week's step date). Version 0 is the engine at default
  settings with no research.

## Arms and worlds

- **Arms:** `gpt` (an OpenAI model proposes ideas, reading the ledger) and `random` (random dial
  changes through the same gate: what search without thought buys).
- **Worlds:** the real one, plus **5 placebo worlds**. In a placebo world every offer keeps its real
  listing pattern and the same set of day-to-day price changes, but the order of those changes in time
  is shuffled (`data.make_placebo`, seeds 1–5). Nothing about the timing of changes can be learned
  there. What the engine knows about today's list (who is in the middle, how far the neighbours allow
  it to move) is still true there, so placebo results separate gains from list structure from gains
  from timing.
- All arms and worlds run in the same for-the-record run.

## What we will claim

- **Headline:** live skill over the three blind weeks (version k scored on week k), against drift, for
  the real world. Reported as total skill, per-series average, median of series, and the share of
  truths below / inside / above the range.
- **Improvement:** later versions against version 0 and against each other on the same blind cells.
- **Uncertainty:** differences in error against drift, resampled by series and by target day
  (2,000 resamples), reported as 90% intervals.
- **Rule:** we say "beats drift" only if the 90% interval for the error difference excludes zero.
  Otherwise the headline is "not distinguishable from drift". Three weeks is little independent
  evidence and we will say so.
- **Placebo:** the same numbers for the 5 placebo worlds, side by side. If the scientist's
  improvements in the real world are no larger than in placebo worlds, we say its discoveries about
  timing are not established.
- No settings or code that affect forecasts change after the for-the-record run. A bug found afterwards
  is fixed, the run repeated, and both runs disclosed.

## Rehearsal

Before the for-the-record run the whole pipeline is rehearsed on steps 2 Oct and 9 Oct, using only
data up to 16 Oct. Rehearsal results are development feedback, not evidence.

What the rehearsal changed: on blind weeks after 25 Sep, 2 Oct and 9 Oct, the default engine scored
total skill 0.111 / 0.188 / 0.303 with no per-series switch to drift, and 0.115 / 0.148 / 0.261 with
the switch ("engine worse than drift on this series so far"). Stricter versions of the switch
(worse by 10% or 20% over 60+ or 150+ cells) never beat no switch. The switch was removed.

## Submission

Step 4 runs 8 more ideas on data up to 6 Nov and freezes the final version. `reproduce.sh` rebuilds
the forecast file from the pack and that frozen version with no AI calls and no Modal.

## RL arm (extra)

A small open model is trained with reinforcement learning (on Tinker) to propose better ideas.

- **Episode:** one prompt (the scientist's instructions plus a practice world's diagnostics at default
  settings), one idea, one reward.
- **Reward:** percentage fall in total error on that practice world's later period, which the prompt
  does not show; clipped to ±5. No change scores 0. Unreadable output scores −5.
- **Practice worlds:** built only from data up to 16 Oct: the real data cut at several earlier dates,
  and shuffled (placebo) copies with their own seeds. In a shuffled world restraint is the right answer.
- **Stop rule:** a trial of about 200 episodes on one world first; continue to about 2,000 only if the
  average reward clearly rises and the model does not collapse to one repeated answer.
- **Test (blind):** at each time-machine step, in the real world, each of three proposers (the small
  model before training, after training, and the random control) gives 16 single ideas from the
  diagnostics; each idea is scored on the following blind week. We report the share of ideas that
  reduce error and the average change. The trained model's weights and this recipe are fixed before
  its test is run.
- If the trial fails the stop rule, that is reported as the result.

## Deviations

Written after the for-the-record run (commit `354fc20` holds the protocol as it stood before it).

1. **RL trial longer than planned, and continued despite the stop rule.** The planned trial (3 rounds,
   192 tries, one practice world) gave average rewards of +1.21%, +0.55% and +0.96%: too few updates
   to judge. Two further trials of 8 rounds (512 tries each, learning rates 4e-5 and 1e-4) both rose
   to about +5% and both collapsed onto one repeated answer. Read strictly, the stop rule says stop.
   We continued because on a single practice world one best answer is the expected end point; the
   collapse that matters is across the 16 practice worlds, and it is reported: the full run ended
   with two answers ("30-day lookback" and "no change").
2. **The submission is version 0, not the scientist's step-4 version.** The blind test showed the
   scientist's versions were not better than the engine at default settings, so the default is
   submitted. The scientist's final version and the RL model's idea are in `submission/` as alternates.
3. **Analyses added afterwards, labelled as such in the findings:** the RL model's idea scored as a
   full version with intervals; the breakdown of what moves a median; the roll-forward test.
4. **No code or setting that affects forecasts changed after the run.** `engine.py`, `harness.py`,
   `core.py` and `scientist.py` are as committed in `354fc20`.
5. **Two forecasters were added after the blind test:** a pretrained time-series model (Chronos-2) and a
   crowd of five AI agents that forecast each price directly. Neither is in the protocol above. Both
   were first scored on the three test weeks after those weeks had been looked at. The agents'
   instructions were written once and not changed afterwards.
6. **More test periods were added after the blind test,** because one three-week window is thin: the
   eight weeks before it (21 Aug to 15 Oct) and four years of AWS spot prices. The eight earlier weeks
   were visible while the engine was being designed, so they are not blind for the engine.
7. **The submission is the agent crowd,** not the scientist's step-4 version (the protocol) and not the
   engine alone (deviation 2). It beat drift in all three tests and beat the engine on our own
   prices. The engine alone is `submission/alt_engine_2026-11-06.csv`.
8. **The main forecast is rebuilt from recorded AI answers.** `reproduce.sh` averages the answers saved
   in `runs/agents/final_answers.json`; asking the agents again gives slightly different numbers.
