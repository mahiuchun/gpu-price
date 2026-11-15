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
9. **Two further improvement loops were added last** (14 Nov), neither in the protocol above: a coach AI
   that proposes changes to the agent crowd (`gpuval.coach`), and RL training of an open model to
   forecast directly (`rl/train_forecaster.py`). Their rules were fixed in code before their tests ran:
   - Coach: practice days and hidden-check days are AWS dates in 2024–25 and our prices on 26 Sep, 2, 8
     and 14 Oct. A change is kept only if total error on the practice days falls by at least 0.5% and
     error on the check days does not rise, within a small random tolerance of about half a percent
     (the coach is told only pass or fail). The exam days (our
     prices on 18, 21, 26, 29 Oct and 2 Nov; AWS on 24 Jul, 21 Aug and 2 Oct 2026) were named in the
     code before the loop ran, were never shown to the coach, and were scored once, after the loop ended.
   - RL forecaster: trained on AWS dates up to 30 Jun 2025, watched on AWS dates to 20 Feb 2026, tested
     on our prices from 24 Sep to 5 Nov and on AWS Fridays from 27 Feb 2026. One training run per
     model, 60 rounds, no settings tuned on the test.
   - Added after the exam had been read: the second check of the coach on the 14 days and 12 AWS days
     the first crowd had already answered; the split of the gain between the two kept changes; and the
     trial of the trained open model as a sixth agent (weight one sixth, set after seeing it on those
     14 days and before scoring it on the exam days).
10. **The submission changed from the first crowd to the improved crowd,** because the improved crowd
    had less error on the exam days (+4.0%, +0.5% to +7.6% on our prices) and on the second check.
    Four agents keep the answers recorded for the first crowd; the rewritten agent was asked again from
    the same facts. The first crowd's file is `submission/other/agents_original_2026-11-06.csv`.
11. **The RL forecaster was re-run on Modal GPUs** (`rl/modal_forecaster.py`), with our own sampling and
    policy-gradient code in place of Tinker's service. Written before any Modal run had finished:
    - Same task, prompts, reward, practice prices (same draw for the same seed), 60 rounds of 16 prices
      x 8 answers, LoRA rank 32. Our LoRA set-up is not Tinker's, so the learning rate is not
      comparable; we run four (4e-5, 1e-4, 2.5e-4, 6e-4), each with two random seeds.
    - The learning rate is chosen on the watch set only (256 later AWS practice prices, dates to
      20 Feb 2026): the highest average of the round-50 and round-60 watch scores over the two seeds.
    - Test-day results are reported for all eight runs, with the chosen learning rate marked. Containers
      are sent test prompts without outcomes; scoring is done locally with the same code as before.
    - Outcome of the choice, written while the test step was still running (no Modal test-day result
      existed yet): watch scores were +0.038 at 4e-5, +0.042 at 1e-4, -0.018 at 2.5e-4 and -0.153 at
      6e-4 (both of its runs collapsed onto one repeated answer). Chosen: 1e-4. The first test step
      crashed (out of memory, our bug); the trained adapters were already saved, and the test step
      was re-run from them with no retraining.
    - Two follow-ups were then started at 1e-4, also before any test-day result: the same recipe on
      Qwen3.5-9B (two seeds), and three times the practice on Qwen3-8B (180 rounds, two seeds).
    - The first Qwen3.5-9B attempt ran out of GPU memory mid-training and saved nothing. It was
      restarted with smaller working batches (`micro=8`, `token_budget=70_000`); these change memory
      use, not the update.
    - **One follow-up chosen after seeing test-day results** (so it is not blind): the Qwen3.5-9B runs
      at 1e-4 gave +5.9% and +3.2% on our prices, against +11.3% for the Tinker run of the same model.
      The Tinker run used 4e-5, so the two differ in learning rate. Written before it started: we
      re-run Qwen3.5-9B on Modal at 4e-5 with seeds 0 and 1 (seed 0 has the Tinker run's practice
      prices). Both results are reported whatever they show. If seed 0 lands near +11%, the gap was
      the learning rate; if it lands near +3% to +6%, the Tinker figure was a lucky run.
    - Outcome: +9.3% (seed 0) and +9.7% (seed 1) on our prices; −2.9% and +2.8% on AWS. Nearer +11%
      than +3% to +6%, so we read the gap as mostly the learning rate. Modal ran both on A100 80GB
      cards, not H100s. `rl/runs/modal/newer3/rate_and_model_comparison.json` holds the comparisons
      of the two rates and the two models on the same forecasts (each rate's two seeds averaged).
12. **Added on 15 Nov, after the forecast was handed in and after reading the brief's reference
    list: corrections to the write-up and further tests on past data.** The forecast file is not
    changed: its target days (7 to 13 Nov) had passed, so a change could not be blind. None of the
    tests below is blind either: every day they use had been looked at. What can still be fixed in
    advance is the rule of each test, so each rule is written here and committed before that test
    is first run.
    - **Range memory** (`gpuval.range_memory`, no AI calls). A forecaster's ranges are stretched by
      one number per horizon, set from its own earlier forecasts. On each forecast day, for each
      horizon, the memory holds every earlier forecast by the same forecaster whose outcome was
      published on or before that day. The stretch is the value in 0.50, 0.55, ..., 3.00 with the
      lowest total pinball loss in dollars on the memory (ties: the value nearest 1). With fewer
      than 50 forecasts in the memory it is 1. A stretch s turns low into mid × (low ÷ mid)^s and
      high likewise, as the coach's stretch does. One stretch per horizon is shared by all prices.
      - Scored on the 19 days of our prices that the first crowd answered outside the coach's
        practice and check days, counting every forecast whose outcome is known by 6 Nov.
      - Compared on the first crowd's recorded answers: no stretch, the coach's fixed
        1.0 / 1.25 / 1.65, and the memory. The crowd's memory also holds its answers on the
        coach's four practice and check days of our prices; those days are never scored here.
      - The same rule is run on the engine (memory: its own forecasts from 21 Aug) and on the open
        models before and after RL training (memory: their own forecasts from 24 Sep). "Untrained
        plus memory against trained" asks whether RL taught more than range width.
      - The grid, the minimum memory and the scoring days do not change after the first run.
      - Outcome of the first run (commit `9563862` holds the rule, `d7ca03a` the numbers):
        first crowd +14.7% against drift with no stretch, +17.0% with the coach's stretch, +15.9%
        with the memory; memory against none +1.4% (−1.5% to +4.1%), against the coach −1.4%
        (−3.2% to +0.2%). Nothing was changed and re-run. Two readings in the findings were made
        after seeing these numbers and are labelled so: the split of days at 12 Oct, and the share
        of misses that are zero-width ranges.
    - **Recall probe** (`gpuval.recall_probe`, about 280 AI calls; written before its first run).
      The agents' model is asked, with no data shown, for the AWS spot price of each machine type
      on each of the agents' AWS days (median across zones, USD per GPU-hour), and whether it was
      higher, lower or the same (within 1%) a week later. "I don't know" is allowed; a best guess
      is still asked for. The 11 days before April 2026 are the test and the 4 days after it,
      which the model cannot have seen, are the control. Same model and effort as the agents.
      - We will say the model shows recall if, on the days before April 2026, either (a) its
        prices are closer to the truth of their own date than to the truth of other dates for the
        same machine type (dates shuffled within machine type, 5,000 shuffles, p below 0.05), or
        (b) where it names a direction and the price moved, it is right clearly more than half
        the time (90% interval above 0.5). Otherwise we say no recall was found, and report how
        close its guesses were, before and after April 2026, side by side.
    - **Agent memory** (`gpuval.agent_memory`, about 10,600 AI calls on the cheaper flex tier,
      stopped by code at an estimated $45; written before its first run). The five agents as first
      written, lived through the 19 scoring days in date order, in three arms that differ only in
      what is added to the facts: nothing; the agent's own earlier forecasts for that price whose
      outcomes are known, each marked inside, below or above its range (the last 12, with a
      count of all); or that record plus a lesson of at most three sentences that the agent
      writes for itself before each day from its whole scored record (once it has 30 outcomes).
      Each arm's memory holds that arm's own answers. No range stretch is applied in any arm.
      - The "nothing" arm is asked again now, so that the three arms are the same age. Its gap to
        the answers recorded on 14 Nov shows how far the score moves when nothing changes.
      - We will say a memory helps only if its arm has less error than the "nothing" arm on the
        same forecasts with a 90% interval that excludes zero (all 19 days, dollars, resampled by
        price and target day). We also report the later half of the days, each horizon, each
        agent, and the lessons as written.
      - Expectation, from the reading: the record helps a little or not at all, and the written
        lesson does no better than the record. One run; prompts are not changed after seeing a
        result. A crash is resumed from the last finished day and reported.
      - Outcome of the one run (10,635 calls, all on the flex tier, an estimated $24; no crash):
        record against nothing +1.9% (−0.2% to +4.2%), lesson against nothing +2.1% (−0.2% to
        +4.5%), lesson against record +0.2% (−1.0% to +1.3%). By the rule, neither memory is
        shown to help. The "nothing" arm against the answers of 14 Nov: −0.6% (−2.1% to +0.8%).
    - **The ML model, a second look** (`gpuval.ml_challenger crossrun`, a Modal T4; written before
      its first run). Same model and version, nothing tuned. Three new arms on the same 3,769
      forecasts as the plain run: `cross` (every scored price observed on the forecast day goes in
      one batch with cross-learning on; one batch per forecast day, so no later reading of any
      price is in it), `cov` (one price at a time, plus four past-only facts about its list: the
      cheapest offer, the lower and upper quartile, the number of offers), and `cross_cov` (both).
      - We will say an arm repairs the model only if it beats drift in both blocks (the eight
        earlier weeks and the three test weeks) with 90% intervals that exclude zero. We also
        report each arm against the plain run and against the engine on the same forecasts.
      - Outcome of the first run: no arm repairs the model. In the eight earlier weeks `cross`
        scored −2.4% against drift (−11% to +4%) and `cross_cov` −2.1%; in the test weeks +8.4%
        and +8.8%, with intervals that include zero. Cross-learning was better than the plain run
        in the earlier weeks (+4.4%, +0.9% to +7.7%) and no different in the test weeks.
    - **Recall probe, outcome of the first run:** 212 questions. The model said it remembered
      none of them and answered "don't know" to every week-later question. Its price guesses were
      no closer to their own date than to other dates (p = 0.12 before April 2026, 0.37 after).
      By the rule above, no recall was found.
    - **An analyst that writes its own test** (`gpuval.analyst`, 12 AI calls and a Modal CPU
      container; written before its first run). The agents' model, at medium effort, proposes 12
      ideas one at a time about when or in what order prices and lists change, each as a Python
      function that returns one number. It is given a neutral description of the data and no
      hint from our own findings.
      - Its code runs in a container with no keys and no network, on the real data and on 39
        shuffled copies (`data.make_placebo`, seeds 1 to 39). Discovery uses data up to 7 Oct:
        an idea passes if the real number is outside all 39 shuffled ones. The analyst is told
        each result.
      - Confirmation uses 8 Oct to 6 Nov, is never shown to the analyst, and is run once after
        its last idea, only for ideas that passed: confirmed if the real number is again outside
        all 39 shuffled ones (seeds 101 to 139) on the same side.
      - By luck an idea passes discovery 1 time in 20 and both steps about 1 time in 800. We
        report every idea, passed or not. A confirmed idea is a pattern in timing, not yet a
        better forecast.
      - Outcome of the first run: 12 ideas, all ran, 5 passed discovery, 3 confirmed (repricing
        by weekday; an offer repricing again two to four days later; one provider's changes
        arriving together across its GPUs). Two passed discovery and failed confirmation.
        Afterwards, as a reading and not part of the test, the three confirmed functions were
        re-run in the same container with one provider's offers removed
        (`gpuval.analyst without scaleway`); all three still passed in both periods.
    - **Weekday rule** (`gpuval.weekday_rule`, no AI calls; written after the analyst's result and
      a count of price moves by weekday, and before this rule's first run). For 1-day forecasts
      only: if, in this price's own history up to the forecast day, the published price has never
      moved into the weekday being forecast, in at least 8 past cases, the forecast becomes "no
      change" with no range; otherwise it is left alone. Applied to the engine and to drift on
      every forecast day from 21 Aug to 5 Nov, and to the first crowd's recorded answers on its 19
      days. We will say the rule helps a forecaster only if its 1-day error falls with a 90%
      interval that excludes zero. The 8 and the horizon do not change after the first run.
      - Outcome of the first run: no gain. It fired on 146 of 2,058 one-day forecasts; engine with
        the rule against without −0.1% (−0.5% to +0.1%); first crowd −0.2% (−0.9% to +0.2%).
    - **Checks with no rule to fix** (`gpuval.checks`, no AI calls). Counts and splits of results
      that were already recorded: how often ranges held when the price moved, how many prices and
      days carry the pooled figures, each agent alone, the AWS days before and after the agents'
      model's knowledge ends, the prices that supply the coach's gain, offers that leave a list
      and return, one provider's price by weekday, and the engine on every other daily-median
      price in the pack. These describe; they were chosen after reading the references and after
      looking at the data. One of them is a what-if and is labelled so: the ML model with its
      middle held at the last value.
    - **Corrections to FINDINGS.md.** Sentences that claimed more than the tests show were reworded
      (the ML model's numbers do reach one agent; the agents' different jobs and the averaging are
      not shown to matter; the coach's gain is nearly all range width; the 7-day engine figure is
      72 forecasts from three days). Each is marked "added 15 Nov" or "corrected" where it stands.
