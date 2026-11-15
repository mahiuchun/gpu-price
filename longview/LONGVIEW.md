# Long view: AWS spot rent for H100 and A100, 1-3 years out

**Question.** What will an H100 and an A100 rent for on AWS spot in 1, 2 and 3 years, and how far ahead can a forecast be trusted?

**Instrument.** AWS EC2 Linux spot, USD per GPU-hour, end of day, median across a fixed zone set: `p5.48xlarge` (H100, 28 zones), `p4d.24xlarge` (A100, 19). AWS sets it, in small steps. Not on-demand list price, not resale.

**Method.** `longview.py` (31 s, fixed seed) simulates 20,000 futures day by day: each day's change is drawn from past days that looked alike (same last move; same side of the chip's usual price) and fed back in. Replayed from every month-start on record, then each range widened until it caught 80% of that chip's test outcomes (arxiv:1702.06810, arxiv:2402.05203, arxiv:2509.02844 - Paper Lantern, 116 papers surveyed). All numbers: `out/*.csv`.

**Revised 15 Nov 2026**, after reading the brief's references. Added: the list price (section 0), every older-chip and successor pair (section 1), an out-of-sample test of the successor adjustment (section 3), a ceiling on every path and three hardware-trend paths (section 4). The long view is not scored against the 7-13 Nov truth. Every earlier number is still produced: rows named `*_nocap` in `out/outlook.csv` are the paths as first published.

## 0. Spot is a share of list price (`out/spot_to_list.csv`, `out/aws_list_prices.csv`)

AWS sets two prices for the same machine: the on-demand list price and the spot price. The pack holds the list price by region from 12 Jan 2026 (one collector's rows). It never changed in that time: H100 $6.88 (US) to $11.56 across 15 regions, A100 $2.74 to $4.44.

- **List is a ceiling.** Since 5 Jun 2025 no zone has traded above its own region's list price: 0 of 11,363 H100 zone-days and 0 of 10,739 A100 zone-days. The highest is exactly 1.000 of list. In each month of 2026, 4% to 23% of H100 zone-days and 20% to 27% of A100 zone-days sat at the ceiling.
- **AWS cut the list in June 2025: H100 by 44%, A100 by 33%.** The size comes from the pack: one tracking site still showed the earlier list prices on 28 Sep to 2 Oct 2026 ($12.29 and $4.10) and the current ones from 3 Oct ($6.88 and $2.74). The spot data gives the day: until 4 Jun 2025 two H100 zones sat at exactly the earlier ceiling, and from 5 Jun none is above today's. AWS's announcement dates the list change to 1 Jun 2025 (brief reference 22, as read by an agent on 15 Nov 2026; the pack does not hold the announcement).
- **Spot fell with it.** The H100's zone-median spot price was $3.58 in May 2025 and $2.16 in July (-40%). As a share of the list price in force it was 0.29 both times.
- **The 2026 rise is the share rising, with list unchanged.** Median share of list over zone-days:

| | 90 days before the cut | Jul 2025 to Feb 2026 | Jul to Sep 2026 | Oct 2026 |
|---|---|---|---|---|
| H100 | 0.28 | 0.33 | 0.37 | 0.47 |
| A100 | 0.30 | 0.39 | 0.80 | 0.72 |

On 31 Oct 2026 the H100 composite was $3.86, 0.51 of its composite list price ($7.64, the median of its 28 zones' list prices). The A100 was $2.12, 0.64 of $3.29. Over the whole archive the monthly share for these two chips has ranged from 0.15 to 0.84.

What we could not do: model the list price itself. The archive holds one change of it, so there is nothing to learn a pattern from. We treat today's list as fixed and say so under Limits.

## 1. When the newer chip arrived (`out/chart1_price_paths.png`, `out/successor_pairs.csv`)

The two pairs first reported, on windows chosen by hand:

| Older chip | Before its successor | After |
|---|---|---|
| A100 (H100 arrived 26 Jul 2023) | -20%/yr ($1.56 to $1.23) | +1%/yr to Feb 2026 (zones -10% to +1%) |
| H100 (H200 12 Dec 2024, B200 24 Jul 2025) | -49%/yr, then -73%/yr ($8.77 to $2.03) | -1%/yr from B200 to Feb 2026 |

We first wrote "older chips did not fall faster after a successor arrived; their declines stopped". Two things weaken that.

- **The list cut sits inside the H100's windows.** Its -73%/yr leg (12 Dec 2024 to 23 Jul 2025) contains the 44% cut of 5 Jun 2025, seven weeks before the B200 arrived. The flat spell "after the B200" is spot settling at 0.3 of the new list price. It cannot be read as an effect of the B200.
- **Other pairs do not agree.** The archive supports 14 pairs: an older chip, and a later chip of the same line whose first spot price falls inside the archive, with 180 or more days of the older chip's price on each side. Each compares the yearly change in the 365 days before the arrival with the 365 days after (fewer after the B200, which arrived 220 days before the gap in the archive).

| Older chip | Newer chip, first spot price | Before | After | Change (log points/yr) | June 2025 list cut | Share of other dates with a lower change |
|---|---|---|---|---|---|---|
| K80 | H100, 26 Jul 2023 | 0%/yr | +32%/yr | +0.28 | not a cut chip | 1.00 |
| V100 | H100 | -3% | +16% | +0.18 | not a cut chip | 0.94 |
| A100 | H100 | -41% | +8% | +0.61 | outside both windows | 0.71 |
| V100 | H200, 12 Dec 2024 | -13% | -30% | -0.22 | not a cut chip | 0.41 |
| A100 | H200 | +62% | -33% | -0.88 | in the after window | 0.00 |
| H100 | H200 | -57% | -60% | -0.09 | in the after window | 0.71 |
| V100 | B200, 24 Jul 2025 | -18% | -69% | -0.97 | not a cut chip | 0.06 |
| A100 | B200 | -21% | -13% | +0.10 | in the before window | 0.47 |
| H100 | B200 | -63% | +1% | +1.01 | in the before window | 1.00 |
| H200 | B200 | +7% | +33% | +0.22 | in the before window | none to compare |
| T4 | L4, 4 Apr 2024 | +18% | -11% | -0.27 | not a cut chip | 0.04 |
| A10G | L4 | +61% | -24% | -0.75 | not a cut chip | 0.04 |
| T4 | L40S, 15 Aug 2024 | +16% | 0% | -0.14 | not a cut chip | 0.17 |
| A10G | L40S | +28% | +7% | -0.18 | not a cut chip | 0.09 |

- **In 8 of the 14 the older chip's yearly change was lower after the arrival; in 6 it was higher.** Leaving out the five with the list cut in a window: 6 of 9 lower, middle change -0.18 log points a year, range -0.97 to +0.61.
- **The two chip lines differ.** Large training chips: 4 of 10 lower. Small inference chips (T4 and A10G when the L4 and L40S arrived): 4 of 4 lower, and lower than at 83% to 96% of other dates.
- **So the first sentence does not hold in general.** It describes the A100 when the H100 arrived. Read the table as a description, not a test: each "other date" is a month-start at least 90 days from any arrival (11 to 24 of them per row; 7 and 2 for the two H100 rows), the windows overlap heavily, and three or four years hold only two or three separate ones.
- The A100 row differs from the hand-chosen windows above (-41%/yr before, against -20%) because its window starts on 26 Jul 2022, in a spike: the zone-median price was $1.33 in July 2022, $2.05 in August and $1.23 in September. Step prices make any such rate sensitive to its end dates.
- The K80 has no pair after 2023: its spot prices stop on 1 Jun 2025. The pack's daily series carries the last price forward to February 2026, and we cut that off.

Per unit of work (one H100 = 2.46 A100s) the H100 launched 3.1 times dearer than the A100; since mid-2025 it is a third cheaper, and the cheapest work on AWS spot fell 35%/yr to early 2026. July to October 2026 then lifted the H100 from $2.13 to $3.86 and the A100 from $1.69 to $2.12. Measured to July-October 2026 instead, the fall is 19%/yr.

The speed ratios are the pack's capability units, a composite derived from published MLPerf® Inference results. Result not verified by MLCommons Association. The MLPerf® name and logo are registered and unregistered trademarks of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use strictly prohibited. See www.mlcommons.org for more information.

## 2. Trust horizon (`out/chart2_trust_horizon.png`)

A100 / H100 (three oldest zones):

| Ahead | Inside 10-90% range (target 80%) | Width, % of price | Loss vs no change | Independent windows |
|---|---|---|---|---|
| 30 days | 73%/50% | 43/24 | 0.71/0.80 | 49/33 |
| 180 days | 73%/52% | 71/41 | 0.72/0.55 | 7/5 |
| 1 year | 61%/41% | 80/40 | 0.65/0.50 | 4/2 |
| 2 years | 81%/20% | 83/17 | 0.71/0.66 | 2/1 |

Loss below 1 beats "price stays put" (pinball loss, a score for ranges). The simulator wins everywhere, mostly by giving a range at all; its raw ranges are too narrow, badly so for the H100. **Supported to about six months. One year rests on 2-4 independent windows, two years on 1-2; three years is untested.** The test has no ceiling in it: the list price before 2026 is not in the pack.

## 3. "New chip arrives" adjustment (`out/successor_test.csv`)

- **In sample.** After the H100 arrived the A100's trend improved by 26%/yr (-6% to +31% across measures). Adding that changed one- and two-year test loss by -6% to 0%, using a number fitted on the same history: indistinguishable from no effect. The opposite sign scored 5-17% worse.
- **Out of sample, it hurt.** We took the A100's shift as it could be measured the day before the H200 arrived (+0.42 log points a year) and applied it to the H100 from that day. One-year loss rose 20% (17 forecasts; 27% on the simulator with no pull-back). The opposite sign scored 2% to 5% better. Two-year results rest on 5 forecasts and point both ways.
- **Reading.** A number measured on one chip did not carry to the next. The H100's year after the H200 holds the list cut, which no successor rule knows about. We do not use the adjustment in the main outlook.

## 4. Outlook from 31 Oct 2026 (`out/chart3_outlook.png`, `out/outlook.csv`)

USD per GPU-hour, middle (10% to 90%). Every path is now capped at the composite list price ($7.64 H100, $3.29 A100):

| | Main, widened | With successor effect | Other reading: no pull-back |
|---|---|---|---|
| **H100**, now $3.86: +1 yr | 2.69 (1.05-7.64) | 2.76 (1.07-7.64) | 3.73 (2.36-5.87) |
| +2 yr | 2.56 (0.97-7.14) | 2.61 (1.00-7.56) | 3.64 (1.90-6.42) |
| +3 yr | 2.54 (0.94-6.89) | 2.58 (0.98-7.23) | 3.53 (1.61-6.53) |
| **A100**, now $2.12: +1 yr | 1.26 (0.78-2.84) | 1.28 (0.79-3.00) | 1.95 (0.96-3.06) |
| +2 yr | 1.24 (0.75-2.57) | 1.26 (0.76-2.73) | 1.73 (0.69-3.00) |
| +3 yr | 1.24 (0.74-2.53) | 1.26 (0.76-2.69) | 1.56 (0.52-2.95) |

Main pulls prices back to their usual level ($2.53 H100, $1.28 A100). Four years of A100 history favour it; for the H100 the test cannot choose (its two one-year composite tests both missed high).

**What the ceiling changed.**

- Main: the middles barely move (A100 +1 yr $1.27 to $1.26). The tops come down: H100 +1 yr $8.25 to $7.64; A100 $3.18, $2.80, $2.70 to $2.84, $2.57, $2.53.
- No pull-back: the A100's tops were $4.33, $5.83 and $7.38. All three are above its composite list price ($3.29), and the last two are above the dearest regional list price in the pack ($4.44). They are now $3.06, $3.00 and $2.95. Its middle no longer holds at $2.03: it falls to $1.56 by year three, because a price that can fall freely but cannot rise past list drifts down. The H100's year-three middle moves from $3.58 to $3.53.

**Three paths that follow the hardware trend** (comparison, not a forecast; no range). The main middle is flat after year one, which assumes cheaper computing stops pulling old chips' rent down. A steady yearly fall from today's price gives:

| | -18% a year | -25% a year | -35% a year | Main middle |
|---|---|---|---|---|
| H100 +1 / +2 / +3 yr | 3.17 / 2.60 / 2.13 | 2.90 / 2.17 / 1.63 | 2.51 / 1.63 / 1.06 | 2.69 / 2.56 / 2.54 |
| A100 +1 / +2 / +3 yr | 1.74 / 1.43 / 1.17 | 1.59 / 1.19 / 0.89 | 1.38 / 0.90 / 0.58 | 1.26 / 1.24 / 1.24 |

- **Where the rates come from.** Epoch AI reports that the cost of a unit of computing halves every 2.1 to 2.5 years, which is 24% to 28% a year (brief references 24 and 25, as read by an agent on 15 Nov 2026; we did not reproduce that figure). Our own straight-line fits on the pack's copy of Epoch's table (`key_numbers.json`, `hardware_trend_fit_on_pack`) bracket it: 14% a year (8% to 20%) for single-precision speed per launch dollar, 54 GPUs released 2008 to 2025, and 33% a year (29% to 36%) for each chip's fastest number format, 57 GPUs. On AWS spot the cheapest work fell 35% a year to early 2026 and 19% a year to July-October 2026 (section 1).
- **Epoch measures the launch price of new chips, not the rent of installed ones.** A new chip being cheaper per unit of work does not force an old chip's rent down at the same rate; section 1 shows how unevenly that happens.
- **Reading.** In year one the main middle falls 30% (H100) and 41% (A100) as the 2026 rise fades, then stays flat. By year three it is above all three paths for both chips: $2.54 against $1.06 to $2.13, and $1.24 against $0.58 to $1.17. If cheaper computing keeps pulling old chips' rent down, the main middle is too high at three years.

## Assumptions and limits

- The 2026 rise fades (main) or persists (other); the data cannot say which.
- The list price stays where it is. The ceiling is today's; AWS has changed it once in the archive, downward. A further cut would lower every path and a rise would lift the tops.
- One history, one cloud, two chips.
- No data 1 Mar-30 Jun 2026; nothing carried across.
- Days when many new zones were ramping up from floor prices are not learned from (else the H100 shows a false +73%/yr trend).
- H100 ranges are doubled each end on two independent one-year windows, then stopped at the ceiling.
- The list-price rows come from one collector and start on 12 Jan 2026. For June 2025 to January 2026 we assume the same list price; the spot ceilings agree (no zone above it).
- Rent, not resale value.
