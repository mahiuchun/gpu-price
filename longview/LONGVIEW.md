# Long view: AWS spot rent for H100 and A100, 1-3 years out

**Question.** What will an H100 and an A100 rent for on AWS spot in 1, 2 and 3 years, and how far ahead can a forecast be trusted?

**Instrument.** AWS EC2 Linux spot, USD per GPU-hour, end of day, median across a fixed zone set: `p5.48xlarge` (H100, 28 zones), `p4d.24xlarge` (A100, 19). AWS sets it, in small steps. Not on-demand list price, not resale.

**Method.** `longview.py` (23 s, fixed seed) simulates 20,000 futures day by day: each day's change is drawn from past days that looked alike (same last move; same side of the chip's usual price) and fed back in. Replayed from every month-start on record, then each range widened until it caught 80% of that chip's test outcomes (arxiv:1702.06810, arxiv:2402.05203, arxiv:2509.02844 - Paper Lantern, 116 papers surveyed). All numbers: `out/*.csv`.

## 1. When the newer chip arrived (`out/chart1_price_paths.png`)

| Older chip | Before its successor | After |
|---|---|---|
| A100 (H100 arrived 26 Jul 2023) | -20%/yr ($1.56 to $1.23) | +1%/yr to Feb 2026 (zones -10% to +1%) |
| H100 (H200 12 Dec 2024, B200 24 Jul 2025) | -49%/yr, then -73%/yr ($8.77 to $2.03) | -1%/yr from B200 to Feb 2026 |

Older chips did not fall faster after a successor arrived; their declines stopped. Per unit of work (one H100 = 2.46 A100s) the H100 launched 3.1 times dearer than the A100; since mid-2025 it is a third cheaper, and the cheapest work on AWS spot fell 35%/yr to early 2026. July to October 2026 then lifted the H100 from $2.13 to $3.86 and the A100 from $1.69 to $2.12.

## 2. Trust horizon (`out/chart2_trust_horizon.png`)

A100 / H100 (three oldest zones):

| Ahead | Inside 10-90% range (target 80%) | Width, % of price | Loss vs no change | Independent windows |
|---|---|---|---|---|
| 30 days | 73%/50% | 43/24 | 0.71/0.80 | 49/33 |
| 180 days | 73%/52% | 71/41 | 0.72/0.55 | 7/5 |
| 1 year | 61%/41% | 80/40 | 0.65/0.50 | 4/2 |
| 2 years | 81%/20% | 83/17 | 0.71/0.66 | 2/1 |

Loss below 1 beats "price stays put" (pinball loss, a score for ranges). The simulator wins everywhere, mostly by giving a range at all; its raw ranges are too narrow, badly so for the H100. **Supported to about six months. One year rests on 2-4 independent windows, two years on 1-2; three years is untested.**

## 3. "New chip arrives" adjustment

After the H100 arrived the A100's trend improved by 26%/yr (-6% to +31% across measures). Adding that changed one- and two-year test loss by -6% to 0%, using a number fitted on the same history: indistinguishable from no effect. The opposite sign scored 5-17% worse.

## 4. Outlook from 31 Oct 2026 (`out/chart3_outlook.png`)

USD per GPU-hour, middle (10% to 90%):

| | Main, widened | With successor effect | Other reading: no pull-back |
|---|---|---|---|
| **H100**, now $3.86: +1 yr | 2.69 (1.05-8.25) | 2.76 (1.07-8.61) | 3.73 (2.36-5.90) |
| +2 yr | 2.56 (0.97-7.15) | 2.61 (1.00-7.56) | 3.64 (1.90-6.98) |
| +3 yr | 2.54 (0.94-6.89) | 2.58 (0.98-7.24) | 3.58 (1.61-7.93) |
| **A100**, now $2.12: +1 yr | 1.27 (0.78-3.18) | 1.29 (0.79-3.42) | 2.05 (0.97-4.33) |
| +2 yr | 1.25 (0.75-2.80) | 1.27 (0.77-3.07) | 2.03 (0.71-5.83) |
| +3 yr | 1.24 (0.74-2.70) | 1.26 (0.76-2.97) | 2.03 (0.55-7.38) |

Main pulls prices back to their usual level ($2.53 H100, $1.28 A100). Four years of A100 history favour it; for the H100 the test cannot choose (its two one-year composite tests both missed high).

## Assumptions and limits

- The 2026 rise fades (main) or persists (other); the data cannot say which.
- One history, one cloud, two chips; spot looks capped at AWS's list price (zones sit flat at ceilings), so range tops may be unreachable.
- No data 1 Mar-30 Jun 2026; nothing carried across.
- Days when many new zones were ramping up from floor prices are not learned from (else the H100 shows a false +73%/yr trend).
- H100 ranges are doubled each end on two independent one-year windows.
- Rent, not resale value.
