# Resale: a used H100 or A100, 1-3 years out

**Question.** What will a used H100 or A100 sell for in 1, 2 and 3 years, given the rental outlook in `LONGVIEW.md`? Prices are per bare GPU module. Rent only feeds a model; it is never joined to a resale series.

**Revised 15 Nov 2026**, after reading the brief's references: a floor under the low, middle and high figures; the rental input capped at AWS's list price; the same calculation on three hardware-trend rent paths; one weak source dropped. Not scored against the 7-13 Nov truth. `uv run python longview/resale.py --rent-variant main_nocap --no-floor` reproduces the figures as first published.

## Public evidence (`data/resale_quotes.csv`)

| Publisher, date | What it says | Link |
|---|---|---|
| American Compute, 10 Oct 2026 | Used bare module: H100 SXM $10,000-14,000; A100 80GB SXM $4,700-5,400 | American Compute, GPU Prices, October 2026, https://www.amcompute.com/gpu-prices |
| GPU Appraisals, 26 Sep 2026 | One H100 server appraised at $214K, sold for $121K | [blog post](https://gpuappraisals.com/blog/h100-residual-value-2026/) |
| Hashrate Index, as reported by Blockspace, 1 Sep 2026 | Refurbished H100 server, median asking price $246,000 | [news report](https://blockspace.media/insight/hashrate-index-ai-hardware-price-benchmark/) |
| Five West, 5 Nov 2026 | One used A100 server listed at $94,999 | [blog post](https://www.fivewestfinancial.com/blog/refurbished-gpu-server-financing.html) |

A server holds eight GPUs. American Compute's ranges are the middle of resale prices over the 12 months to August 2026. Its CSV header line reads "updated 2026-10-10. Free to use with attribution." Today's quote is each range's midpoint: H100 $12,000, A100 $5,050.

A fifth publisher's row, a LinkedIn post quoting a used A100 server price, was removed in this revision. It was read only through a search index's copy of the post, and the model never used it.

Other places that publish resale figures, which we point to but do not copy: [American Compute's appraisal pages](https://www.amcompute.com/appraisals/gpus), [its residual-value bands](https://www.amcompute.com/gpu-resale-index/residual-value), [Compute Exchange](https://compute.exchange/hardware-market/used-gpus). The pack lists these as pointers only, with no values copied: their terms do not allow copying, or they state no licence.

## Model (`resale.py`, `out/resale_assumptions.csv`)

A GPU's value in use is the rent it can still earn: rent times utilisation, less running cost, to the end of its earning life, discounted. Rent follows the AWS spot outlook, capped at AWS's list price; each input has a low, central and high value. The outlook is today's quote times the model's future value as a share of its value today, and never less than the floor below.

The model values the rent an installed GPU can still earn: $38,500 per H100 ($27,500-51,300), $13,900 per A100 ($9,100-19,700). Today's quote is the resale price of a bare module: $12,000 and $5,050. These are different things. We show them side by side, not as a check on the model.

**The floor.** On its own the model lets a GPU that has stopped earning reach $0. Old chips do not: they still sell for parts and for cheap second use. The low, middle and high figures are therefore floored at 1%, 2% and 5% of launch price.

| | Launch price (pack hardware table, Epoch AI) | Floor: low / middle / high |
|---|---|---|
| H100 | $33,600 | $340 / $670 / $1,680 |
| A100 | $15,000 | $150 / $300 / $750 |

- The three levels are a judgement range, not a measurement. The one figure we can quote: a July 2026 paper builds a working cluster from nine-year-old V100 16GB cards that it prices at $60 each (Cao et al., "DumpsterCluster: From Dumpster Diving to Serving LLaMA-70B on $60 GPUs", arXiv:2608.14614; title as listed in the pack's `resale_pointers.csv`). That is 0.6% of the V100's $10,664 launch price in the pack, a little under our low floor. Other old-chip prices are on the pages we point to above and are not copied here.
- The launch prices are rough. The H100's is a server price divided by eight (the pack records no official price). The A100's is the 40 GB part, the only A100 module with a price in the table, while today's quote is for the 80 GB part.
- The floor binds only where the model reaches zero: the H100's low figure at +3 years, the A100's low figure at +2 years, and the A100's low and middle figures at +3 years.

## Outlook (`out/resale_outlook.csv`, `out/chart4_resale.png`)

Share of today's value from the model, and USD per bare module after the floor: low / middle / high (10%, 50%, 90% points of 20,000 draws).

| | H100 share | H100 USD | A100 share | A100 USD |
|---|---|---|---|---|
| +1 yr | 8% / 70% / 235% | 900 / 8,400 / 28,100 | 16% / 56% / 164% | 800 / 2,900 / 8,300 |
| +2 yr | 4% / 45% / 158% | 500 / 5,400 / 18,900 | 0% / 26% / 90% | 150 / 1,300 / 4,500 |
| +3 yr | 0% / 19% / 90% | 340 / 2,300 / 10,800 | 0% / 0% / 35% | 150 / 300 / 1,800 |

As first published (no cap on rent, no floor): H100 900 / 8,400 / 30,300, then 500 / 5,400 / 20,200, then 0 / 2,300 / 11,300; A100 800 / 2,900 / 9,200, then 0 / 1,300 / 4,900, then 0 / 0 / 1,800. The middles moved by at most $300. The cap took up to $2,200 off the H100's tops and $900 off the A100's.

Corrected on 15 Nov after a code review. The cap first reached only the three numbers read from the rental outlook. Each of the 20,000 draws takes a place in that range, and a draw above the 90% point was extended past it, so one draw in ten had the H100 renting above AWS's list price (one in thirteen for the A100). Every drawn rent path is now stopped at the ceiling. That took the H100's tops from $29,800, $20,200 and $11,300 to the figures above; its lows and middles did not move. The A100's one-year middle now reads $2,900, not $2,800: the simulator correction in `LONGVIEW.md` section 4 moved its share from 56.4% to 56.5%.

**If rent follows the hardware trend instead.** The main rent path is flat after year one. `LONGVIEW.md` section 4 sets three steady falls beside it. On those the model's middle, in USD per bare module after the floor, at +1 / +2 / +3 years:

| Rent falls by | H100 | A100 |
|---|---|---|
| 18% a year | 8,000 / 4,700 / 2,100 | 3,000 / 1,300 / 300 |
| 25% a year | 7,100 / 3,600 / 1,300 | 2,700 / 1,000 / 300 |
| 35% a year | 5,600 / 1,900 / 670 | 2,100 / 500 / 300 |
| Main outlook | 8,400 / 5,400 / 2,300 | 2,900 / 1,300 / 300 |

Here the fall continues for the GPU's whole life and the range (in the CSV) is over the other assumptions only. A 25% yearly fall takes $1,000 off the H100's middle at three years.

**Beside the published forward bands.** American Compute publishes forward residual-value bands for the H100 as a percent of price new, by year to 2030 (link above; a pointer, its figures are not copied here). In the same unit our H100 middle is 25%, 16% and 7% of the $33,600 launch price at +1, +2 and +3 years, and 21%, 11% and 4% on the 25%-a-year path. As read by an agent on 15 Nov 2026, not re-checked by us: our main middle sits above that page's bands two and three years out, and the 25%-a-year path sits closer to them.

## What would change it

- **Today's quote.** Unfloored dollar figures scale with it: each extra $1,000 adds 8.3% to the H100's and 19.8% to the A100's. At the top of the published range ($14,000) the H100 middles are $9,800 / $6,300 / $2,700. The one executed sale in our file works out at $15,125 per GPU, but that is a whole server divided by eight, with chassis, processors and memory in the price; from $15,125 the middles would be $10,600 / $6,800 / $2,900. Newer monthly figures exist on the pages we point to. `--anchor-h100` and `--anchor-a100` re-run the outlook from any quote without storing it here.
- **Rent path**, most. On its own it moves the H100 share at +2 years from 7% to 168%.
- **Earning life**, next. A two-year life leaves the H100 at its floor at +2 years. NVIDIA's data-centre chips double in FP16 speed about every 1.4 years ([arxiv:2601.20115](https://arxiv.org/abs/2601.20115), Jan 2026 - Paper Lantern, 120 papers surveyed).
- Utilisation, running cost and discount rate move the share by under four points.

## Limits

- Thin evidence: four publishers, six rows. Today's quote rests on one publisher's rounded range.
- We assume a bare module's price moves in step with what an installed GPU earns. That is untested.
- High ends assume rents near the top of the rental outlook and may be out of reach, even after the cap.
- The floor is a judgement range on rough launch prices. It replaces "zero means stopped earning" with a small number, not with evidence about H100s, none of which is yet old.
- One model, one rental benchmark, tested to six months. Staff and repair costs left out.
- Dates are the pages' own: the search index returned current copies, not snapshots as of 6 Nov.
- One row (Hashrate Index as reported by Blockspace) was read only through the search index's copy of the page, not the page itself.
