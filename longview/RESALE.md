# Resale: a used H100 or A100, 1-3 years out

**Question.** What will a used H100 or A100 sell for in 1, 2 and 3 years, given the rental outlook in `LONGVIEW.md`? Prices are per bare GPU module. Rent only feeds a model; it is never joined to a resale series.

## Public evidence (`data/resale_quotes.csv`, 7 rows)

| Publisher, date | What it says | Link |
|---|---|---|
| American Compute, 10 Oct 2026 | Used H100 SXM, bare module: $10,000-14,000 | American Compute, GPU Prices, October 2026, https://www.amcompute.com/gpu-prices |
| American Compute, 10 Oct 2026 | Used A100 80GB SXM, bare module: $4,700-5,400 | Same |
| GPU Appraisals, 26 Sep 2026 | One H100 server appraised at $214K | [What an H100 Is Worth Today](https://gpuappraisals.com/blog/h100-residual-value-2026/) |
| GPU Appraisals, 26 Sep 2026 | The same server sold for $121K | Same |
| Hashrate Index, as reported by Blockspace, 1 Sep 2026 | Refurbished H100 server, median asking price $246,000 | [Hashrate Index launches AI hardware benchmark](https://blockspace.media/insight/hashrate-index-ai-hardware-price-benchmark/) |
| Five West, 5 Nov 2026 | One used A100 server listed at $94,999 | [Refurbished GPU Server Financing](https://www.fivewestfinancial.com/blog/refurbished-gpu-server-financing.html) |
| Stoa, 12 Oct 2026 | Used A100 server, weekly price $51,763-58,314 | [A 3-year GPU commit costs more than owning the servers](https://www.linkedin.com/posts/beratcelik1_stoa-intelligence-market-note-q3-2026-activity-7515502016429723651-6sSd) |

A server holds eight GPUs. American Compute's ranges are the middle of resale prices over the 12 months to August 2026. Its CSV header line reads "updated 2026-10-10. Free to use with attribution." Today's quote is the midpoint: H100 $12,000, A100 $5,050.

Other places that publish resale figures, which we point to but do not copy: [American Compute's appraisal pages](https://www.amcompute.com/appraisals/gpus), [its residual-value bands](https://www.amcompute.com/gpu-resale-index/residual-value), [Compute Exchange](https://compute.exchange/hardware-market/used-gpus).

## Model (`resale.py`, `out/resale_assumptions.csv`)

A GPU's value in use is the rent it can still earn: rent times utilisation, less running cost, each month of its remaining earning life, discounted. Rent follows the AWS spot outlook; the other inputs have low, central and high values, and ranges are the 10%, 50% and 90% points of 20,000 draws. The outlook is today's quote times the model's value at each horizon as a share of its value today.

The model values the rent an installed GPU can still earn: $38,500 per H100 ($27,500-51,300) and $14,000 per A100 ($9,100-19,800). Today's quote is the resale price of a bare module: $12,000 and $5,050. These are different things, so we show them side by side and do not treat the gap as a check on the model.

## Outlook (`out/resale_outlook.csv`, `out/chart4_resale.png`)

Low / middle / high are the 10%, 50% and 90% points. USD is per bare module.

| | H100 share of today's value | H100, USD | A100 share of today's value | A100, USD |
|---|---|---|---|---|
| +1 yr | 8% / 70% / 253% | 900 / 8,400 / 30,300 | 17% / 57% / 182% | 800 / 2,900 / 9,200 |
| +2 yr | 4% / 45% / 169% | 500 / 5,400 / 20,200 | 0% / 26% / 96% | 0 / 1,300 / 4,900 |
| +3 yr | 0% / 19% / 94% | 0 / 2,300 / 11,300 | 0% / 0% / 36% | 0 / 0 / 1,800 |

## What would change it

- **Today's quote.** Dollar figures scale with it. The ends of the published range move them by 17% (H100) and 7% (A100).
- **Rent path**, most. With other inputs central, the H100 share at +2 years runs from 7% on the low path to 168% on the high one.
- **Earning life**, second: at two years the H100 is worthless by +2. NVIDIA's data-centre chips double in FP16 speed about every 1.4 years ([arxiv:2601.20115](https://arxiv.org/abs/2601.20115), Jan 2026 - Paper Lantern, 120 papers surveyed).
- Utilisation, running cost and discount rate move the share by under four points.

## Limits

- Thin evidence: five publishers, seven rows. Today's quote rests on one publisher's rounded range.
- The share comes from what an installed GPU earns. We assume a bare module's price moves in step. That is untested.
- High ends assume rents near the top of the rental outlook and may be out of reach.
- One model, one rental benchmark, tested to six months. No scrap value (zero means "stopped earning"); staff and repair costs left out.
- Dates are the pages' own: the search index returned current copies, not snapshots as of 6 Nov.
- Two rows (Hashrate Index as reported by Blockspace; Stoa) were read only through the search index's copy of the page, not the page itself.
