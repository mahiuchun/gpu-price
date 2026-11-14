# Resale: a used H100 or A100, 1-3 years out

**Question.** What will a used H100 or A100 sell for in 1, 2 and 3 years, given the rental outlook in `LONGVIEW.md`? Rent only feeds a model; it is never joined to a resale series.

## Public evidence (`data/resale_quotes.csv`, 15 rows dated by 6 Nov 2026)

| Publisher, date | What it says | Title, link |
|---|---|---|
| American Compute, Oct 2026 | Used 8-GPU server, per GPU: H100 $24,400, A100 $11,900. Bare module: $14,250, $5,400 | [H100 Price 2026](https://www.amcompute.com/appraisals/h100), [A100 Price 2026](https://www.amcompute.com/appraisals/a100) |
| American Compute, Jun 2026 | Forward band, % of list price: H100 27-39 (2026), 11-19 (2027), 5-7 (2028) | [H100 Price 2026](https://www.amcompute.com/appraisals/h100) |
| Compute Exchange, Q3 2026 | Indicative bare module: H100 $22,000-27,000, A100 $7,000-9,500 | [Used GPUs for Sale](https://compute.exchange/hardware-market/used-gpus) |
| GPU Appraisals, 26 Sep 2026 | H100 server appraised at $214K; sold for $121K | [What an H100 Is Worth Today](https://gpuappraisals.com/blog/h100-residual-value-2026/) |
| Hashrate Index via Blockspace, 1 Sep 2026 | Median ask, H100 server: refurbished $246,000, new $324,438 | [Hashrate Index launches AI hardware benchmark](https://blockspace.media/insight/hashrate-index-ai-hardware-price-benchmark/) |
| Five West, 5 Nov 2026 | Used A100 server listed at $94,999 | [Refurbished GPU Server Financing](https://www.fivewestfinancial.com/blog/refurbished-gpu-server-financing.html) |
| Stoa, 12 Oct 2026 | Used A100 server $51,763-58,314 | [A 3-year GPU commit costs more than owning the servers](https://www.linkedin.com/posts/beratcelik1_stoa-intelligence-market-note-q3-2026-activity-7515502016429723651-6sSd) |

## Model (`resale.py`, `out/resale_assumptions.csv`)

A used GPU is worth the rent it can still earn: rent x utilisation, less running cost, monthly over its remaining earning life, discounted. Rent is the AWS spot outlook's 10/50/90% paths. Other inputs, low/central/high: utilisation 50/70/90%; running cost $150/280/450 per kW-month on 1.0/1.275/1.6 kW per H100 (A100 0.65/0.81/1.0); discount 8/12/20%; earning life 2/4/6 years (A100 1/3/5); ranges are the 10/50/90% points of 20,000 draws.

## Check against the quotes (nothing fitted)

The model puts an H100 at $38,500 per GPU ($27,500-51,300) and an A100 at $14,000 ($9,100-19,800): 1.6 and 1.2 times American Compute's server values. So the central case looks generous: a two-year H100 life or 50% utilisation would match, and American Compute's band falls faster (about 45%, then 18%, of its 2026 level).

## Outlook (`out/resale_outlook.csv`, `out/chart4_resale.png`)

Today's quote x the model's share of today's value, per GPU in a working server; middle (10%-90%):

| | H100 share | H100, from $24,400 | A100 share | A100, from $11,900 |
|---|---|---|---|---|
| +1 yr | 70% (8-253%) | $17,100 (1,900-61,700) | 57% (17-182%) | $6,800 (2,000-21,700) |
| +2 yr | 45% (4-169%) | $11,000 (1,000-41,100) | 26% (0-96%) | $3,100 (0-11,400) |
| +3 yr | 19% (0-94%) | $4,600 (0-23,100) | 0% (0-36%) | $0 (0-4,300) |

Bare modules: the same shares of $14,250 and $5,400.

## What would change it

- **Rent path**, most. If the 2026 rise holds, the model's value today is 44% (H100) and 67% (A100) higher. AWS spot is one posted price: the neocloud rate-card median ($3.95, $2.32; FastGPU, pack, 31 Oct 2026) held flat gives +57% and +95%, mostly from the flat path, not the level (+3%, +14%).
- **Earning life**, second: at two years the H100 is worthless by +2. NVIDIA's data-centre chips double in FP16 speed about every 1.4 years ([arxiv:2601.20115](https://arxiv.org/abs/2601.20115), Jan 2026 - Paper Lantern, 120 papers surveyed).
- Utilisation and cost move the level, hardly the share.

## Limits

- Thin evidence: six publishers, eight rows from one. No transaction-level prices: publishers' summaries of sales, asks and appraisals, and one individual sale.
- H100 tops at +1 and +2 years exceed a new server's asking price per GPU: out of reach.
- One model, one rental benchmark, tested to six months. No scrap value (zero means "stopped earning"); staff and repair costs left out.
- Dates are the pages' own: the search index returned current copies, not snapshots as of 6 Nov. American Compute and Compute Exchange show only a month or a quarter.
- Two rows (Hashrate Index as reported by Blockspace; Stoa) were read only through the search index's copy of the page, not the page itself.
- Reuse: figures from American Compute's appraisal pages and from Compute Exchange are short attributed quotations from pages whose terms do not grant redistribution; the pack lists both as pointers only. Remove `data/resale_quotes.csv` rows for them before any wider reuse.
