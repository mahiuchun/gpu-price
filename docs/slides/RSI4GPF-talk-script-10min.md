# RSI4GPF: 10-minute talk script

The spoken script for the 10-minute deck ([PDF](RSI4GPF-slides-10min.pdf)). The same text is in each slide's speaker
notes. About 1104 words: at a relaxed pace, with pauses and slide changes, it runs about 10 minutes.
Times are cues, not limits; slide 6, the training loop, is the one to take your time on.

## 1. RSI4GPF  <sub>0:00–0:55</sub>

Hi everyone. This is RSI4GPF: a crew of AI agents and one coach AI, with recursive self-improvement, for GPU price forecasting. We forecast thirty-seven posted GPU rental prices, one, three and seven days ahead. And alongside the forecast, we asked a harder question: which kinds of AI help actually survive when you test them on data the system has never seen? I’ll give you the answer up front. The forecast we handed in made twenty percent less error than the standard baseline on fresh days. In the next ten minutes: why this matters, how it works, and what held up.

## 2. Thin prices, and AI that can fool itself  <sub>0:55–2:13</sub>

Two problems motivate this work. First, GPU prices are thin. Compute is now rented, leased and even lent against like a commodity. But a published GPU price is usually just the median of one website’s list of offers. It can move because an offer joins or leaves the list, because one provider reprices every day, or because a euro price is converted to dollars each morning. If you’re setting a lease rate or valuing collateral, you need a forecast with honest ranges. Second, self-improving AI is easy to fool. A loop that keeps whatever scores well on the data it tuned on will happily improve on noise. So the real question is not whether an AI can change its forecaster, but whether those changes still help on days it has never seen. That needs a verifier the AI cannot change, and a final exam on unseen data.

## 3. The forecast we hand in cut error by 20% on days no model had seen  <sub>2:13–3:16</sub>

Here is the headline. This chart puts every forecaster we built on exactly the same four hundred and seventy-nine forecasts, on days no agent and no coach had answered. Zero is drift, the standard baseline, which simply replays how the median moved before. Our submitted forecast, five AI agents after the coach’s changes, is at the top: twenty percent less error than drift, with a likely range of ten to twenty-eight percent. Below it you can see reinforcement learning lifting two small open models well above their untrained versions. And the plain engine, with no AI at all, is at about ten percent. Every number is scored exactly the way the organisers’ own scorer scores it.

## 4. Most movement in a published median is offers coming and going  <sub>3:16–3:56</sub>

Why does looking inside the list help so much? Because of this. For neocloud list prices, sixty-six percent of the day-to-day movement in the published median comes from offers joining or leaving the list. Another twenty-five percent comes from one or two offers that reprice almost daily. Only nine percent is everyone else actually changing their price. So if you only watch the median, you’re mostly watching the sample change.

## 5. Replay what each offer did, then take the middle again  <sub>3:56–4:48</sub>

That insight gives us the engine, which uses no AI at all. It starts from today’s list, so it knows which offer sits in the middle and how far its neighbours let the middle move. Then, for each past pair of days, it replays what every individual offer did, who repriced, who left, who came back, and takes the median again. Do that across many past days and you get a whole spread of possible medians. The tenth, fiftieth and ninetieth percentiles of that spread are the forecast. It’s classic historical simulation, and nothing is trained.

## 6. Training: a self-improvement loop with verifiers  <sub>4:48–6:31</sub>

This is the heart of the project. On top of the engine sits a crew of five AI agents, each with one job: a list reader, a repricer watcher, a base-rate keeper, a reference weigher and a jump spotter. Their forecasts are averaged. Training improves this crew in a recursive loop, and it never touches model weights; what changes is the crew’s configuration. Each round, the crew forecasts six practice days, and a verifier scores it against drift and lists the worst misses. A coach AI, the generator, reads that and proposes exactly one change: rewrite an agent, add one, drop one, or stretch the ranges. A second verifier then re-scores the changed crew. The change is kept only if the practice days get better and five hidden check days get no worse, and the coach only ever learns pass or fail on those. The verifiers are fixed code that the AI cannot change. We ran ten rounds. Round one kept wider ranges, round three kept a rewritten base-rate keeper, and the rest were dropped. A final exam on days the loop never saw scored the final crew four percent better than where it started.

## 7. Inference: the frozen crowd forecasts 6 Nov  <sub>6:31–7:08</sub>

At inference, the crew is frozen. Nothing learns and nothing changes. For each of the thirty-seven prices, the facts, the last twenty-one days and today’s list of offers, go to the five agents. Their lows, middles and highs are averaged, and the ranges are stretched at three and seven days. One pass per price gives the one hundred and eleven forecasts we handed in.

## 8. The agents and the engine beat drift in every test  <sub>7:08–7:56</sub>

One test window is thin, so we checked three: the eight weeks before our test, the three test weeks themselves, and four years of AWS spot prices, almost thirty-six thousand forecasts. The engine was a steady eight percent ahead of drift in all three. The five agents were fifteen to nineteen percent ahead in all three. And notice the orange bars: an off-the-shelf machine learning model swung from worse than drift to clearly ahead, depending on the period, which is exactly why we didn’t rely on it.

## 9. Seven ways to make the system improve itself, tested on fresh days  <sub>7:56–9:00</sub>

Now the honest part. We tried seven ways to make the system improve itself. Two held up on fresh days: the coach, and reinforcement learning that teaches a small open model to forecast. Three did not. An AI scientist tuning the engine actually made it slightly worse on unseen weeks. And two gave small gains we can’t call established. Two more caveats. Our ranges held the truth eighty percent of the time overall, but only sixty-six percent of the time when the price actually moved, and that’s the number a lender cares about. On the other hand, the result doesn’t hang on one model: four other API models in the same crew cut error by fifteen to twenty-two percent.

## 10. Three things to remember  <sub>9:00–10:00</sub>

So, three things to remember. One: look inside the list. A median moves mostly because offers come and go. Two: a crew of AI agents plus a coach cut error by twenty percent against drift on days no model had seen. Three: most self-improvement ideas did not survive a fair test. The gate is the product. A verifier the AI can’t change is what makes recursive self-improvement trustworthy, and that same propose, verify, keep loop can be pointed at other agent crews and other forecasting problems. Everything is open and reproducible: the forecast rebuilds in about fifteen seconds with no AI calls. Thank you, and I’m happy to take questions.
