# RL fine-tuning on Tinker: scaffold and briefing

Checked against the live service on 2026-11-14 with `tinker` 0.39.0 and the docs at tinker-docs.thinkingmachines.ai.
`policy_gradient.py` is the reusable `Trainer` / `Sampler`; `smoke_rl.py` is a toy RL loop that uses it.

## Install and auth
- `cd rl && uv sync && uv run python smoke_rl.py` (own uv project: `tinker`, `python-dotenv`, `jinja2`; about 90 seconds, 1 cent).
- Auth is the `TINKER_API_KEY` env var (or `uv run tinker auth login`). The module loads the nearest `.env` (here `../.env`) if it is unset.
- `tinker-cookbook` (renderers, `rl.train`, recipes) is optional and not installed here; it pulls in `torch`.
- New since SDK 0.38.0 (5 Nov 2026): every call returns an `APIFuture`, so use `await x` or `x.result()`. The `*_async` methods are deprecated and client factories return futures too. Llama and most Qwen3 models were retired in June 2026.

## Model and prices (USD per million tokens, list prices today)
| Model | Prefill | Sample | Train | Note |
|---|---|---|---|---|
| `Qwen/Qwen3-8B` (chosen) | 0.195 | 0.60 | 0.44 | cheapest model that can answer without a reasoning block |
| `openai/gpt-oss-20b` | 0.18 | 0.45 | 0.396 | always reasons first, so more tokens per episode; not tested |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | 0.184 | 0.495 | 0.44 | "limited-time 50% discount" price; not tested |
| `Qwen/Qwen3.5-4B` | 0.33 | 1.005 | 0.737 | the docs' quickstart model; `smoke_rl.py --model` ran unchanged |

Qwen3-8B with `enable_thinking=False` emitted well-formed JSON in all 3,000+ toy samples. It is the last Qwen3-generation model listed, so it may be retired; Qwen3.5-4B is the fallback. Live model list: `tinker.RestClient().get_server_capabilities().result()`. Prices: `https://tinker-docs.thinkingmachines.ai/tinker/models.json`.

## The lines that matter (raw SDK, run verbatim today; `rs` = your rewards for the 8 completions)
```python
svc = tinker.ServiceClient()                                              # reads TINKER_API_KEY
tc = svc.create_lora_training_client(base_model="Qwen/Qwen3-8B", rank=32).result()
tok = tc.get_tokenizer()
policy = tc.save_weights_and_get_sampling_client().result()               # current weights; redo after every optim_step
p = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                            enable_thinking=False, tokenize=True, return_dict=False)
seqs = policy.sample(tinker.ModelInput.from_ints(p), num_samples=8,
                     sampling_params=tinker.SamplingParams(max_tokens=256, temperature=1.0)).result().sequences
adv = [r - sum(rs) / len(rs) for r in rs]                                 # reward minus the prompt group's mean
data = [tinker.Datum(model_input=tinker.ModelInput.from_ints((p + s.tokens)[:-1]), loss_fn_inputs={
            "target_tokens": (p + s.tokens)[1:], "logprobs": [0.0] * (len(p) - 1) + s.logprobs,
            "advantages": [0.0] * (len(p) - 1) + [a] * len(s.tokens)}) for s, a in zip(seqs, adv)]
fb, opt = tc.forward_backward(data, "importance_sampling"), tc.optim_step(tinker.AdamParams(learning_rate=4e-5))
fb.result(); opt.result()                                                 # submit both, then wait
path = tc.save_weights_for_sampler("v1", ttl_seconds=7 * 86400).result().path   # tinker://<run>/sampler_weights/v1
later = svc.create_sampling_client(model_path=path).result()              # base model: base_model="Qwen/Qwen3-8B"
```
The cookbook's RL code (`recipes/rl_loop.py`, `rl/train.py`) is this same loop: group-mean advantages with no std division, all-equal groups dropped, losses summed over tokens, Adam at 4e-5, rank 32, temperature 1. The loss is one string: `importance_sampling` (`-(p/q)*A`, the `rl/train.py` default), `ppo`, `cispo`, `gspo`, `dro`, or `stable_reinforce` (alpha, now the `rl_loop.py` default; needs provenance spans and `record_stability_info=True`).

## Measured (Qwen3-8B, rank 32)
- Toy task (4 prompts x 8 samples, ~60 prompt / 10 completion tokens, 9 runs): sample batch 2.4-3.6 s; update 3.8-6.2 s, of which 2.5-4 s is `forward_backward`+`optim_step` and 1-2 s refreshing the sampler (medians per run). Setup 13-18 s. An 8-update run takes 85-111 s.
- Realistic shape (8 prompts x 8 samples, 706 prompt / ~210 completion tokens, 2 rounds): sample batch 3.4-4.8 s, update 5.6-5.7 s. Latency is mostly fixed overhead per request, so bigger batches cost little extra time. Save 14 s, resume from state 21 s.
- Default smoke run: mean reward -0.060 -> -0.018 over 8 updates; control on 64 fresh samples each: base -0.053, saved checkpoint -0.018.

## Cost (token counts x list price; the billing API lags by hours and showed no rows yet)
- Default smoke run: 2.5k prefill + 4.2k sampled + 10.6k trained tokens = $0.008. Everything run today: about $0.17.
- 2,000 episodes at 600 prompt / 150 completion tokens: sampling $0.18 + training $0.66 + prefill $0.03 (8 samples share one prompt) = about $0.90, or $1.10 if prefill turns out to be billed per completion. About $1.50 on Qwen3.5-4B.
- At 64 episodes per round that is ~32 rounds x ~10 s, so 5-6 minutes of Tinker time before your reward harness.

## Concurrency and rate limits
- No numeric limits are published. The SDK queues client-side (400 sample requests in flight) and retries with backoff; `tinker.RateLimitError` is the HTTP 429 class. Eight concurrent requests and three parallel training runs hit no limit here.
- One request per prompt with `num_samples=n`; submit all requests, then collect. Submit `forward_backward` and `optim_step` back to back. The docs say not to wrap calls in your own timeouts or retries.

## Gotchas hit
- `apply_chat_template` needs `jinja2`, which `tinker` does not install. The "PyTorch was not found" warning is harmless.
- No diversity, no gradient: asked plainly for a number, the base model answered `{"x": 0.5}` 24 times out of 24 at temperature 1. `update` skips all-equal groups and reports `n_constant_groups` and `skipped`.
- The policy collapses to one of the better answers each prompt already produced (here 0.75 / 0.45 / 0.71 / 0.57) within 2-8 updates at learning rates 2e-5 to 1e-4, so reward plateaus near -0.02. One run sampling at temperature 1.6 kept exploring: -0.067 -> -0.006 in 12 updates (control at temperature 1: -0.055 -> -0.005).
- `sample` returns the model's untempered logprobs at any temperature (checked at 0.3, 1.0, 1.8), so the importance ratio does not correct for temperature.
- With one update per fresh batch the sampler equals the learner (gap under 0.01 nats per token), so `ppo` and `cispo` should match `importance_sampling`; both were accepted and improved in 2-update runs.
- Checkpoints are big: 0.37 GB per rank-32 sampler checkpoint, 1.1 GB per training state; the free plan caps at 100 checkpoints or 100 GB. `Trainer.save` defaults to a 7-day TTL. Today's 13 checkpoints all expire within 24 hours.
- Small groups (2-8 per prompt) make the mean a noisy baseline; a median baseline gained up to 4.6 points at group size 2 (arxiv:2601.22582 - Paper Lantern, 107 papers surveyed). Not implemented here.

## The self-improvement loop on Modal GPUs
`rsi_forecaster.py` runs the loop on Tinker; `modal_forecaster.py` runs the same loop in one GPU container (function `rsi`). Each generation trains for 8 rounds from the kept LoRA weights, answers a fixed watch set (128 later AWS price-days x 4 answers), and replaces the kept weights only if the watch score is higher and stays higher in at least 90% of 2,000 resamples of the price-days. Otherwise the kept weights are restored from a copy held in memory and the learning rate is halved or doubled within 1e-5 to 1.6e-4. Our own 37 prices are not used.

    cd rl
    uv run modal deploy modal_forecaster.py                                   # once, and after any change to the container code
    uv run python modal_forecaster.py rsi-launch --name rsi_m1 --generations 5 --rounds 8 --plateau 4 --minutes 100   # returns at once
    uv run python modal_forecaster.py rsi-collect --name rsi_m1               # the generations so far, or the finished log

- Set `GPUVAL_RL_APP=<another name>` on all three commands to deploy under a second app name and leave the first app alone.
- `rsi-collect` writes the finished log to `runs/modal/rsi/<name>.json`, with the fields of the Tinker log (`start`, `generations`, `stop`, `kept_generation`, `kept_watch`, `minutes`) plus `gpu` and `cost_estimate_usd` (container minutes x the GPU's hourly price). While the loop is running it prints the log the container writes to the volume after every generation.
- The kept adapter is saved on the volume `gpuval-rl-out` at `rsi/<name>/adapter` after each acceptance and at the end.
- Differences from the Tinker loop: the training round is this file's own (per-price baseline, advantages rescaled to unit spread), and a rollback starts a fresh Adam optimizer.
