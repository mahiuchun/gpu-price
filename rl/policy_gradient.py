"""Single-step policy-gradient (GRPO-style) LoRA fine-tuning on Tinker.

Episode shape: prompt -> one completion -> scalar reward. Written against tinker 0.39.0.

    trainer = Trainer("Qwen/Qwen3-8B", lora_rank=32, learning_rate=4e-5)
    groups = trainer.sample(prompts, n_per_prompt=8)            # list[list[Completion]]
    batch = [(p, c, my_reward(c.text)) for p, g in zip(prompts, groups) for c in g]
    stats = trainer.update(batch)                               # one optimiser step
    path = trainer.save("round-10")                             # tinker://.../sampler_weights/round-10
    control = trainer.base_sampler().sample(prompts, 8)         # untrained base model
    later = Sampler(checkpoint=path).sample(prompts, 8)         # any process, any time

Sync / async: everything here is plain blocking code. The SDK returns `APIFuture`s that work
both ways (`.result()` or `await`); this module only uses `.result()`, and gets concurrency by
submitting all requests before waiting on any of them. Do NOT call these methods from inside a
running asyncio event loop (the SDK warns: it blocks the loop). From async code use
`await asyncio.to_thread(trainer.sample, ...)`.

Auth: `TINKER_API_KEY` must be in the environment. If it is not, the nearest `.env` in this
directory or a parent directory is loaded with python-dotenv. The key is never logged.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import tinker

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"

# USD per million tokens, copied 2026-11-14 from
# https://tinker-docs.thinkingmachines.ai/tinker/models.json (the machine-readable price list).
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "Qwen/Qwen3-8B": {"prefill": 0.195, "sample": 0.60, "train": 0.44},
    "openai/gpt-oss-20b": {"prefill": 0.18, "sample": 0.45, "train": 0.396},
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": {"prefill": 0.184, "sample": 0.495, "train": 0.44},
    "Qwen/Qwen3.5-4B": {"prefill": 0.33, "sample": 1.005, "train": 0.737},
    "Qwen/Qwen3.5-9B": {"prefill": 0.30, "sample": 0.807, "train": 1.463},
}


@dataclass(frozen=True)
class Completion:
    """One sampled completion plus everything `Trainer.update` needs to train on it."""

    text: str  # decoded completion, special tokens stripped
    tokens: list[int]  # sampled token ids (includes the end-of-turn token when stop_reason == "stop")
    logprobs: list[float]  # sampler log-prob of each sampled token (q in the importance ratio p/q)
    prompt_tokens: list[int]  # the rendered chat prompt the completion was sampled from
    stop_reason: str  # "stop" (model ended its answer) or "length" (cut off at max_tokens)


def _ensure_api_key() -> None:
    if os.environ.get("TINKER_API_KEY"):
        return
    from dotenv import load_dotenv

    for folder in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (folder / ".env").is_file():
            load_dotenv(folder / ".env")
            break
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set and no .env file providing it was found")


def _new_usage() -> dict[str, int]:
    return {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0}


def estimate_cost(usage: dict[str, int], base_model: str) -> float | None:
    """USD for a usage dict ({prefill,sample,train}_tokens); None if the model has no price entry.

    Assumption, not confirmed against an invoice: prefill is billed once per sample request
    (the prompt is shared by its `num_samples` completions), at the uncached rate. If it were
    billed per completion, multiply prefill_tokens by n_per_prompt for the upper bound.
    """
    price = PRICES_PER_MTOK.get(base_model)
    if price is None:
        return None
    return sum(usage[f"{kind}_tokens"] * price[kind] for kind in ("prefill", "sample", "train")) / 1e6


class Sampler:
    """Sampling-only handle on the base model or on a saved checkpoint (no training run).

    Sampler(base_model="Qwen/Qwen3-8B")             -> untrained base model (control)
    Sampler(checkpoint="tinker://.../sampler_weights/<name>") -> weights saved by Trainer.save
    """

    def __init__(
        self,
        base_model: str | None = None,
        checkpoint: str | None = None,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        service_client: tinker.ServiceClient | None = None,
        _client: tinker.SamplingClient | None = None,
    ) -> None:
        if _client is None:
            if (base_model is None) == (checkpoint is None):
                raise ValueError("pass exactly one of base_model or checkpoint")
            _ensure_api_key()
            service_client = service_client or tinker.ServiceClient(display_name="rl-sampler")
            if checkpoint is not None:
                _client = service_client.create_sampling_client(model_path=checkpoint).result()
            else:
                _client = service_client.create_sampling_client(base_model=base_model).result()
        self._client = _client
        self._tokenizer = _client.get_tokenizer()
        self.base_model = _client.get_base_model()
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.usage = _new_usage()

    def render(self, prompt: str) -> list[int]:
        """Token ids of the chat-formatted prompt, ending where the assistant's answer starts."""
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,  # hybrid models: False = answer without a <think> block
            tokenize=True,
            return_dict=False,
        )
        return [int(i) for i in ids]

    def sample(
        self,
        prompts: Sequence[str],
        n_per_prompt: int,
        temperature: float = 1.0,
        max_tokens: int = 256,
        stop: Sequence[str] | None = None,
    ) -> list[list[Completion]]:
        """Sample `n_per_prompt` completions for each prompt; result[i] is the group for prompts[i].

        One request per prompt (`num_samples=n_per_prompt`); all requests are submitted before
        any result is awaited, so they run concurrently on the service.
        """
        params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=stop)
        rendered = [self.render(prompt) for prompt in prompts]
        futures = [
            self._client.sample(
                prompt=tinker.ModelInput.from_ints(ids), num_samples=n_per_prompt, sampling_params=params
            )
            for ids in rendered
        ]
        groups: list[list[Completion]] = []
        for ids, future in zip(rendered, futures):
            response = future.result()
            group = []
            for seq in response.sequences:
                if seq.logprobs is None:
                    raise RuntimeError("sampler returned no logprobs; cannot build a policy-gradient update")
                tokens = [int(t) for t in seq.tokens]
                group.append(
                    Completion(
                        text=self._tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                        tokens=tokens,
                        logprobs=[float(lp) for lp in seq.logprobs],
                        prompt_tokens=ids,
                        stop_reason=str(seq.stop_reason),
                    )
                )
                self.usage["sample_tokens"] += len(tokens)
            self.usage["prefill_tokens"] += len(ids)
            groups.append(group)
        return groups


class Trainer:
    """LoRA policy-gradient trainer: sample from the current policy, then update on rewards.

    Each `update` is one `forward_backward` + one Adam `optim_step`, with advantages
    `reward - mean(reward of the same prompt's group)` (GRPO-style centering, no std division,
    as in the Tinker cookbook) spread over every sampled token of the completion.

    loss_fn: "importance_sampling" (default here and in the cookbook's rl/train.py), "ppo" or
    "cispo". They take the same inputs; pass clip thresholds via `loss_fn_config`.
    """

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        lora_rank: int = 32,
        learning_rate: float = 4e-5,
        *,
        loss_fn: str = "importance_sampling",
        loss_fn_config: dict[str, float | str] | None = None,
        normalize_advantages: bool = False,
        skip_constant_groups: bool = True,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        seed: int | None = None,
        resume_from: str | None = None,
        session_name: str = "rl-policy-gradient",
    ) -> None:
        """resume_from: a `tinker://<run>/weights/<name>` path from `save(..., with_state=True)`;
        training continues from those weights with fresh optimiser state, and the checkpoint's
        own base model and LoRA rank apply instead of the arguments."""
        _ensure_api_key()
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.loss_fn_config = loss_fn_config
        self.normalize_advantages = normalize_advantages
        self.skip_constant_groups = skip_constant_groups
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.usage = _new_usage()
        self.steps = 0
        self.last_state_path: str | None = None

        self._service = tinker.ServiceClient(display_name=session_name)
        if resume_from is not None:
            self._training = self._service.create_training_client_from_state(resume_from).result()
        else:
            self._training = self._service.create_lora_training_client(
                base_model=base_model, rank=lora_rank, seed=seed
            ).result()
        self._policy = self._wrap(self._training.save_weights_and_get_sampling_client().result())
        self.base_model = self._policy.base_model  # authoritative when resuming from a checkpoint

    def _wrap(self, client: tinker.SamplingClient) -> Sampler:
        sampler = Sampler(
            _client=client, system_prompt=self.system_prompt, enable_thinking=self.enable_thinking
        )
        sampler.usage = self.usage  # share one running token count with the trainer
        return sampler

    # -- sampling -------------------------------------------------------------------------

    def sample(
        self,
        prompts: Sequence[str],
        n_per_prompt: int,
        temperature: float = 1.0,
        max_tokens: int = 256,
        stop: Sequence[str] | None = None,
    ) -> list[list[Completion]]:
        """Sample from the CURRENT policy (weights as of the last `update`).

        The service returns the model's untempered logprobs at any temperature, so away from
        1.0 the importance ratio does not correct for it (plain REINFORCE on off-policy
        samples). 1.0 is the cookbook default; above 1 buys exploration (see README gotchas).
        """
        return self._policy.sample(prompts, n_per_prompt, temperature, max_tokens, stop)

    def base_sampler(self) -> Sampler:
        """The untrained base model, for control comparisons (tokens count towards `usage`)."""
        return self._wrap(self._service.create_sampling_client(base_model=self.base_model).result())

    def load_sampler(self, checkpoint: str) -> Sampler:
        """A saved checkpoint (`tinker://.../sampler_weights/<name>`), e.g. one from `save`."""
        return self._wrap(self._service.create_sampling_client(model_path=checkpoint).result())

    # -- training -------------------------------------------------------------------------

    def update(
        self,
        batch: Iterable[tuple[str, Completion, float]],
        learning_rate: float | None = None,
    ) -> dict[str, float]:
        """One policy-gradient step on (prompt, completion, reward) triples.

        Triples are grouped by prompt string; advantage = reward - group mean. Groups whose
        rewards are all equal carry no signal and are skipped (so a group needs >= 2 samples
        with different rewards to contribute). Returns training stats; `stats["skipped"]` is
        1.0 when nothing had signal and no step was taken.
        """
        t_start = time.time()
        groups: dict[str, list[tuple[Completion, float]]] = {}
        for prompt, completion, reward in batch:
            groups.setdefault(prompt, []).append((completion, float(reward)))
        if not groups:
            raise ValueError("empty batch")

        all_rewards = [reward for members in groups.values() for _, reward in members]
        rows: list[tuple[Completion, float]] = []
        constant_groups = 0
        for members in groups.values():
            rewards = [reward for _, reward in members]
            if max(rewards) - min(rewards) < 1e-12:
                constant_groups += 1
                if self.skip_constant_groups:
                    continue
            baseline = sum(rewards) / len(rewards)
            rows.extend((completion, reward - baseline) for completion, reward in members)

        stats: dict[str, float] = {
            "n_samples": float(len(all_rewards)),
            "n_groups": float(len(groups)),
            "n_constant_groups": float(constant_groups),
            "mean_reward": statistics.fmean(all_rewards),
            "reward_std": statistics.pstdev(all_rewards),
            "skipped": 0.0,
        }
        advantages = [advantage for _, advantage in rows]
        scale = statistics.pstdev(advantages) if rows else 0.0
        if scale < 1e-12:
            stats.update(skipped=1.0, n_datums=0.0, seconds=time.time() - t_start)
            return stats
        if self.normalize_advantages:  # one batch-wide scale, not per group (avoids GRPO's difficulty bias)
            advantages = [advantage / scale for advantage in advantages]

        datums = [self._datum(completion, adv) for (completion, _), adv in zip(rows, advantages)]
        lr = self.learning_rate if learning_rate is None else learning_rate
        # Submit everything before waiting on anything: the three requests run in order on the
        # service and share one "clock cycle" instead of costing three.
        fwd_bwd_future = self._training.forward_backward(datums, self.loss_fn, self.loss_fn_config)
        optim_future = self._training.optim_step(tinker.AdamParams(learning_rate=lr))
        policy_future = self._training.save_weights_and_get_sampling_client()
        fwd_bwd = fwd_bwd_future.result()
        optim_future.result()
        t_trained = time.time()
        self._policy = self._wrap(policy_future.result())
        t_done = time.time()

        # Learner log-probs come back per position; compare with the sampler's on the completion.
        gaps = []
        for (completion, _), output in zip(rows, fwd_bwd.loss_fn_outputs):
            learner = output["logprobs"].tolist()[-len(completion.tokens) :]
            gaps.extend(q - p for q, p in zip(completion.logprobs, learner))
        train_tokens = sum(len(c.prompt_tokens) + len(c.tokens) - 1 for c, _ in rows)
        self.usage["train_tokens"] += train_tokens
        self.steps += 1
        stats.update(
            n_datums=float(len(datums)),
            train_tokens=float(train_tokens),
            mean_abs_advantage=statistics.fmean(abs(a) for a in advantages),
            loss_sum=float(fwd_bwd.metrics.get("loss:sum", float("nan"))),
            sampler_learner_logprob_gap=statistics.fmean(gaps),  # ~0 when sampling was on-policy
            learning_rate=lr,
            fwd_bwd_optim_seconds=t_trained - t_start,
            refresh_sampler_seconds=t_done - t_trained,
            seconds=t_done - t_start,
        )
        return stats

    @staticmethod
    def _datum(completion: Completion, advantage: float) -> tinker.Datum:
        """Position i of the input predicts token i+1, so inputs/targets are the sequence shifted by one.

        Prompt positions get advantage 0 (which masks them); every sampled token gets the
        completion's advantage. The RL losses accept exactly these three keys.
        """
        full = completion.prompt_tokens + completion.tokens
        n_prefix = len(completion.prompt_tokens) - 1
        n_sampled = len(completion.tokens)
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(full[:-1]),
            loss_fn_inputs={
                "target_tokens": full[1:],
                "logprobs": [0.0] * n_prefix + list(completion.logprobs),
                "advantages": [0.0] * n_prefix + [advantage] * n_sampled,
            },
        )

    # -- checkpoints ----------------------------------------------------------------------

    def save(self, name: str, ttl_seconds: int | None = 7 * 24 * 3600, with_state: bool = False) -> str:
        """Save current weights for sampling; returns `tinker://<run_id>/sampler_weights/<name>`.

        Load it later with `Sampler(checkpoint=path)` or `trainer.load_sampler(path)`.
        with_state=True also saves a resumable training checkpoint (`.last_state_path`, for
        `Trainer(resume_from=...)`). Checkpoints expire after `ttl_seconds` (None = keep
        forever; the free plan holds 100 checkpoints / 100 GB per organisation).
        """
        sampler_future = self._training.save_weights_for_sampler(name, ttl_seconds=ttl_seconds)
        state_future = (
            self._training.save_state(name, ttl_seconds=ttl_seconds) if with_state else None
        )
        if state_future is not None:
            self.last_state_path = state_future.result().path
        return sampler_future.result().path

    def cost(self) -> float | None:
        """Estimated USD spent by this trainer so far (see `estimate_cost`)."""
        return estimate_cost(self.usage, self.base_model)
