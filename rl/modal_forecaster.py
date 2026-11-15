"""The RL forecaster trained on Modal GPUs instead of Tinker.

Same task as train_forecaster.py: an open model reads the facts about one price and answers with
low / mid / high percent moves for 1, 3 and 7 days; its reward is how much less error it made than
drift. Here the model's weights are loaded on a rented GPU and this file does the sampling and the
policy-gradient update itself (Hugging Face transformers plus a LoRA adapter). Each run is its own
container, so many runs go at once: several learning rates, several random seeds.

    cd rl
    uv run modal deploy modal_forecaster.py                    # once, and after any change to the container code
    uv run python modal_forecaster.py launch --plan smoke      # start the runs; returns at once
    uv run python modal_forecaster.py collect --plan smoke     # fetch what has finished, score it, print the tables

What a container is sent: prompts, and for practice prices only, the outcomes needed to score an
answer. For the test days it is sent prompts only; the outcomes stay on this machine.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import modal

APP = os.environ.get("GPUVAL_RL_APP", "gpuval-rl-forecaster")  # a second name lets a fix be deployed while runs are in flight
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.15.0", "transformers==5.23.0", "peft==0.21.3", "accelerate==1.15.0", "numpy==2.5.4")
         .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"}))
hf_cache = modal.Volume.from_name("gpuval-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("gpuval-rl-out", create_if_missing=True)

HORIZONS = (1, 3, 7)
CLIP, UNREADABLE = 3.0, -3.0
DEFAULTS = dict(base_model="Qwen/Qwen3-8B", lr=4e-5, seed=0, rounds=60, prompts=16, samples=8, lora_rank=32, lora_alpha=32,
                micro=16, check_every=10, check_samples=4, eval_samples=8, max_new=90, gen_batch=16,
                token_budget=170_000)  # answers x tokens held on the GPU at once while sampling


# ------------------------------------------------- scoring an answer (mirrors forecast_env) ----
def parse(text: str, last: float) -> dict | None:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(text[a:b + 1])
        out = {}
        for h in HORIZONS:
            lo, mid, hi = (float(x) for x in obj[f"h{h}"][:3])
            if not all(math.isfinite(v) for v in (lo, mid, hi)) or max(abs(lo), abs(mid), abs(hi)) > 60:
                return None
            lo, hi = min(lo, mid), max(hi, mid)
            out[h] = (last * (1 + lo / 100), last * (1 + mid / 100), last * (1 + hi / 100))
        return out
    except Exception:
        return None


def _pin(y: float, f: float, tau: float) -> float:
    d = y - f
    return max(tau * d, (tau - 1.0) * d)


def cell_loss(y: float, q) -> float:
    return (_pin(y, q[0], 0.1) + _pin(y, q[1], 0.5) + _pin(y, q[2], 0.9)) / 3.0


def reward_of(item: dict, text: str) -> tuple[float, bool]:
    q = parse(text, item["last"])
    if q is None:
        return UNREADABLE, False
    base = statistics.fmean(cell_loss(item["y"][str(h)], item["drift"][str(h)]) for h in HORIZONS)
    mine = statistics.fmean(cell_loss(item["y"][str(h)], q[h]) for h in HORIZONS)
    return max(-CLIP, min(CLIP, 100.0 * (base - mine) / item["last"])), True


# ------------------------------------------------------------------ the model on the GPU ----
class Policy:
    """The open model plus a LoRA adapter: sample answers, and take policy-gradient steps."""

    def __init__(self, cfg: dict, system: str, train: bool, adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.cfg, self.system = torch, cfg, system
        torch.manual_seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        self.tok = AutoTokenizer.from_pretrained(cfg["base_model"])
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16, device_map={"": 0})
        except ValueError:  # a model that also reads images is registered under a different class; we use its text part
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(cfg["base_model"], dtype=torch.bfloat16, device_map={"": 0})
        self.dev = next(model.parameters()).device
        self.opt = None
        if train:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(model, LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, bias="none",
                                                     target_modules="all-linear", task_type="CAUSAL_LM"))
            core = model.base_model.model
            core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            core.enable_input_require_grads()
            self.params = [p for p in model.parameters() if p.requires_grad]
            self.opt = torch.optim.Adam(self.params, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8)
        elif adapter:  # a trained adapter saved by an earlier run, for answering only
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            core = model.base_model.model
        else:
            core = model
        self.model, self.head = model, core.lm_head
        self.decoder = core.model.language_model if hasattr(core.model, "language_model") else core.model
        eos = self.tok.eos_token_id
        gen_eos = getattr(core.generation_config, "eos_token_id", None)
        ids = set([eos] if isinstance(eos, int) else []) | set([gen_eos] if isinstance(gen_eos, int) else (gen_eos or []))
        self.eos = torch.tensor(sorted(ids), device=self.dev)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else int(self.eos[0])
        self.share_prefix = self._prefix_sharing_works()

    def render(self, prompt: str) -> list[int]:
        out = self.tok.apply_chat_template([{"role": "system", "content": self.system}, {"role": "user", "content": prompt}],
                                           add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
        return [int(i) for i in (out["input_ids"] if isinstance(out, dict) else out)]

    # -- sampling ---------------------------------------------------------------------------
    def _prefill(self, prompts: list[list[int]], n: int, share: bool):
        torch = self.torch
        B, L = len(prompts), max(len(p) for p in prompts)
        ids = torch.full((B, L), self.pad, dtype=torch.long, device=self.dev)
        mask = torch.zeros((B, L), dtype=torch.long, device=self.dev)
        for i, p in enumerate(prompts):  # left padding, so every prompt ends at the last position
            ids[i, L - len(p):] = torch.tensor(p, device=self.dev)
            mask[i, L - len(p):] = 1
        if not share:
            ids, mask = ids.repeat_interleave(n, 0), mask.repeat_interleave(n, 0)
        out = self.decoder(input_ids=ids, attention_mask=mask, use_cache=True)
        past, logits = out.past_key_values, self.head(out.last_hidden_state[:, -1])
        if share:  # the prompt is run once and its cache copied for the n answers
            past.batch_repeat_interleave(n)
            logits, mask = logits.repeat_interleave(n, 0), mask.repeat_interleave(n, 0)
        return past, logits, mask

    def _prefix_sharing_works(self) -> bool:
        torch = self.torch
        try:
            prompts = [self.render("Say yes."), self.render("Count from one to five, slowly.")]
            self.model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                step = torch.full((4, 1), int(self.tok.encode(" the")[0]), dtype=torch.long, device=self.dev)
                outs = []
                for share in (True, False):
                    past, logits, mask = self._prefill(prompts, 2, share)
                    mask = torch.cat([mask, torch.ones((4, 1), dtype=torch.long, device=self.dev)], 1)
                    h = self.decoder(input_ids=step, attention_mask=mask, past_key_values=past, use_cache=True).last_hidden_state[:, -1]
                    outs.append((logits.float(), self.head(h).float()))
            same_top = all(bool((a.argmax(-1) == b.argmax(-1)).all()) for a, b in zip(outs[0], outs[1]))
            gap = max(float((a.log_softmax(-1).exp() - b.log_softmax(-1).exp()).abs().max()) for a, b in zip(outs[0], outs[1]))
            print(f"prefix sharing self-test: same top token {same_top}, largest probability gap {gap:.4f}", flush=True)
            return same_top and gap < 0.02
        except Exception as e:  # fall back to running every copy of the prompt
            print(f"prefix sharing unavailable ({type(e).__name__}: {e}); using plain batching", flush=True)
            return False

    def sample(self, prompts: list[list[int]], n: int) -> list[list[dict]]:
        """n answers for each prompt at temperature 1 from the full distribution. Each answer: text, tokens, logprobs."""
        torch = self.torch
        self.model.eval()
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out: list = [None] * len(prompts)
        pos = 0
        while pos < len(order):
            longest = len(prompts[order[min(pos + self.cfg["gen_batch"], len(order)) - 1]]) + self.cfg["max_new"]
            k = max(1, min(self.cfg["gen_batch"], self.cfg["token_budget"] // (n * longest)))  # prompts are sorted, so this bounds the batch's memory
            idx = order[pos:pos + k]
            pos += k
            o = None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                past, logits, mask = self._prefill([prompts[i] for i in idx], n, self.share_prefix)
                N = logits.shape[0]
                done = torch.zeros(N, dtype=torch.bool, device=self.dev)
                toks, lps = [], []
                for _ in range(self.cfg["max_new"]):
                    logp = torch.log_softmax(logits.float(), -1)
                    nxt = torch.multinomial(logp.exp(), 1).squeeze(1)
                    toks.append(torch.where(done, torch.full_like(nxt, -1), nxt))
                    lps.append(logp.gather(1, nxt[:, None]).squeeze(1))
                    done = done | torch.isin(nxt, self.eos)
                    if bool(done.all()):
                        break
                    mask = torch.cat([mask, torch.ones((N, 1), dtype=torch.long, device=self.dev)], 1)
                    o = self.decoder(input_ids=nxt[:, None], attention_mask=mask, past_key_values=past, use_cache=True)
                    past, logits = o.past_key_values, self.head(o.last_hidden_state[:, -1])
            T, P = torch.stack(toks, 1).tolist(), torch.stack(lps, 1).tolist()
            del past, logits, mask, o  # o holds the cache too; without this the next batch runs out of memory
            torch.cuda.empty_cache()
            for b, i in enumerate(idx):
                group = []
                for j in range(n):
                    row, lp = T[b * n + j], P[b * n + j]
                    m = row.index(-1) if -1 in row else len(row)
                    group.append({"text": self.tok.decode(row[:m], skip_special_tokens=True).strip(), "tokens": row[:m], "logprobs": lp[:m],
                                  "finished": bool(m and row[m - 1] in set(self.eos.tolist()))})
                out[i] = group
        return out

    # -- one policy-gradient step ------------------------------------------------------------
    def update(self, rows: list[tuple[list[int], dict, float]]) -> dict:
        """rows: (prompt tokens, sampled answer, advantage). Loss = -sum(advantage * log prob of the answer) / number of answers."""
        torch = self.torch
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        rows = sorted(rows, key=lambda r: len(r[0]) + len(r[1]["tokens"]))
        total, gaps = 0.0, []
        for k in range(0, len(rows), self.cfg["micro"]):
            mb = rows[k:k + self.cfg["micro"]]
            L = max(len(p) + len(c["tokens"]) for p, c, _ in mb) - 1
            ids = torch.full((len(mb), L), self.pad, dtype=torch.long, device=self.dev)
            mask = torch.zeros((len(mb), L), dtype=torch.long, device=self.dev)
            bi, ti, tgt, adv, old = [], [], [], [], []
            for i, (p, c, a) in enumerate(mb):
                full = p + c["tokens"]
                ids[i, :len(full) - 1] = torch.tensor(full[:-1], device=self.dev)
                mask[i, :len(full) - 1] = 1
                for j, t in enumerate(c["tokens"]):  # position len(p)-1+j predicts answer token j
                    bi.append(i); ti.append(len(p) - 1 + j); tgt.append(t); adv.append(a); old.append(c["logprobs"][j])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = self.decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                logits = self.head(h[torch.tensor(bi, device=self.dev), torch.tensor(ti, device=self.dev)])
            lp = torch.log_softmax(logits.float(), -1).gather(1, torch.tensor(tgt, device=self.dev)[:, None]).squeeze(1)
            loss = -(torch.tensor(adv, device=self.dev, dtype=torch.float32) * lp).sum() / len(rows)
            loss.backward()
            total += float(loss)
            gaps.append(float((lp.detach() - torch.tensor(old, device=self.dev)).abs().mean()))
            del h, logits, lp, loss
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.params, 1e9))
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)  # free the gradients before the next round's sampling
        torch.cuda.empty_cache()
        return {"loss": total, "grad_norm": gnorm, "sampler_learner_gap": statistics.fmean(gaps)}

    def adapter_norm(self) -> float:
        return float(sum(float(p.detach().float().pow(2).sum()) for n, p in self.model.named_parameters() if "lora_B" in n) ** 0.5)


@app.function(image=image, volumes={"/cache": hf_cache}, timeout=3600)
def prefetch(base_model: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(base_model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    hf_cache.commit()
    return path


@app.function(image=image, gpu=["H100", "A100-80GB"], timeout=4 * 3600, volumes={"/cache": hf_cache, "/out": out_vol}, max_containers=16)
def run(cfg: dict, system: str, train_rounds: list, check_items: list, eval_items: list) -> dict:
    """Train (if cfg["rounds"] > 0), watch a fixed set of later practice prices, then answer the test prompts."""
    import importlib.metadata as md

    import torch

    cfg = {**DEFAULTS, **cfg}
    t0 = time.time()
    hf_cache.reload()
    pol = Policy(cfg, system, train=cfg["rounds"] > 0)
    res = {"cfg": cfg, "gpu": torch.cuda.get_device_name(0), "prefix_sharing": pol.share_prefix,
           "versions": {p: md.version(p) for p in ("torch", "transformers", "peft")}, "rounds": [], "checks": [], "load_seconds": round(time.time() - t0, 1)}
    print(f"{cfg['name']}: loaded {cfg['base_model']} on {res['gpu']} in {res['load_seconds']}s; versions {res['versions']}", flush=True)
    out_dir = Path("/out") / cfg["plan"] / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        (out_dir / "result.json").write_text(json.dumps(res))
        out_vol.commit()

    check_ids = [pol.render(it["prompt"]) for it in check_items]

    def check(r: int) -> None:
        groups = pol.sample(check_ids, cfg["check_samples"])
        per_item = [[reward_of(it, c["text"]) for c in g] for it, g in zip(check_items, groups)]
        flat = [x for g in per_item for x in g]
        first = [x for g in per_item[:64] for x in g]
        res["checks"].append({"round": r, "mean_reward": round(statistics.fmean(x[0] for x in flat), 4), "readable": round(sum(x[1] for x in flat) / len(flat), 4),
                              "mean_reward_first64": round(statistics.fmean(x[0] for x in first), 4), "n": len(flat), "minutes": round((time.time() - t0) / 60, 1)})
        print(f"{cfg['name']} check at round {r}: {res['checks'][-1]}", flush=True)

    if check_items:
        check(0)
    for r in range(1, cfg["rounds"] + 1):
        t1 = time.time()
        items = train_rounds[r - 1]
        pids = [pol.render(it["prompt"]) for it in items]
        groups = pol.sample(pids, cfg["samples"])
        t2 = time.time()
        rows, rs, ok, spread = [], [], [], []
        for it, p, g in zip(items, pids, groups):
            sc = [reward_of(it, c["text"]) for c in g]
            base = statistics.fmean(x[0] for x in sc)
            rs += [x[0] for x in sc]; ok += [x[1] for x in sc]
            if max(x[0] for x in sc) - min(x[0] for x in sc) < 1e-12:
                continue  # every answer scored the same: nothing to learn from this price
            spread += [x[0] - base for x in sc]
            rows += [(p, c, x[0] - base) for c, x in zip(g, sc)]
        row = {"round": r, "mean_reward": round(statistics.fmean(rs), 4), "readable": round(sum(ok) / len(ok), 4)}
        if rows:
            sd = statistics.pstdev(spread) or 1.0
            up = pol.update([(p, c, a / sd) for p, c, a in rows])  # advantages rescaled to unit spread across the batch
            row.update(within_group_sd=round(sd, 4), grad_norm=round(up["grad_norm"], 4), sampler_learner_gap=round(up["sampler_learner_gap"], 4),
                       adapter_norm=round(pol.adapter_norm(), 4))
        row.update(sample_seconds=round(t2 - t1, 1), update_seconds=round(time.time() - t2, 1))
        res["rounds"].append(row)
        print(f"{cfg['name']} round {r}: {row}", flush=True)
        if r % cfg["check_every"] == 0 or r == cfg["rounds"]:
            if check_items:
                check(r)
            save()
    res["train_minutes"] = round((time.time() - t0) / 60, 1)
    if cfg["rounds"] > 0:
        pol.model.save_pretrained(str(out_dir / "adapter"))
        save()  # the training record and the adapter are safe before the test prompts are answered
    if pol.opt is not None:
        pol.opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    t3 = time.time()
    groups = pol.sample([pol.render(it["prompt"]) for it in eval_items], cfg["eval_samples"]) if eval_items else []
    res["eval"] = {it["id"]: [c["text"] for c in g] for it, g in zip(eval_items, groups)}
    res["eval_minutes"] = round((time.time() - t3) / 60, 1)
    res["total_minutes"] = round((time.time() - t0) / 60, 1)
    save()
    return res


@app.function(image=image, gpu=["H100", "A100-80GB"], timeout=2 * 3600, volumes={"/cache": hf_cache, "/out": out_vol}, max_containers=16)
def evaluate(cfg: dict, system: str, eval_items: list) -> dict:
    """Answer the test prompts with a run's saved adapter (or with the untrained model if it has none)."""
    import importlib.metadata as md

    import torch

    cfg = {**DEFAULTS, **cfg}
    t0 = time.time()
    hf_cache.reload()
    out_vol.reload()
    out_dir = Path("/out") / cfg["plan"] / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = out_dir / "adapter"
    assert cfg["rounds"] == 0 or adapter.exists(), f"no saved adapter for {cfg['name']}"
    pol = Policy(cfg, system, train=False, adapter=str(adapter) if adapter.exists() else None)
    res = json.loads((out_dir / "result.json").read_text()) if (out_dir / "result.json").exists() else {"cfg": cfg, "rounds": [], "checks": []}
    res.update(gpu=torch.cuda.get_device_name(0), prefix_sharing=pol.share_prefix, versions={p: md.version(p) for p in ("torch", "transformers", "peft")})
    groups = pol.sample([pol.render(it["prompt"]) for it in eval_items], cfg["eval_samples"])
    res["eval"] = {it["id"]: [c["text"] for c in g] for it, g in zip(eval_items, groups)}
    res["eval_minutes"] = round((time.time() - t0) / 60, 1)
    res["eval_done_separately"] = True
    (out_dir / "result.json").write_text(json.dumps(res))
    out_vol.commit()
    return res


# ----------------------------------------------------------- everything below runs locally ----
HERE = Path(__file__).resolve().parent
LOCAL = HERE / "runs" / "modal"


def plans(lr: float | None = None) -> dict:
    """lr: the learning rate chosen on the watch set in sweep1; used by the follow-up plans."""
    sweep = [dict(name="base_qwen3_8b", rounds=0)]
    sweep += [dict(name=f"lr{lr:g}_s{seed}", lr=lr, seed=seed) for lr in (4e-5, 1e-4, 2.5e-4, 6e-4) for seed in (0, 1)]
    return {"smoke": [dict(name="smoke_base", rounds=0, n_check=8, n_eval=12),
                      dict(name="smoke_train", rounds=2, prompts=4, samples=4, lr=1e-4, check_every=1, n_check=8, n_eval=12)],
            "smoke9b": [dict(name="smoke9b_base", base_model="Qwen/Qwen3.5-9B", rounds=0, n_check=8, n_eval=12),
                        dict(name="smoke9b_train", base_model="Qwen/Qwen3.5-9B", rounds=2, prompts=4, samples=4, lr=1e-4, check_every=1, n_check=8, n_eval=12)],
            "sweep1": sweep,
            # follow-ups at the learning rate chosen in sweep1: the newer model, and three times the practice
            "newer": [dict(name="base_qwen3_5_9b", base_model="Qwen/Qwen3.5-9B", rounds=0)]
                     + [dict(name=f"qwen3_5_9b_s{seed}", base_model="Qwen/Qwen3.5-9B", lr=lr, seed=seed) for seed in (0, 1)],
            # second attempt: the first ran out of memory mid-training (this model's layers need more working memory)
            "newer2": [dict(name=f"qwen3_5_9b_s{seed}", base_model="Qwen/Qwen3.5-9B", lr=lr, seed=seed, micro=8, token_budget=70_000) for seed in (0, 1)],
            "longer": [dict(name=f"rounds180_s{seed}", lr=lr, seed=seed, rounds=180, check_every=20) for seed in (0, 1)]}


def _slim(it: dict) -> dict:
    return {"id": it["id"], "prompt": it["prompt"], "last": it["last"], "y": {str(h): v for h, v in it["y"].items()},
            "drift": {str(h): list(v) for h, v in it["drift"].items()}}


def payloads(cfgs: list[dict]) -> tuple[str, dict, list, list]:
    """Practice prices per run (the same draw as the Tinker run for the same seed), the watch set, the test prompts."""
    import forecast_env as env

    world = env.aws_world()
    train = {}
    for c in cfgs:
        c = {**DEFAULTS, **c}
        key = (c["seed"], c["rounds"], c["prompts"])
        if c["rounds"] and key not in train:
            rng = random.Random(c["seed"])
            train[key] = [[_slim(it) for it in env.sample_items(world, c["prompts"], "2022-08-05", env.TRAIN_END, rng)] for _ in range(c["rounds"])]
    check = env.sample_items(world, 64, env.CHECK_FROM, env.CHECK_TO, random.Random(123))  # the 64 the Tinker runs were watched on
    seen = {it["id"] for it in check}
    for it in env.sample_items(world, 400, env.CHECK_FROM, env.CHECK_TO, random.Random(124)):
        if it["id"] not in seen and len(check) < 256:
            check.append(it); seen.add(it["id"])
    return env.SYSTEM, train, [_slim(it) for it in check], test_items()


def test_items() -> list[dict]:
    import eval_forecaster as ev
    from gpuval import coach

    return [{"id": it["id"], "prompt": it["facts"] + "\n\nGive your forecast as JSON.", "last": it["last"]} for it in coach.build_items(ev.DAYS).values()]


def launch(plan: str, lr: float | None = None) -> None:
    cfgs = plans(lr)[plan]
    assert all(c.get("lr", 1) is not None for c in cfgs), "this plan needs --lr"
    system, train, check, test = payloads(cfgs)
    fn = modal.Function.from_name(APP, "run")
    pre = modal.Function.from_name(APP, "prefetch")
    for m in sorted({{**DEFAULTS, **c}["base_model"] for c in cfgs}):
        print("model files ready:", m, pre.remote(m), flush=True)
    calls = {}
    for c in cfgs:
        full = {**DEFAULTS, **c, "plan": plan}
        n_check, n_eval = full.pop("n_check", None), full.pop("n_eval", None)
        tr = train.get((full["seed"], full["rounds"], full["prompts"]), [])
        call = fn.spawn(full, system, tr, check[:n_check] if n_check else check, test[:n_eval] if n_eval else test)
        calls[full["name"]] = call.object_id
        print("started", full["name"], call.object_id, flush=True)
    (LOCAL / plan).mkdir(parents=True, exist_ok=True)
    (LOCAL / plan / "calls.json").write_text(json.dumps(calls, indent=1))


def launch_eval(plan: str, lr: float | None = None) -> None:
    """Re-run only the test step, from the adapters the training runs saved on the Modal volume."""
    cfgs = plans(lr)[plan]
    fn = modal.Function.from_name(APP, "evaluate")
    import forecast_env as env

    test = test_items()
    d = LOCAL / plan
    if (d / "calls.json").exists() and not (d / "calls_train.json").exists():
        (d / "calls_train.json").write_text((d / "calls.json").read_text())
    calls = {}
    for c in cfgs:
        full = {**DEFAULTS, **c, "plan": plan}
        n_eval = full.pop("n_eval", None); full.pop("n_check", None)
        calls[full["name"]] = fn.spawn(full, env.SYSTEM, test[:n_eval] if n_eval else test).object_id
        print("started test step for", full["name"], calls[full["name"]], flush=True)
    (d / "calls.json").write_text(json.dumps(calls, indent=1))


def collect(plan: str) -> None:
    import pandas as pd

    import eval_forecaster as ev
    import forecast_env as env
    from gpuval import coach, report

    d = LOCAL / plan
    calls = json.loads((d / "calls.json").read_text())
    results = {}
    for name, cid in calls.items():
        path = d / f"{name}.json"
        if not path.exists():
            try:
                path.write_text(json.dumps(modal.FunctionCall.from_id(cid).get(timeout=0)))
            except TimeoutError:
                print(f"{name}: still running")
                continue
            except Exception as e:
                print(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}")
                continue
        results[name] = json.loads(path.read_text())
    if not results:
        return
    print("\nWatch set (later AWS practice dates; mean reward, 0 = as good as drift)")
    for name, r in results.items():
        c = r["cfg"]
        print(f"  {name:16s} lr {c['lr']:<8g} seed {c['seed']} | " + " ".join(f"r{k['round']}:{k['mean_reward']:+.3f}" for k in r["checks"])
              + f" | {r['gpu']}, {r.get('total_minutes')} min, prefix sharing {r['prefix_sharing']}")
    items = {it["id"]: it for it in coach.build_items(ev.DAYS).values()}
    parsed = {name: {iid: [({str(h): list(q[h]) for h in env.HORIZONS} if q else None) for q in (env.parse(t, items[iid]["last"]) for t in texts)]
                     for iid, texts in r["eval"].items()} for name, r in results.items() if r.get("eval")}
    def parse_all(r: dict) -> dict:
        return {iid: [({str(h): list(q[h]) for h in env.HORIZONS} if q else None) for q in (env.parse(t, items[iid]["last"]) for t in texts)]
                for iid, texts in r["eval"].items()}

    bases = {}  # the untrained model's answers, from whichever plan ran it
    for path in sorted(LOCAL.glob("*/base_*.json")):
        r = json.loads(path.read_text())
        if r["cfg"]["rounds"] == 0 and r.get("eval") and len(r["eval"]) > 100:
            bases[r["cfg"]["base_model"]] = parse_all(r)
    f, summary = report.fmt_ci, {}
    for name, r in results.items():
        base = bases.get(r["cfg"]["base_model"])
        if r["cfg"]["rounds"] == 0 or name not in parsed or base is None:
            continue
        ids = [i for i in parsed[name] if i in base]
        df = ev.cells([items[i] for i in ids], {"base": base, "trained": parsed[name]})
        df.to_parquet(d / f"{name}_cells.parquet", index=False)
        t = ev.table(df)
        summary[name] = {"cfg": r["cfg"], "checks": r["checks"], "gpu": r["gpu"], "total_minutes": r.get("total_minutes"), "table": t}
        print(f"\n== {name} (lr {r['cfg']['lr']:g}, seed {r['cfg']['seed']}, final watch score {r['checks'][-1]['mean_reward']:+.3f})")
        for w in ("ours", "aws", "all"):
            b = t[w]
            print(f"  {w:5s} {b['n']:5d}: before {f(b['base_vs_drift'])} | after {f(b['trained_vs_drift'])} | after vs before {f(b['trained_vs_base'])} "
                  f"| vs engine {f(b['trained_vs_engine'])} | inside {b['inside']['base']:.2f} -> {b['inside']['trained']:.2f}")
    (d / "summary.json").write_text(json.dumps(summary, indent=1))


H100_USD_PER_HOUR = 3.95  # modal.com/pricing, read 14 Nov 2026 ($0.001097 per second)


def summarize() -> None:
    """One file with every Modal run next to the Tinker runs: findings/data/modal_rl.json."""
    import eval_forecaster as ev
    import forecast_env as env
    from gpuval import coach, data, report

    out = {"gpu_usd_per_hour": H100_USD_PER_HOUR, "runs": [], "tinker": {}}
    for tag, run_name in (("Qwen/Qwen3-8B", "fc1"), ("Qwen/Qwen3.5-9B", "fc2")):
        e, l = json.loads((HERE / "runs" / run_name / "eval.json").read_text()), json.loads((HERE / "runs" / run_name / "log.json").read_text())
        out["tinker"][tag] = {"train_usd": l["cost_estimate_usd"], "test_usd": e["cost_estimate_usd"], "train_minutes": l["minutes"],
                              "watch_first64": [c["mean_reward"] for c in l["checks"]],
                              **{w: {k: e["table"][w][k] for k in ("n", "base_vs_drift", "trained_vs_drift", "trained_vs_base", "trained_vs_engine", "engine_vs_drift", "inside")} for w in ("ours", "aws", "all")}}
    for plan in ("sweep1", "longer", "newer2"):
        d = LOCAL / plan
        if not (d / "summary.json").exists():
            continue
        for name, b in json.loads((d / "summary.json").read_text()).items():
            r = json.loads((d / f"{name}.json").read_text())
            minutes = (r["checks"][-1]["minutes"] if r["checks"] else 0) + (r.get("eval_minutes") or 0) if r.get("eval_done_separately") else r.get("total_minutes")
            c = b["cfg"]
            out["runs"].append({"plan": plan, "name": name, "base_model": c["base_model"], "lr": c["lr"], "seed": c["seed"], "rounds": c["rounds"],
                                "watch": {str(k["round"]): k["mean_reward"] for k in b["checks"]}, "collapsed": sum("grad_norm" not in x for x in r["rounds"]) > len(r["rounds"]) / 4,
                                "gpu": b["gpu"], "gpu_minutes": minutes, "usd": round(minutes * H100_USD_PER_HOUR / 60, 2),
                                **{w: {k: b["table"][w][k] for k in ("n", "base_vs_drift", "trained_vs_drift", "trained_vs_base", "trained_vs_engine", "engine_vs_drift", "inside")} for w in ("ours", "aws", "all")}})
    sw = [r for r in out["runs"] if r["plan"] == "sweep1"]
    by_lr = {}
    for r in sw:
        by_lr.setdefault(r["lr"], []).append((r["watch"]["50"] + r["watch"]["60"]) / 2)
    out["watch_rule"] = {f"{lr:g}": statistics.fmean(v) for lr, v in by_lr.items()}
    out["chosen_lr"] = max(by_lr, key=lambda lr: statistics.fmean(by_lr[lr]))
    out["total_usd"] = round(sum(r["usd"] for r in out["runs"]), 2)
    (data.REPO / "findings" / "data" / "modal_rl.json").write_text(json.dumps(out, indent=1))
    f = report.fmt_ci
    print(f"chosen learning rate (watch set): {out['chosen_lr']:g}  {out['watch_rule']}")
    for tag, t in out["tinker"].items():
        print(f"Tinker {tag:16s} ours {f(t['ours']['base_vs_drift'])} -> {f(t['ours']['trained_vs_drift'])} | aws {f(t['aws']['base_vs_drift'])} -> {f(t['aws']['trained_vs_drift'])} | ${t['train_usd'] + t['test_usd']:.2f}")
    for r in out["runs"]:
        print(f"Modal  {r['name']:16s} {r['base_model'].split('/')[1]:11s} lr {r['lr']:<7g} s{r['seed']} r{r['rounds']:<3d} | ours {f(r['ours']['trained_vs_drift'])} (vs before {f(r['ours']['trained_vs_base'])}) "
              f"| aws {f(r['aws']['trained_vs_drift'])} | inside {r['ours']['inside']['trained']:.2f} | {r['gpu_minutes']:.0f} min ${r['usd']:.2f}{' | COLLAPSED' if r['collapsed'] else ''}")
    print("Modal total: $", out["total_usd"])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["launch", "launch-eval", "collect", "summarize"])
    ap.add_argument("--plan")
    ap.add_argument("--lr", type=float)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    {"launch": lambda: launch(a.plan, a.lr), "launch-eval": lambda: launch_eval(a.plan, a.lr), "collect": lambda: collect(a.plan), "summarize": summarize}[a.cmd]()
