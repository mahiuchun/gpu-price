"""Modal app: every research step runs in its own container.

The container is sent one thing as data: a world cut at the step's wall date. It has no volume,
no copy of the pack and no later data, so the wall is physical, not a promise in code. It is also
how the real world and the placebo worlds run side by side.
"""
from __future__ import annotations

import os

import modal

app = modal.App("gpuval-timemachine")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=1.24", "pandas>=2.0", "pyarrow>=15", "openai>=1.50", "python-dotenv>=1.0")
    .add_local_python_source("gpuval")
)

# only the one key the scientist needs is sent
_secret = modal.Secret.from_dict({"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")})


@app.function(image=image, secrets=[_secret], timeout=3600, cpu=2.0, memory=4096, max_containers=40)
def research_step(blob: bytes, state: dict, kind: str, n_ideas: int, seed: int) -> dict:
    from gpuval.timemachine import research_local

    return research_local(blob, state, kind, n_ideas, seed)
