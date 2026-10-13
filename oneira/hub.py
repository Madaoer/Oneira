"""Resolve `hf://<org>/<repo>/<path>` to a local path, downloading on first use."""

import os

BASE_MODEL = "hf://MiniMaxAI/MiniMax-H3/Ref2VA"
CHECKPOINT = "hf://Madaoer-Otaku/Oneira_example_weights/model.safetensors"


def resolve(uri):
    if not uri.startswith("hf://"):
        return uri
    parts = uri[len("hf://") :].split("/")
    repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
    from huggingface_hub import hf_hub_download, snapshot_download

    if os.path.splitext(sub)[1]:
        return hf_hub_download(repo, sub)
    root = snapshot_download(repo, allow_patterns=[f"{sub}/*"] if sub else None)
    return os.path.join(root, sub)
