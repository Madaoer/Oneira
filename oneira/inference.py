"""Generate a video for each sample folder given to --input.

A sample is a single segment (first_frame.png, cond.mp4, prompt.json in the folder) or a chain of
segments (segment_1/, segment_2/, ...). In a chain, only segment_1 has first_frame.png; every later
segment starts from the last frame of the previous segment's saved video.

    python -m oneira.inference --config configs/inference.yaml --input examples/multi_instance --output results
"""

import argparse
import gc
import os

import numpy as np
import torch
from PIL import Image

from . import attention, hub
from .config import parse
from .data import load_segment, read_video, segment_dirs, write_video
from .pipeline import encode_sample, generate, load_pipeline

DEFAULTS = {
    "base_model": hub.BASE_MODEL,
    "checkpoint": hub.CHECKPOINT,
    "num_steps": 20,
    "seed": 128,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}


def release_memory():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", nargs="+", required=True, help="one or more sample folders")
    ap.add_argument("--output", required=True)
    args = parse(ap, DEFAULTS)

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    folders = {os.path.basename(os.path.normpath(p)): p for p in args.input}
    names = sorted(folders)[rank::world]
    if not names:
        return
    base, lora = hub.resolve(args.base_model), hub.resolve(args.checkpoint)
    attention.install()

    segments = {n: segment_dirs(folders[n]) for n in names}
    videos = {n: [] for n in names}
    # one round per segment index: encode with the text encoder, then denoise with the DiT
    for k in range(max(len(s) for s in segments.values())):
        todo = [n for n in names if k < len(segments[n])]
        pipe = load_pipeline(base, text_encoder=True, dit=False, device=device)
        encoded, cache = {}, {}
        for n in todo:
            first = Image.fromarray(read_video(videos[n][-1], size=None)[-1]) if k else None
            s = load_segment(segments[n][k], first)
            encoded[n] = encode_sample(
                pipe, s["first_frame"], s["cond"], s["global_caption"], s["captions"], s["memory"], cache
            )
            print(f"[encode] {n} segment {k + 1}", flush=True)
        del pipe
        release_memory()

        pipe = load_pipeline(base, lora=lora, device=device)
        for n in todo:
            frames = generate(
                pipe,
                encoded[n],
                seed=args.seed,
                num_steps=args.num_steps,
                flow_shift=args.flow_shift,
                audio_flow_shift=args.audio_flow_shift,
            )
            name = f"segment_{k + 1}.mp4" if len(segments[n]) > 1 else "generated.mp4"
            videos[n].append(write_video(frames, os.path.join(args.output, n, name)))
            print(f"[generate] {n} segment {k + 1}", flush=True)
        del pipe
        release_memory()

    for n in names:
        if len(videos[n]) > 1:
            frames = np.concatenate([read_video(v, size=None) for v in videos[n]])
            write_video(frames, os.path.join(args.output, n, "generated.mp4"))
        print(f"[done] {n} -> {os.path.join(args.output, n, 'generated.mp4')}", flush=True)


if __name__ == "__main__":
    main()
