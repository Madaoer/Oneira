"""Encode the dataset (videos, condition videos, captions) into LMDB shards, one per rank.

torchrun --nproc_per_node=8 -m oneira.data_process --data-root Oneira_example_data --output lmdb/
"""

import argparse
import json
import os
import zlib

import torch
from PIL import Image

from . import hub
from .data import load_prompt, memory_frames, read_video
from .pipeline import FPS, NUM_FRAMES, block_dims, encode_chunks, encode_text, load_pipeline, references


def tensor_to_bytes(t):
    t = t.contiguous().cpu()
    dtype = str(t.dtype).replace("torch.", "")
    raw = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
    return raw, {"shape": list(t.shape), "dtype": dtype}


class LMDBWriter:
    def __init__(self, path, map_size_gb=1024):
        import lmdb

        os.makedirs(path, exist_ok=True)
        self.env = lmdb.open(path, map_size=map_size_gb << 30)
        with self.env.begin() as txn:
            ids = txn.get(b"__ids")
        self.ids = json.loads(ids) if ids else []

    def put(self, clip_id, tensors, meta):
        with self.env.begin(write=True) as txn:
            for name, t in tensors.items():
                raw, m = tensor_to_bytes(t)
                txn.put(f"{clip_id}/{name}".encode(), raw)
                txn.put(f"{clip_id}/{name}__meta".encode(), json.dumps(m).encode())
            txn.put(f"{clip_id}/__row_meta".encode(), json.dumps(meta).encode())
            if clip_id not in self.ids:
                self.ids.append(clip_id)
            txn.put(b"__ids", json.dumps(self.ids).encode())

    def close(self):
        self.env.sync()
        self.env.close()


class Clips(torch.utils.data.Dataset):
    def __init__(self, root, ids):
        self.root, self.ids = root, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = os.path.join(self.root, self.ids[i])
        try:
            return {"id": self.ids[i], "video": read_video(f"{d}/video.mp4"), "cond": read_video(f"{d}/cond.mp4")}
        except Exception as e:  # noqa: BLE001
            return {"id": self.ids[i], "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-model", default=hub.BASE_MODEL)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    ids = sorted(
        f"{s}/{c}"
        for s in ("navigation", "interaction")
        if os.path.isdir(f"{args.data_root}/{s}")
        for c in os.listdir(f"{args.data_root}/{s}")
    )
    writer = LMDBWriter(os.path.join(args.output, f"shard_{rank:03d}"))
    todo = [c for c in ids[rank::world] if c not in set(writer.ids)]
    print(f"[rank {rank}] {len(todo)} clips to encode", flush=True)
    if not todo:
        return

    pipe = load_pipeline(hub.resolve(args.base_model), text_encoder=True, dit=False, device=device)
    pipe.scheduler.training = True  # makes the pipeline encode input_video into input_latents

    # silent audio track, the same for every clip
    sr = int(getattr(pipe.audio_vae, "sample_rate", 32000))
    wav = torch.zeros(2, int(round(NUM_FRAMES / FPS * sr)))
    audio = pipe.audio_vae.encode_audio(wav.to(pipe.device), dtype=pipe.torch_dtype).to("cpu", torch.bfloat16)
    audio = audio.squeeze(0) if audio.dim() == 4 else audio

    loader = torch.utils.data.DataLoader(Clips(args.data_root, todo), batch_size=None, num_workers=args.num_workers)
    cache = {}
    for n, item in enumerate(loader):
        clip_id = item["id"]
        if "error" in item:
            print(f"[rank {rank}] skip {clip_id}: {item['error']}", flush=True)
            continue
        folder = os.path.join(args.data_root, clip_id)
        global_caption, captions = load_prompt(f"{folder}/prompt.json")
        video = [Image.fromarray(f) for f in item["video"].numpy()]
        cond = [Image.fromarray(f) for f in item["cond"].numpy()]
        refs = references(video[0], cond, memory_frames(folder))
        seed = zlib.crc32(os.path.basename(clip_id).encode()) % 10000
        g_emb, g_tags, shared = encode_text(pipe, global_caption, references=refs, input_video=video, seed=seed)
        vae = shared["input_latents"].to("cpu", torch.bfloat16)
        tensors = {
            "vae_latent": vae.squeeze(0) if vae.dim() == 5 else vae,
            "audio_latent": audio,
            "global_emb": g_emb,
            "global_tags": g_tags,
            "ref_visual_anchor": shared["ref_visual_anchor"].to("cpu", torch.bfloat16),
        }
        for j, (emb, tags) in enumerate(encode_chunks(pipe, captions, cache)):
            tensors[f"seg{j}_emb"], tensors[f"seg{j}_tags"] = emb, tags
        writer.put(
            clip_id,
            tensors,
            {
                "n_chunks": len(captions),
                "ref_block_dims": block_dims(shared),
                "imgvid_cond_noise_aug": float(pipe.imgvid_cond_noise_aug),
                "global_caption": global_caption,
                "captions": captions,
            },
        )
        print(f"[rank {rank}] {n + 1}/{len(todo)} {clip_id}", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
