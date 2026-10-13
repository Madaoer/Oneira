"""Sample folders, video IO and the LMDB latent cache."""

import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from .pipeline import HEIGHT, NUM_FRAMES, WIDTH

TENSOR_COLS = ["vae_latent", "audio_latent", "global_emb", "global_tags", "ref_visual_anchor"]
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int64": torch.int64, "int32": torch.int32}


def read_video(path, num_frames=NUM_FRAMES, size=(WIDTH, HEIGHT)):
    """uint8 [T, H, W, 3]; pads with the last frame if the video is short."""
    import av

    frames = []
    with av.open(path) as c:
        for f in c.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) >= num_frames:
                break
    if not frames:
        raise ValueError(f"no frames in {path}")
    frames += [frames[-1]] * (num_frames - len(frames))
    arr = np.stack(frames)
    if size is not None and arr.shape[1:3] != (size[1], size[0]):
        arr = np.stack([np.asarray(Image.fromarray(f).resize(size, Image.LANCZOS)) for f in arr])
    return arr


def write_video(frames, path, fps=24, crf=18):
    import av

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with av.open(path, "w", options={"movflags": "+faststart"}) as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        s.options = {"crf": str(crf)}
        for f in frames:
            c.mux(s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")))
        c.mux(s.encode())
    return path


def load_prompt(path):
    p = json.load(open(path))
    chunks = sorted(p["chunks"], key=lambda c: c["k"])
    return p["global_caption"], [c["caption"] for c in chunks]


def memory_frames(folder):
    return [Image.open(p).convert("RGB") for p in sorted(glob.glob(f"{folder}/memory_*.png"))]


def segment_dirs(folder):
    """A sample is one segment (files in the folder) or several (segment_1/, segment_2/, ...)."""
    segs = sorted(glob.glob(f"{folder}/segment_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
    return segs or [folder]


def load_segment(folder, first_frame=None):
    """first_frame.png (or the given image), cond.mp4, prompt.json, optional memory_*.png."""
    global_caption, captions = load_prompt(f"{folder}/prompt.json")
    if first_frame is None:
        first_frame = Image.open(f"{folder}/first_frame.png").convert("RGB")
    return {
        "first_frame": first_frame,
        "cond": [Image.fromarray(f) for f in read_video(f"{folder}/cond.mp4")],
        "global_caption": global_caption,
        "captions": captions,
        "memory": memory_frames(folder),
    }


def bytes_to_tensor(buf, meta):
    return torch.frombuffer(bytearray(buf), dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"])


def lmdb_shards(root):
    """A single LMDB or a folder of LMDB shards."""
    if os.path.isfile(f"{root}/data.mdb"):
        return [root]
    return sorted(os.path.dirname(p) for p in glob.glob(f"{root}/*/data.mdb"))


class LatentDataset(torch.utils.data.Dataset):
    """Reads encoded clips from LMDB and builds the DiT inputs."""

    def __init__(self, root):
        import lmdb

        self.dirs = lmdb_shards(root)
        assert self.dirs, f"no LMDB under {root}"
        self.items = []
        for k, d in enumerate(self.dirs):
            env = lmdb.open(d, readonly=True, lock=False, readahead=False)
            with env.begin() as txn:
                self.items += [(k, i) for i in json.loads(txn.get(b"__ids"))]
            env.close()
        self.envs = [None] * len(self.dirs)

    def __len__(self):
        return len(self.items)

    def fetch(self, k, clip_id):
        import lmdb

        if self.envs[k] is None:
            self.envs[k] = lmdb.open(self.dirs[k], readonly=True, lock=False, readahead=False)
        with self.envs[k].begin() as txn:
            meta = json.loads(txn.get(f"{clip_id}/__row_meta".encode()))
            names = TENSOR_COLS + [f"seg{j}_{x}" for j in range(meta["n_chunks"]) for x in ("emb", "tags")]
            out = {}
            for n in names:
                m = json.loads(txn.get(f"{clip_id}/{n}__meta".encode()))
                out[n] = bytes_to_tensor(txn.get(f"{clip_id}/{n}".encode()), m)
        return out, meta

    def __getitem__(self, i):
        from .pipeline import pack_text

        k, clip_id = self.items[i]
        d, meta = self.fetch(k, clip_id)
        segs = [(d[f"seg{j}_emb"], d[f"seg{j}_tags"]) for j in range(meta["n_chunks"])]
        prompt_embeds, packed, mask = pack_text(d["global_emb"], d["global_tags"], segs, meta["ref_block_dims"])
        shared = {
            "imgvid_cond_noise_aug": float(meta["imgvid_cond_noise_aug"]),
            "audio_cond_noise_aug": 1.0,
            "use_gradient_checkpointing": True,
            "use_gradient_checkpointing_offload": False,
            "input_latents": d["vae_latent"].unsqueeze(0),
            "audio_input_latents": d["audio_latent"],
            "ref_visual_anchor": d["ref_visual_anchor"],
            "ref_audio_anchor": None,
        }
        return shared, {"prompt_embeds": prompt_embeds, "packed": packed}, mask
