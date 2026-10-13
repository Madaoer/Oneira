"""MiniMax-H3 Ref2VA with chunk-wise causal captions and a pinned first frame: load, encode inputs, generate."""

import glob
import inspect

import numpy as np
import torch

import diffsynth.pipelines.minimax_h3_audio_video as h3

from . import attention

NUM_FRAMES, FPS, HEIGHT, WIDTH = 243, 24, 480, 832
N_LAT, N_CHUNK, CHUNK_LAT = 72, 8, 9
LAT_H, LAT_W = 30, 52
FRAME_ROWS = (LAT_H // 2) * (LAT_W // 2)
AUDIO_T = round(NUM_FRAMES / FPS * 40)
REF_IMAGE_SHORT_EDGE = 480
REF_VIDEO_SHORT_EDGE, REF_VIDEO_MAX_PIXELS = 256, 448 * 256
ENCODE_SEED = 42


def _build_model_fn():
    """model_fn_minimax_h3 with an extra `_pin_rows`: the first rows keep timestep max(t, cond_aug)."""
    src = inspect.getsource(h3.model_fn_minimax_h3)
    src = src.replace("def model_fn_minimax_h3(", "def model_fn_pinned(", 1)
    a = "    dtype, device = video_latents.dtype, video_latents.device\n"
    b = "    timesteps[audio_pos[:ref_audio_rows_count]] = max(float(t_audio), audio_cond_noise_aug)\n"
    assert src.count(a) == 1 and src.count(b) == 1, "unsupported DiffSynth-Studio version"
    src = src.replace(a, a + "    _pin_rows = int(kwargs.get('_pin_rows', 0) or 0)\n")
    src = src.replace(
        b,
        b
        + (
            "    if _pin_rows:\n"
            "        timesteps[img_pos[cond_rows_count:cond_rows_count + _pin_rows]] = "
            "max(float(t_video), float(imgvid_cond_noise_aug))\n"
        ),
    )
    ns = dict(vars(h3))
    exec(compile(src, "<oneira_model_fn>", "exec"), ns)
    return ns["model_fn_pinned"]


model_fn_pinned = _build_model_fn()


def load_pipeline(base_dir, text_encoder=False, dit=True, lora=None, device="cuda"):
    configs = []
    if text_encoder:
        configs.append(h3.ModelConfig(path=sorted(glob.glob(f"{base_dir}/text_encoder/model-*.safetensors"))))
    if dit:
        configs.append(h3.ModelConfig(path=sorted(glob.glob(f"{base_dir}/transformer/model-*.safetensors"))))
    configs += [
        h3.ModelConfig(path=[f"{base_dir}/video_vae/source/model.safetensors"]),
        h3.ModelConfig(path=[f"{base_dir}/audio_vae/model.safetensors"]),
    ]
    pipe = h3.MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=configs,
        processor_config=h3.ModelConfig(path=f"{base_dir}/processor/"),
    )
    if lora:
        pipe.load_lora(pipe.dit, lora, alpha=1)
    return pipe


def run_units(pipe, prompt, references=None, input_video=None, seed=ENCODE_SEED):
    shared = {
        "cfg_scale": 1,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "seed": seed,
        "rand_device": "cpu",
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
        "keyframes": None,
        "keyframe_indices": None,
        "references": references,
        "input_video": input_video,
        "input_audio": None,
        "ref_image_short_edge": REF_IMAGE_SHORT_EDGE,
        "ref_video_short_edge": REF_VIDEO_SHORT_EDGE,
        "ref_video_max_pixels": REF_VIDEO_MAX_PIXELS,
        "imgvid_cond_noise_aug": pipe.imgvid_cond_noise_aug,
        "audio_cond_noise_aug": pipe.audio_cond_noise_aug,
    }
    posi, nega = {"prompt": prompt}, {"negative_prompt": " "}
    for unit in pipe.units:
        shared, posi, nega = pipe.unit_runner(unit, pipe, shared, posi, nega)
    return shared, posi


def encode_text(pipe, prompt, **kw):
    shared, posi = run_units(pipe, prompt, **kw)
    n = int(posi["packed"]["text_pos"].shape[0])
    emb = posi["prompt_embeds"].to("cpu", torch.bfloat16)
    tags = posi["packed"]["token_tags"][:n].to("cpu")
    return emb, tags, shared


def block_dims(shared):
    return [{k: b[k] for k in ("kind", "latent_t", "latent_h", "latent_w")} for b in shared["ref_blocks"]]


def references(first_frame, cond_frames, memory_frames=()):
    """Reference order: first frame, memory frames, condition video."""
    refs = [{"type": "image", "image": first_frame}]
    refs += [{"type": "image", "image": m} for m in memory_frames]
    refs.append({"type": "video", "video": cond_frames})
    return refs


def encode_chunks(pipe, captions, cache=None):
    cache = {} if cache is None else cache
    out = []
    for c in captions:
        if c not in cache:
            emb, tags, _ = encode_text(pipe, c)
            cache[c] = (emb, tags)
        out.append(cache[c])
    return out


@torch.no_grad()
def encode_first_frame(pipe, image):
    from PIL import Image

    img = pipe.preprocess_image(
        image.convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS), torch_dtype=torch.float32, min_value=0
    )
    pipe.load_models_to_device(["video_vae"])
    return pipe.video_vae.encode_video(img, dtype=pipe.torch_dtype, process_image=True)


@torch.no_grad()
def encode_sample(pipe, first_frame, cond_frames, global_caption, captions, memory_frames=(), cache=None):
    """Everything the DiT needs for one segment, computed without the DiT."""
    refs = references(first_frame, cond_frames, memory_frames)
    g_emb, g_tags, shared = encode_text(pipe, global_caption, references=refs)
    return {
        "global_emb": g_emb,
        "global_tags": g_tags,
        "segs": encode_chunks(pipe, captions, cache),
        "ref_block_dims": block_dims(shared),
        "ref_visual_anchor": shared["ref_visual_anchor"].to("cpu", torch.bfloat16),
        "ff_lat": encode_first_frame(pipe, first_frame).to("cpu", torch.bfloat16),
        "imgvid_cond_noise_aug": float(pipe.imgvid_cond_noise_aug),
    }


def ref_rows(dims):
    return sum(b["latent_t"] * (b["latent_h"] // 2) * (b["latent_w"] // 2) for b in dims)


def pack_text(global_emb, global_tags, segs, dims):
    """Global caption followed by the chunk captions, plus packed positions and the causal groups."""
    embs, tags, spans = [global_emb], [global_tags], []
    off = int(global_emb.shape[0])
    for e, t in segs:
        embs.append(e)
        tags.append(t)
        spans.append((off, off + int(e.shape[0])))
        off += int(e.shape[0])
    prompt_embeds, text_tags = torch.cat(embs), torch.cat(tags)
    packed = h3.MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_ref2va(
        off, N_LAT, LAT_H, LAT_W, AUDIO_T, [dict(b) for b in dims]
    )
    packed["token_tags"][packed["text_pos"]] = text_tags
    total, groups = attention.build_groups(packed, spans, ref_rows(dims), N_LAT, len(segs), CHUNK_LAT, FRAME_ROWS)
    return prompt_embeds, packed, (total, groups)


@torch.no_grad()
def generate(pipe, enc, seed=42, num_steps=20, flow_shift=12.0, audio_flow_shift=3.0):
    """Denoise one encoded sample (see encoding.encode_sample). Returns uint8 frames [T, H, W, 3]."""
    device = pipe.device
    prompt_embeds, packed, (total, groups) = pack_text(
        enc["global_emb"], enc["global_tags"], enc["segs"], enc["ref_block_dims"]
    )
    attention.set_groups(total, [(q.to(device), k.to(device)) for q, k in groups])
    packed = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in packed.items()}

    # frame 0 is pinned to the first frame for the whole trajectory
    g = torch.Generator("cpu").manual_seed(seed * 100 + 7)
    aug = enc["imgvid_cond_noise_aug"]
    eps = torch.randn(enc["ff_lat"].shape, generator=g).to(device, torch.bfloat16)
    pinned = aug * enc["ff_lat"].to(device, torch.bfloat16) + (1.0 - aug) * eps

    g = torch.Generator("cpu").manual_seed(seed)
    video = torch.randn((1, 24, N_LAT, LAT_H, LAT_W), generator=g).to(device, torch.bfloat16)
    audio = torch.randn((2, 32, AUDIO_T), generator=g).to(device, torch.bfloat16)
    video[:, :, 0:1] = pinned
    shared = {
        "video_latents": video,
        "audio_latents": audio,
        "ref_visual_anchor": enc["ref_visual_anchor"].to(device, torch.bfloat16),
        "ref_audio_anchor": None,
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
    }
    posi = {"prompt_embeds": prompt_embeds.to(device, torch.bfloat16), "packed": packed}

    pipe.scheduler.set_timesteps(num_steps, shift=flow_shift)
    pipe.scheduler_audio.set_timesteps(num_steps, shift=audio_flow_shift)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {n: getattr(pipe, n) for n in pipe.in_iteration_models}
    for i in range(len(pipe.scheduler.timesteps)):
        v_pred, a_pred = model_fn_pinned(
            **models,
            **shared,
            **posi,
            t_video=float(1.0 - pipe.scheduler.sigmas[i]),
            t_audio=float(1.0 - pipe.scheduler_audio.sigmas[i]),
            device=device,
            _pin_rows=FRAME_ROWS,
        )
        shared["video_latents"] = pipe.step(pipe.scheduler, shared["video_latents"], i, noise_pred=v_pred)
        shared["audio_latents"] = pipe.step(pipe.scheduler_audio, shared["audio_latents"], i, noise_pred=a_pred)
        shared["video_latents"][:, :, 0:1] = pinned
    attention.clear()

    pipe.load_models_to_device(["video_vae"])
    frames = pipe.video_vae.decode_video(
        shared["video_latents"], dtype=pipe.torch_dtype, tiled=True, tile_size=256, tile_overlap=64
    )
    return np.stack([np.asarray(f) for f in pipe.vae_output_to_video(frames, min_value=0, max_value=1)])
