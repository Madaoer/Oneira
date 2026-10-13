"""LoRA fine-tuning of MiniMax-H3 Ref2VA on encoded clips.

torchrun --nnodes=8 --nproc_per_node=8 -m oneira.train --config configs/train.yaml --lmdb lmdb/ --output runs/oneira
"""

import argparse
import glob
import json
import os
import time

import torch
import torch.distributed as dist
from diffsynth.diffusion.loss import FlowMatchSFTMiniMaxH3AudioVideoLoss
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

from . import attention, hub
from .config import parse
from .data import LatentDataset
from .pipeline import FRAME_ROWS, model_fn_pinned

DEFAULTS = {
    "base_model": hub.BASE_MODEL,
    "lora_rank": 128,
    "lora_targets": "qkv_proj,out_proj",
    "i2v_rate": 0.8,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "epochs": 15,
    "seed": 42,
    "shard_size": 8,
    "max_steps": 0,
}

attention.install()


def i2v_loss(pipe, i2v_rate=0.8, **inputs):
    """With probability i2v_rate, pin frame 0 to the clean first frame and drop it from the video loss."""
    if float(torch.rand(())) >= i2v_rate:
        return FlowMatchSFTMiniMaxH3AudioVideoLoss(pipe, **inputs)

    n = len(pipe.scheduler.timesteps)
    tid = torch.randint(
        int(inputs.get("min_timestep_boundary", 0) * n), int(inputs.get("max_timestep_boundary", 1) * n), (1,)
    )
    t_video = pipe.scheduler.timesteps[tid].to(dtype=pipe.torch_dtype, device=pipe.device)
    t_audio = pipe.scheduler_audio.timesteps[tid].to(dtype=pipe.torch_dtype, device=pipe.device)

    clean = inputs["input_latents"]
    noise = torch.randn_like(clean)
    video = pipe.scheduler.add_noise(clean, noise, t_video)
    target = pipe.scheduler.training_target(clean, noise, t_video)
    aug = float(inputs.get("imgvid_cond_noise_aug", 0.999))
    video[:, :, 0:1] = aug * clean[:, :, 0:1] + (1.0 - aug) * torch.randn_like(clean[:, :, 0:1])
    inputs["video_latents"] = video

    a_clean = inputs["audio_input_latents"]
    a_noise = torch.randn_like(a_clean)
    inputs["audio_latents"] = pipe.scheduler_audio.add_noise(a_clean, a_noise, t_audio)
    a_target = pipe.scheduler_audio.training_target(a_clean, a_noise, t_audio)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    v_pred, a_pred = model_fn_pinned(
        **models,
        **inputs,
        t_video=1.0 - float(t_video) / pipe.scheduler.num_train_timesteps,
        t_audio=1.0 - float(t_audio) / pipe.scheduler_audio.num_train_timesteps,
        device=pipe.device,
        _pin_rows=FRAME_ROWS,
    )
    loss = torch.nn.functional.mse_loss(v_pred[:, :, 1:].float(), target[:, :, 1:].float())
    loss = loss * pipe.scheduler.training_weight(t_video)
    loss_audio = torch.nn.functional.mse_loss(a_pred.float(), a_target.float())
    return loss + loss_audio * pipe.scheduler_audio.training_weight(t_audio)


class LoRAModule(DiffusionTrainingModule):
    def __init__(self, base_dir, lora_rank, lora_targets, i2v_rate, device):
        super().__init__()
        self.pipe = MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[ModelConfig(path=sorted(glob.glob(f"{base_dir}/transformer/model-*.safetensors")))],
            processor_config=ModelConfig(path=f"{base_dir}/processor/"),
        )
        self.pipe = self.split_pipeline_units("sft:train", self.pipe, lora_base_model="dit")
        self.switch_pipe_to_training_mode(
            self.pipe, lora_base_model="dit", lora_target_modules=lora_targets, lora_rank=lora_rank, task="sft:train"
        )
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=12.0)
        self.pipe.scheduler_audio.set_timesteps(1000, training=True, shift=3.0)
        self.i2v_rate = i2v_rate

    def forward(self, shared, posi):
        inputs = self.transfer_data_to_device((shared, posi, {}), self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return i2v_loss(self.pipe, self.i2v_rate, **inputs[0], **inputs[1])


def save_lora(module, path, rank):
    from safetensors.torch import save_file
    from torch.distributed.tensor import DTensor

    state = {}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        t = p.data.full_tensor() if isinstance(p.data, DTensor) else p.data
        if rank == 0:
            state[name.removeprefix("pipe.dit.")] = t.detach().to("cpu", torch.bfloat16).contiguous()
    if rank == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(state, path)
        print(f"saved {path}", flush=True)
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--output", required=True)
    args = parse(ap, DEFAULTS)

    rank, local, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed * 1000 + rank)

    # HSDP: shard within a node, replicate across nodes
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    shard = min(args.shard_size, world)
    mesh = init_device_mesh("cuda", (world // shard, shard), mesh_dim_names=("replicate", "shard"))

    if local == 0:
        hub.resolve(args.base_model)
    dist.barrier()
    base = hub.resolve(args.base_model)
    module = LoRAModule(base, args.lora_rank, args.lora_targets, args.i2v_rate, str(device))
    dit = module.pipe.dit
    for blk in dit.blocks:
        fully_shard(blk, mesh=mesh)
    for blk in getattr(getattr(dit, "token_refiner", None), "blocks", []):
        fully_shard(blk, mesh=mesh)
    fully_shard(dit, mesh=mesh)
    torch.cuda.empty_cache()

    dataset = LatentDataset(args.lmdb)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=0)
    trainable = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    if rank == 0:
        os.makedirs(args.output, exist_ok=True)
        json.dump({**vars(args), "world": world}, open(f"{args.output}/config.json", "w"), indent=2)
        print(f"{len(dataset)} clips, {len(loader)} steps per epoch per rank", flush=True)

    step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for shared, posi, (total, groups) in loader:
            tic = time.time()
            attention.set_groups(total, [(q.to(device), k.to(device)) for q, k in groups])
            loss = module(shared, posi)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, float("inf"))
            gnorm = gnorm.full_tensor() if isinstance(gnorm, DTensor) else gnorm
            optimizer.step()
            optimizer.zero_grad()
            attention.clear()
            step += 1
            mean = loss.detach().float().clone()
            dist.all_reduce(mean)
            if rank == 0 and step % 10 == 1:
                print(
                    f"epoch {epoch} step {step} loss {float(mean) / world:.4f} "
                    f"grad {float(gnorm):.3f} {time.time() - tic:.1f}s",
                    flush=True,
                )
            if args.max_steps and step >= args.max_steps:
                save_lora(module, f"{args.output}/step-{step}.safetensors", rank)
                dist.destroy_process_group()
                return
        save_lora(module, f"{args.output}/epoch-{epoch}.safetensors", rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
