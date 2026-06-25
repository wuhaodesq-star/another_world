#!/usr/bin/env python
"""Multi-process DDP smoke test.

Wraps the toy transformer in :class:`torch.nn.parallel.DistributedDataParallel`,
runs a handful of steps, and prints the per-rank loss after an all-reduce so
we can confirm gradients are actually synchronised.

Two launch modes are supported:

1. **Self-spawn (default, works everywhere including Windows CPU)**::

       python scripts/ddp_smoke.py --nproc 2 --device cpu --steps 8

   This script uses :mod:`torch.multiprocessing` to spawn ``--nproc`` workers
   and sets up the env vars they need for ``init_process_group``.

2. **torchrun (preferred on Linux clusters)**::

       torchrun --standalone --nproc_per_node=8 \
           scripts/ddp_smoke.py --device cuda --precision bf16 --steps 50

   When ``RANK`` / ``WORLD_SIZE`` are present in the environment we honour
   them and skip the self-spawn step.

This is *not* the production trainer (that's TorchTitan / FSDP2 in stage 3);
it's a Gate-0 check that our environment, NCCL/gloo, DDP wrapping, and CLI
plumbing all work end-to-end.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from another_world.data.datasets import DummyTokenDataset
from another_world.models.dynamics import ToyTransformerConfig, build_toy_transformer
from another_world.utils.distributed import (
    all_reduce_sum,
    init_distributed,
    shutdown_distributed,
)
from another_world.utils.logging import get_logger

_LOG = get_logger("ddp_smoke")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("ddp_smoke")
    p.add_argument("--nproc", type=int, default=2,
                   help="processes for self-spawn mode (ignored when launched "
                        "via torchrun)")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--precision", type=str, default="fp32",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--master-port", type=int, default=29500)
    return p


def _resolve_device(spec: str, local_rank: int) -> torch.device:
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    if spec == "cuda":
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _worker_main(args: argparse.Namespace) -> None:
    """Per-process entry point. Reads RANK / WORLD_SIZE from env vars."""

    torch.manual_seed(args.seed)
    info = init_distributed()
    device = _resolve_device(args.device, info.local_rank)

    if info.is_main:
        _LOG.info(
            "[main] world_size=%d backend=%s device=%s",
            info.world_size, info.backend, device,
        )

    model_cfg = ToyTransformerConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=max(64, args.seq_len),
    )
    model = build_toy_transformer(model_cfg).to(device)

    ddp_model: torch.nn.Module
    if info.world_size > 1:
        device_ids = [info.local_rank] if device.type == "cuda" else None
        ddp_model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0] if device_ids else None,
        )
    else:
        ddp_model = model

    dataset = DummyTokenDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        length=max(args.batch_size * args.steps * max(info.world_size, 1), 16),
        seed=args.seed,
    )
    sampler: DistributedSampler | None = None
    if info.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=info.world_size,
            rank=info.rank,
            shuffle=True,
            seed=args.seed,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        num_workers=0,
    )

    optim = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)

    losses_first: list[float] = []
    losses_last: list[float] = []
    data_iter = iter(loader)
    t0 = time.perf_counter()
    for step in range(args.steps):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step)
            data_iter = iter(loader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device)
        targets = targets.to(device)
        out = ddp_model(inputs, targets=targets)
        loss = out["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        local_loss = float(loss.detach().cpu())
        if step < 3:
            losses_first.append(local_loss)
        if step >= args.steps - 3:
            losses_last.append(local_loss)
        if info.is_main and (
            step % max(1, args.steps // 5) == 0 or step == args.steps - 1
        ):
            _LOG.info("step=%3d  rank=%d  loss=%.4f", step, info.rank, local_loss)

    dt = time.perf_counter() - t0

    last_loss = losses_last[-1] if losses_last else 0.0
    summed = all_reduce_sum(last_loss)
    mean = summed / max(info.world_size, 1)

    if info.is_main:
        first_mean = sum(losses_first) / max(len(losses_first), 1)
        last_mean = sum(losses_last) / max(len(losses_last), 1)
        _LOG.info(
            "[main] dt=%.2fs  first=%.4f  last=%.4f  all_reduce_mean=%.4f",
            dt, first_mean, last_mean, mean,
        )

    shutdown_distributed()


def _spawn_entrypoint(local_rank: int, world_size: int,
                      args: argparse.Namespace, master_port: int) -> None:
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    # Some Windows + CPU-only PyTorch builds don't ship libuv.
    os.environ.setdefault("USE_LIBUV", "0")
    _worker_main(args)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    launched_externally = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if launched_externally:
        _worker_main(args)
        return 0

    if args.nproc <= 1:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        _worker_main(args)
        return 0

    _LOG.info("Self-spawning %d processes (master_port=%d)",
              args.nproc, args.master_port)
    mp.spawn(
        _spawn_entrypoint,
        args=(args.nproc, args, args.master_port),
        nprocs=args.nproc,
        join=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
