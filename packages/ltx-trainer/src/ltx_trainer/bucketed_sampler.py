"""
Bucketed samplers for variable-length video training.

Groups samples by exact latent frame count so each batch — and each DDP step —
uses clips of the same length. Bucket keys come directly from the data.

Distributed note:
  ``BucketBatchSampler`` emits the *global* batch list. ``accelerate`` wraps it in
  ``BatchSamplerShard``, which performs the per-rank split. Do not slice by rank
  inside this sampler or batches will be consumed twice (epoch length ≈ 1/N).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator

import torch.distributed as dist
from torch.utils.data import Sampler

from ltx_trainer import logger


def _resolve_num_replicas(num_replicas: int | None) -> int:
    if num_replicas is not None:
        return num_replicas
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _resolve_rank_for_logging() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def build_bucket_index(num_frames_list: list[int]) -> dict[int, list[int]]:
    """Group dataset indices by their exact latent frame count."""
    buckets: dict[int, list[int]] = defaultdict(list)
    unknown: list[int] = []

    for idx, nf in enumerate(num_frames_list):
        if nf <= 0:
            unknown.append(idx)
        else:
            buckets[nf].append(idx)

    if unknown:
        fallback_key = min(buckets.keys()) if buckets else 1
        buckets[fallback_key].extend(unknown)
        logger.warning(
            f"{len(unknown)} samples had unknown frame count (0) and were "
            f"merged into bucket {fallback_key}."
        )

    return dict(buckets)


def unwrap_bucket_batch_sampler(dataloader) -> BucketBatchSampler | None:
    """Find ``BucketBatchSampler`` under accelerate's ``BatchSamplerShard`` wrapper."""
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    visited: set[int] = set()
    while batch_sampler is not None and id(batch_sampler) not in visited:
        visited.add(id(batch_sampler))
        if isinstance(batch_sampler, BucketBatchSampler):
            return batch_sampler
        batch_sampler = getattr(batch_sampler, "batch_sampler", None)
    return None


class BucketBatchSampler(Sampler[list[int]]):
    """Yield batches of indices with identical latent frame counts.

    Each global DDP step uses ``num_replicas`` consecutive batches from the same
    bucket (one batch per GPU). Buckets smaller than ``num_replicas * batch_size``
    are padded by random resampling *within the bucket* so every original index
    appears at least once per epoch and all ranks stay aligned.

    After ``accelerator.prepare``, ``BatchSamplerShard`` assigns batch ``i`` to
    rank ``i % num_replicas``; this sampler must therefore expose the full
    global batch list (no per-rank slicing here).
    """

    def __init__(
        self,
        num_frames_list: list[int],
        batch_size: int,
        num_replicas: int | None = None,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.batch_size = batch_size
        self.num_replicas = _resolve_num_replicas(num_replicas)
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._cached_epoch: int | None = None
        self._cached_flat_batches: list[list[int]] | None = None

        self._buckets = build_bucket_index(num_frames_list)
        self._dataset_size = len(num_frames_list)
        bucket_info = {k: len(v) for k, v in sorted(self._buckets.items())}
        logger.info(
            f"BucketBatchSampler: {len(self._buckets)} fine-grained buckets "
            f"(latent frames→count): {bucket_info}, "
            f"batch_size={batch_size}, replicas={self.num_replicas} "
            f"(global batch list; shard handled by accelerate)"
        )

    def _pad_bucket_indices(self, indices: list[int], rng: random.Random) -> list[int]:
        """Pad so len(indices) is a multiple of ``num_replicas * batch_size``.

        All original indices are kept; extras are drawn with replacement from the
        same bucket only.
        """
        if not indices:
            return indices

        step_size = self.num_replicas * self.batch_size
        remainder = len(indices) % step_size
        if remainder == 0:
            return list(indices)
        shortfall = step_size - remainder
        extras = rng.choices(indices, k=shortfall)
        return list(indices) + extras

    def _make_batches_for_bucket(self, indices: list[int]) -> list[list[int]]:
        batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        return batches

    def _build_epoch_batches(self, rng: random.Random) -> list[list[int]]:
        flat_batches: list[list[int]] = []
        bucket_keys = sorted(self._buckets.keys())
        rng.shuffle(bucket_keys)

        for key in bucket_keys:
            indices = list(self._buckets[key])
            if not indices:
                continue

            indices = self._pad_bucket_indices(indices, rng)
            rng.shuffle(indices)
            batches = self._make_batches_for_bucket(indices)
            if not batches:
                continue

            # Align to full DDP steps: each step consumes num_replicas consecutive batches.
            if self.drop_last:
                n = (len(batches) // self.num_replicas) * self.num_replicas
                batches = batches[:n]
            else:
                remainder = len(batches) % self.num_replicas
                if remainder:
                    batches += batches[: self.num_replicas - remainder]

            flat_batches.extend(batches)

        if _resolve_rank_for_logging() == 0:
            seen: set[int] = set()
            for batch in flat_batches:
                seen.update(batch)
            per_rank = len(flat_batches) // self.num_replicas if self.num_replicas else len(flat_batches)
            logger.info(
                f"BucketBatchSampler epoch {self.epoch}: "
                f"{len(flat_batches)} global batches "
                f"(~{per_rank} per rank after BatchSamplerShard), "
                f"{len(seen)}/{self._dataset_size} unique indices "
                f"({'all covered' if len(seen) >= self._dataset_size else 'INCOMPLETE'})"
            )
            if len(seen) < self._dataset_size:
                logger.warning(
                    "Not every dataset index appears in this epoch plan — "
                    "check bucket sizes and drop_last settings."
                )

        return flat_batches

    def _flat_batches_for_epoch(self) -> list[list[int]]:
        if self._cached_epoch != self.epoch or self._cached_flat_batches is None:
            rng = random.Random(self.seed + self.epoch)
            self._cached_flat_batches = self._build_epoch_batches(rng)
            self._cached_epoch = self.epoch
        return self._cached_flat_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._cached_epoch = None
        self._cached_flat_batches = None

    def __len__(self) -> int:
        return len(self._flat_batches_for_epoch())

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._flat_batches_for_epoch()
