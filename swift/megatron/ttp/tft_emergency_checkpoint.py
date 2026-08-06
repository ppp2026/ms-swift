# Phase 6: Atomic Emergency Checkpoint Write
# 实现原子性的 emergency checkpoint 写入
#
# 核心功能：
# 1. write_emergency_checkpoint: 旧版本，gather 后 writer 单独写入 optimizer_0.pt
# 2. write_emergency_checkpoint_parallel: 新版本，每个 rank 并行写入自己的 shard
# 3. 使用临时目录 + 原子 rename 保证原子性
# 4. 写入 manifest.json 和 COMPLETED 标记
#
# 目录结构（旧版本，单文件）：
# checkpoint-0000005-ttp-tmp/
#   manifest.json
#   common.pt
#   model.pt
#   optimizer_0.pt
#   COMPLETED
#
# 目录结构（新版本，并行写入）：
# checkpoint-0000005-ttp-tmp/
#   manifest.json
#   common.pt
#   model.pt
#   optimizer_shard_0.pt
#   optimizer_shard_1.pt
#   COMPLETED
#
# 原子 rename 后：
# checkpoint-0000005-ttp/
#   manifest.json
#   common.pt
#   model.pt
#   optimizer_0.pt  或  optimizer_shard_*.pt
#   COMPLETED

import os
import json
import random
import torch
import numpy as np
from logging import getLogger
from pathlib import Path
from typing import Dict, Any, Optional

ttp_logger = getLogger(__name__)


def write_emergency_checkpoint(
    step: int,
    output_dir: str,
    context: Any,
    optimizer_state: Dict[str, Any],
    selected_ranks: tuple,
    writer_rank: int,
    replica_num: int,
):
    """Write emergency checkpoint to disk (only called by writer rank).

    This function writes the emergency checkpoint in a temporary directory,
    then atomically renames it to the final directory. This ensures that
    the checkpoint is either fully written or not present at all.

    Args:
        step: Current training iteration
        output_dir: Base output directory for checkpoints
        context: TTPTrainContext containing trainer, models, optimizer, etc.
        optimizer_state: Complete optimizer state (gathered from selected ranks)
        selected_ranks: Tuple of selected ranks (canonicalized to shard order)
        writer_rank: Rank that is writing the checkpoint
        replica_num: Number of replicas
    """
    cur_rank = torch.distributed.get_rank()

    if cur_rank != writer_rank:
        ttp_logger.warning(
            f"[TTP] Rank {cur_rank} is not the writer (writer={writer_rank}), "
            f"skipping checkpoint write"
        )
        return

    ttp_logger.info(
        f"[TTP] Rank {cur_rank} writing emergency checkpoint at step {step}"
    )

    # Create temporary directory
    tmp_dir = os.path.join(output_dir, f'checkpoint-{step:07d}-ttp-tmp')
    final_dir = os.path.join(output_dir, f'checkpoint-{step:07d}-ttp')

    if os.path.exists(tmp_dir):
        ttp_logger.warning(f"[TTP] Temporary directory already exists: {tmp_dir}")
        import shutil
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Write common.pt (iteration, scheduler state, args, RNG state)
        common = _build_common_state(step, context)
        torch.save(common, os.path.join(tmp_dir, 'common.pt'))
        ttp_logger.info(f"[TTP] Written common.pt")

        # Write model.pt (model state dict)
        model_state = _build_model_state(context)
        torch.save(model_state, os.path.join(tmp_dir, 'model.pt'))
        ttp_logger.info(f"[TTP] Written model.pt")

        # Write optimizer_0.pt (complete optimizer state)
        torch.save(optimizer_state, os.path.join(tmp_dir, 'optimizer_0.pt'))
        ttp_logger.info(f"[TTP] Written optimizer_0.pt")

        # Write manifest.json
        manifest = _build_manifest(
            step=step,
            context=context,
            selected_ranks=selected_ranks,
            writer_rank=writer_rank,
            replica_num=replica_num,
        )
        with open(os.path.join(tmp_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        ttp_logger.info(f"[TTP] Written manifest.json")

        # Write COMPLETED marker
        Path(os.path.join(tmp_dir, 'COMPLETED')).touch()
        ttp_logger.info(f"[TTP] Written COMPLETED marker")

        # Atomic rename
        if os.path.exists(final_dir):
            ttp_logger.warning(f"[TTP] Final directory already exists: {final_dir}")
            ttp_logger.warning(f"[TTP] Removing existing directory")
            import shutil
            shutil.rmtree(final_dir)

        os.replace(tmp_dir, final_dir)
        ttp_logger.info(f"[TTP] Atomic rename: {tmp_dir} -> {final_dir}")

        # Update latest_ttp_checkpoint.txt
        latest_file = os.path.join(output_dir, 'latest_ttp_checkpoint.txt')
        with open(latest_file, 'w') as f:
            f.write(final_dir)
        ttp_logger.info(f"[TTP] Updated {latest_file}")

        ttp_logger.info(
            f"[TTP] Emergency checkpoint written successfully at step {step}"
        )

    except Exception as e:
        ttp_logger.error(
            f"[TTP] Failed to write emergency checkpoint: {e}",
            exc_info=True
        )
        # Clean up temporary directory on failure
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
        raise


def write_emergency_checkpoint_parallel(
    step: int,
    output_dir: str,
    context: Any,
    local_shard: Dict[str, Any],
    selected_ranks: tuple,
    shard_map: dict,
    writer_rank: int,
    replica_num: int,
    save_group,
):
    """Write emergency checkpoint in parallel (each rank writes its own shard).

    This function replaces the gather-based write_emergency_checkpoint() with
    a parallel approach:
    - Each rank writes its own optimizer_shard_{N}.pt (no gather needed)
    - Writer rank also writes common.pt, model.pt
    - Barrier ensures all shards written before COMPLETED marker

    Args:
        step: Current training iteration
        output_dir: Base output directory for checkpoints
        context: Trainer instance
        local_shard: Local optimizer shard data (from build_local_optimizer_shard)
        selected_ranks: Tuple of selected ranks (canonicalized to shard order)
        shard_map: Dict mapping rank -> shard index
        writer_rank: Rank that writes common/model/manifest
        replica_num: Number of replicas
        save_group: Gloo group for barrier synchronization
    """
    cur_rank = torch.distributed.get_rank()
    shard_idx = shard_map[cur_rank]

    tmp_dir = os.path.join(output_dir, f'checkpoint-{step:07d}-ttp-tmp')
    final_dir = os.path.join(output_dir, f'checkpoint-{step:07d}-ttp')

    ttp_logger.info(
        f"[TTP] Rank {cur_rank} (shard {shard_idx}) parallel write started at step {step}"
    )

    # Step 1: Writer creates tmp directory
    if cur_rank == writer_rank:
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

    # Barrier: ensure tmp_dir is created before any rank writes
    torch.distributed.barrier(save_group)

    # Step 2: Each rank writes its own shard file
    shard_file = os.path.join(tmp_dir, f'optimizer_shard_{shard_idx}.pt')
    shard_data = {
        'shard_idx': shard_idx,
        'rank': cur_rank,
        'data': local_shard,
    }
    torch.save(shard_data, shard_file)
    ttp_logger.info(f"[TTP] Rank {cur_rank} written {shard_file}")

    # Step 3: Writer also writes common.pt and model.pt
    if cur_rank == writer_rank:
        try:
            common = _build_common_state(step, context)
            torch.save(common, os.path.join(tmp_dir, 'common.pt'))
            ttp_logger.info(f"[TTP] Written common.pt")

            model_state = _build_model_state(context)
            torch.save(model_state, os.path.join(tmp_dir, 'model.pt'))
            ttp_logger.info(f"[TTP] Written model.pt")
        except Exception as e:
            ttp_logger.error(f"[TTP] Failed to write common/model: {e}", exc_info=True)
            raise

    # Barrier: ensure all shards and common/model are written
    torch.distributed.barrier(save_group)

    # Step 4: Writer writes manifest.json + COMPLETED + atomic rename
    if cur_rank == writer_rank:
        try:
            # Build shard files list for manifest
            shard_files = []
            for rank in selected_ranks:
                s_idx = shard_map[rank]
                shard_files.append({
                    'shard_idx': s_idx,
                    'rank': rank,
                    'file': f'optimizer_shard_{s_idx}.pt',
                })

            manifest = _build_manifest_parallel(
                step=step,
                context=context,
                selected_ranks=selected_ranks,
                writer_rank=writer_rank,
                replica_num=replica_num,
                shard_files=shard_files,
            )
            with open(os.path.join(tmp_dir, 'manifest.json'), 'w') as f:
                json.dump(manifest, f, indent=2)
            ttp_logger.info(f"[TTP] Written manifest.json")

            Path(os.path.join(tmp_dir, 'COMPLETED')).touch()
            ttp_logger.info(f"[TTP] Written COMPLETED marker")

            # Atomic rename
            if os.path.exists(final_dir):
                ttp_logger.warning(f"[TTP] Final directory already exists: {final_dir}")
                import shutil
                shutil.rmtree(final_dir)

            os.replace(tmp_dir, final_dir)
            ttp_logger.info(f"[TTP] Atomic rename: {tmp_dir} -> {final_dir}")

            # Update latest_ttp_checkpoint.txt
            latest_file = os.path.join(output_dir, 'latest_ttp_checkpoint.txt')
            with open(latest_file, 'w') as f:
                f.write(final_dir)
            ttp_logger.info(f"[TTP] Updated {latest_file}")

            ttp_logger.info(
                f"[TTP] Parallel emergency checkpoint written successfully at step {step}"
            )

        except Exception as e:
            ttp_logger.error(
                f"[TTP] Failed to finalize parallel checkpoint: {e}",
                exc_info=True
            )
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir)
            raise

    ttp_logger.info(
        f"[TTP] Rank {cur_rank} parallel checkpoint write completed at step {step}"
    )


def _extract_optimizer_non_param_state(optimizer):
    """Extract optimizer non-param state, handling MindSpeed AdamW incompatibility.

    MindSpeed AdamW stores 'step' in param_groups as tensor. mcore
    DistributedOptimizer.state_dict() uses set() to deduplicate tensors,
    which fails because tensors hash by id (same value, different objects).
    This function manually extracts the non-param state in mcore's expected format,
    converting step to int to avoid the set() deduplication issue.
    """
    # Get inner optimizer (MindSpeed AdamW) via DistributedOptimizer.optimizer
    inner_opt = getattr(optimizer, 'optimizer', optimizer)
    inner_state_dict = inner_opt.state_dict()

    # Extract step from param_groups (MindSpeed's location), keep as CPU tensor
    # MindSpeed AdamW's step() expects group['step'] to be a tensor (calls .is_cpu)
    step = None
    for g in inner_state_dict.get('param_groups', []):
        if 'step' in g:
            s = g['step']
            if hasattr(s, 'cpu'):
                step = s.cpu()
            else:
                import torch
                step = torch.tensor(int(s), dtype=torch.int64)
            break

    # Build state dict in mcore DistributedOptimizer.state_dict() format
    state_dict = {}
    state_dict['optimizer'] = {k: v for k, v in inner_state_dict.items() if k != 'state'}
    for param_group in state_dict['optimizer']['param_groups']:
        param_group.pop('params', None)
        if step is not None:
            param_group['step'] = step

    # Grad scaler
    grad_scaler = getattr(optimizer, 'grad_scaler', None)
    if grad_scaler is not None:
        state_dict['grad_scaler'] = grad_scaler.state_dict()

    return state_dict

def _build_common_state(step: int, context: Any) -> Dict[str, Any]:
    """Build common state dict for checkpoint.

    Args:
        step: Current training iteration
        context: Trainer instance (BaseMegatronTrainer)

    Returns:
        Dict with common state including optimizer non-param state and full RNG
    """
    # MS-Swift trainer uses opt_param_scheduler, not scheduler
    scheduler = getattr(context, 'scheduler', None)
    if scheduler is None:
        scheduler = getattr(context, 'opt_param_scheduler', None)

    # Optimizer non-param state (step, param_groups, grad_scaler)
    # In MCore DistributedOptimizer, state_dict() only saves non-param state;
    # param/exp_avg/exp_avg_sq are saved separately in optimizer_0.pt
    optimizer_state = None
    if context.optimizer is not None:
        try:
            optimizer_state = _extract_optimizer_non_param_state(context.optimizer)
        except Exception as e:
            ttp_logger.warning(f"[TTP] Failed to save optimizer non-param state: {e}", exc_info=True)

    common = {
        'iteration': step,
        'consumed_train_samples': context.state.consumed_train_samples,
        'optimizer_update_state': getattr(context.optimizer, 'ttp_update_state', 'IDLE'),
        'optimizer_committed_step': getattr(context.optimizer, 'ttp_committed_step', step),
        'optimizer_non_param_state': optimizer_state,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'trainer_state': {
            'iteration': context.state.iteration,
            'consumed_train_samples': context.state.consumed_train_samples,
            'consumed_valid_samples': getattr(context.state, 'consumed_valid_samples', 0),
        },
        'args': {
            'tensor_model_parallel_size': context.args.tensor_model_parallel_size,
            'pipeline_model_parallel_size': context.args.pipeline_model_parallel_size,
            'context_parallel_size': getattr(context.args, 'context_parallel_size', 1),
            'expert_model_parallel_size': getattr(context.args, 'expert_model_parallel_size', 1),
            'optimizer_replica_num': getattr(context.args, 'optimizer_replica_num', 2),
        },
        'rng_state': {
            'random_rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
        },
    }

    # Add device RNG state (NPU or CUDA)
    try:
        if hasattr(torch, 'npu') and torch.npu.is_available():
            common['rng_state']['device_rng_state'] = torch.npu.get_rng_state()
        elif torch.cuda.is_available():
            common['rng_state']['device_rng_state'] = torch.cuda.get_rng_state()
    except Exception:
        pass

    # Add Megatron RNG tracker states
    try:
        from megatron.core import tensor_parallel
        common['rng_state']['rng_tracker_states'] = (
            tensor_parallel.get_cuda_rng_tracker().get_states()
        )
    except Exception:
        common['rng_state']['rng_tracker_states'] = None

    return common


def _build_model_state(context: Any) -> Dict[str, Any]:
    """Build model state dict for checkpoint.

    Saves ALL model chunks (not just the first one).
    Prioritizes unwrapped models to avoid DDP wrapper state in checkpoint.

    Args:
        context: Trainer instance (BaseMegatronTrainer)

    Returns:
        Dict with model state (keyed by model index)
    """
    # Priority: unwrapped_models > wrapped_models > models
    models = getattr(context, 'unwrapped_models', None)
    if not models:
        models = getattr(context, 'wrapped_models', None)
    if not models:
        models = getattr(context, 'models', None)
    if not models:
        raise ValueError("[TTP] No model found in context")

    return {f'model_{index}': model.state_dict() for index, model in enumerate(models)}


def _build_manifest(
    step: int,
    context: Any,
    selected_ranks: tuple,
    writer_rank: int,
    replica_num: int,
) -> Dict[str, Any]:
    """Build manifest for emergency checkpoint.

    Args:
        step: Current training iteration
        context: TTPTrainContext
        selected_ranks: Tuple of selected ranks
        writer_rank: Rank that wrote the checkpoint
        replica_num: Number of replicas

    Returns:
        Dict with manifest
    """
    args = context.args

    manifest = {
        'format': 'ms-swift-ttp-emergency-v1',
        'iteration': step,
        'world_size': args.world_size,
        'data_parallel_size': args.world_size // (
            args.tensor_model_parallel_size *
            args.pipeline_model_parallel_size *
            getattr(args, 'context_parallel_size', 1)
        ),
        'tensor_model_parallel_size': args.tensor_model_parallel_size,
        'pipeline_model_parallel_size': args.pipeline_model_parallel_size,
        'context_parallel_size': getattr(args, 'context_parallel_size', 1),
        'expert_model_parallel_size': getattr(args, 'expert_model_parallel_size', 1),
        'replica_num': replica_num,
        'optimizer_shard_count': args.world_size // (
            args.tensor_model_parallel_size *
            args.pipeline_model_parallel_size *
            getattr(args, 'context_parallel_size', 1) *
            replica_num
        ),
        'selected_ranks': list(selected_ranks),
        'writer_rank': writer_rank,
        'safe_step': True,
        'files': ['common.pt', 'model.pt', 'optimizer_0.pt'],
    }

    return manifest


def _build_manifest_parallel(
    step: int,
    context: Any,
    selected_ranks: tuple,
    writer_rank: int,
    replica_num: int,
    shard_files: list,
) -> Dict[str, Any]:
    """Build manifest for parallel-written emergency checkpoint.

    Args:
        step: Current training iteration
        context: Trainer instance
        selected_ranks: Tuple of selected ranks
        writer_rank: Rank that wrote common/model/manifest
        replica_num: Number of replicas
        shard_files: List of {shard_idx, rank, file} dicts

    Returns:
        Dict with manifest
    """
    args = context.args

    shard_file_names = [f['file'] for f in shard_files]

    manifest = {
        'format': 'ms-swift-ttp-emergency-v2-parallel',
        'iteration': step,
        'world_size': args.world_size,
        'data_parallel_size': args.world_size // (
            args.tensor_model_parallel_size *
            args.pipeline_model_parallel_size *
            getattr(args, 'context_parallel_size', 1)
        ),
        'tensor_model_parallel_size': args.tensor_model_parallel_size,
        'pipeline_model_parallel_size': args.pipeline_model_parallel_size,
        'context_parallel_size': getattr(args, 'context_parallel_size', 1),
        'expert_model_parallel_size': getattr(args, 'expert_model_parallel_size', 1),
        'replica_num': replica_num,
        'optimizer_shard_count': len(shard_files),
        'selected_ranks': list(selected_ranks),
        'writer_rank': writer_rank,
        'safe_step': True,
        'files': ['common.pt', 'model.pt'] + shard_file_names,
        'shard_files': shard_files,
        'write_mode': 'parallel',
    }

    return manifest


def _merge_shard_files(shard_dict: Dict[int, Any]) -> Dict[str, Any]:
    """Merge multiple shard files into complete DP-zero state.

    This is the inverse of build_local_optimizer_shard(): it concatenates
    local shards from all selected ranks (ordered by shard_idx) to rebuild
    the complete optimizer state, matching the format of
    gather_and_reorder_local_shards() output on writer rank.

    Args:
        shard_dict: {shard_idx: shard_data} where shard_data is the output
                    of build_local_optimizer_shard()

    Returns:
        Complete optimizer state dict:
        {
            'buckets_coalesced': True,
            gbuf_idx: {
                dtype: {
                    'param': complete_cpu_tensor,
                    'exp_avg': complete_cpu_tensor,
                    'exp_avg_sq': complete_cpu_tensor,
                    'numel_unpadded': int,
                }
            }
        }
    """
    if not shard_dict:
        return None

    # Get structure from first shard (all shards have same structure)
    first_shard = next(iter(shard_dict.values()))

    state = {'buckets_coalesced': True}

    for gbuf_idx, dtype_state in first_shard.items():
        dtype_result = {}
        for dtype, shard_info in dtype_state.items():
            buffer_numel_unpadded = shard_info['numel_unpadded']
            num_buckets = len(shard_info['buckets'])

            # Allocate complete tensors
            world_tensors = {
                key: torch.zeros(buffer_numel_unpadded, dtype=torch.float32, device='cpu')
                for key in ('param', 'exp_avg', 'exp_avg_sq')
            }
            world_tensors['numel_unpadded'] = buffer_numel_unpadded

            # For each bucket, concatenate shards in logical shard_idx order
            for bucket_idx in range(num_buckets):
                # Get bucket metadata from first shard
                bucket_meta = shard_info['buckets'][bucket_idx]
                offset = bucket_meta['offset']
                world_numel_unpadded = bucket_meta['world_numel_unpadded']

                for key in ('param', 'exp_avg', 'exp_avg_sq'):
                    # Collect shard tensors in logical shard_idx order
                    logical_tensors = []
                    for shard_idx in sorted(shard_dict.keys()):
                        shard_data = shard_dict[shard_idx]
                        shard_bucket = shard_data[gbuf_idx][dtype]['buckets'][bucket_idx]
                        logical_tensors.append(shard_bucket[key])

                    concatenated = torch.cat(logical_tensors)
                    copy_len = min(world_numel_unpadded, concatenated.numel())
                    world_tensors[key][offset:offset + copy_len].copy_(
                        concatenated[:copy_len]
                    )

            dtype_result[dtype] = world_tensors
        state[gbuf_idx] = dtype_result

    return state


def load_emergency_checkpoint(
    checkpoint_dir: str,
    map_location: str = 'cpu',
    load_optimizer: bool = True,
) -> Dict[str, Any]:
    """Load emergency checkpoint from disk.

    Supports both v1 (single optimizer_0.pt) and v2-parallel (multi shard)
    formats. For v2-parallel, loads all shard files and merges them into
    complete DP-zero state.

    Args:
        checkpoint_dir: Path to checkpoint directory
        map_location: Device to load tensors to
        load_optimizer: If False, skip loading the large optimizer file(s)

    Returns:
        Dict with checkpoint data (optimizer is complete DP-zero state)
    """
    manifest = validate_emergency_checkpoint(checkpoint_dir)

    # Load common state (weights_only=False because it contains numpy RNG state)
    common_file = os.path.join(checkpoint_dir, 'common.pt')
    common = torch.load(common_file, map_location=map_location, weights_only=False)

    # Load model state
    model_file = os.path.join(checkpoint_dir, 'model.pt')
    model_state = torch.load(model_file, map_location=map_location, weights_only=False)

    # Load optimizer state (optional)
    optimizer_state = None
    if load_optimizer:
        fmt = manifest['format']

        if fmt == 'ms-swift-ttp-emergency-v1':
            # Old format: single optimizer_0.pt
            optimizer_file = os.path.join(checkpoint_dir, 'optimizer_0.pt')
            optimizer_state = torch.load(optimizer_file, map_location=map_location, weights_only=False)

        elif fmt == 'ms-swift-ttp-emergency-v2-parallel':
            # New format: multiple optimizer_shard_*.pt files
            shard_files = manifest['shard_files']
            shard_dict = {}
            for shard_info in shard_files:
                shard_file = os.path.join(checkpoint_dir, shard_info['file'])
                shard_idx = shard_info['shard_idx']
                shard_data = torch.load(shard_file, map_location=map_location, weights_only=False)
                shard_dict[shard_idx] = shard_data['data']
                ttp_logger.info(f"[TTP] Loaded shard {shard_idx} from {shard_file}")

            # Merge shards into complete DP-zero state
            optimizer_state = _merge_shard_files(shard_dict)
            ttp_logger.info(
                f"[TTP] Merged {len(shard_dict)} shards into complete optimizer state"
            )

    ttp_logger.info(
        f"[TTP] Loaded emergency checkpoint from {checkpoint_dir} "
        f"(iteration={manifest['iteration']}, format={manifest['format']})"
    )

    return {
        'manifest': manifest,
        'common': common,
        'model': model_state,
        'optimizer': optimizer_state,
    }


def validate_emergency_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
    """Validate emergency checkpoint directory structure and manifest.

    Supports two formats:
    - v1 (single file): requires optimizer_0.pt
    - v2-parallel (multi shard): requires optimizer_shard_*.pt files

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Manifest dict

    Raises:
        FileNotFoundError: If directory or required files are missing
        RuntimeError: If checkpoint is incomplete or format is unsupported
    """
    SUPPORTED_FORMATS = ('ms-swift-ttp-emergency-v1', 'ms-swift-ttp-emergency-v2-parallel')
    BASE_REQUIRED_FILES = ('manifest.json', 'common.pt', 'model.pt', 'COMPLETED')

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"[TTP] Checkpoint directory not found: {checkpoint_dir}")

    # Check base required files
    missing = [
        name for name in BASE_REQUIRED_FILES
        if not os.path.exists(os.path.join(checkpoint_dir, name))
    ]
    if missing:
        raise RuntimeError(f"[TTP] Incomplete emergency checkpoint, missing: {missing}")

    with open(os.path.join(checkpoint_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    fmt = manifest.get('format')
    if fmt not in SUPPORTED_FORMATS:
        raise RuntimeError(f"[TTP] Unsupported checkpoint format: {fmt}")
    if not manifest.get('safe_step', False):
        raise RuntimeError("[TTP] Refusing to load a non-safe-step checkpoint")
    if manifest.get('optimizer_shard_count', 0) <= 0:
        raise RuntimeError("[TTP] Invalid optimizer_shard_count")

    # Check optimizer files based on format
    if fmt == 'ms-swift-ttp-emergency-v1':
        # Old format: single optimizer_0.pt
        opt_file = os.path.join(checkpoint_dir, 'optimizer_0.pt')
        if not os.path.exists(opt_file):
            raise RuntimeError(f"[TTP] Missing optimizer_0.pt for v1 format")
    elif fmt == 'ms-swift-ttp-emergency-v2-parallel':
        # New format: multiple optimizer_shard_*.pt files
        shard_files = manifest.get('shard_files', [])
        if not shard_files:
            raise RuntimeError("[TTP] v2-parallel format missing shard_files in manifest")
        for shard_info in shard_files:
            shard_file = os.path.join(checkpoint_dir, shard_info['file'])
            if not os.path.exists(shard_file):
                raise RuntimeError(f"[TTP] Missing shard file: {shard_info['file']}")

    return manifest


def save_ttp_checkpoint(trainer, step, selected_ranks=None):
    """Save checkpoint in TTP emergency format (parallel write).

    Unified entry point for both normal save_steps and emergency callback.
    Uses parallel write: each rank writes its own optimizer shard file,
    eliminating the gather communication step.

    Args:
        trainer: BaseMegatronTrainer instance
        step: Current training iteration (for normal save) or committed_step (for emergency)
        selected_ranks: Ranks to include in save. If None, all DP ranks participate
                        (normal save). For emergency save, pass the fault-survived ranks.

    This function:
    1. Canonicalizes selected ranks to shard order
    2. Creates dynamic Gloo dump_group for barrier synchronization
    3. Each rank extracts local optimizer shard (no gather!)
    4. Each rank writes its own optimizer_shard_{N}.pt in parallel
    5. Writer rank writes common.pt, model.pt, manifest.json, COMPLETED
    """
    args = trainer.args
    optimizer = trainer.optimizer
    cur_rank = torch.distributed.get_rank()

    from .tft_replica_group import ttp_get_dp_ranks
    from .tft_shard_map import (
        build_optimizer_shard_map,
        canonicalize_selected_save_ranks,
        get_writer_rank,
    )
    from .tft_emergency_group import (
        destroy_dump_group,
        get_emergency_save_group_for_save_info,
    )
    from .tft_optimizer_shard import build_local_optimizer_shard

    dp_ranks = ttp_get_dp_ranks()
    if dp_ranks is None:
        raise RuntimeError("[TTP] dp_ranks is None, TTP not initialized?")

    replica_num = getattr(args, 'optimizer_replica_num', None) or 2

    # Default: select first replica group (covers all shards exactly once)
    if selected_ranks is None:
        shard_count = len(dp_ranks) // replica_num
        selected_ranks = list(dp_ranks[:shard_count])

    ttp_logger.info(
        f"[TTP] save_ttp_checkpoint (parallel): step={step}, "
        f"selected_ranks={selected_ranks}, replica_num={replica_num}"
    )

    # Canonicalize selected ranks to shard order
    try:
        selected_ranks = canonicalize_selected_save_ranks(
            dp_ranks, selected_ranks, replica_num
        )
        shard_map = build_optimizer_shard_map(dp_ranks, replica_num)
        writer_rank = get_writer_rank(selected_ranks, dp_ranks, replica_num)
    except ValueError as e:
        ttp_logger.error(f"[TTP] Failed to canonicalize selected ranks: {e}")
        raise

    ttp_logger.info(
        f"[TTP] Canonicalized: selected_ranks={selected_ranks}, writer={writer_rank}"
    )

    # Non-selected ranks skip the save
    if cur_rank not in selected_ranks:
        ttp_logger.info(
            f"[TTP] Rank {cur_rank} not selected for save; skipping"
        )
        return

    # Set dump args for TTP (for compatibility with save_parameter_state)
    if hasattr(optimizer, 'set_dump_args'):
        optimizer.set_dump_args(writer_rank, step, list(selected_ranks))

    # Create dynamic Gloo dump_group (for barrier only, no gather needed)
    save_group = None
    try:
        canonicalized, save_group, group_ranks = get_emergency_save_group_for_save_info(
            dp_ranks, list(selected_ranks), replica_num
        )

        ttp_logger.info(
            f"[TTP] Rank {cur_rank} created Gloo group for barrier "
            f"(group_ranks={group_ranks})"
        )

        # Each rank extracts its local shard (no gather!)
        replica_group_size = len(selected_ranks)
        local_shard = build_local_optimizer_shard(optimizer, replica_group_size)

        ttp_logger.info(
            f"[TTP] Rank {cur_rank} extracted local shard (shard_idx={shard_map[cur_rank]})"
        )

        # Parallel write: each rank writes its own shard file
        write_emergency_checkpoint_parallel(
            step=step,
            output_dir=args.output_dir,
            context=trainer,
            local_shard=local_shard,
            selected_ranks=selected_ranks,
            shard_map=shard_map,
            writer_rank=writer_rank,
            replica_num=replica_num,
            save_group=save_group,
        )

    finally:
        # Always destroy the temporary Gloo group to prevent accumulation
        destroy_dump_group(save_group)

    ttp_logger.info(
        f"[TTP] Rank {cur_rank} save_ttp_checkpoint (parallel) completed at step {step}"
    )
