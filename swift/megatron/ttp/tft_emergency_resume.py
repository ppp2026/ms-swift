# TTP Emergency Checkpoint Resume
# 实现从 emergency checkpoint 恢复训练状态
#
# 按七阶段手册 T2/T3/T4 实现：
# T2: 恢复模型和 Trainer 公共状态
# T3: 恢复 optimizer 并重新切分到副本组
# T4: 恢复 scheduler、RNG 并验证 loss 连续性
#
# 核心思路（T3）：
#   不手动切分 tensor，复用 mcore 的 load_parameter_state_from_dp_zero()
#   临时把 optimizer.data_parallel_group 切换到副本组
#   让 mcore 自动处理副本组内的 scatter

import os
import json
import random
from logging import getLogger

import numpy as np
import torch

from .tft_emergency_checkpoint import load_emergency_checkpoint

logger = getLogger(__name__)


# ========== T2: Model and Common State Restore ==========

def _get_models(trainer):
    """Get model list from trainer, preferring unwrapped models."""
    models = getattr(trainer, 'unwrapped_models', None)
    if not models:
        models = getattr(trainer, 'wrapped_models', None)
    if not models:
        models = getattr(trainer, 'models', None)
    if not models:
        raise RuntimeError('[TTP RESUME] Trainer has no models')
    return models


def _validate_runtime_layout(args, manifest):
    """Validate that runtime parallel layout matches checkpoint manifest."""
    expected = {
        'world_size': args.world_size,
        'tensor_model_parallel_size': args.tensor_model_parallel_size,
        'pipeline_model_parallel_size': args.pipeline_model_parallel_size,
        'context_parallel_size': getattr(args, 'context_parallel_size', 1),
        'expert_model_parallel_size': getattr(args, 'expert_model_parallel_size', 1),
        'replica_num': getattr(args, 'optimizer_replica_num', 2),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f'[TTP RESUME] Runtime layout mismatch: {mismatches}')


def _restore_models(trainer, model_state):
    """Restore model parameters from checkpoint."""
    models = _get_models(trainer)
    if len(models) != len(model_state):
        raise RuntimeError(
            f'[TTP RESUME] Model chunk count mismatch: '
            f'{len(model_state)} checkpoint vs {len(models)} runtime'
        )
    for index, model in enumerate(models):
        key = f'model_{index}'
        if key not in model_state:
            raise KeyError(f'[TTP RESUME] Missing {key} in model.pt')
        incompatible = model.load_state_dict(model_state[key], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f'[TTP RESUME] Model state mismatch: {incompatible}')
    logger.info(f'[TTP RESUME] Restored {len(models)} model chunks')


# ========== T3: Optimizer Restore with Re-sharding ==========

def _unwrap_single_distributed_optimizer(optimizer):
    """Unwrap chained_optimizers and validate DistributedOptimizer interface."""
    chained = getattr(optimizer, 'chained_optimizers', None)
    if chained is not None:
        if len(chained) != 1:
            raise RuntimeError(
                f'[TTP RESUME] MVP requires one optimizer, got {len(chained)}'
            )
        optimizer = chained[0]

    required = (
        'gbuf_ranges',
        'buffers',
        'data_parallel_group',
        'data_parallel_group_gloo',
    )
    missing = [name for name in required if not hasattr(optimizer, name)]
    if missing:
        raise RuntimeError(f'[TTP RESUME] Incompatible optimizer, missing: {missing}')
    return optimizer


def _validate_optimizer_parameter_state(distributed_optimizer, state_dict):
    """Validate that optimizer state dict matches runtime gbuf layout."""
    for gbuf_idx, gbuf_range_maps in enumerate(distributed_optimizer.gbuf_ranges):
        if gbuf_idx not in state_dict:
            raise KeyError(f'[TTP RESUME] Missing optimizer gbuf {gbuf_idx}')
        for dtype in gbuf_range_maps:
            if dtype not in state_dict[gbuf_idx]:
                raise KeyError(f'[TTP RESUME] Missing optimizer dtype {dtype}')
            saved = state_dict[gbuf_idx][dtype]
            expected_numel = distributed_optimizer.buffers[gbuf_idx].numel_unpadded
            if saved.get('numel_unpadded') != expected_numel:
                raise RuntimeError(
                    f'[TTP RESUME] gbuf {gbuf_idx} numel mismatch: '
                    f"{saved.get('numel_unpadded')} checkpoint vs {expected_numel} runtime"
                )
            for key in ('param', 'exp_avg', 'exp_avg_sq'):
                if key not in saved:
                    raise KeyError(f'[TTP RESUME] Missing optimizer tensor {key}')


def _restore_optimizer(trainer, checkpoint_dir, common):
    """Restore optimizer state from emergency checkpoint.

    Core idea (T3):
    - Each replica group's local rank 0 reads the complete optimizer_0.pt
    - Temporarily switch data_parallel_group to replica group
    - Call mcore's load_parameter_state_from_dp_zero() to scatter
    - Restore original group in finally block
    """
    if getattr(trainer.args, 'no_load_optim', False):
        logger.warning('[TTP RESUME] --no_load_optim=true; optimizer restore skipped')
        return

    non_param_state = common.get('optimizer_non_param_state')
    if non_param_state is None:
        logger.warning(
            '[TTP RESUME] optimizer_non_param_state not found in common.pt '
            '(old checkpoint format); skipping non-param state restore. '
            'Optimizer step will be 0, scheduler may be inconsistent.'
        )
    else:
        # Step 1: Load non-param state (step, param_groups, grad_scaler)
        # This allocates optimizer tensor state with correct shapes
        trainer.optimizer.load_state_dict(non_param_state)

        # Convert int step back to tensor for MindSpeed AdamW compatibility
        # (old checkpoints saved step as int; MindSpeed AdamW expects tensor)
        import torch
        _inner_opt = getattr(trainer.optimizer, 'optimizer', trainer.optimizer)
        for _g in _inner_opt.param_groups:
            if 'step' in _g and isinstance(_g['step'], int):
                _g['step'] = torch.tensor(_g['step'], dtype=torch.int64)

        logger.info('[TTP RESUME] Loaded optimizer non-param state (step, param_groups)')

    distributed_optimizer = _unwrap_single_distributed_optimizer(trainer.optimizer)

    # Step 2: Get replica groups
    from .tft_replica_group import (
        ttp_get_dp_cp_replica_group,
        ttp_get_dp_cp_replica_group_gloo,
    )

    replica_group = ttp_get_dp_cp_replica_group()
    replica_group_gloo = ttp_get_dp_cp_replica_group_gloo()
    if replica_group is None or replica_group_gloo is None:
        raise RuntimeError('[TTP RESUME] Replica optimizer groups are unavailable')

    # Step 3: Each replica group's local rank 0 reads complete optimizer state
    parameter_state = None
    local_rank_in_replica = torch.distributed.get_rank(replica_group_gloo)
    if local_rank_in_replica == 0:
        # Check checkpoint format (v1 single-file vs v2-parallel multi-shard)
        manifest_path = os.path.join(checkpoint_dir, 'manifest.json')
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if manifest.get('format') == 'ms-swift-ttp-emergency-v2-parallel':
            # New format: load and merge multiple shard files
            from .tft_emergency_checkpoint import load_emergency_checkpoint
            payload = load_emergency_checkpoint(checkpoint_dir, load_optimizer=True)
            parameter_state = payload['optimizer']
            logger.info(
                f'[TTP RESUME] Loaded and merged {len(manifest.get("shard_files", []))} '
                f'shard files (v2-parallel format)'
            )
        else:
            # Old format: single optimizer_0.pt
            optimizer_path = os.path.join(checkpoint_dir, 'optimizer_0.pt')
            logger.info(f'[TTP RESUME] Loading optimizer state from {optimizer_path}')
            parameter_state = torch.load(optimizer_path, map_location='cpu', weights_only=False)

        _validate_optimizer_parameter_state(distributed_optimizer, parameter_state)
        logger.info('[TTP RESUME] Optimizer state loaded and validated')
    else:
        logger.info(f'[TTP RESUME] Waiting for local rank 0 to broadcast (local_rank={local_rank_in_replica})')

    # Step 4: Temporarily switch DP group to replica group for scatter
    old_group = distributed_optimizer.data_parallel_group
    old_group_gloo = distributed_optimizer.data_parallel_group_gloo
    distributed_optimizer.data_parallel_group = replica_group
    distributed_optimizer.data_parallel_group_gloo = replica_group_gloo

    try:
        # Step 5: Call mcore's built-in scatter method
        # load_parameter_state_from_dp_zero() will:
        # - Use parameter_state from local rank 0
        # - Scatter to all ranks in the (now replica) group
        # - Each rank gets its corresponding shard
        distributed_optimizer.load_parameter_state_from_dp_zero(parameter_state)
        logger.info('[TTP RESUME] Optimizer parameter state scattered to replica group')
    finally:
        # Step 6: Restore original DP group (size=4)
        distributed_optimizer.data_parallel_group = old_group
        distributed_optimizer.data_parallel_group_gloo = old_group_gloo
        logger.info('[TTP RESUME] Restored original DP group')


def _restore_scheduler_and_rng(trainer, common):
    """Restore scheduler state and RNG states."""
    # Restore scheduler
    if not getattr(trainer.args, 'no_load_optim', False):
        scheduler_state = common.get('scheduler')
        if scheduler_state is not None and trainer.opt_param_scheduler is not None:
            trainer.opt_param_scheduler.load_state_dict(scheduler_state)
            logger.info('[TTP RESUME] Restored scheduler state')

    # Restore RNG
    if getattr(trainer.args, 'no_load_rng', False):
        logger.warning('[TTP RESUME] --no_load_rng=true; RNG restore skipped')
        return

    rng_state = common.get('rng_state')
    if rng_state is None:
        # Fallback to old format (torch_rng_state, npu_rng_state)
        logger.warning('[TTP RESUME] rng_state not found, trying legacy format')
        if 'torch_rng_state' in common:
            torch.set_rng_state(common['torch_rng_state'])
        if 'npu_rng_state' in common:
            try:
                if hasattr(torch, 'npu') and torch.npu.is_available():
                    torch.npu.set_rng_state(common['npu_rng_state'])
            except Exception:
                pass
        return

    random.setstate(rng_state['random_rng_state'])
    np.random.set_state(rng_state['np_rng_state'])
    torch.set_rng_state(rng_state['torch_rng_state'])

    device_rng_state = rng_state.get('device_rng_state')
    if device_rng_state is not None:
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.set_rng_state(device_rng_state)
        elif torch.cuda.is_available():
            torch.cuda.set_rng_state(device_rng_state)

    tracker_state = rng_state.get('rng_tracker_states')
    if tracker_state is not None:
        try:
            from megatron.core import tensor_parallel
            tensor_parallel.get_cuda_rng_tracker().set_states(tracker_state)
        except Exception as e:
            logger.warning(f'[TTP RESUME] Failed to restore rng_tracker: {e}')

    logger.info('[TTP RESUME] Restored RNG states (random, numpy, torch, device, tracker)')


# ========== Main Entry Point ==========

def load_ttp_model_and_common_state(trainer, checkpoint_dir):
    """Main entry: Restore model, optimizer, scheduler, RNG from emergency checkpoint.

    Implements T2 + T3 + T4 from the seven-stage manual:
    - T2: Restore model and common state (iteration, consumed_samples)
    - T3: Restore optimizer with re-sharding to replica groups
    - T4: Restore scheduler and RNG

    Args:
        trainer: BaseMegatronTrainer instance
        checkpoint_dir: Path to emergency checkpoint directory

    Returns:
        Dict with checkpoint payload (manifest, common, model, optimizer)
    """
    logger.info(f'[TTP RESUME] Starting emergency checkpoint restore from {checkpoint_dir}')

    # T2: Load checkpoint (without optimizer for now)
    payload = load_emergency_checkpoint(
        checkpoint_dir,
        map_location='cpu',
        load_optimizer=False,
    )
    _validate_runtime_layout(trainer.args, payload['manifest'])
    _restore_models(trainer, payload['model'])

    common = payload['common']

    # T3: Restore optimizer (reads optimizer_0.pt inside)
    _restore_optimizer(trainer, checkpoint_dir, common)

    # T4: Restore scheduler and RNG
    _restore_scheduler_and_rng(trainer, common)

    # Restore iteration and consumed samples
    iteration = int(common['iteration'])
    consumed = int(common.get('consumed_train_samples', 0))
    trainer.state.iteration = iteration
    trainer.state.consumed_train_samples = consumed
    trainer.args.consumed_train_samples = consumed

    logger.info(
        '[TTP RESUME] Emergency checkpoint restore completed: '
        'iteration=%s, consumed_samples=%s',
        iteration, consumed,
    )
    return payload
