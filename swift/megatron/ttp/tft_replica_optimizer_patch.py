# TTP Replica Optimizer Patch for ms-swift Megatron backend
# Uses buffer group swap + method binding (not composition wrapper)
# Replaces the old TTPReplicaOptimizerWrapper approach
# Date: 2026-07-24

import torch
from logging import getLogger

ttp_logger = getLogger(__name__)

try:
    from mindio_ttp.framework_ttp import tft_start_updating_os, tft_end_updating_os
    from mindio_ttp.utils import tft_set_update_start_time, tft_set_update_end_time
except ImportError:
    ttp_logger.warning("mindio_ttp not available, TTP update hooks disabled")
    def tft_start_updating_os(): pass
    def tft_end_updating_os(step): pass
    def tft_set_update_start_time(): pass
    def tft_set_update_end_time(): pass

# Global TTP state
_TTP_STATE = {
    'enabled': False,
    'os_shard_group': None,
}


# ========== Buffer group swap functions ==========

def swap_buffer_groups(wrapped_models, os_shard_group):
    """Temporarily swap data_parallel_group on all buffers to os_shard_group.

    Called before optimizer creation so that _build_model_gbuf_range uses
    os_shard_group (replica subgroup) instead of the original DP group.
    This makes each replica subgroup hold full optimizer state.

    Args:
        wrapped_models: List of DDP models (self.wrapped_models from trainer)
        os_shard_group: The replica subgroup process group

    Returns:
        list: Saved original data_parallel_group for each buffer (for restoration)
    """
    saved_groups = []
    for model in wrapped_models:
        # Dense buffers
        buffers = getattr(model, 'buffers', [])
        for buffer in buffers:
            saved_groups.append(buffer.data_parallel_group)
            buffer.data_parallel_group = os_shard_group

        # Expert parallel buffers (MoE)
        expert_buffers = getattr(model, 'expert_parallel_buffers', [])
        for buffer in expert_buffers:
            saved_groups.append(buffer.data_parallel_group)
            buffer.data_parallel_group = os_shard_group

    ttp_logger.info(
        f"[TTP] Swapped buffer groups to os_shard_group "
        f"(size={torch.distributed.get_world_size(os_shard_group)}, "
        f"{len(saved_groups)} buffers)"
    )
    return saved_groups


def restore_buffer_groups(wrapped_models, saved_groups):
    """Restore original data_parallel_group on all buffers.

    Called after optimizer creation so gradient all-reduce uses the original
    DP group during training.

    Args:
        wrapped_models: List of DDP models
        saved_groups: List of original groups from swap_buffer_groups()
    """
    idx = 0
    for model in wrapped_models:
        # Dense buffers
        buffers = getattr(model, 'buffers', [])
        for buffer in buffers:
            buffer.data_parallel_group = saved_groups[idx]
            idx += 1

        # Expert parallel buffers (MoE)
        expert_buffers = getattr(model, 'expert_parallel_buffers', [])
        for buffer in expert_buffers:
            buffer.data_parallel_group = saved_groups[idx]
            idx += 1

    ttp_logger.info(f"[TTP] Restored buffer groups ({idx} buffers)")


# ========== Patch optimizer creation ==========

def patch_optimizer_creation(trainer_cls):
    """Patch get_optimizer_and_scheduler to swap buffer groups during optimizer creation.

    This must be called BEFORE prepare_trainer() so the patch is active when
    BaseMegatronTrainer.__init__ calls get_optimizer_and_scheduler().

    Args:
        trainer_cls: The trainer class (e.g., BaseMegatronTrainer)
    """
    orig_method = trainer_cls.get_optimizer_and_scheduler

    def patched_get_optimizer_and_scheduler(self):
        if _TTP_STATE['enabled'] and _TTP_STATE['os_shard_group'] is not None:
            ttp_logger.info("[TTP] Patching optimizer creation: swapping buffer groups")
            saved = swap_buffer_groups(self.wrapped_models, _TTP_STATE['os_shard_group'])
            try:
                result = orig_method(self)
            finally:
                restore_buffer_groups(self.wrapped_models, saved)
            ttp_logger.info("[TTP] Optimizer created with os_shard_group sharding")
            return result
        return orig_method(self)

    trainer_cls.get_optimizer_and_scheduler = patched_get_optimizer_and_scheduler
    ttp_logger.info(f"[TTP] Patched {trainer_cls.__name__}.get_optimizer_and_scheduler")


def _set_dump_args(optimizer, rank, step, rank_list):
    """Set parameters for emergency save."""
    optimizer.save_args['step'] = step
    optimizer.save_args['rank'] = rank
    optimizer.save_args['rank_list'] = rank_list
    optimizer.error_dump = True
    optimizer.current_step = step

    ttp_logger.info(
        f"[TTP] set_dump_args: rank={rank}, step={step}, rank_list={rank_list}"
    )


def _need_write_file(optimizer):
    """Check if current rank should write checkpoint file."""
    if not optimizer.error_dump:
        return False

    cur_rank = torch.distributed.get_rank()
    save_rank = optimizer.save_args.get('rank')
    should_write = (save_rank == cur_rank)

    if should_write:
        ttp_logger.info(f"[TTP] Rank {cur_rank} will write checkpoint")

    return should_write


def _save_parameter_state(optimizer, filename):
    """Save optimizer parameter state."""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if optimizer.error_dump:
        cur_rank = torch.distributed.get_rank()
        save_rank = optimizer.save_args.get('rank')

        if cur_rank == save_rank:
            state_dict = _get_parameter_state_dp_zero_for_ttp(optimizer)
            torch.save(state_dict, filename)
            ttp_logger.info(
                f"[TTP] Rank {cur_rank} saved optimizer state to {filename}"
            )
    else:
        if hasattr(optimizer, 'save_parameter_state'):
            optimizer.save_parameter_state(filename)
        else:
            state_dict = optimizer.state_dict()
            torch.save(state_dict, filename)


def _get_parameter_state_dp_zero_for_ttp(optimizer):
    """Gather optimizer state from all surviving ranks for TTP."""
    if hasattr(optimizer, 'get_parameter_state_dp_zero'):
        return optimizer.get_parameter_state_dp_zero()

    return {
        'optimizer': optimizer.state_dict(),
        'step': optimizer.current_step,
    }


def _begin_to_update(optimizer, iteration):
    """Notify TTP that optimizer update is starting."""
    optimizer.current_step = iteration
    tft_start_updating_os()
    tft_set_update_start_time()


def _end_to_update(optimizer):
    """Notify TTP that optimizer update is complete."""
    tft_set_update_end_time()
    optimizer.current_step += 1
    tft_end_updating_os(optimizer.current_step)


def _sync_gather_all_model_params(optimizer, force_sync=False):
    """Force synchronize and gather all model parameters across ranks."""
    if hasattr(optimizer, 'model_chunks'):
        for model_chunk in optimizer.model_chunks:
            if hasattr(model_chunk, 'start_param_sync'):
                model_chunk.start_param_sync(force_sync=force_sync)
    else:
        ttp_logger.warning(
            "[TTP] optimizer.model_chunks not found, sync_gather_all_model_params skipped"
        )


def _show_replica_inc_memsize(optimizer):
    """Calculate and log memory increase due to replica optimizer."""
    if not hasattr(optimizer, 'gbuf_ranges'):
        ttp_logger.warning("[TTP] optimizer.gbuf_ranges not found, cannot calculate memory increase")
        return

    fp32_bytes = 4
    bytes_to_gb = 1024 * 1024 * 1024

    # Use os_shard_group size for local numel calculation
    if hasattr(optimizer, 'os_shard_group') and optimizer.os_shard_group is not None:
        data_parallel_world_size = torch.distributed.get_world_size(optimizer.os_shard_group)
    else:
        data_parallel_world_size = len(optimizer.ori_dp_list)

    total_local_numel = 0
    for gbuf_idx, gbuf_range_maps in enumerate(optimizer.gbuf_ranges):
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            for bucket_idx, _ in enumerate(gbuf_range_map_for_all_buckets):
                gbuf_world_numel = optimizer.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                total_local_numel += gbuf_local_numel

    total_inc_bytes = total_local_numel * 3 * fp32_bytes * (optimizer.replica_num - 1) / optimizer.replica_num
    total_inc_gb = total_inc_bytes / bytes_to_gb

    ttp_logger.warning(
        f"[TTP] Replica optimizer increases Memory On Chip usage by: {total_inc_gb:.4f} GB"
    )


# ========== Bind TTP methods to optimizer instance ==========

def bind_ttp_methods(optimizer, ori_dp_group, replica_num, os_shard_group=None, initial_step=0):
    """Bind TTP methods to an existing optimizer instance.

    Replaces the old TTPReplicaOptimizerWrapper composition approach.
    Instead of wrapping the optimizer, we add TTP state and methods directly
    to the optimizer instance.

    Args:
        optimizer: mcore DistributedOptimizer instance (already created with os_shard_group sharding)
        ori_dp_group: Original data parallel process group (for gradient sync)
        replica_num: Number of replicas (default 2)
        os_shard_group: The replica subgroup (for memory calculation)
        initial_step: Initial training iteration (from trainer.state.iteration, for resume)
    """
    # Set TTP state on optimizer instance
    optimizer.ori_dp_group = ori_dp_group
    optimizer.ori_dp_list = torch.distributed.get_process_group_ranks(ori_dp_group) if ori_dp_group else [0]
    optimizer.replica_num = replica_num
    optimizer.os_shard_group = os_shard_group
    optimizer.error_dump = False
    optimizer.save_args = {}
    optimizer.current_step = initial_step

    # Phase 5: Initialize transaction state
    from .tft_optimizer_state import init_optimizer_transaction_state
    init_optimizer_transaction_state(optimizer, initial_step=initial_step)

    # Validate replica configuration
    dp_world_size = len(optimizer.ori_dp_list)
    if dp_world_size == 1:
        ttp_logger.warning(f"[TTP] DP world size is 1, TTP replica not effective")
    elif dp_world_size % replica_num != 0:
        raise ValueError(
            f"TTP: DP world size ({dp_world_size}) must be divisible by "
            f"replica_num ({replica_num})"
        )

    # Bind TTP methods
    optimizer.set_dump_args = lambda r, s, rl: _set_dump_args(optimizer, r, s, rl)
    optimizer.need_write_file = lambda: _need_write_file(optimizer)
    optimizer.save_parameter_state = lambda fn: _save_parameter_state(optimizer, fn)
    optimizer.get_parameter_state_dp_zero_for_ttp = lambda: _get_parameter_state_dp_zero_for_ttp(optimizer)
    optimizer.begin_to_update = lambda it: _begin_to_update(optimizer, it)
    optimizer.end_to_update = lambda: _end_to_update(optimizer)
    optimizer.sync_gather_all_model_params = lambda fs=False: _sync_gather_all_model_params(optimizer, fs)
    optimizer.show_replica_inc_memsize = lambda: _show_replica_inc_memsize(optimizer)

    # Phase 5: Wrap step() with transaction state tracking
    # Replaces old patched_step that incorrectly called end_to_update() on failure
    from .tft_optimizer_state import wrap_step_with_transaction_state
    orig_step = optimizer.step
    optimizer.step = wrap_step_with_transaction_state(optimizer, orig_step)

    # Log memory increase
    optimizer.show_replica_inc_memsize()

    ttp_logger.info(
        f"[TTP] TTP methods bound to optimizer "
        f"(replica_num={replica_num}, dp_world_size={dp_world_size})"
    )
