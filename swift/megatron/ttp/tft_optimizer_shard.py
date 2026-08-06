# Phase 6: Export Local Optimizer Shard
# 从 optimizer 中提取本地 shard 的数据（param, exp_avg, exp_avg_sq）
#
# 核心功能：
# 1. build_local_optimizer_shard: 从 optimizer 中提取本地 shard 的数据
# 2. gather_and_reorder_local_shards: 使用 Gloo gather 收集完整 optimizer state
# 3. 支持 Adam/AdamW 优化器（param, exp_avg, exp_avg_sq）
#
# 数据来源：
# - optimizer.gbuf_ranges: 定义每个 rank 负责的参数范围
# - optimizer.buffers: 实际的参数和梯度缓冲区
# - optimizer 内部状态: main_param, exp_avg, exp_avg_sq

import torch
from logging import getLogger
from typing import Dict, List, Any, Optional

ttp_logger = getLogger(__name__)


def _unwrap_distributed_optimizer(optimizer):
    """Unwrap ChainedOptimizer to get the DistributedOptimizer.

    First phase only supports single optimizer (dense/full SFT).
    MoE with multiple chained optimizers is not supported yet.
    """
    if hasattr(optimizer, 'chained_optimizers'):
        if len(optimizer.chained_optimizers) != 1:
            raise ValueError(
                "[TTP] Emergency save currently supports one optimizer only, "
                f"got {len(optimizer.chained_optimizers)}"
            )
        return optimizer.chained_optimizers[0]
    return optimizer


def _get_optimizer_shard_world_size(optimizer):
    """Get the optimizer shard world size.

    This is the size of os_shard_group (replica subgroup), NOT the global DP size.
    For 4DP/replica=2, this returns 2 (not 4).

    The optimizer was created with buffer groups swapped to os_shard_group,
    so its internal sharding is based on os_shard_group size.
    """
    os_shard_group = getattr(optimizer, 'os_shard_group', None)
    if os_shard_group is not None:
        return torch.distributed.get_world_size(os_shard_group)
    # Fallback: use DP group size (should not happen with V3 patch enabled)
    ttp_logger.warning("[TTP] os_shard_group not found on optimizer, using global world_size")
    return torch.distributed.get_world_size()


def _get_main_param_and_optimizer_states(optimizer, model_param):
    """Extract main parameter and optimizer states for a given model parameter.

    MCore DistributedOptimizer binds Adam states to FP32 `main_param`,
    not to the original `model_param`. This function finds the correct
    main_param and extracts its state.

    Args:
        optimizer: MCore DistributedOptimizer instance
        model_param: Model parameter (torch.nn.Parameter)

    Returns:
        Dict with keys: 'param', 'exp_avg', 'exp_avg_sq'
    """
    # Try to use MCore's method if available
    if hasattr(optimizer, '_get_main_param_and_optimizer_states'):
        return optimizer._get_main_param_and_optimizer_states(model_param)

    # Fallback: manually extract from optimizer state
    if not hasattr(optimizer, 'model_param_group_index_map'):
        raise ValueError("[TTP] optimizer.model_param_group_index_map not found")

    group_index, group_order = optimizer.model_param_group_index_map[model_param]
    main_param = optimizer.optimizer.param_groups[group_index]['params'][group_order]
    state = optimizer.optimizer.state[main_param]

    return {
        'param': main_param,
        'exp_avg': state['exp_avg'],
        'exp_avg_sq': state['exp_avg_sq'],
    }


def build_local_optimizer_shard(optimizer, replica_group_size):
    """Extract local optimizer shard data from this rank's optimizer.

    This is the per-rank extraction logic factored out of
    gather_and_reorder_local_shards(). Each rank calls this to get
    its own shard data, then writes it directly to disk (parallel write)
    instead of gathering to a writer rank.

    Args:
        optimizer: MCore DistributedOptimizer instance (or ChainedOptimizer)
        replica_group_size: Number of ranks in the replica subgroup
            (= len(selected_ranks) = shard_world_size)

    Returns:
        Dict with local shard data:
        {
            gbuf_idx: {
                dtype: {
                    'param': local_cpu_tensor,
                    'exp_avg': local_cpu_tensor,
                    'exp_avg_sq': local_cpu_tensor,
                    'numel_unpadded': int,
                }
            }
        }
    """
    optimizer = _unwrap_distributed_optimizer(optimizer)

    state = {}

    for gbuf_idx, gbuf_range_maps in enumerate(optimizer.gbuf_ranges):
        dtype_state = {}

        for dtype, bucket_range_maps in gbuf_range_maps.items():
            buffer_numel_unpadded = optimizer.buffers[gbuf_idx].numel_unpadded

            local_shards = {}
            offset = 0

            for bucket_idx, gbuf_range_map in enumerate(bucket_range_maps):
                bucket = optimizer.buffers[gbuf_idx].buckets[bucket_idx]
                world_numel = bucket.grad_data.numel()
                local_numel = world_numel // replica_group_size
                world_numel_unpadded = bucket.numel_unpadded

                # Build local shard for this bucket
                bucket_shards = {
                    key: torch.zeros(local_numel, dtype=torch.float32, device='cpu')
                    for key in ('param', 'exp_avg', 'exp_avg_sq')
                }

                for model_param, range_map in gbuf_range_map['param_map'].items():
                    tensors = _get_main_param_and_optimizer_states(optimizer, model_param)
                    start = range_map['gbuf_local'].start
                    end = range_map['gbuf_local'].end
                    for key in bucket_shards:
                        bucket_shards[key][start:end].copy_(tensors[key].detach().cpu())

                # Store bucket shards with offset info for merge
                if 'buckets' not in local_shards:
                    local_shards['buckets'] = []
                local_shards['buckets'].append({
                    'param': bucket_shards['param'],
                    'exp_avg': bucket_shards['exp_avg'],
                    'exp_avg_sq': bucket_shards['exp_avg_sq'],
                    'offset': offset,
                    'world_numel_unpadded': world_numel_unpadded,
                })

                offset += world_numel_unpadded

            local_shards['numel_unpadded'] = buffer_numel_unpadded
            dtype_state[dtype] = local_shards
        state[gbuf_idx] = dtype_state

    return state


def merge_shards_to_dp_zero_state(shard_dict, optimizer, replica_group_size):
    """Merge multiple shard files into complete DP-zero state.

    This is the inverse of build_local_optimizer_shard(): it concatenates
    local shards from all selected ranks (ordered by shard_idx) to rebuild
    the complete optimizer state, matching MCore's get_parameter_state_dp_zero()
    format.

    Used by load_emergency_checkpoint() when loading a parallel-written
    checkpoint (multiple optimizer_shard_*.pt files).

    Args:
        shard_dict: {shard_idx: shard_data} where shard_data is the output
                    of build_local_optimizer_shard()
        optimizer: MCore DistributedOptimizer instance (for gbuf_ranges)
        replica_group_size: Number of shards (= len(selected_ranks))

    Returns:
        Complete optimizer state dict (same format as
        gather_and_reorder_local_shards() output on writer rank):
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
    optimizer = _unwrap_distributed_optimizer(optimizer)

    state = {'buckets_coalesced': True}

    for gbuf_idx, gbuf_range_maps in enumerate(optimizer.gbuf_ranges):
        dtype_state = {}

        for dtype, bucket_range_maps in gbuf_range_maps.items():
            buffer_numel_unpadded = optimizer.buffers[gbuf_idx].numel_unpadded

            # Allocate complete tensors
            world_tensors = {
                key: torch.zeros(buffer_numel_unpadded, dtype=torch.float32, device='cpu')
                for key in ('param', 'exp_avg', 'exp_avg_sq')
            }
            world_tensors['numel_unpadded'] = buffer_numel_unpadded

            # Concatenate shards in logical shard_idx order
            for shard_idx in sorted(shard_dict.keys()):
                shard_data = shard_dict[shard_idx]
                shard_dtype_state = shard_data[gbuf_idx][dtype]
                shard_buckets = shard_dtype_state['buckets']

                for bucket_entry in shard_buckets:
                    offset = bucket_entry['offset']
                    world_numel_unpadded = bucket_entry['world_numel_unpadded']

                    for key in ('param', 'exp_avg', 'exp_avg_sq'):
                        shard_tensor = bucket_entry[key]
                        copy_len = min(world_numel_unpadded, shard_tensor.numel())
                        world_tensors[key][offset:offset + copy_len].copy_(
                            shard_tensor[:copy_len]
                        )

            dtype_state[dtype] = world_tensors
        state[gbuf_idx] = dtype_state

    return state


def gather_and_reorder_local_shards(
    optimizer,
    selected_ranks,
    writer_rank,
    save_group,
    group_ranks,
    shard_map,
):
    """Gather optimizer shards and rebuild MCore DP-zero state on writer.

    This function uses torch.distributed.gather on the Gloo save_group to
    collect optimizer state from all selected ranks. The gathered shards
    are reordered by logical shard index (shard 0, shard 1, ...) and
    concatenated to form the complete optimizer state.

    Args:
        optimizer: MCore DistributedOptimizer instance (or ChainedOptimizer)
        selected_ranks: Tuple of selected ranks (canonicalized to shard order)
        writer_rank: Rank that will write the checkpoint
        save_group: Pre-created Gloo group for gathering
        group_ranks: Tuple of ranks in the ProcessGroup (stable global-rank order)
        shard_map: Dict mapping rank -> shard index

    Returns:
        Complete optimizer state dict (only on writer_rank, None on other ranks)

    The returned state structure matches MCore's get_parameter_state_dp_zero()
    coalesced format:
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
    shard_world_size = _get_optimizer_shard_world_size(optimizer)
    optimizer = _unwrap_distributed_optimizer(optimizer)
    cur_rank = torch.distributed.get_rank()
    selected_ranks = tuple(selected_ranks)
    group_ranks = tuple(group_ranks)

    if cur_rank not in selected_ranks:
        return None

    replica_group_size = len(selected_ranks)
    if shard_world_size != replica_group_size:
        raise ValueError(
            f"[TTP] Optimizer shard group size {shard_world_size} does not match "
            f"selected rank count {replica_group_size}"
        )

    # Map global rank -> index in the ProcessGroup
    recv_index_by_rank = {rank: index for index, rank in enumerate(group_ranks)}
    if set(group_ranks) != set(selected_ranks):
        raise ValueError(
            f"[TTP] ProcessGroup ranks {group_ranks} do not match selected ranks {selected_ranks}"
        )

    state = {'buckets_coalesced': True}

    for gbuf_idx, gbuf_range_maps in enumerate(optimizer.gbuf_ranges):
        dtype_state = {}

        for dtype, bucket_range_maps in gbuf_range_maps.items():
            buffer_numel_unpadded = optimizer.buffers[gbuf_idx].numel_unpadded

            # Writer allocates complete (DP-zero) tensors
            world_tensors = {}
            if cur_rank == writer_rank:
                world_tensors = {
                    key: torch.zeros(buffer_numel_unpadded, dtype=torch.float32, device='cpu')
                    for key in ('param', 'exp_avg', 'exp_avg_sq')
                }
                world_tensors['numel_unpadded'] = buffer_numel_unpadded

            offset = 0
            for bucket_idx, gbuf_range_map in enumerate(bucket_range_maps):
                bucket = optimizer.buffers[gbuf_idx].buckets[bucket_idx]
                world_numel = bucket.grad_data.numel()
                local_numel = world_numel // replica_group_size
                world_numel_unpadded = bucket.numel_unpadded

                # Build local shard for this bucket
                local_shards = {
                    key: torch.zeros(local_numel, dtype=torch.float32, device='cpu')
                    for key in ('param', 'exp_avg', 'exp_avg_sq')
                }

                for model_param, range_map in gbuf_range_map['param_map'].items():
                    tensors = _get_main_param_and_optimizer_states(optimizer, model_param)
                    start = range_map['gbuf_local'].start
                    end = range_map['gbuf_local'].end
                    for key in local_shards:
                        local_shards[key][start:end].copy_(tensors[key].detach().cpu())

                # Gather each key (param, exp_avg, exp_avg_sq) separately
                for key, send_tensor in local_shards.items():
                    recv_tensors = None
                    if cur_rank == writer_rank:
                        recv_tensors = [torch.empty_like(send_tensor) for _ in group_ranks]

                    torch.distributed.gather(
                        send_tensor,
                        gather_list=recv_tensors,
                        dst=writer_rank,
                        group=save_group,
                    )

                    if cur_rank == writer_rank:
                        # Reorder: ProcessGroup order -> logical shard order
                        logical_tensors = [
                            recv_tensors[recv_index_by_rank[rank]]
                            for rank in selected_ranks
                        ]
                        concatenated = torch.cat(logical_tensors)
                        # Copy unpadded portion (drop padding)
                        copy_len = min(world_numel_unpadded, concatenated.numel())
                        world_tensors[key][offset:offset + copy_len].copy_(
                            concatenated[:copy_len]
                        )

                offset += world_numel_unpadded

            dtype_state[dtype] = world_tensors
        state[gbuf_idx] = dtype_state

    if cur_rank == writer_rank:
        ttp_logger.info(
            f"[TTP] Rank {cur_rank} gathered complete optimizer state from {selected_ranks}"
        )
        return state
    return None
