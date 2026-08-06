"""
Phase 4: Emergency save groups for fault-tolerant checkpoint.

Based on MindSpeed-LLM's approach (avoids combinatorial explosion):
1. At init time: only create replica sub-groups (replica_num groups)
2. At fault time: dynamically create dump_group using use_local_synchronization=True

Previous approach (pre-create all combinations) had combinatorial explosion:
  4DP/replica=2: 4 groups (OK)
  8DP/replica=2: 35 groups (bad)
  16DP/replica=2: 6435 groups (infeasible)

New approach (dynamic creation):
  Any DP/replica: only replica_num groups at init + 1 dynamic group at fault
"""

import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Dict
from logging import getLogger


ttp_logger = getLogger(__name__)

# Global state
_DUMP_WORLD_GROUP: Optional[dist.ProcessGroup] = None  # Dynamically created dump group
_REPLICA_GROUPS: Dict[int, Tuple[List[int], dist.ProcessGroup]] = {}  # Pre-created replica groups
_REPLICA_GROUPS_GLOO: Dict[int, Tuple[List[int], dist.ProcessGroup]] = {}  # Gloo version


def initialize_emergency_save_groups(
    dp_ranks: List[int],
    replica_num: int,
) -> None:
    """Initialize replica sub-groups for emergency save.

    Only creates replica_num sub-groups (NOT all combinations).
    This follows MindSpeed-LLM's approach to avoid combinatorial explosion.

    For 4DP/replica=2: creates 2 groups [0,1] and [2,3]
    For 8DP/replica=2: creates 2 groups [0,1,2,3] and [4,5,6,7]

    Args:
        dp_ranks: List of global ranks in the DP group
        replica_num: Number of replicas
    """
    global _REPLICA_GROUPS, _REPLICA_GROUPS_GLOO

    if _REPLICA_GROUPS:
        return  # Already initialized

    cur_rank = dist.get_rank()

    # Divide dp_ranks into replica_num sub-groups
    if len(dp_ranks) % replica_num != 0:
        raise ValueError(
            f"dp_ranks size {len(dp_ranks)} must be divisible by replica_num {replica_num}"
        )

    replica_group_size = len(dp_ranks) // replica_num
    replica_lists = [
        dp_ranks[i * replica_group_size : (i + 1) * replica_group_size]
        for i in range(replica_num)
    ]

    ttp_logger.info(f"[TTP] Building {replica_num} replica groups (dp_ranks={dp_ranks}):")
    for i, replica_list in enumerate(replica_lists):
        ttp_logger.info(f"[TTP]   replica_group[{i}]: {replica_list}")

    # Create process groups for each replica sub-group
    # Following MindSpeed-LLM: create both NCCL and Gloo groups
    for i, replica_list in enumerate(replica_lists):
        # NCCL group (for gradient sync within replica)
        nccl_group = dist.new_group(replica_list, use_local_synchronization=True)
        # Gloo group (for emergency save coordination)
        gloo_group = dist.new_group(
            replica_list, backend='gloo', use_local_synchronization=False
        )

        # Only store groups that current rank belongs to
        if cur_rank in replica_list:
            _REPLICA_GROUPS[i] = (replica_list, nccl_group)
            _REPLICA_GROUPS_GLOO[i] = (replica_list, gloo_group)

    ttp_logger.info(
        f"[TTP] Pre-created {len(replica_lists)} replica groups for emergency save "
        f"(avoiding combinatorial explosion)"
    )


def create_dump_group_for_ranks(
    dump_group_ranks: List[int],
) -> Tuple[dist.ProcessGroup, Tuple[int, ...]]:
    """Dynamically create a dump group for the given ranks.

    This is called at FAULT TIME, not at init time.
    Uses use_local_synchronization=True to avoid hanging on fault ranks.

    Key insight from MindSpeed-LLM:
    - use_local_synchronization=True means only participating ranks need to call new_group
    - This avoids the problem where fault ranks cannot participate in new_group
    - backend='gloo' bypasses HCCL which may be broken after fault

    Args:
        dump_group_ranks: List of ranks to include in the dump group

    Returns:
        Tuple of (dump_group, group_ranks_tuple)
    """
    global _DUMP_WORLD_GROUP

    cur_rank = dist.get_rank()
    group_ranks = tuple(sorted(set(dump_group_ranks)))
    if len(group_ranks) != len(dump_group_ranks):
        raise ValueError(f"Duplicate ranks in dump group: {dump_group_ranks}")
    if cur_rank not in group_ranks:
        raise RuntimeError(
            f"Rank {cur_rank} must not create dump group for non-members {group_ranks}"
        )

    ttp_logger.info(
        f"[TTP] Creating dynamic Gloo dump group for ranks: {group_ranks}"
    )

    # Create dump group with Gloo backend and local synchronization
    # Gloo backend bypasses HCCL which may be broken after fault
    # use_local_synchronization=True means only participating ranks call new_group
    dump_group = dist.new_group(
        ranks=list(group_ranks),
        backend='gloo',
        use_local_synchronization=True,
    )
    _DUMP_WORLD_GROUP = dump_group
    ttp_logger.info(f"[TTP] Rank {cur_rank} joined Gloo dump group successfully")
    return dump_group, group_ranks


def destroy_dump_group(group: Optional[dist.ProcessGroup]) -> None:
    """Destroy the temporary fault-time group after optimizer collection.

    This should be called in a finally block to ensure cleanup on both
    success and failure paths. Prevents ProcessGroup accumulation across
    multiple fault saves.
    """
    global _DUMP_WORLD_GROUP
    if group is not None:
        try:
            dist.destroy_process_group(group)
        except Exception as e:
            ttp_logger.warning(f"[TTP] Failed to destroy dump group: {e}")
    if _DUMP_WORLD_GROUP is group:
        _DUMP_WORLD_GROUP = None


def get_dump_world_group() -> Optional[dist.ProcessGroup]:
    """Get the current dump world group.

    Returns:
        The dump world group if set, None otherwise.
    """
    return _DUMP_WORLD_GROUP


def set_dump_world_group(group: Optional[dist.ProcessGroup]) -> None:
    """Set the dump world group.

    Args:
        group: Process group to set as dump world group
    """
    global _DUMP_WORLD_GROUP
    _DUMP_WORLD_GROUP = group


def get_replica_groups() -> Dict[int, Tuple[List[int], dist.ProcessGroup]]:
    """Get all pre-created replica groups (for debugging).

    Returns:
        Dict mapping replica_idx to (rank_list, ProcessGroup)
    """
    return _REPLICA_GROUPS.copy()


def get_replica_groups_gloo() -> Dict[int, Tuple[List[int], dist.ProcessGroup]]:
    """Get all pre-created replica Gloo groups (for debugging).

    Returns:
        Dict mapping replica_idx to (rank_list, GlooProcessGroup)
    """
    return _REPLICA_GROUPS_GLOO.copy()


def find_replica_group_for_rank(
    rank: int,
) -> Optional[Tuple[int, List[int], dist.ProcessGroup]]:
    """Find which replica group a rank belongs to.

    Args:
        rank: Global rank to look up

    Returns:
        Tuple of (replica_idx, rank_list, ProcessGroup) or None if not found
    """
    for replica_idx, (rank_list, group) in _REPLICA_GROUPS.items():
        if rank in rank_list:
            return replica_idx, rank_list, group
    return None


def clear_emergency_save_groups() -> None:
    """Clear all groups.

    This should be called during cleanup to avoid memory leaks.
    """
    global _REPLICA_GROUPS, _REPLICA_GROUPS_GLOO, _DUMP_WORLD_GROUP
    _REPLICA_GROUPS.clear()
    _REPLICA_GROUPS_GLOO.clear()
    _DUMP_WORLD_GROUP = None


def get_emergency_save_group_for_save_info(
    dp_ranks: List[int],
    selected_ranks: List[int],
    replica_num: int,
) -> Tuple[Tuple[int, ...], dist.ProcessGroup, Tuple[int, ...]]:
    """Get or create emergency save group for the given save_info.

    Dynamically creates a Gloo dump group instead of looking up
    a pre-created group. This avoids combinatorial explosion.

    Args:
        dp_ranks: List of global ranks in the DP group
        selected_ranks: List of selected ranks from save_info
        replica_num: Number of replicas

    Returns:
        Tuple of (selected_ranks_tuple, dump_group, group_ranks_tuple)
    """
    dump_group, group_ranks = create_dump_group_for_ranks(selected_ranks)
    return tuple(selected_ranks), dump_group, group_ranks


# Backward compatibility: keep old function name but mark as deprecated
def get_emergency_save_group(selected_ranks: Tuple[int, ...]) -> dist.ProcessGroup:
    """[DEPRECATED] Get pre-created group for selected ranks.

    This function is kept for backward compatibility but should not be used.
    Use create_dump_group_for_ranks() instead for dynamic group creation.
    """
    raise RuntimeError(
        "get_emergency_save_group is deprecated. "
        "Use create_dump_group_for_ranks() for dynamic group creation. "
        "This avoids the combinatorial explosion problem."
    )


def get_all_emergency_groups() -> Dict[Tuple[int, ...], dist.ProcessGroup]:
    """[DEPRECATED] Get all pre-created emergency save groups.

    This function is kept for backward compatibility. The new implementation
    does not pre-create all combinations.
    """
    ttp_logger.warning(
        "[TTP] get_all_emergency_groups is deprecated. "
        "New implementation only creates replica_num groups at init."
    )
    return {}
