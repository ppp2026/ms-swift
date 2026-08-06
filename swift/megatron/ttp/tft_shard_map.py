"""
Phase 3: Logical optimizer shard mapping and selected rank validation.

This module provides pure functions for mapping global ranks to logical
optimizer shards and validating/canonicalizing selected save ranks.

For 4DP, replica_num=2:
  dp_ranks = [0, 1, 2, 3]
  replica_num = 2
  shard_count = 4 / 2 = 2

  shard_map = {0: 0, 1: 1, 2: 0, 3: 1}
  - rank 0 holds shard 0
  - rank 1 holds shard 1
  - rank 2 holds shard 0 (mirror of rank 0)
  - rank 3 holds shard 1 (mirror of rank 1)

If we kill rank 1, save_info might return [0, 3]:
  - rank 0 → shard 0
  - rank 3 → shard 1
  - canonicalized: [0, 3] (already in shard order)

If we kill rank 0, save_info might return [1, 2]:
  - rank 1 → shard 1
  - rank 2 → shard 0
  - canonicalized: [2, 1] (shard 0 first, then shard 1)
"""


def build_optimizer_shard_map(dp_ranks, replica_num):
    """Build mapping from global rank to logical optimizer shard index.

    Args:
        dp_ranks: List of global ranks in the DP group (e.g., [0, 1, 2, 3])
        replica_num: Number of replicas (e.g., 2)

    Returns:
        Dict mapping global_rank -> shard_idx

    Example:
        >>> build_optimizer_shard_map([0, 1, 2, 3], 2)
        {0: 0, 1: 1, 2: 0, 3: 1}
    """
    if len(dp_ranks) % replica_num != 0:
        raise ValueError(
            f'DP size ({len(dp_ranks)}) must be divisible by replica_num ({replica_num}).'
        )

    shard_count = len(dp_ranks) // replica_num
    shard_map = {}

    for replica_idx in range(replica_num):
        start = replica_idx * shard_count
        replica_ranks = dp_ranks[start:start + shard_count]
        for shard_idx, global_rank in enumerate(replica_ranks):
            shard_map[global_rank] = shard_idx

    return shard_map


def validate_selected_save_ranks(dp_ranks, selected_ranks, replica_num):
    """Validate that selected ranks cover all logical optimizer shards exactly once.

    Args:
        dp_ranks: List of global ranks in the DP group
        selected_ranks: List of selected ranks from save_info
        replica_num: Number of replicas

    Returns:
        shard_map: Dict mapping global_rank -> shard_idx

    Raises:
        ValueError: If selected ranks don't cover all shards exactly once
    """
    shard_map = build_optimizer_shard_map(dp_ranks, replica_num)
    shard_count = len(dp_ranks) // replica_num

    if len(selected_ranks) != shard_count:
        raise ValueError(
            f'Expected {shard_count} selected ranks, got {len(selected_ranks)}: {selected_ranks}'
        )

    # Check that all selected ranks are in dp_ranks
    for rank in selected_ranks:
        if rank not in shard_map:
            raise ValueError(
                f'Selected rank {rank} is not in dp_ranks {dp_ranks}'
            )

    # Check that each shard is covered exactly once
    selected_shards = [shard_map[r] for r in selected_ranks]
    if sorted(selected_shards) != list(range(shard_count)):
        raise ValueError(
            f'Selected ranks {selected_ranks} do not contain exactly one owner '
            f'for every optimizer shard: shards={selected_shards}, '
            f'expected {list(range(shard_count))}'
        )

    return shard_map


def canonicalize_selected_save_ranks(dp_ranks, selected_ranks, replica_num):
    """Canonicalize selected ranks to shard0, shard1, ... order.

    MindIO may return ranks in global rank order (e.g., [1, 2]), but we need
    them in logical shard order (e.g., [2, 1] for shard 0 and shard 1).

    Args:
        dp_ranks: List of global ranks in the DP group
        selected_ranks: List of selected ranks from save_info
        replica_num: Number of replicas

    Returns:
        Tuple of ranks ordered by shard index (shard 0 first, then shard 1, etc.)

    Example:
        >>> canonicalize_selected_save_ranks([0, 1, 2, 3], [1, 2], 2)
        (2, 1)  # rank 2 holds shard 0, rank 1 holds shard 1
    """
    shard_map = validate_selected_save_ranks(dp_ranks, selected_ranks, replica_num)

    # Sort by shard index
    return tuple(sorted(selected_ranks, key=lambda rank: shard_map[rank]))


def get_writer_rank(selected_ranks, dp_ranks, replica_num):
    """Determine which rank should write the checkpoint.

    Writer is the rank that holds shard 0 (for deterministic file output).

    Args:
        selected_ranks: Canonicalized selected ranks (shard 0 first)
        dp_ranks: List of global ranks in the DP group
        replica_num: Number of replicas

    Returns:
        Global rank of the writer
    """
    shard_map = build_optimizer_shard_map(dp_ranks, replica_num)

    # Find the rank that holds shard 0
    for rank in selected_ranks:
        if shard_map[rank] == 0:
            return rank

    raise ValueError(f'No rank in {selected_ranks} holds shard 0')
