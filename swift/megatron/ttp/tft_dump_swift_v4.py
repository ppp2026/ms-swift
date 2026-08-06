# TTP Emergency Save Callback V4
# Phase 7: Emergency save callback using the new kill-rank checkpoint logic
#
# This is a replacement for the old tft_dump_swift.py that:
# 1. Dynamically creates Gloo dump_group at fault time (bypasses broken HCCL)
# 2. Uses shard map to validate and canonicalize selected ranks
# 3. Gathers optimizer state from selected ranks using Gloo backend
# 4. Writes atomic emergency checkpoint with manifest and COMPLETED marker
# 5. Only writer rank writes the checkpoint
#
# Key differences from old V3 implementation:
# - Does NOT use save_mcore_checkpoint() (which hangs after kill)
# - Does NOT use original DP group for communication (uses dynamic Gloo group)
# - Does NOT allow all ranks to write (only writer rank writes)
# - Does NOT use placeholder communication (real torch.distributed.gather)

import torch
from logging import getLogger

ttp_logger = getLogger(__name__)


def tft_save_callback_swift_v4(step: int, save_info: list, train_args, ctx):
    """Emergency save callback for ms-swift (V4 kill-rank implementation).

    Called by TTP when a fault is detected. This callback now delegates to
    save_ttp_checkpoint() which is shared with normal save_steps, ensuring
    a unified checkpoint format.

    Args:
        step: current training iteration
        save_info: list of dicts with {"type": optim_idx, "ranks": rank_list}
        train_args: overwritten by @tft_exception_handler, NOT used.
        ctx: save context (not used)
    """
    from .tft_optimizer_state import check_transaction_state_before_save
    from .tft_emergency_checkpoint import save_ttp_checkpoint
    from .ttp_init_swift import get_ttp_trainer

    cur_rank = torch.distributed.get_rank()
    trainer = get_ttp_trainer()
    if trainer is None:
        raise RuntimeError("[TTP V4] Trainer context is unavailable")
    optimizer = trainer.optimizer

    ttp_logger.info(f"[TTP V4] Rank {cur_rank} emergency save started at iteration {step}")

    # Check transaction state before save (reject if UPDATING or ABORTED)
    check_transaction_state_before_save(optimizer)

    # Parse save_info to get selected ranks
    if len(save_info) == 0:
        ttp_logger.error(f"[TTP V4] Rank {cur_rank} save_info is empty")
        return

    info = save_info[0]
    raw_selected_ranks = info.get('ranks', [cur_rank])

    ttp_logger.info(f"[TTP V4] Rank {cur_rank} received save_info: {raw_selected_ranks}")

    # Use committed step (last successfully completed step)
    committed_step = getattr(optimizer, 'ttp_committed_step', step)
    if committed_step != step:
        ttp_logger.warning(
            f"[TTP V4] Fault reported at step {step}, "
            f"saving last committed step {committed_step}"
        )

    # Delegate to unified save function with fault-injected selected_ranks
    save_ttp_checkpoint(trainer, committed_step, selected_ranks=raw_selected_ranks)

    ttp_logger.info(f"[TTP V4] Rank {cur_rank} emergency save completed at iteration {step}")


def tft_rename_callback_swift_v4(step: int, train_args):
    """Rename callback is no longer needed in V4.

    The emergency checkpoint is already written to the final directory
    with atomic rename in write_emergency_checkpoint().
    """
    cur_rank = torch.distributed.get_rank()
    ttp_logger.info(
        f"[TTP V4] Rank {cur_rank} rename callback called (no-op in V4)"
    )
