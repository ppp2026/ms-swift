# Phase 5: Optimizer Update Transaction State Fix
# 修复 optimizer.step() 失败时仍调用 end_to_update() 的问题
#
# 问题：当前 patched_step 在异常时仍调用 end_to_update()，导致 TTP 认为更新已成功提交
# 解决：引入状态机 IDLE -> UPDATING -> COMMITTED/ABORTED，只在成功时调用 end_to_update()
#
# 状态转换：
#   IDLE -> UPDATING (step 开始)
#   UPDATING -> COMMITTED (step 成功，调用 end_to_update)
#   UPDATING -> ABORTED (step 失败，不调用 end_to_update)
#   COMMITTED -> IDLE (下一次 step 开始)
#   ABORTED -> IDLE (下一次 step 开始)

import traceback
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


# ========== Transaction State Constants ==========

STATE_IDLE = 'IDLE'
STATE_UPDATING = 'UPDATING'
STATE_COMMITTED = 'COMMITTED'
STATE_ABORTED = 'ABORTED'


def is_safe_state_for_emergency_save(optimizer):
    """Check if optimizer is in a safe state for emergency checkpoint.

    Emergency save is only allowed when:
    1. State is IDLE (between steps) or COMMITTED (step just completed)
    2. NOT in UPDATING state (step in progress)
    3. NOT in ABORTED state (step failed)

    Args:
        optimizer: Optimizer instance with ttp_update_state attribute

    Returns:
        bool: True if safe for emergency save
    """
    state = getattr(optimizer, 'ttp_update_state', STATE_IDLE)
    return state in (STATE_IDLE, STATE_COMMITTED)


def init_optimizer_transaction_state(optimizer, initial_step=0):
    """Initialize optimizer transaction state tracking.

    Args:
        optimizer: Optimizer instance
        initial_step: Initial committed step (from trainer.state.iteration)
    """
    optimizer.ttp_update_state = STATE_IDLE
    optimizer.ttp_committed_step = initial_step
    optimizer.ttp_updating_step = None

    ttp_logger.info(
        f"[TTP] Initialized transaction state: "
        f"state={STATE_IDLE}, committed_step={initial_step}"
    )


def wrap_step_with_transaction_state(optimizer, orig_step):
    """Wrap optimizer.step() with transaction state tracking.

    State transitions:
    - Before step: IDLE -> UPDATING
    - After step success: UPDATING -> COMMITTED, increment committed_step
    - After step failure: UPDATING -> ABORTED, do NOT call end_to_update()

    Args:
        optimizer: Optimizer instance
        orig_step: Original optimizer.step() method

    Returns:
        Wrapped step function
    """
    def patched_step(*args, **kwargs):
        # Transition: IDLE/COMMITTED -> UPDATING
        optimizer.ttp_update_state = STATE_UPDATING
        optimizer.ttp_updating_step = optimizer.ttp_committed_step

        # Notify TTP: update starting
        tft_start_updating_os()
        tft_set_update_start_time()

        try:
            # Call original optimizer step
            result = orig_step(*args, **kwargs)

            # Synchronize to ensure update is complete
            # Use NPU sync if available, fallback to CUDA
            try:
                if torch.npu.is_available():
                    torch.npu.synchronize()
                elif torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass

            # Transition: UPDATING -> COMMITTED
            optimizer.ttp_committed_step += 1
            optimizer.ttp_update_state = STATE_COMMITTED
            optimizer.ttp_updating_step = None

            # Notify TTP: update complete (ONLY on success)
            tft_set_update_end_time()
            tft_end_updating_os(optimizer.ttp_committed_step)

            return result

        except Exception as e:
            # Transition: UPDATING -> ABORTED
            optimizer.ttp_update_state = STATE_ABORTED
            optimizer.ttp_updating_step = None

            # DO NOT call end_to_update() on failure
            # This prevents TTP from thinking the update was committed

            ttp_logger.error(
                f"[TTP] Optimizer step failed at step {optimizer.ttp_committed_step}, "
                f"state transitioned to {STATE_ABORTED}. "
                f"Emergency save will be rejected if state is not safe."
            )
            ttp_logger.error(f"[TTP] Actual exception: {type(e).__name__}: {e}")
            ttp_logger.error(f"[TTP] Full traceback:\n{traceback.format_exc()}")

            raise e

    return patched_step


def check_transaction_state_before_save(optimizer):
    """Check transaction state before emergency save.

    Raises RuntimeError if optimizer is not in a safe state.

    Args:
        optimizer: Optimizer instance with transaction state

    Raises:
        RuntimeError: If state is UPDATING or ABORTED
    """
    state = getattr(optimizer, 'ttp_update_state', STATE_IDLE)
    committed_step = getattr(optimizer, 'ttp_committed_step', 0)

    if state == STATE_UPDATING:
        raise RuntimeError(
            f"[TTP] Emergency save rejected: optimizer is in {STATE_UPDATING} state "
            f"(updating step {optimizer.ttp_updating_step}). "
            f"Cannot save during an in-progress optimizer step."
        )

    if state == STATE_ABORTED:
        raise RuntimeError(
            f"[TTP] Emergency save rejected: optimizer is in {STATE_ABORTED} state "
            f"(last committed step {committed_step}). "
            f"The last optimizer step failed, state may be inconsistent."
        )

    # Safe states: IDLE or COMMITTED
    ttp_logger.info(
        f"[TTP] Transaction state check passed: state={state}, "
        f"committed_step={committed_step}"
    )


# ========== Integration with bind_ttp_methods ==========

def bind_ttp_methods_with_transaction_state(
    optimizer,
    ori_dp_group,
    replica_num,
    os_shard_group=None,
    initial_step=0,
):
    """Bind TTP methods to optimizer with transaction state tracking.

    This is the Phase 5 fix for the optimizer step wrapper.

    Args:
        optimizer: mcore DistributedOptimizer instance
        ori_dp_group: Original data parallel process group
        replica_num: Number of replicas
        os_shard_group: The replica subgroup
        initial_step: Initial committed step (from trainer.state.iteration)
    """
    # Import the original bind_ttp_methods to reuse other functionality
    from .tft_replica_optimizer_patch import (
        _set_dump_args,
        _need_write_file,
        _save_parameter_state,
        _get_parameter_state_dp_zero_for_ttp,
        _begin_to_update,
        _end_to_update,
        _sync_gather_all_model_params,
        _show_replica_inc_memsize,
    )

    # Initialize transaction state
    init_optimizer_transaction_state(optimizer, initial_step)

    # Set TTP state on optimizer instance
    optimizer.ori_dp_group = ori_dp_group
    optimizer.ori_dp_list = torch.distributed.get_process_group_ranks(ori_dp_group) if ori_dp_group else [0]
    optimizer.replica_num = replica_num
    optimizer.os_shard_group = os_shard_group
    optimizer.error_dump = False
    optimizer.save_args = {}
    optimizer.current_step = initial_step

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

    # Add transaction state methods
    optimizer.is_safe_for_emergency_save = lambda: is_safe_state_for_emergency_save(optimizer)
    optimizer.check_transaction_state = lambda: check_transaction_state_before_save(optimizer)

    # Wrap step() with transaction state tracking (Phase 5 fix)
    orig_step = optimizer.step
    optimizer.step = wrap_step_with_transaction_state(optimizer, orig_step)

    # Log memory increase
    optimizer.show_replica_inc_memsize()

    ttp_logger.info(
        f"[TTP] TTP methods bound with transaction state tracking "
        f"(replica_num={replica_num}, dp_world_size={dp_world_size}, "
        f"initial_step={initial_step})"
    )
