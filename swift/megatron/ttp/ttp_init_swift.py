# TTP initialization adapted for ms-swift Megatron backend
# Based on MindSpeed-LLM's tft_train_initialize.py
# Date: 2026-07-23

import os
import torch
from functools import wraps
from logging import getLogger

from mindio_ttp.framework_ttp import (
    tft_exception_handler,
    tft_init_controller,
    tft_start_controller,
    tft_init_processor,
    tft_start_processor,
    tft_register_rename_handler,
    set_mindio_export_version,
    tft_register_save_ckpt_handler,
    tft_set_optimizer_replica,
    tft_set_dp_group_info,
    tft_register_stop_handler,
    tft_register_clean_handler,
    tft_register_repair_handler,
    tft_register_rollback_handler,
    tft_register_rebuild_group_handler,
    tft_is_reboot_node,
    tft_register_stream_sync_handler,
    tft_pause_train,
    tft_set_step_args,
)

ttp_logger = getLogger(__name__)

# Global trainer reference for TTP callbacks.
# @tft_exception_handler overwrites save_handler._args with (train_dataset, val_dataset),
# so callbacks cannot use save_handler._args to access the trainer.
# Instead, callbacks use this global variable.
_TTP_TRAINER = None


def get_ttp_trainer():
    """Get the global trainer reference for TTP callbacks."""
    return _TTP_TRAINER

try:
    from mindio_ttp.framework_ttp import tft_register_exception_handler
except ImportError:
    ttp_logger.warning(
        "Warning: tft_register_exception_handler does not take effect, "
        "please install the latest mindio_ttp."
    )
    tft_register_exception_handler = lambda *args, **kwargs: None


def tft_init_controller_processor_swift(args):
    """Initialize TTP Controller (rank 0) and Processor (all ranks).

    Adapted from MindSpeed-LLM's tft_init_controller_processor().
    Uses ms-swift args instead of megatron get_args().
    """
    default_ip = '127.0.0.1'
    ttp_ip = os.getenv('TTP_ADDR', default_ip)
    controller_ip = os.getenv('CONTROLLER_ADDR', default_ip)
    if controller_ip == default_ip:
        controller_ip = ttp_ip
    processor_ip = os.getenv('PROCESSOR_ADDR', default_ip)
    if processor_ip == default_ip:
        processor_ip = ttp_ip
    port = int(os.getenv("TTP_CONTROLLER_PORT", "18000"))

    cur_rank = args.rank
    world_size = args.world_size

    enable_worker_reboot = getattr(args, 'enable_worker_reboot', False)
    enable_hbmfault_repair = getattr(args, 'enable_hbmfault_repair', False)
    enable_elastic_training = getattr(args, 'enable_elastic_training', False)

    # Only rank 0 starts Controller (when not in K8s/MindX environment)
    enable_mindx = os.getenv('MINDX_TASK_ID')
    if cur_rank == 0 and enable_mindx is None:
        ttp_logger.info(f"[TTP] Rank 0 initializing Controller (ip={controller_ip}, port={port})")
        tft_init_controller(cur_rank, world_size, False, enable_worker_reboot, enable_elastic_training)
        tft_start_controller(controller_ip, port, False, '')
        ttp_logger.info("[TTP] Controller started successfully")
    else:
        ttp_logger.info(f"[TTP] Rank {cur_rank} skipping Controller init (rank 0 only)")

    # All ranks start Processor
    ttp_logger.info(f"[TTP] Rank {cur_rank} initializing Processor (ip={processor_ip}, port={port})")
    tft_init_processor(
        cur_rank, world_size, False, False, '',
        enable_hbmfault_repair, enable_worker_reboot, enable_elastic_training
    )
    tft_start_processor(processor_ip, port)
    ttp_logger.info(f"[TTP] Rank {cur_rank} Processor started successfully")


def register_empty_callbacks():
    """Register callbacks for TTP.

    save/rename: V4 implementation from tft_dump_swift_v4 (dynamic Gloo group)
    others: empty for MVP (Step 5 will implement)
    """
    # V4 Save checkpoint callback - real implementation with Gloo gather
    from .tft_dump_swift_v4 import tft_save_callback_swift_v4, tft_rename_callback_swift_v4

    # Repair callback - not implemented in MVP
    def empty_repair_callback(*args, **kwargs):
        ttp_logger.info("[TTP] empty_repair_callback called")

    # Rollback callback - not implemented in MVP
    def empty_rollback_callback(*args, **kwargs):
        ttp_logger.info("[TTP] empty_rollback_callback called")

    # Rebuild group callback - not implemented in MVP
    def empty_rebuild_group_callback(*args, **kwargs):
        ttp_logger.info("[TTP] empty_rebuild_group_callback called")

    # Register all callbacks (V4 for save/rename)
    tft_register_save_ckpt_handler(tft_save_callback_swift_v4)
    tft_register_rename_handler(tft_rename_callback_swift_v4)
    tft_register_repair_handler(empty_repair_callback)
    tft_register_rollback_handler(empty_rollback_callback)
    tft_register_rebuild_group_handler(empty_rebuild_group_callback)

    # Register stop/clean/stream_sync from tft_stop_clean.py
    from .tft_stop_clean import stop_callback, clean_callback, torch_sync
    tft_register_stop_handler(stop_callback)
    tft_register_clean_handler(clean_callback)
    tft_register_stream_sync_handler(torch_sync)

    ttp_logger.info("[TTP] All callbacks registered (V4 save/rename: Gloo gather, others: empty for MVP)")


def _validate_safe_point_mvp_config(args):
    """Reject configurations that cannot produce a complete MVP checkpoint.

    First phase (safe-point MVP) only supports:
    - Full-parameter SFT (no LoRA)
    - Distributed optimizer enabled
    - TP/PP/CP/EP = 1
    - overlap_grad_reduce/param_gather = False
    - optimizer_replica_num = 2
    """
    if not getattr(args, 'use_distributed_optimizer', False):
        raise ValueError("[TTP] Safe-point emergency save requires distributed optimizer")

    tuner_type = getattr(args, 'tuner_type', 'full')
    if tuner_type not in (None, 'full'):
        raise ValueError(f"[TTP] Only full-parameter SFT is supported, got {tuner_type}")

    parallel_sizes = {
        'tensor_model_parallel_size': getattr(args, 'tensor_model_parallel_size', 1),
        'pipeline_model_parallel_size': getattr(args, 'pipeline_model_parallel_size', 1),
        'context_parallel_size': getattr(args, 'context_parallel_size', 1),
        'expert_model_parallel_size': getattr(args, 'expert_model_parallel_size', 1),
    }
    unsupported = {name: size for name, size in parallel_sizes.items() if size != 1}
    if unsupported:
        raise ValueError(f"[TTP] Safe-point MVP requires TP/PP/CP/EP=1, got {unsupported}")

    # overlap_flags = (
    #     'overlap_grad_reduce',
    #     'overlap_param_gather',
    #     'overlap_param_gather_with_optimizer_step',
    # )
    # enabled_overlap = [name for name in overlap_flags if getattr(args, name, False)]
    # if enabled_overlap:
    #     raise ValueError(f"[TTP] Safe-point MVP requires overlap disabled: {enabled_overlap}")

    replica_num = getattr(args, 'optimizer_replica_num', None) or 2
    if replica_num != 2:
        raise ValueError(f"[TTP] Safe-point MVP requires optimizer_replica_num=2, got {replica_num}")
    if args.world_size < 2 or args.world_size % replica_num != 0:
        raise ValueError(
            f"[TTP] DP world size {args.world_size} must be divisible by replica_num {replica_num}"
        )


def tft_register_processor_swift(args):
    """Register TTP processor with replica info and callbacks.

    Adapted from MindSpeed-LLM's tft_register_processor().
    Uses ms-swift args and mpu API directly.

    NOTE: For single-card testing, replica info will be minimal.
    For DP>=2, full replica group setup is needed.
    """
    # Set global args first so tft_replica_group can access them via get_args()
    from .tft_replica_group import set_ttp_args
    set_ttp_args(args)

    from .tft_replica_group import (
        ttp_get_replica_dp_num,
        ttp_get_dp_cp_ranks,
        ttp_get_dp_ep_ranks,
        ttp_get_dp_ranks,
    )

    cur_rank = args.rank
    replica_info = []

    dp_cp_ranks = ttp_get_dp_cp_ranks()
    dp_ranks = ttp_get_dp_ranks()

    if dp_cp_ranks is None:
        # Single-card or replica group not initialized yet
        # Use minimal replica info for testing
        ttp_logger.warning(f"[TTP] Rank {cur_rank} dp_cp_ranks is None, using minimal replica info")
        dp_cp_ranks = [cur_rank]
        dp_ranks = [cur_rank]

    dense_replica_cnt = ttp_get_replica_dp_num() if args.use_distributed_optimizer else len(dp_cp_ranks)
    # Clamp replica_cnt to not exceed rank_list size (single-card: replica_cnt=1)
    dense_replica_cnt = min(dense_replica_cnt, len(dp_cp_ranks))
    replica_offset = 0

    replica_dict = {
        "rank_list": dp_cp_ranks,
        "replica_cnt": dense_replica_cnt,
        "replica_shift": replica_offset
    }
    replica_info.append(replica_dict)

    # MoE support (if applicable)
    moe_flag = getattr(args, 'expert_model_parallel_size', 1) > 1
    if moe_flag:
        dp_ep_ranks = ttp_get_dp_ep_ranks()
        if dp_ep_ranks is None:
            dp_ep_ranks = [cur_rank]
        moe_replica_cnt = ttp_get_replica_dp_num() if args.use_distributed_optimizer else len(dp_ep_ranks)
        moe_replica_cnt = min(moe_replica_cnt, len(dp_ep_ranks))
        replica_dict = {
            "rank_list": dp_ep_ranks,
            "replica_cnt": moe_replica_cnt,
            "replica_shift": replica_offset
        }
        replica_info.append(replica_dict)

    # Register replica info
    ttp_logger.info(f"[TTP] Rank {cur_rank} registering replica_info: {replica_info}")
    tft_set_optimizer_replica(cur_rank, replica_info)
    tft_set_dp_group_info(cur_rank, dp_ranks)

    # Register callbacks
    register_empty_callbacks()

    ttp_logger.info(f"[TTP] Rank {cur_rank} processor registered successfully")


def init_ttp_for_swift(args):
    """Main entry point for TTP initialization in ms-swift.

    Call this after distributed process group is initialized.

    Args:
        args: ms-swift MegatronSftArguments (must have .rank, .world_size,
              .use_distributed_optimizer, .enable_high_availability)
    """
    if not getattr(args, 'enable_high_availability', False):
        ttp_logger.info("[TTP] enable_high_availability is False, skipping TTP init")
        return

    # Validate configuration before any TTP initialization
    _validate_safe_point_mvp_config(args)

    ttp_logger.info("=" * 60)
    ttp_logger.info("[TTP] Starting TTP initialization for ms-swift")
    ttp_logger.info(f"[TTP] rank={args.rank}, world_size={args.world_size}")
    ttp_logger.info(f"[TTP] use_distributed_optimizer={args.use_distributed_optimizer}")
    ttp_logger.info("=" * 60)

    # Step 0: Build replica DP groups (required for DP>=2)
    # Must be called after initialize_model_parallel but before TTP init
    _init_replica_dp_group_if_needed(args)

    # Step 1: Initialize Controller and Processor
    tft_init_controller_processor_swift(args)

    # Step 2: Register processor (replica info + callbacks)
    tft_register_processor_swift(args)

    ttp_logger.info("[TTP] TTP initialization completed successfully")


def _init_replica_dp_group_if_needed(args):
    """Build replica DP groups for TTP (required for DP>=2).

    Must be called after megatron's initialize_model_parallel().
    For single-card testing, this is skipped (dp_cp_ranks will be None).
    """
    world_size = args.world_size
    if world_size < 2:
        ttp_logger.info("[TTP] world_size < 2, skipping replica group init (single-card mode)")
        return

    from .tft_replica_group import ttp_initialize_replica_dp_group, set_ttp_args
    set_ttp_args(args)

    tp_size = getattr(args, 'tensor_model_parallel_size', 1)
    pp_size = getattr(args, 'pipeline_model_parallel_size', 1)
    cp_size = getattr(args, 'context_parallel_size', 1)
    ep_size = getattr(args, 'expert_model_parallel_size', 1)
    etp_size = getattr(args, 'expert_tensor_parallel_size', None)

    ttp_logger.info(f"[TTP] Building replica DP groups "
                    f"(tp={tp_size}, pp={pp_size}, cp={cp_size}, ep={ep_size}, world={world_size})")
    ttp_initialize_replica_dp_group(
        pipeline_model_parallel_size=pp_size,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
        world_size=world_size,
    )
    ttp_logger.info("[TTP] Replica DP groups built successfully")


def register_ttp_for_trainer(trainer):
    """Register trainer with TTP and bind TTP methods to optimizer.

    This allows TTP callbacks to access the trainer (model, optimizer, etc.)
    when triggered by a fault. Also binds TTP methods (begin_to_update,
    end_to_update, set_dump_args, etc.) to the optimizer instance.

    Call this after BaseMegatronTrainer.__init__ completes, before training starts.

    Args:
        trainer: BaseMegatronTrainer instance with .args, .models, .optimizer
    """
    if not getattr(trainer.args, 'enable_high_availability', False):
        return

    # Set mindio export version (must be MindSpeed-LLM, the only supported value)
    set_mindio_export_version("MindSpeed-LLM")

    # Store trainer in global variable for TTP callbacks.
    # NOTE: We cannot rely on save_handler._args because @tft_exception_handler
    # overwrites it with (train_dataset, val_dataset) when training starts.
    # Callbacks (clean_callback, save_callback, etc.) use get_ttp_trainer() instead.
    global _TTP_TRAINER
    _TTP_TRAINER = trainer
    ttp_logger.info(f"[TTP] Trainer stored in global _TTP_TRAINER (rank={trainer.args.rank})")

    # === Bind TTP methods to optimizer (replaces old TTPReplicaOptimizerWrapper) ===
    from .tft_replica_optimizer_patch import bind_ttp_methods
    from .tft_replica_group import ttp_get_replica_dp_num, ttp_get_dp_cp_replica_group
    from megatron.core import parallel_state as mpu

    replica_num = ttp_get_replica_dp_num() if trainer.args.use_distributed_optimizer else 2
    ori_dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
    os_shard_group = ttp_get_dp_cp_replica_group()

    ttp_logger.info(f"[TTP] Binding TTP methods to optimizer "
                    f"(replica_num={replica_num}, os_shard_group size={torch.distributed.get_world_size(os_shard_group) if os_shard_group else 'N/A'})")

    # V3 patches（enable_ttp_optimizer_patch_v3 + fix_optimizer_for_checkpoint）
    # 已在 sft.py 的 prepare_trainer 之前启用，这里不再重复调用
    # （之前在 register_ttp_for_trainer 里调用太晚，_load_checkpoint 已经执行过了）

    # 绑定紧急保存方法（V2/V3 通用）
    bind_ttp_methods(
        trainer.optimizer,
        ori_dp_group=ori_dp_group,
        replica_num=replica_num,
        os_shard_group=os_shard_group,
        initial_step=trainer.state.iteration,
    )

    ttp_logger.info(f"[TTP] TTP methods bound to optimizer successfully")


def ttp_step_hook(iteration):
    """Step 2: Hook to call after each training step.

    Lets TTP check if training should pause (elastic training scenario).
    For MVP, this is a no-op unless TTP requests a pause.

    Args:
        iteration: current training iteration (1-based after increment)
    """
    if iteration < 0:
        return
    try:
        tft_pause_train(iteration)
    except RuntimeError as e:
        if "STEP FINISH" in str(e):
            ttp_logger.info(f"[TTP] Step finish requested at iteration {iteration}")
            raise
        ttp_logger.warning(f"[TTP] tft_pause_train warning: {e}")
    except Exception as e:
        ttp_logger.warning(f"[TTP] tft_pause_train warning: {e}")


def ttp_train_wrapper(fn):
    """Wrap a training function with TTP exception handler.

    Usage:
        train_fn = ttp_train_wrapper(trainer.train)
        train_fn(train_dataset, val_dataset)

    Key: tft_exception_handler must decorate the ACTUAL training function (fn),
    not ttp_train_wrapper itself. Otherwise the exception handler's try/except
    only wraps the wrapper-creation step, not the actual training, and
    RuntimeErrors from training are NOT caught.
    """
    set_mindio_export_version("MindSpeed-LLM")

    # Directly decorate fn with tft_exception_handler
    # This ensures the exception handler wraps the actual training function
    decorated_fn = tft_exception_handler(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return decorated_fn(*args, **kwargs)
    return wrapper
