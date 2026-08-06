# TTP Replica Optimizer Patch V3
# 基于迁移手册思路，结合 MCore 0.15.0 实际情况
#
# 与 V2 的核心区别：
# 1. 不永久交换 bucket group 的 intra_* 属性（V2 的 fix 有梯度同步 bug）
# 2. start_param_sync: 每次调用时临时换 intra_*，用完即恢复（手册方案）
# 3. start_grad_sync: 临时关 use_distributed_optimizer 强制 All-Reduce（MCore 0.15 无 force_all_reduce）
# 4. buffer group swap 仅用于优化器创建期间（和 V2 一样），不碰 bucket group
#
# 修复的 V2 bug：
# V2 fix 把 bucket group 的 intra_* 永久换成 os_shard_group 且不恢复，
# 导致 start_grad_sync 也用 os_shard_group 做梯度同步 → 梯度只在副本组内同步，错误！
# V3 通过 per-call swap 彻底避免这个问题。

import torch
from logging import getLogger
from functools import wraps

ttp_logger = getLogger(__name__)

# 复用 V2 的 bind_ttp_methods（紧急保存方法绑定，与数据面无关）
try:
    from .tft_replica_optimizer_patch import bind_ttp_methods
except ImportError:
    from tft_replica_optimizer_patch import bind_ttp_methods

# 全局 TTP 状态
_TTP_STATE_V3 = {
    'enabled': False,
    'os_shard_group': None,       # 副本子组
    'original_dp_group': None,    # 原始 DP 组（梯度 all-reduce 用）
    'dump_group': None,           # 故障保存时的通信组（只包含存活 rank）
}


# ========== DDP 通信 patch（手册核心方案：per-call swap）==========

_DDP_PATCHED = False
_ORIG_START_GRAD_SYNC = None
_ORIG_START_PARAM_SYNC = None


def install_ddp_patches():
    """Patch _ParamAndGradBucketGroup 的通信方法。

    - start_grad_sync: 强制 All-Reduce（每个 rank 需要完整梯度来算副本优化器状态）
    - start_param_sync: 临时用副本组做参数 All-Gather，用完恢复
    """
    global _DDP_PATCHED, _ORIG_START_GRAD_SYNC, _ORIG_START_PARAM_SYNC

    if _DDP_PATCHED:
        return

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    _ORIG_START_GRAD_SYNC = _ParamAndGradBucketGroup.start_grad_sync
    _ORIG_START_PARAM_SYNC = _ParamAndGradBucketGroup.start_param_sync

    @wraps(_ORIG_START_GRAD_SYNC)
    def start_grad_sync_for_ttp(self):
        """梯度同步：强制 All-Replace 替代 Reduce-Scatter。

        TTP 需要每个 rank 拿到完整梯度（不只是自己的分片），
        因为副本组内每个 rank 要独立计算自己那份优化器状态。

        MCore 0.15 没有 force_all_reduce 参数，所以临时关掉
        use_distributed_optimizer 让它走 All-Reduce 分支。

        注意：MCore 0.15 的 _ParamAndGradBucketGroup.__init__ 是 if/else：
        - use_distributed_optimizer=True 时只设 intra_* 属性
        - use_distributed_optimizer=False 时才设 data_parallel_group
        所以我们临时关闭 use_distributed_optimizer 时，必须同时临时设置
        data_parallel_group，否则会 AttributeError。
        """
        if not _TTP_STATE_V3['enabled']:
            return _ORIG_START_GRAD_SYNC(self)

        # 临时关闭 distributed_optimizer → start_grad_sync 走 All-Reduce 分支
        old_val = self.ddp_config.use_distributed_optimizer
        self.ddp_config.use_distributed_optimizer = False

        # 临时设置 data_parallel_group（因为 __init__ 时 use_distributed_optimizer=True，
        # 所以 BucketGroup 上没有这个属性）
        old_dp_group = getattr(self, 'data_parallel_group', None)
        self.data_parallel_group = _TTP_STATE_V3['original_dp_group']

        try:
            return _ORIG_START_GRAD_SYNC(self)
        finally:
            # 恢复原始状态
            self.ddp_config.use_distributed_optimizer = old_val
            if old_dp_group is None:
                # 恢复时删除临时添加的属性
                if hasattr(self, 'data_parallel_group'):
                    delattr(self, 'data_parallel_group')
            else:
                self.data_parallel_group = old_dp_group

    @wraps(_ORIG_START_PARAM_SYNC)
    def start_param_sync_for_ttp(self, *args, **kwargs):
        """参数同步：临时用副本组做 All-Gather，调用完立刻恢复。

        优化器按副本组分片（gbuf_ranges 用 os_shard_group），
        所以参数 All-Gather 也必须在副本组内进行，才能对齐分片边界。

        故障保存时优先使用 dump_group（只包含存活 rank），避免和故障 rank 通信死锁。
        用 try/finally 确保属性一定被恢复，不存在残留风险。
        """
        if not _TTP_STATE_V3['enabled']:
            return _ORIG_START_PARAM_SYNC(self, *args, **kwargs)

        # 故障保存时优先使用 dump_group（只含存活 rank），否则用 os_shard_group
        target_group = _TTP_STATE_V3.get('dump_group')
        if target_group is None:
            target_group = _TTP_STATE_V3['os_shard_group']
        if target_group is None:
            return _ORIG_START_PARAM_SYNC(self, *args, **kwargs)

        # 保存原始属性
        old_group = self.intra_distributed_optimizer_instance_group
        old_size = self.intra_distributed_optimizer_instance_size
        old_rank = self.intra_distributed_optimizer_instance_rank

        # 临时换成目标组（dump_group 或 os_shard_group）
        self.intra_distributed_optimizer_instance_group = target_group
        self.intra_distributed_optimizer_instance_size = torch.distributed.get_world_size(target_group)
        self.intra_distributed_optimizer_instance_rank = target_group.rank()

        # 清空缓存（旧缓存是按原始 DP 大小算的，分片不对）
        self.cached_param_buffer_shard_list = [None] * len(self.buckets)

        try:
            return _ORIG_START_PARAM_SYNC(self, *args, **kwargs)
        finally:
            # 立刻恢复，保证属性不残留
            self.intra_distributed_optimizer_instance_group = old_group
            self.intra_distributed_optimizer_instance_size = old_size
            self.intra_distributed_optimizer_instance_rank = old_rank

    _ParamAndGradBucketGroup.start_grad_sync = start_grad_sync_for_ttp
    _ParamAndGradBucketGroup.start_param_sync = start_param_sync_for_ttp
    _DDP_PATCHED = True

    ttp_logger.info("[TTP V3] DDP patches installed (start_grad_sync + start_param_sync)")


# ========== 优化器创建 patch（buffer group swap，仅用于创建期间）==========


def _swap_buffer_groups_v3(wrapped_models, os_shard_group):
    """临时交换 buffer 的 data_parallel_group（仅用于优化器创建）。

    和 V2 一样，让 _build_gbuf_range_map 读到 os_shard_group。
    但和 V2 fix 不同：不碰 bucket group 的 intra_* 属性。
    bucket group 的通信由 DDP patches 在运行时处理。
    """
    saved = []
    for model in wrapped_models:
        for buffer in getattr(model, 'buffers', []):
            saved.append((buffer, buffer.data_parallel_group, buffer.data_parallel_world_size))
            buffer.data_parallel_group = os_shard_group
            buffer.data_parallel_world_size = torch.distributed.get_world_size(os_shard_group)

        for buffer in getattr(model, 'expert_parallel_buffers', []):
            saved.append((buffer, buffer.data_parallel_group, buffer.data_parallel_world_size))
            buffer.data_parallel_group = os_shard_group
            buffer.data_parallel_world_size = torch.distributed.get_world_size(os_shard_group)

    ttp_logger.info(
        f"[TTP V3] Swapped {len(saved)} buffers to os_shard_group "
        f"(size={torch.distributed.get_world_size(os_shard_group)})"
    )
    return saved


def _restore_buffer_groups_v3(saved):
    """恢复 buffer 的 data_parallel_group（创建完后立刻恢复）。"""
    for buffer, orig_group, orig_world_size in saved:
        buffer.data_parallel_group = orig_group
        buffer.data_parallel_world_size = orig_world_size
    ttp_logger.info(f"[TTP V3] Restored {len(saved)} buffers to original DP group")


def patch_optimizer_creation_v3(trainer_cls):
    """Patch get_optimizer_and_scheduler：创建优化器时临时 swap buffer group。

    和 V2 一样的 swap/restore 模式，但不碰 bucket group。
    创建完 optimizer 后立即调用 fix_optimizer_for_checkpoint，
    因为 _load_checkpoint 紧接着 __init__ 调用，需要 sharded_state_dict 已被 patch。
    """
    orig_method = trainer_cls.get_optimizer_and_scheduler

    def patched_get_optimizer_and_scheduler(self):
        if _TTP_STATE_V3['enabled'] and _TTP_STATE_V3['os_shard_group'] is not None:
            ttp_logger.info("[TTP V3] Patching optimizer creation: swapping buffer groups")
            saved = _swap_buffer_groups_v3(self.wrapped_models, _TTP_STATE_V3['os_shard_group'])
            try:
                result = orig_method(self)
            finally:
                _restore_buffer_groups_v3(saved)
            ttp_logger.info("[TTP V3] Optimizer created with os_shard_group sharding")

            # 创建完 optimizer 后立即 patch checkpoint 相关方法
            # 因为 _load_checkpoint 紧接着 __init__ 调用，需要 sharded_state_dict 已被 patch
            # get_optimizer_and_scheduler 返回 (optimizer, opt_param_scheduler) 元组
            if isinstance(result, tuple):
                optimizer = result[0]
            else:
                optimizer = result
            fix_optimizer_for_checkpoint(optimizer)

            return result
        return orig_method(self)

    trainer_cls.get_optimizer_and_scheduler = patched_get_optimizer_and_scheduler
    ttp_logger.info(f"[TTP V3] Patched {trainer_cls.__name__}.get_optimizer_and_scheduler")


# ========== 创建后修正 ==========


class _DPGroupSizeOverride:
    """ProcessGroup wrapper：只覆盖 size()，rank() 和其他方法委托给原始 group。

    用于 checkpoint 保存时：size() 返回 os_shard_group size（匹配 gbuf_ranges），
    rank() 返回原始 DP rank（保证 key 唯一性）。
    """

    def __init__(self, original_group, override_size):
        self._original = original_group
        self._override_size = override_size

    def size(self):
        return self._override_size

    def rank(self):
        return self._original.rank()

    def __getattr__(self, name):
        return getattr(self._original, name)


def fix_optimizer_for_checkpoint(optimizer):
    """修正优化器以支持 TTP 副本组的 checkpoint 保存。

    问题：gbuf_ranges 按 os_shard_group（size=2）计算，但 data_parallel_group
    是原始 DP group（size=4），导致 checkpoint 保存时维度计算错误。

    解决方案：patch sharded_state_dict 方法，临时用 _DPGroupSizeOverride 包装
    data_parallel_group，使 size() 返回 os_shard_group size，rank() 返回原始 DP rank。
    这样既保证维度计算正确，又保证 checkpoint key 唯一。

    同时修正 grad_stats_parallel_group 用于 grad norm 统计。
    """
    if not _TTP_STATE_V3['enabled']:
        return

    os_group = _TTP_STATE_V3['os_shard_group']
    os_world_size = torch.distributed.get_world_size(os_group)

    def fix_single_optimizer(opt):
        # 修正 grad_stats_parallel_group（用于 grad norm 统计）
        if hasattr(opt, 'grad_stats_parallel_group'):
            opt.grad_stats_parallel_group = os_group
            ttp_logger.info("[TTP V3] Fixed grad_stats_parallel_group")

        # Patch sharded_state_dict 用于 checkpoint 保存
        if hasattr(opt, 'sharded_state_dict'):
            orig_sharded_sd = opt.sharded_state_dict

            def patched_sharded_state_dict(*args, **kwargs):
                old_group = opt.data_parallel_group
                opt.data_parallel_group = _DPGroupSizeOverride(old_group, os_world_size)
                try:
                    return orig_sharded_sd(*args, **kwargs)
                finally:
                    opt.data_parallel_group = old_group

            opt.sharded_state_dict = patched_sharded_state_dict
            ttp_logger.info(
                f"[TTP V3] Patched sharded_state_dict "
                f"(size override: {os_world_size})"
            )

    # 处理 ChainedOptimizer
    if hasattr(optimizer, 'chained_optimizers'):
        for opt in optimizer.chained_optimizers:
            fix_single_optimizer(opt)
        ttp_logger.info("[TTP V3] Fixed optimizer for checkpoint on chained_optimizers")
    else:
        fix_single_optimizer(optimizer)
        ttp_logger.info("[TTP V3] Fixed optimizer for checkpoint on optimizer")


# ========== 入口函数 ==========


def enable_ttp_optimizer_patch_v3(os_shard_group, original_dp_group=None):
    """启用 TTP 优化器 patch（V3 方案）。

    在 init_ttp_for_swift() 之后、prepare_trainer() 之前调用。

    与 V2 的区别：
    - DDP 通信用 per-call swap（手册方案），不永久改 bucket group 属性
    - 梯度同步用临时关 use_distributed_optimizer 强制 All-Reduce
    - 修复了 V2 fix 的梯度同步 bug

    Args:
        os_shard_group: 副本子组 ProcessGroup
        original_dp_group: 原始 DP 组（梯度 all-reduce 用）
                          None 则自动从 mpu 获取
    """
    _TTP_STATE_V3['enabled'] = True
    _TTP_STATE_V3['os_shard_group'] = os_shard_group

    if original_dp_group is None:
        from megatron.core import parallel_state as mpu
        original_dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
    _TTP_STATE_V3['original_dp_group'] = original_dp_group

    # 安装 DDP 通信 patches（class-level，per-call swap）
    install_ddp_patches()

    # 安装优化器创建 patch（buffer group swap，仅创建期间）
    from swift.megatron.trainers.base import BaseMegatronTrainer
    patch_optimizer_creation_v3(BaseMegatronTrainer)

    shard_size = torch.distributed.get_world_size(os_shard_group)
    dp_size = torch.distributed.get_world_size(original_dp_group)
    ttp_logger.info(
        f"[TTP V3] Optimizer patch enabled "
        f"(os_shard_group size={shard_size}, original_dp size={dp_size})"
    )

