# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import time
from logging import getLogger

import torch
import torch_npu
from mindio_ttp.framework_ttp.ttp_decorator import tft_get_repair_type

from .ha_utils import ha_constant

ttp_logger = getLogger(__name__)


def stop_callback(train_args, ctx):
    # stop and clean device
    device = torch.npu.current_device()
    ret = torch_npu.npu.stop_device(device)
    if ret is not None and ret != ha_constant.RET_OK:
        raise RuntimeError("stop failed,end stop callback")


def clean_callback(is_uce_error: bool, train_args, ctx):
    """
    this function do:
    1) get UCE check result from torch_npu
    2) do some clear before rebuild (avoid OOM) when the check result is UCE_HIGH_LEVEL
    3) HCCL resume and restart device

    NOTE: train_args is NOT the trainer object. @tft_exception_handler overwrites
    save_handler._args with (train_dataset, val_dataset). We use get_ttp_trainer()
    to access the trainer instead.
    """
    from .ttp_init_swift import get_ttp_trainer
    trainer = get_ttp_trainer()
    if trainer is None:
        ttp_logger.error("[TTP] clean_callback: _TTP_TRAINER is None, cannot clean")
        return ha_constant.RET_ERROR

    device = torch.npu.current_device()
    rank = torch.distributed.get_rank()
    ret = ha_constant.RET_OK
    start_time = time.time()
    unset_gather_handle(trainer)
    if is_uce_error:
        check_memory_result = torch_npu.npu.check_uce_in_memory(device)
        ttp_logger.info(f"rank {rank} check uce memory result: {check_memory_result}")
        if check_memory_result == ha_constant.UCE_LOW_LEVEL:  # no need rebuild
            ret = ha_constant.RET_NO_REBUILD
        elif check_memory_result == ha_constant.UCE_HIGH_LEVEL:  # need rebuild
            model = trainer.wrapped_models
            optim = trainer.optimizer
            optim.update_npu_tensor_to_safe()
            model_update_npu_tensor_to_safe(model)
            ret = ha_constant.RET_OK
        else:  # exit
            ret = ha_constant.RET_ERROR
            ttp_logger.error(f"rank {rank} check uce memory result {ret} is abnormal, exiting...")

    clean_type = tft_get_repair_type()
    if clean_type == "retry":
        torch.distributed.reinit_process_group(group=None, rebuild_link=False)
    torch.npu.restart_device(device)

    ttp_logger.info(f'[clean] rank:{rank}, type:{clean_type}, cost:{time.time() - start_time:.3f}s, ret:{ret}')
    return ret


def model_update_npu_tensor_to_safe(models):
    from torch_npu.npu._recovery import update_npu_tensor_to_safe as update_tensor_to_safe
    for model in models:
        for buffer in model.buffers:
            for bucket in buffer.buckets:
                if bucket.param_data is not None:
                    update_tensor_to_safe(bucket.param_data)
                if bucket.grad_data is not None:
                    update_tensor_to_safe(bucket.grad_data)
        for buffer in model.expert_parallel_buffers:
            for bucket in buffer.buckets:
                if bucket.param_data is not None:
                    update_tensor_to_safe(bucket.param_data)
                if bucket.grad_data is not None:
                    update_tensor_to_safe(bucket.grad_data)

def unset_gather_handle(trainer):
    """Clear gather handles on all model buffers.

    Args:
        trainer: BaseMegatronTrainer instance (NOT a tuple).
                 Uses trainer.wrapped_models to access models.
    """
    models = trainer.wrapped_models
    for model in models:
        for bucket_group in model.bucket_groups:
            bucket_group.grad_reduce_handle = None
            bucket_group.param_gather_handle = None
            if bucket_group.next_param_gather_bucket_group:
                bucket_group.next_param_gather_bucket_group.param_gather_handle = None

        for bucket_group in model.expert_parallel_bucket_groups:
            bucket_group.grad_reduce_handle = None
            bucket_group.param_gather_handle = None
            if bucket_group.next_param_gather_bucket_group:
                bucket_group.next_param_gather_bucket_group.param_gather_handle = None


def torch_sync():
    rank = torch.distributed.get_rank()
    torch.cuda.synchronize()
    ttp_logger.debug(f"[Pause] rank: {rank} finish synchronize")
