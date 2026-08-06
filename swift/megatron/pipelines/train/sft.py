# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import torch
import torch.distributed as dist
from dataclasses import asdict
from transformers.utils import is_torch_npu_available
from typing import List, Optional, Union

from swift.megatron.arguments import MegatronSftArguments
from swift.megatron.trainers import MegatronEmbeddingTrainer, MegatronRerankerTrainer, MegatronTrainer
from swift.pipelines import SwiftSft
from swift.utils import append_to_jsonl, get_logger, is_last_rank, plot_images

if is_torch_npu_available():
    # Enable Megatron on Ascend NPU
    from mindspeed.megatron_adaptor import repatch

    from swift.model.npu_patcher import patch_mindspeed_te_cp_implementation
else:
    repatch = None
    patch_mindspeed_te_cp_implementation = None

logger = get_logger()


class MegatronSft(SwiftSft):
    args_class = MegatronSftArguments
    args: args_class

    def prepare_trainer(self):
        args = self.args
        if args.task_type == 'embedding':
            return MegatronEmbeddingTrainer(self.args, self.template)
        elif args.task_type in {'reranker', 'generative_reranker'}:
            return MegatronRerankerTrainer(self.args, self.template)
        else:
            return MegatronTrainer(self.args, self.template)

    def _set_seed(self):
        pass

    def __init__(self, args: Optional[Union[List[str], MegatronSftArguments]] = None) -> None:
        self.train_msg = {}
        super(SwiftSft, self).__init__(args)
        args = self.args
        if repatch is not None:
            megatron_args = asdict(self.args)
            if args.attention_backend != 'local':
                # MindSpeed requires passing `use_flash_attn` to Megatron
                # to enable flash attention on Ascend NPU.
                args.use_flash_attn = True
                megatron_args['use_flash_attn'] = True
            patch_mindspeed_te_cp_implementation(megatron_args)
            repatch(megatron_args)
        template_cls = args.template_meta.template_cls
        if args.model_meta.is_multimodal and template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self.model, self.processor = args.get_model_processor(**kwargs, download_model=args.mcore_model is None)
        self._prepare_template()
        args.save_args(args.output_dir)
        self.template.use_megatron = True

    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()
        args.init_iters(train_dataset, val_dataset)

        # === TTP 注入开始（必须在 prepare_trainer 之前）===
        if getattr(args, 'enable_high_availability', False):
            import torch
            from swift.megatron.ttp.ttp_init_swift import (
                init_ttp_for_swift, register_ttp_for_trainer, ttp_step_hook
            )
            from swift.megatron.ttp.tft_replica_optimizer_v3 import enable_ttp_optimizer_patch_v3
            from swift.megatron.ttp.tft_replica_group import ttp_get_dp_cp_replica_group

            logger.info('[TTP] enable_high_availability=True, starting TTP initialization')
            init_ttp_for_swift(args)

            # 获取 os_shard_group 并启用 optimizer patch
            # 这会 patch get_optimizer_and_scheduler，在优化器创建时自动交换 buffer 组
            os_shard_group = ttp_get_dp_cp_replica_group()
            if os_shard_group is not None:
                # V3: 必须在 prepare_trainer 之前启用，因为 _load_checkpoint 在
                # MegatronTrainer.__init__ 里调用，需要 sharded_state_dict 已被 patch
                from megatron.core import parallel_state as mpu
                ori_dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
                enable_ttp_optimizer_patch_v3(os_shard_group, ori_dp_group)
                shard_size = torch.distributed.get_world_size(os_shard_group)
                dp_size = torch.distributed.get_world_size(ori_dp_group)
                logger.info(f'[TTP] V3 optimizer patch enabled (os_shard_group size={shard_size}, dp size={dp_size})')
            else:
                logger.warning('[TTP] os_shard_group is None, optimizer patch skipped')
        # === TTP 注入结束 ===

        trainer = self.prepare_trainer()

        # === TTP 注册（prepare_trainer 之后）===
        if getattr(args, 'enable_high_availability', False):
            register_ttp_for_trainer(trainer)
            logger.info('[TTP] TTP initialization completed')

            # Patch run_train_step to call ttp_step_hook
            original_run_train_step = type(trainer).run_train_step
            def patched_run_train_step(self_trainer, train_data_iterator, val_data_iterator):
                result = original_run_train_step(self_trainer, train_data_iterator, val_data_iterator)
                ttp_step_hook(self_trainer.state.iteration)
                return result
            type(trainer).run_train_step = patched_run_train_step

        try:
            # Phase 7: 使用 ttp_train_wrapper 包装训练函数（触发 TTP 异常处理）
            if getattr(args, 'enable_high_availability', False):
                from swift.megatron.ttp.ttp_init_swift import ttp_train_wrapper
                train_fn = ttp_train_wrapper(trainer.train)
            else:
                train_fn = trainer.train

            train_fn(train_dataset, val_dataset)
        finally:
            state = trainer.state
            self._handle_trainer_state(trainer, is_last_rank())
            self.train_msg.update({
                'last_model_checkpoint': state.last_model_checkpoint,
                'best_model_checkpoint': state.best_model_checkpoint,
                'best_metric': state.best_metric,
            })
            # Visualization
            if is_last_rank():
                images_dir = os.path.join(args.output_dir, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, args.tensorboard_dir)

                jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
                append_to_jsonl(jsonl_path, self.train_msg, strict=False, write_on_rank='last')
        # Exceptions may cause the process to hang, preventing the exception from being propagated.
        # Therefore, destroy_process_group() should not be placed inside the finally block.
        if dist.is_initialized():
            dist.destroy_process_group()
        return self.train_msg

def megatron_sft_main(args: Optional[Union[List[str], MegatronSftArguments]] = None):
    return MegatronSft(args).main()
