# TTP (临终遗言) module for ms-swift Megatron backend
# Migration from MindSpeed-LLM

from .ttp_init_swift import (
    init_ttp_for_swift,
    tft_init_controller_processor_swift,
    tft_register_processor_swift,
    register_ttp_for_trainer,
    ttp_step_hook,
    ttp_train_wrapper,
)
