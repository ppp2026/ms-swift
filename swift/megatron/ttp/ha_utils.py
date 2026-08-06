# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
from dataclasses import dataclass


@dataclass
class HighAvailabilityConstant:
    RET_OK = 0
    RET_ERROR = 1
    RET_NO_REBUILD = 2

    MODEL_INDEX = 1
    OPTIM_INDEX = 2
    SCHEDULER_INDEX = 3
    TRAIN_DATA_INDEX = 4
    VALID_DATA_INDEX = 5
    CONFIG_INDEX = -1

    UCE_LOW_LEVEL = 2
    UCE_HIGH_LEVEL = 3

    DEFAULT_MIN_FILE_SIZE = 1
    DEFAULT_MAX_FILE_SIZE = 1024 * 1024 * 1024


ha_constant = HighAvailabilityConstant()
