import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner.hooks.optimizer import Fp16OptimizerHook
from mmcv.runner.hooks import HOOKS


@HOOKS.register_module()
class CustomFp16OptimizerHook(Fp16OptimizerHook):

    def __init__(self,
                custom_fp16={},
                *args,
                **kwargs):
        super(CustomFp16OptimizerHook, self).__init__(*args, **kwargs)
        self.custom_fp16 = custom_fp16

    def before_run(self, runner) -> None:
        super().before_run(runner)
        for module_name, v in self.custom_fp16.items():
            runner.model.module._modules[module_name].fp16_enabled = v
    