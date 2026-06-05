import torch
from torch.optim import Optimizer

class IIPG(Optimizer):
    """ Inertial Incremental proximal gradient optimizer as described in Hammernik et al 2016 """
    def __init__(self, optimizer, params, **kwargs):
        # 必须先初始化基类，否则 Lightning 的某些钩子函数（如 lr_scheduler）会报错
        self.opt = optimizer(params, **kwargs)
        super().__init__(self.opt.param_groups, self.opt.defaults) 
        
        self.params = list(params)  # 转换为列表方便后续遍历
        self.param_groups = self.opt.param_groups
        self.state = self.opt.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        
        # 投影梯度下降 (Proximal Mapping)
        for param in self.params:   
            if len(param.shape) == 0:
                # 1. 数据权重项约束 (保证非负)
                param.data = torch.max(input=param.data, other=torch.zeros_like(param.data))

            else:
                if param.shape[0] > 1:
                    # 2. 卷积核约束
                    # 【核心修复】：原代码将 param.data 错写成了 param_data
                    param.data = zero_mean_norm_ball(param.data, axis=(1, 2, 3, 4))

        return loss
