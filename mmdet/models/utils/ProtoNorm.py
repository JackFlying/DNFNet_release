import torch
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

class _PrototypeNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(_PrototypeNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.register_buffer("target_prototypes", None)
        self.register_buffer("norm_flag", None)
        self.register_buffer("iou", None)
            
    def register_target_prototypes(self, targets, norm_flag, iou):
        self.target_prototypes = targets
        self.norm_flag = norm_flag
        self.iou = iou
        
    def reset_target_prototypes(self):
        self.target_prototypes = None
        self.norm_flag = None
        
    def forward(self, input):
        self._check_input_dim(input)
        ndim = input.ndim
        
        dim_s = [i for i in range(ndim)]; dim_s.remove(1) # 1d: [0] 2d: [0,2,3]
        dim_o = [1]*ndim; dim_o[1] = -1 # 1d: [1,-1] 2d: [1, -1, 1, 1]
        
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            targets = self.target_prototypes.long().to(input.device)
            targets_unique = torch.unique(targets)
            prototypes = []
            # weights = []

            # process labelled
            # for t in targets_unique:
            #     # weight = self.iou[targets==t]
            #     # weight /= weight.sum()
            #     # prototype = torch.sum(torch.mul(weight.unsqueeze(1), input[targets==t]), dim=0)
            #     prototype = input[targets==t].mean([0])
            #     prototypes.append(prototype)

            # # keep the best IoU instance for each class
            for t, nf in zip(input, self.norm_flag):
                if nf:
                    prototypes.append(t)
            
            prototypes = torch.stack(prototypes)
    
            n = input.numel() / input.size(1)
            mean = prototypes.mean(dim_s)
            var = ((input-mean.view(dim_o))**2).sum(dim_s)/n

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
            self.reset_target_prototypes()

        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean.view(dim_o)) / (torch.sqrt(var.view(dim_o) + self.eps))
        if self.affine:
            input = input * self.weight.view(dim_o) + self.bias.view(dim_o)

        return input
    
class PrototypeNorm1d(_PrototypeNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )
            
class PrototypeNorm2d(_PrototypeNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )

def convert_bn_to_pn(module, module_names=None):

    mod = module
    for pth_module, sync_module in zip([torch.nn.modules.batchnorm.BatchNorm1d,
                                        torch.nn.modules.batchnorm.BatchNorm2d],
                                       [PrototypeNorm1d,
                                        PrototypeNorm2d]):
        if isinstance(module, pth_module):
            mod = sync_module(module.num_features, module.eps, module.momentum, module.affine)
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

    if module_names is not None:
        for name, child in module.named_children():
            if name in module_names: 
                mod.add_module(name, convert_bn_to_pn(child))
    else:
        for name, child in module.named_children():
            mod.add_module(name, convert_bn_to_pn(child))
        
    return mod

def register_targets_for_pn(module, targets, norm_flag, iou):
    for mod in module.modules():
        if isinstance(mod, _PrototypeNorm):
            mod.register_target_prototypes(targets, norm_flag, iou)
    