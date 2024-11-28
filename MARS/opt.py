import torch
import collections
"""
Adapted from askerlee@github: https://github.com/KellerJordan/modded-nanogpt/issues/9
"""
def separate_params(param_groups):
    param_groups_2d     = []
    param_groups_non2d  = []
    total_param_2d_count      = 0
    total_param_non2d_count   = 0


    # Convert iterators to lists
    if isinstance(param_groups, collections.abc.Iterable):
        param_groups = list(param_groups)

    # Check if param_groups is a list of dicts or list of params
    if (isinstance(param_groups, list) and isinstance(param_groups[0], dict)) \
      or isinstance(param_groups, dict):
        if isinstance(param_groups, dict):
            param_groups = [param_groups]
        # param_groups is a list of dicts
        for group in param_groups:
            params_2d, params_non2d, param_2d_count, param_non2d_count = separate_params(group['params'])
            param_group_2d      = {'params': params_2d}
            param_group_non2d   = {'params': params_non2d}
            # Copy the group dict and replace the 'params' key with the separated params
            for k in group.keys():
                if k != 'params':
                    param_group_2d[k]    = group[k]
                    param_group_non2d[k] = group[k]

            param_groups_2d.append(param_group_2d)
            param_groups_non2d.append(param_group_non2d)
            total_param_2d_count    += param_2d_count
            total_param_non2d_count += param_non2d_count

        return param_groups_2d, param_groups_non2d, total_param_2d_count, total_param_non2d_count

    elif isinstance(param_groups, list) and isinstance(param_groups[0], torch.Tensor):
        params_2d    = []
        params_non2d = []
        param_group  = param_groups
        # param_group is a list of param tensors
        for param in param_group:
            if param.ndim >= 2:
                params_2d.append(param)
            else:
                params_non2d.append(param)
        return params_2d, params_non2d, len(params_2d), len(params_non2d)
    else:
        breakpoint()

'''
# CombinedOptimizer is now a torch.optim.Optimizer, compatible with pytorch lightning.
# Original Example:
    optimizer = CombinedOptimizer([
        torch.optim.AdamW(self.lm_head.parameters(), lr=learning_rate, betas=betas, weight_decay=0, fused=True),
        OrthogonalNesterov(self.transformer.h.parameters(), lr=0.1*learning_rate, momentum=0.95)
    ])
# Refactored Example:
    optimizer = CombinedOptimizer(\
        self.parameters(),
        [OrthogonalNesterov, torch.optim.AdamW],
        [{'lr': 0.1*learning_rate, 'momentum': 0.95}, 
         {'lr': learning_rate, 'betas': betas, 'weight_decay': 0, 'fused': True}
        ])
'''

class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_types, configs, raw_model = False):
        # Separate 2D and non-2D parameters.
        # If params is a list of tensors, then each of param_groups_2d and param_groups_non2d 
        # will be a list of tensors.
        # If params is a list of dicts, then each of param_groups_2d and param_groups_non2d
        # will be a list of dicts.
        # If params is a dict, then each of param_groups_2d and param_groups_non2d will 
        # be a list of dicts containing only one dict.
        if raw_model:
            params_others = list(params.transformer.h.parameters())
            param_groups_2d, param_groups_non2d, total_param_2d_count, total_param_non2d_count \
            = separate_params(params_others)
            param_groups_non2d.extend(list(params.lm_head.parameters()))
            total_param_non2d_count += 2
        else:
            param_groups_2d, param_groups_non2d, total_param_2d_count, total_param_non2d_count \
                = separate_params(params)
        param_groups_2d_non2d = (param_groups_non2d, param_groups_2d)
        print(f"Total 2D params: {total_param_2d_count}, Total non-2D params: {total_param_non2d_count}")

        assert len(optimizer_types) == len(configs) == 2
        self.optimizers = [ optimizer_types[i](param_groups_2d_non2d[i], **configs[i]) for i in range(2) ]
        self.param_groups = [pg for opt in self.optimizers for pg in opt.param_groups]
        self.base_lrs = [opt.param_groups[0]['lr'] for opt in self.optimizers]
        # Combine the state dicts of all opt in self.optimizers into a single dict
        self.state = {k: v for opt in self.optimizers for k, v in opt.state.items()}
        # Initially all states are empty. So no point to print their counts.
        # Only use the defaults of the OrthogonalNesterov optimizer
        self.defaults = self.optimizers[0].defaults

    def step(self, *args, **kwargs):
        for opt in self.optimizers:
            opt.step(*args, **kwargs)

    def zero_grad(self, **kwargs):
        for opt in self.optimizers:
            opt.zero_grad(**kwargs)

    def scale_lrs(self, lr_scale):
        for base_lr, opt in zip(self.base_lrs, self.optimizers):
            opt.param_groups[0]['lr'] = base_lr * lr_scale

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]