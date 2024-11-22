import math
import torch
from torch.optim.optimizer import Optimizer

def exists(val):
    return val is not None


class ADOPT(Optimizer):
    """Implements Adopt algorithm.
    https://github.com/stockeh/mlx-optimizers/blob/e1b49cd325ec055a8786dcb01b81866ce3fa2363/mlx_optimizers/adopt.py
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(ADOPT, self).__init__(params, defaults)
        self.eps = eps

    def __setstate__(self, state):
        super(ADOPT, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        """
        c = step, m = exp_avg, v = exp_avg_sq in original implementation
        """
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad.add_(p.data, alpha=group['weight_decay'])
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = grad.mul(grad)
                    continue
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if 'step' in state:
                    state['step'] += 1
                else:
                    state['step'] = 1                
                beta1, beta2 = group['betas']
                denom = torch.maximum(exp_avg_sq.sqrt(), torch.tensor(self.eps))
                if state['step'] == 1:
                    exp_avg = grad.div(denom)
                else:
                    exp_avg.mul_(beta1).addcdiv_(1. - beta1, grad, denom)
                    
                p.data.add_(state['exp_avg'], alpha=-group['lr'])
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        return loss



