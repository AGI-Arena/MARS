import math
import torch

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    
    This version allows for G to be more than 2D.
    """
    # assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X

class MARS_M(torch.optim.Optimizer):
    """
    MARS_M: MARS on Matrix-Level

    Arguments:
        lr: The learning rate. The updates will have spectral norm of `lr`.
        wd: The weight decay.
        muon_params: The parameters to be optimized by Muon.
        momentum: The momentum used by the internal SGD.
        ns_steps: The number of Newton-Schulz iterations to run.
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas used by AdamW.
        adamw_eps: The epsilon used by AdamW.
        gamma: The gamma parameter in MARS.
        clip_c: Whether to clip the c_t vector to have norm at most 1.
        is_approx: Whether to use the approximate version of MARS-M (True) or the exact version (False).
            The approximate version does not require any extra gradient computations, while the exact version
            requires one extra gradient computation per parameter per step. 
            However, the exact version may yield slightly better performance.
            See the MARS-M paper for details.

    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        gamma=0.025,
        clip_c=False,
        is_approx=True,
    ):
        mars_factor = gamma * momentum / (1-momentum)
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            gamma=gamma,
            mars_factor=mars_factor,
            clip_c=clip_c,
            is_approx=is_approx
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        self.is_approx = is_approx
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            # assert p.ndim >= 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def update_last_grad(self):
        if not self.is_approx:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    ## only update previous grad for muon params
                    if not state["use_muon"]:
                        continue
                    ## end skip
                    if "last_grad" not in state:
                        state["last_grad"] = torch.zeros_like(p)
                    state["last_grad"].zero_().add_(state["previous_grad"], alpha=1.0)
    @torch.no_grad()
    def update_previous_grad(self):
        if not self.is_approx:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        print (p, "grad is none")
                        continue
                    state = self.state[p]
                    ## only update previous grad for muon params
                    if not state["use_muon"]:
                        continue
                    ## end skip
                    if "previous_grad" not in state:
                        state['previous_grad'] = torch.zeros_like(p)
                    state['previous_grad'].zero_().add_(p.grad, alpha=1.0)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
                
        If using exact version, the example usage is as follows:
            previous_X, previous_Y = None, None
            for epoch in range(epochs):
                for X, Y in data_loader:
                    if previous_X:
                        logits, loss = model(X, Y)
                        loss.backward()
                        optimizer.update_previous_grad()
                        optimizer.zero_grad(set_to_none=True)
                    logits, loss = model(X, Y)
                    loss.backward()
                    optimizer.step(bs=bs)
                    optimizer.zero_grad(set_to_none=True)
                    optimizer.update_last_grad()
                    iter_num += 1
                    previous_X, previous_Y = X.clone(), Y.clone()
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            gamma = group["gamma"]
            mars_factor = group["mars_factor"]
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                assert g is not None
                state = self.state[p]
                
                # default: MARS, nesterov
                if "last_grad" not in state:
                    state["last_grad"] = torch.zeros_like(g)
                # calc update
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                # mars_factor = gamma * momentum / (1-momentum)
                c_t = (g - state["last_grad"]).mul(mars_factor).add(g)
                c_t_norm = c_t.norm()
                if c_t_norm > 1:
                    c_t.div_(c_t_norm)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(c_t, alpha=(1 - momentum))
                
                u = zeropower_via_newtonschulz5(buf, steps=group["ns_steps"])
                

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)
                if self.is_approx:
                    state["last_grad"] = g

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss