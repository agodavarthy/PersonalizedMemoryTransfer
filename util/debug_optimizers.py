"""
These are the default optimizers but for step we can compute the magnitude of
updates
"""
import math
import torch
import torch.optim as optim
from collections import namedtuple

"""
Param = Norm of Parameters
Grad = Norm of Raw Gradients
Update = Norm of the Update Gradients
Ratio = update / param = ~1e-3
"""
UpdateScale = namedtuple("UpdateScale", ['param', 'grad', 'update', 'ratio'])


class SGD(optim.SGD):

    def step(self, compute_magnitude=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        magnitudes = None
        if compute_magnitude:
            magnitudes = []

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                if compute_magnitude:
                    # Current Parameter Scale
                    param_scale = torch.norm(p.data.view(-1))

                    # Raw gradient Norm
                    grad_norm = torch.norm(p.grad.data.view(-1))

                    # Compute Update
                    update = -group['lr'] * d_p
                    # Norm of our update
                    update_scale = torch.norm(update.data.view(-1))
                    # Ratio of update to current parameters
                    ratio = (update_scale.data / param_scale.data)

                    p.data.add_(update)
                    magnitudes.append(UpdateScale(
                            param=param_scale.item(),
                            grad=grad_norm.item(),
                            update=update_scale.item(),
                            ratio=ratio.item()))

                else:
                    p.data.add_(-group['lr'], d_p)

        return magnitudes



class Adam(optim.Adam):
    def step(self, compute_magnitude=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        magnitudes = None
        if compute_magnitude:
            magnitudes = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                if compute_magnitude:
                    # Current Parameter Scale
                    param_scale = torch.norm(p.data.view(-1))
                    # Compute Update
                    update = -step_size * (exp_avg / denom)
                    update_scale = torch.norm(update.data.view(-1))
                    # Ratio of update to current parameters
                    ratio = update_scale.data / param_scale.data
                    p.data.add_(update.data)
                    grad_norm = torch.norm(p.grad.data.view(-1))
                    magnitudes.append(UpdateScale(
                            param=param_scale.item(),
                            grad=grad_norm.item(),
                            update=update_scale.item(),
                            ratio=ratio.item()))
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return magnitudes