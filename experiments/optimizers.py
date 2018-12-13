import torch
from torch.optim import SGD


class FixedRatioSGD(SGD):

    def __init__(self, params, lr=1, target=1e-3):
        super().__init__(params, lr)
        self._target = target

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                grad_multiplier = self._target * torch.norm(p.data) / torch.norm(d_p)
                p.data.add_(-grad_multiplier * d_p)

        return loss
