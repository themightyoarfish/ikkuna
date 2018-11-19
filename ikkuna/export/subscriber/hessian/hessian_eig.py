
from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus

from hessian_eigenthings import compute_hessian_eigenthings


class HessianEigen(PlotSubscriber):
    def __init__(self, forward_fn, loss_fn, data_loader, batch_size, power_steps=20,
                 message_bus=get_default_bus(), ylims=None, backend='tb'):
        title  = f'Top hessian Eigenvalue'
        ylabel = 'tbd'
        xlabel = 'Train step'
        subscription = Subscription(self, ['batch_finished', 'activations'])
        super().__init__([subscription],
                         message_bus,
                         {'title': title, 'ylabel': ylabel, 'ylims': ylims, 'xlabel': xlabel},
                         backend=backend)

        self._forward_fn = forward_fn
        self._loss_fn = loss_fn
        self._power_steps = power_steps
        self._dataloader = data_loader
        self._parameters = set()

    def compute(self, message):
        if message.kind == 'activations':
            module = message.key.module
            for p in module.parameters():
                self._parameters.add(p)
        else:
            eigenvalues, eigenvectors = compute_hessian_eigenthings(self._forward_fn,
                                                                    self._parameters,
                                                                    self._dataloader,
                                                                    self._loss_fn,
                                                                    power_iter_steps=self._power_steps,
                                                                    num_eigenthings=1)
            self.backend.add_data(f'{self._power_steps}_steps', eigenvalues[0], message.global_step)
