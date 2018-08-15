import torch
from torch.nn.functional import normalize

from ikkuna.export.subscriber import PlotSubscriber, Subscription


class SpectralNormSubscriber(PlotSubscriber):

    def __init__(self, kind, tag=None, subsample=1, ylims=None, backend='tb'):
        '''
        Parameters
        ----------
        kind    :   str
                    Message kind to compute spectral norm on. Doesn't make sense with kinds of
                    non-matrix type.


        For other parameters, see :class:`~ikkuna.export.subscriber.PlotSubscriber`
        '''
        if not isinstance(kind, str):
            raise ValueError('SpectralNormSubscriber only accepts 1 kind')
        subscription = Subscription(self, [kind], tag, subsample)

        title = f'Spectral norms of {kind} per layer'
        xlabel = 'Step'
        ylabel = 'Spectral norm'
        super().__init__(subscription,
                         {'title': title, 'xlabel': xlabel, 'ylims': ylims, 'ylabel': ylabel},
                         tag=tag, backend=backend)
        self.u = dict()

    def _metric(self, module_data):
        '''The spectral norm computation is taken from the `Pytorch implementation of spectral norm
        <https://pytorch.org/docs/master/_modules/torch/nn/utils/spectral_norm.html>`_. It's
        possible to use SVD instead, but we are not interested in the full matrix decomposition,
        merely in the singular values.'''

        module    = module_data.module.name
        # get and reshape the weight tensor to 2d
        weights   = module_data.data[self._subscription.kinds[0]]
        height    = weights.size(0)
        weights2d = weights.reshape(height, -1)

        # buffer for power iteration (don't know what the mahematical purpose is)
        if module not in self.u:
            self.u[module] = normalize(weights2d.new_empty(height).normal_(0, 1), dim=0)

        # estimate singular values
        with torch.no_grad():
            for _ in range(3):      # TODO: Make niter parameter
                v = normalize(torch.matmul(weights2d.t(), self.u[module]), dim=0)
                self.u[module] = normalize(torch.matmul(weights2d, v), dim=0)

        norm = torch.dot(self.u[module], torch.matmul(weights2d, v)).item()

        self._backend.add_data(module, norm, module_data.seq)
