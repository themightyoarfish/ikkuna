import numpy as np
import torch
from torch.nn.functional import normalize

from ikkuna.export.subscriber import LinePlotSubscriber, SynchronizedSubscription

ZERO_TENSOR = torch.tensor(0.0).cuda()


class SpectralNormSubscriber(LinePlotSubscriber):

    def __init__(self, kinds, tag=None, subsample=1, average=1, ylims=None):
        '''
        Parameters
        ----------
        subsample   :   int
                        Factor for subsampling incoming messages. Only every ``subsample``-th
                        message will be processed.
        average :   int
                    Inverse resolution of the plot. For plotting ``average`` norms will be averaged
                    for each module to remove noise.
        '''
        super().__init__(kinds, tag=tag, subsample=subsample, ylims=ylims)
        self._subscription      = SynchronizedSubscription(self, tag)

        self._ax.set_title(f'Spectral norms of {self.kinds[0]} per '
                           f'layer (average of {self._average} batches)')
        self._ax.set_xlabel('Spectral norm')
        self.u = dict()

    def _metric(self, module_data):
        '''The spectral norm computation is taken from the `Pytorch implementation of spectral norm
        <https://pytorch.org/docs/master/_modules/torch/nn/utils/spectral_norm.html>`_. It's
        possible to use SVD instead, but we are not interested in the full matrix decomposition,
        merely in the singular values.'''

        module  = module_data.module.name

        # get and reshape the weight tensor to 2d
        weights = module_data._data[self.kinds[0]]
        height = weights.size(0)
        weights2d = weights.reshape(height, -1)

        # buffer for power iteration (don't know what the mahematical purpose is)
        if module not in self.u:
            self.u[module] = normalize(weights2d.new_empty(height).normal_(0, 1), dim=0)

        # estimate singular values
        with torch.no_grad():
            for _ in range(3):
                v = normalize(torch.matmul(weights2d.t(), self.u[module]), dim=0)
                self.u[module] = normalize(torch.matmul(weights2d, v), dim=0)

        norm = torch.dot(self.u[module], torch.matmul(weights2d, v)).item()

        norms = self._metric_values[module]
        norms.append(norm)

        # every self._average calls, we replace the last self._average elements with their mean
        # possibly a running average would be more efficient, but who's counting
        n_past_norms = self._counter[module] + 1
        if n_past_norms % self._average == 0:
            norms[-self._average:] = (np.mean(norms[-self._average:]),)
