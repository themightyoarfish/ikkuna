import sys
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
from collections import defaultdict
import torch
from torch.nn.functional import normalize

from ikkuna.export.subscriber import Subscriber, SynchronizedSubscription

ZERO_TENSOR = torch.tensor(0.0).cuda()


class SpectralNormSubscriber(Subscriber):

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
        ylims   :   tuple(int, int)
                    Optional Y-axis limits
        '''
        super().__init__(kinds, tag=tag, subsample=subsample)
        self._subscription      = SynchronizedSubscription(self, tag)
        self._norms             = defaultdict(list)
        self._figure, self._ax  = plt.subplots()
        self._plots             = {}
        self._batches_per_epoch = None
        self._ylims             = ylims
        self._average           = int(average)

        self._ax.set_autoscaley_on(True)
        self._ax.set_title(f'Spectral norms of {self.kinds[0]} per '
                           f'layer (average of {self._average} batches)')
        self._ax.set_xlabel('Spectral norm')
        self._ax.set_xlabel('epoch (start)')
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

        norms = self._norms[module]
        norms.append(norm)

        # every self._average calls, we replace the last self._average elements with their mean
        # possibly a running average would be more efficient, but who's counting
        n_past_norms = self._counter[module] + 1
        if n_past_norms % self._average == 0:
            norms[-self._average:] = (np.mean(norms[-self._average:]),)

    def epoch_finished(self, epoch):
        '''The plot is updated, respecting the ``average`` parameter set. Successive norm values
        are averaged. The plot's X-axis labels are in the unit of epochs, but the actual plot
        resolution is ``batches_per_epoch / subsample / average``.'''
        super().epoch_finished(epoch)

        # exit early if nothing to be done
        if len(self._counter) == 0:
            print('Warning: No norms recorded.', file=sys.stderr)
            return

        counters = self._counter.values()

        assert len(set(counters)) == 1, 'Some modules have different counters.'
        if self._batches_per_epoch is None:
            self._batches_per_epoch = list(counters)[0]

        # create the tick positions and labels so we only get one tick label per epoch, but the
        # resolution of batches
        epoch_range = np.arange(epoch + 1)
        ticks       = epoch_range * self._batches_per_epoch / self._subsample / self._average
        tick_labels = [f'{e}' for e in epoch_range]
        # set ticks and labels
        # TODO: Figure out how to do this with LinearLocator or whatever so we need not do it in
        # every redraw
        self._ax.xaxis.set_major_locator(FixedLocator(ticks))
        self._ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))

        for idx, (module, norms) in enumerate(self._norms.items()):

            if module not in self._plots:
                self._plots[module] = self._ax.plot([], [], label=f'{module}')[0]

            n = len(norms)
            # set the extended data for the plots
            x = np.arange(n)
            self._plots[module].set_xdata(x)
            self._plots[module].set_ydata(norms)

        self._figure.subplots_adjust(right=0.7)
        self._ax.legend(bbox_to_anchor=(1, 0.5), ncol=1)

        # set the axes view to accomodate new data
        if not self._ylims:
            self._ax.relim()
            self._ax.autoscale_view()
        else:
            self._ax.relim()
            self._ax.set_ylim(self._ylims)
            self._ax.autoscale_view(scaley=False)

        # redraw the figure
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        self._figure.show()
