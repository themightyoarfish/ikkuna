import sys
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
from collections import defaultdict
import torch

from ikkuna.export.subscriber import Subscriber, SynchronizedSubscription

ZERO_TENSOR = torch.tensor(0.0).cuda()


class RatioSubscriber(Subscriber):

    '''A :class:`Subscriber` which computes the average ratio between two quantities.  The dividend
    will be the first element of th :attr:`Subscriber.kinds` property, the divisor the second.
    Therefore it is vital to pass the message kinds to the
    :class:`ikkuna.export.subsciber.Subscription` object in the correct order.

    Attributes
    ----------
    _ratios :   dict(str, list)
                Per-module record of ratios for each batch
    _figure :   plt.Figure
                Figure to plot ratios in (will update continuously)
    _ax     :   plt.AxesSubplot
                Axes containing the plots
    _batches_per_epoch  :   int
                            Inferred number of batches per epoch. This relies on each epoch being
                            full-sized (no smaller last batch)
    _ylims  :   tuple
                Y-axis limits for the plot. If ``None``, axis will be automatically scaled
    _average    :   int
                    Number of successive ratios to average for the plot
    '''

    def __init__(self, kinds, tag=None, subsample=1, average=1, ylims=None):
        '''
        Parameters
        ----------
        subsample   :   int
                        Factor for subsampling incoming messages. Only every ``subsample``-th
                        message will be processed.
        average :   int
                    Inverse resolution of the plot. For plotting ``average`` ratios will be averaged
                    for each module to remove noise.
        ylims   :   tuple(int, int)
                    Optional Y-axis limits
        '''
        super().__init__(kinds, tag=tag, subsample=subsample)
        self._subscription      = SynchronizedSubscription(self, tag)
        self._ratios            = defaultdict(list)
        self._figure, self._ax  = plt.subplots()
        self._plots             = {}
        self._batches_per_epoch = None
        self._ylims             = ylims
        self._average           = int(average)

        self._ax.set_autoscaley_on(True)
        self._ax.set_title(f'{self.kinds[0]}/{self.kinds[1]} ratios per layer '
                           f'(average of {self._average} batches)')
        self._ax.set_xlabel('Ratio')
        self._ax.set_xlabel('epoch (start)')

    def _metric(self, module_data):
        '''The ratio between the two kinds is computed over the subset of not-NaN values and added
        to the record.'''
        module  = module_data._module

        dividend = module_data._data[self.kinds[0]]
        divisor  = module_data._data[self.kinds[1]]

        ######################################################################################
        #  We need to see how many NaNs we have and compute the mean only over the non-nans  #
        ######################################################################################
        ratio_tensor = dividend.div(divisor)
        n            = float(divisor.numel())
        nan_tensor   = torch.isnan(ratio_tensor)
        n_nans       = nan_tensor.sum().to(torch.float32)
        if n_nans > 0:
            ratio_sum = torch.where(1 - nan_tensor, ratio_tensor, ZERO_TENSOR).sum()
        else:
            ratio_sum = ratio_tensor.sum()
        ratio = (ratio_sum / (n - n_nans)).item()

        if np.isnan(ratio):
            raise ValueError(f'NaN value ratio for {module}')

        ratios = self._ratios[module]
        ratios.append(ratio)

        # every self._average calls, we replace the last self._average elements with their mean
        # possibly a running average would be more efficient, but who's counting
        n_past_ratios = self._counter[module] + 1
        if n_past_ratios % self._average == 0:
            ratios[-self._average:] = (np.mean(ratios[-self._average:]),)

    def epoch_finished(self, epoch):
        '''The plot is updated, respecting the ``average`` parameter set. Successive ratio values
        are averaged. The plot's X-axis labels are in the unit of epochs, but the actual plot
        resolution is ``batches_per_epoch / subsample / average``.'''
        super().epoch_finished(epoch)

        # exit early if nothing to be done
        if len(self._counter) == 0:
            print('Warning: No ratios recorded.', file=sys.stderr)
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
        self._ax.xaxis.set_major_locator(FixedLocator((ticks)))
        self._ax.xaxis.set_major_formatter(FixedFormatter((tick_labels)))

        for idx, (module, ratios) in enumerate(self._ratios.items()):

            if module not in self._plots:
                self._plots[module] = self._ax.plot([], [], label=f'{module}')[0]

            n = len(ratios)
            # set the extended data for the plots
            x = np.arange(n)
            self._plots[module].set_xdata(x)
            self._plots[module].set_ydata(ratios)

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
