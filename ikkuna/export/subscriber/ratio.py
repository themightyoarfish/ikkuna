from ikkuna.export.subscriber import Subscriber
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.ticker import FixedLocator, FixedFormatter
import torch

ZERO_TENSOR = torch.tensor(0.0).cuda()


class RatioSubscriber(Subscriber):

    '''A :class:`Subscriber` which computes the average ratio between weights and updates per epoch

    Attributes
    ----------
    '''

    def __init__(self, subsample=1, average=1, ylims=None):
        super().__init__(subsample=subsample)
        self._ratios            = defaultdict(list)
        self._figure, self._ax  = plt.subplots()
        self._plots             = {}
        self._batches_per_epoch = None
        self._ylims             = ylims
        self._average           = int(average)

        self._ax.set_autoscaley_on(True)
        self._ax.set_title(f'Update/Weight ratios per layer (average of {average})')
        self._ax.set_xlabel('Mean update ratio')
        self._ax.set_xlabel('Epoch start')

    def _process_data(self, module_data):
        module  = module_data._module
        if (self._counter[module] + 1) % self._subsample == 0:
            weights = module_data._data['weights']
            updates = module_data._data['weight_updates']

            ######################################################################################
            #  We need to see how many NaNs we have and compute the mean only over the non-nans  #
            ######################################################################################
            ratio_tensor = updates.div(weights)
            n            = float(weights.numel())
            n_nans       = torch.isnan(ratio_tensor).sum().to(torch.float32)
            if n_nans > 0:
                ratio_sum    = torch.where(1 - torch.isnan(ratio_tensor), ratio_tensor,
                                           ZERO_TENSOR).sum()
            else:
                ratio_sum   = ratio_tensor.sum()
            ratio = (ratio_sum / (n - n_nans)).item()

            # counter = self._counter[module]
            # # moving average of ratios
            # self._ratios[module] += ratio + counter * self._ratios[module] / counter + 1
            if np.isnan(ratio):
                __import__('ipdb').set_trace()
                raise ValueError(f'NaN value ratio for {module}')

            ratios = self._ratios[module]
            ratios.append(ratio)

    def epoch_finished(self, epoch):
        super().epoch_finished(epoch)

        counters = self._counter.values()
        assert len(set(counters)) == 1, 'Some modules have different counters.'
        if self._batches_per_epoch is None:
            self._batches_per_epoch = list(counters)[0]

        # create the tick positions and labels so we only get one label per epoch, but the
        # resolution of batches
        epoch_range = np.arange(epoch + 1)
        ticks = epoch_range * self._batches_per_epoch / self._subsample / self._average
        tick_labels = [f'{e}' for e in epoch_range]
        # set ticks and labels
        # TODO: Figure out how to do this with LinearLocator or whatever so we need not do it in
        # every redraw
        self._ax.xaxis.set_major_locator(FixedLocator((ticks)))
        self._ax.xaxis.set_major_formatter(FixedFormatter((tick_labels)))

        for idx, (module, ratios) in enumerate(self._ratios.items()):

            if module not in self._plots:
                self._plots[module] = self._ax.plot([], [], linewidth=0.5, label=f'{module}')[0]

            n_avg = self._average
            n = len(ratios)
            # create subsequences of length n_avg, dropping elements as necessary
            chunks = [ratios[i:i+n_avg] for i in range(0, n, n_avg) if i+n_avg <= n]
            ratios_averaged = list(map(np.mean, chunks))

            # set the extended data for the plots
            x = np.arange(n // n_avg)
            self._plots[module].set_xdata(x)
            self._plots[module].set_ydata(ratios_averaged)

        # set the axes view to accomodate new data
        self._ax.legend(ncol=2)
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
