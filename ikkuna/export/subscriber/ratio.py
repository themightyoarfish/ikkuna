from ikkuna.export.subscriber import Subscriber
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.ticker import FixedLocator, FixedFormatter


class RatioSubscriber(Subscriber):

    '''A :class:`Subscriber` which computes the average ratio between weights and updates per epoch

    Attributes
    ----------
    '''

    def __init__(self):
        super().__init__()
        self._ratios = defaultdict(list)
        self._figure, self._ax = plt.subplots()
        self._ax.set_autoscaley_on(True)
        self._ax.set_title('Update/Weight ratios per layer')
        self._ax.set_xlabel('Mean update ratio')
        self._ax.set_xlabel('Epoch start')
        self._plots = {}
        self._batches_per_epoch = None

    def __call__(self, module_data):
        super().__call__(module_data)

        module  = module_data._module
        weights = module_data._data['weights']
        updates = module_data._data['weight_updates']
        ratio   = updates.div(weights).mean().item()
        # counter = self._counter[module]
        # # moving average of ratios
        # self._ratios[module] += ratio + counter * self._ratios[module] / counter + 1
        self._ratios[module].append(ratio)

    def epoch_finished(self, epoch):
        super().epoch_finished(epoch)

        counters = self._counter.values()
        assert len(set(counters)) == 1, 'Some modules have different counters.'
        if self._batches_per_epoch is None:
            self._batches_per_epoch = list(counters)[0]

        # create the tick positions and labels so we only get one label per epoch, but the
        # resolution of batches
        ticks = np.arange(epoch + 1) * self._batches_per_epoch
        tick_labels = [f'{e}' for e in range(epoch + 1)]
        # set ticks and labels
        # TODO: Figure out how to do this with LinearLocator or whatever so we need not do it in
        # every redraw
        self._ax.xaxis.set_major_locator(FixedLocator((ticks)))
        self._ax.xaxis.set_major_formatter(FixedFormatter((tick_labels)))

        for idx, (module, ratios) in enumerate(self._ratios.items()):

            if module not in self._plots:
                self._plots[module] = self._ax.plot([], [], linewidth=0.5, label=f'{module}')[0]

            # set the extended data for the plots
            x = np.arange(len(ratios))
            self._plots[module].set_xdata(x)
            self._plots[module].set_ydata(ratios)

        # set the axes view to accomodate new data
        self._ax.legend(ncol=2)
        self._ax.relim()
        self._ax.autoscale_view()

        # redraw the figure
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        self._figure.show()





        # modules  = list(self._ratios.keys())
        # ratioses = list(self._ratios.values())
        # self._ratios.clear()
        # figure   = plt.figure()
        # ax       = figure.gca()
        # for module, ratios in zip(modules, ratioses):
        #     ax.plot(np.arange(self._counter[module]), ratios, label=module)
        # ax.grid(True)
        # ax.legend()
        # figure.show()
