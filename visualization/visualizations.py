from collections import defaultdict
from abc import ABC, abstractmethod


class Handler(ABC):
    '''Abstract base class for all visualisers.'''

    def __init__(self, step):
        self._step = step
        self._current_epoch = 0
        self._modules = []

    def add_module(self, module):
        if module not in self._modules:
            self._modules.append(module)

    @abstractmethod
    def on_epoch_started(self):
        '''Should be invoked each time _before_ a new epoch begins.'''
        self._current_epoch += 1

    @abstractmethod
    def on_epoch_finished(self):
        '''Should be invoked each time _before_ a new epoch begins.'''
        raise NotImplementedError

    @abstractmethod
    def update_display(self):
        '''Should be invoked each time a new epoch _has finished_.'''
        raise NotImplementedError

    @abstractmethod
    def on_training_started(self):
        '''Should be invoked _before_ starting a training session. This serves to distinguish e.g.
        multiple runs of the experiment with different weight initializations and the like.'''
        self._current_epoch = 0
        raise NotImplementedError


class ActivationHandler(Handler):
    '''Base class for activation handlers.'''

    @abstractmethod
    def process_activations(self, module, activations):
        '''Callback for accessing a module's activations during the forward pass.

        Parameters
        ----------
        module  :   nn.Module
        activations :   tuple or torch.Tensor
        '''
        raise NotImplementedError


class GradientHandler(Handler):

    @abstractmethod
    def process_gradients(self, module, gradients):
        '''Callback for accessing a module's activations during the forward pass.

        Parameters
        ----------
        module  :   nn.Module
        gradients :   tuple or torch.Tensor
        '''
        raise NotImplementedError


import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt


class MeanActivationHandler(ActivationHandler):
    '''A handler which computes and plots average activations per layer for each epoch.

    Attributes
    ----------
    _accumulator    :   defaultdict(float)
                        Dictionary with one float value for each module seen. Used to sum up the
                        activations.
    _means  :   defaultdict(list)
                Dictionary keeping per-module average activations for each sequence step
    _fig    :   plt.Figure
                Figure for plotting
    _ax :   plt.AxesSubplot
    _plots  :   dict
                Per-module `Lines2D` object representing the module's plot.
    _datapoints_seen    :   int
                            Total umber of activations seen since last call to :meth:`update_display`
    _monitor_testing    :   bool
    '''

    def __init__(self, step, monitor_testing=False):
        '''
        Parameters
        ----------
        monitor_testing :   bool
                            Whether or not test rungs (where the module's `training` property is
                            `False`) should also be plotted. If `True`, no distinctino is being made
                            and the test activations also increase the counter for averaging.
                            Default is `False`.
        '''
        super().__init__(step)
        self._accumulator     = defaultdict(float)
        self._means           = defaultdict(list)
        self._fig, self._ax   = plt.subplots()
        self._plots           = {}
        self._monitor_testing = monitor_testing
        self._first_module    = None
        self._datapoints_seen = 0

        self._ax.set_autoscaley_on(True)
        self._fig.suptitle(f'Mean activations over every {self._step} data points')
        self._ax.set_xlabel(f'Epoch (end)')
        self._ax.set_ylabel('Mean activation')
        self._ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._ax.set_xticks([], [])

    def process_activations(self, module, activations):

        # record the first module so we always know when a batch is finished. This assumes all
        # modules get activated during each batch, and in the same order.
        if not self._first_module:
            self._first_module = module

        # we are beginning a new set of activations. Whenever we are at the first module, we check
        # if we need to make another step. If yes, update the plot, reset data counter. Else just
        # increment the data counter.
        if module == self._first_module:
            if self._datapoints_seen >= self._step:
                self.update_display()
                self._datapoints_seen = 0

            self._datapoints_seen += activations.shape[0]

        if module.training or self._monitor_testing:
            # TODO: Make float tensor
            self._accumulator[module] += activations.sum().item()
        else:
            # do nothing
            pass

    def update_display(self):

        for module, cum_activation in self._accumulator.items():
            # reset step accumulator
            self._accumulator[module] = 0
            self._means[module].append(cum_activation / self._datapoints_seen)

        for idx, (module, mean_activations) in enumerate(self._means.items()):

            if module not in self._plots:
                self._plots[module] = self._ax.plot([], [],
                                                    label=f'{module.__class__.__name__}-{idx}')[0]

            # set the extended data for the plots
            self._plots[module].set_ydata(mean_activations)
            self._plots[module].set_xdata(np.arange(len(mean_activations)))

        # set the axes view to accomodate new data
        self._ax.legend(ncol=2)
        self._ax.relim()
        self._ax.autoscale_view()

        # redraw the figure
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        self._fig.show()

    def on_training_started(self):
        pass

    def on_epoch_started(self):
        super().on_epoch_started()

    def on_epoch_finished(self):
        plots = list(self._plots.values())
        plot = plots[0]
        xdata = plot.get_xdata()

        current_xticks = self._ax.get_xticks().tolist()
        current_xticks.append(xdata[-1])
        current_xticklabels = list(self._ax.get_xticklabels())
        current_xticklabels.append(self._current_epoch)
        # new_xtick_labels = [str(idx) for idx in range(len(current_xticks))]
        self._ax.set_xticks(current_xticks)
        self._ax.set_xticklabels(current_xticklabels)

        # self._ax.set_xticks([xdata[-1]])
        # ticks = self._ax.get_xticks().tolist()
        # __import__('ipdb').set_trace()
        # xticks = self._ax.get_xticks()
        # last_step = xdata[-1] if len(xdata) > 0 else 0
