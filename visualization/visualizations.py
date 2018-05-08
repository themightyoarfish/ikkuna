from collections import defaultdict
from abc import ABC, abstractmethod
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt
import torch


class Handler(ABC):
    '''Abstract base class for all visualisers.

    Attributes
    ----------
    _step   :   int
                Step size, i.e. the number of samples (*not batches*) to observer before each plot
                update
    _current_epoch  :   int
                        Incremented in :meth:`on_epoch_started()` and is thus 1-based
    _modules    :   list(torch.nn.Module)
                    List of modules seen in order to know which modules are being supervised and
                    when a batch is finished.
    '''

    def __init__(self, step):
        self._step          = step
        self._current_epoch = 0
        self._modules       = []

    def add_module(self, module):
        '''Add a module. Noop if already in the list of modules.

        Parameters
        ----------
        module  :   torch.nn.Module
        '''
        if module not in self._modules:
            self._modules.append(module)

    @abstractmethod
    def on_epoch_started(self):
        '''Should be invoked each time *before* a new epoch begins.'''
        self._current_epoch += 1

    @abstractmethod
    def on_epoch_finished(self):
        '''Should be invoked each time *after* a new epoch begins.'''
        raise NotImplementedError

    @abstractmethod
    def update_display(self):
        '''Should be invoked each time a new epoch has finished.'''
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
    '''Base class for gradient handlers.'''

    @abstractmethod
    def process_gradients(self, module, gradients):
        '''Callback for accessing a module's activations during the forward pass.

        Parameters
        ----------
        module  :   nn.Module
        gradients :   tuple or torch.Tensor
        '''
        raise NotImplementedError


class MeanActivationHandler(ActivationHandler):
    '''A handler which computes and plots average activations per layer for each epoch.

    .. note::
        This is mostly useless and just a proof of concept. Don't try to deduce anything from these
        plots.

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
                            Total umber of activations seen since last call to
                            :meth:`update_display`
    _monitor_testing    :   bool
    '''

    def __init__(self, step, monitor_testing=False):
        '''
        Parameters
        ----------
        step    :   int
        monitor_testing :   bool
                            Whether or not test rungs (where the module's `training` property is
                            `False`) should also be plotted. If `True`, no distinctino is being made
                            and the test activations also increase the counter for averaging.
                            Default is `False`.
        '''
        super().__init__(step)

        def default_tensor():
            return torch.tensor(0, dtype=torch.float32).cuda()

        self._accumulator     = defaultdict(default_tensor)
        self._means           = defaultdict(list)
        self._fig, self._ax   = plt.subplots()
        self._plots           = {}
        self._monitor_testing = monitor_testing
        self._first_module    = None
        self._datapoints_seen = 0

        ###########################################################################################
        #                                    Set up the figure                                    #
        ###########################################################################################
        self._ax.set_autoscaley_on(True)
        self._fig.suptitle(f'Mean activations over every {self._step} '
                           'data points (rounded to batch size)')
        self._ax.set_xlabel(f'Epoch (end)')
        self._ax.set_ylabel('Mean activation')
        self._ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # disable ticks as we handle them manually in on_epoch_finished()
        self._ax.set_xticks([], [])

    def process_activations(self, module, activations):
        '''Receive activations for module. We keep a reference to the first module seen, so we know
        when a batch is finished, since we should get one call for each layer for each batch. When
        we see the first module again, we can check if the step size mandates a plot update and
        increment the counter of samples seen

        .. warning::
            This assumes all modules get activated during each batch, and in the same order.

        '''

        # record the first module so we always know when a batch is finished.
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
            self._accumulator[module] += activations.sum()
        else:
            # do nothing
            pass

    def update_display(self):
        '''Update the plot by appending the mean activation over the most recent ``_step``
        activations to the records and relimiting the graphs.'''

        for module, cum_activation in self._accumulator.items():
            # reset step accumulator
            self._accumulator[module] = 0
            self._means[module].append(cum_activation.item() / self._datapoints_seen)

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
        '''Here happens the tick magic. When an epoch finishes, we need to append a new tick with
        out ``_current_epoch`` as a label, but the value will be something entirely different.'''
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
