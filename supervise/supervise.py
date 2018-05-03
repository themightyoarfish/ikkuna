'''
.. module:: supervise
.. moduleauthor:: Rasmus Diederichsen

This module gives access to supervision functionality, so that declared modules in a network can
automatically be monitored by a supervisor with minimal code adjustment.

.. highlight:: python
    import torch, supervise

    with supervise.capture_modules(torch.nn.Conv2d, torch.nn.ReLU):
        features = nn.Sequential(
            nn.Conv2d(...),
            nn.ReLU(...),
            nn.MaxPool2d(...),
            ...
        )

This will create a supervisor which then captures the modules whose classes are named.

The module maintains a stack of supervisors so that `with` blocks can be nested (for whatever
reason).

.. warning::
    This module is not threadsafe.
'''
_supervisor_stack = []

from patches import nn_module
import torch.nn as nn

def supervisors():
    '''Get the stack of all supervisors'''
    global _supervisor_stack
    return _supervisor_stack

def current_supervisor():
    '''Get topmost supervisor.'''
    global _supervisor_stack
    return None if not _supervisor_stack else _supervisor_stack[-1]

def capture_modules(*modules, allow_subclass=False):
    '''Create a new supervisor for capturing modules.

    Parameters
    ----------
    modules :   list(type)
                Arbitrary types to register on creation
    '''
    return Supervisor(modules, allow_subclass=allow_subclass)

class Supervisor():
    '''A context manager for tracking the creation of nn modules.

    Attributes
    ----------
    _modules    :   list
                    List of modules created in this context manager
    _valid_predicate    :   function
                            Predicate based on the ``allow_subclass`` parameter for determining
                            whether to accept a newly created module for supervision.
    _activation_observers :   set
                            Set of observers for activation updates
    _gradient_observers :   set
                            Set of observers for gradient updates
    _output_observers   :   set
                            Set of observers for activations of only the most recently added module.
                            This makes the assumption that the final module can be treated as the
                            output layer in a classification task.
    '''

    def __init__(self, *allowed_modules, allow_subclass: bool=False):
        '''
        Parameters
        ----------
        allowed_modules :   list(type)
                            List of module classes for supervision
        allow_subclass  :   bool
                            Allow calling :meth:`add_module` with subclass instances of any
                            ``allowed_modules`` member

        '''
        self._modules = []

        if allow_subclass:
            self._valid_predicate = lambda m: m.__class__ in allowed_modules
        else:
            self._valid_predicate = lambda m: any(isinstance(m, cls) for cls in allowed_modules)

        self._gradient_observers   = set()
        self._activation_observers = set()
        self._output_observers = set()
        self._last_forward_hook_handle = None

    def _check_module_is_supervised(self, module: nn.Module):
        '''Check if a module is tracked.

        Raises
        ------
        ValueError
            If ``module`` is not currently supervised.
        '''
        if not module in self._modules:
            raise ValueError(f'Module {module} not registered with this Supervisor')

    def register_activation_observer(self, observer):
        '''Register and observer for handling new activations.

        Parameters
        ----------
        observer    :   object
        '''
        self._activation_observers.add(observer)

    def register_gradient_observer(self, observer):
        '''Register and observer for handling new gradients.

        Parameters
        ----------
        observer    :   object
        '''
        self._gradient_observers.add(observer)

    def register_output_observer(self, observer):
        '''Register and observer for handling outputs of last layer

        Parameters
        ----------
        observer    :   object
        '''
        self._output_observers.add(observer)

    def _process_activations(self, module, in_, out_):
        '''Hook to register on modules for receiving activation updates. See
        :meth:`nn.Module.register_forward_hook`.

        Raises
        ------
        ValueError
            If ``module`` was not previously added with :meth:`add_module`.
        '''
        self._check_module_is_supervised(module)
        for o in self._activation_observers:
            o.process_activations(module, out_)

    def _process_gradients(self, module, grad_in_, grad_out_):
        '''Hook to register on modules for receiving gradient updates. See
        :meth:`nn.Module.register_backward_hook`

        Raises
        ------
        ValueError
            If ``module`` was not previously added with :meth:`add_module`.
        '''
        self._check_module_is_supervised(module)
        for o in self._gradient_observers:
            o.process_gradients(module, grad_out_)

    def _process_outputs(self, module, in_, out_):
        for o in self._output_observers:
            o.process_outputs(module, out_)

    def add_module(self, module: nn.Module):
        '''Add a module to supervise.'''
        # attention: This call can be executed before the module is fully constructed, in case
        # nn.Module.__init__ is patched to call this method. In that case, the concrete module's
        # initializer will call into the super class and we land here. So do not reference any
        # object attributes here, only class ones.
        # TODO: Think about the safety of this injection some more for production purposes
        if self._valid_predicate(module):
            self._modules.append(module)
            module.register_forward_hook(self._process_activations)
            module.register_backward_hook(self._process_gradients)

            # new last module in the pipeline, so update the output hook to only apply to this
            # module
            if self._last_forward_hook_handle:
                self._last_forward_hook_handle.remove()
            self._last_forward_hook_handle = module.register_forward_hook(self._process_outputs)

    def __enter__(self):
        global _supervisor_stack
        _supervisor_stack.append(self)

    def __exit__(self, exc_type, exc_val, traceback):
        global _supervisor_stack
        assert self == _supervisor_stack[-1]    # sanity check
        _supervisor_stack.pop()

from collections import defaultdict
from abc import ABC, abstractmethod

class Handler(ABC):
    '''Abstrac base class for all visualisers.'''

    @abstractmethod
    def on_epoch_started(self):
        '''Should be invoked each time _before_ a new epoch begins.'''
        raise NotImplementedError

    @abstractmethod
    def on_epoch_finished(self):
        '''Should be invoked each time a new epoch _has finished_.'''
        raise NotImplementedError

    @abstractmethod
    def on_training_started(self):
        '''Should be invoked _before_ starting a training session. This serves to distinguish e.g.
        multiple runs of the experiment with different weight initializations and the like.'''
        raise NotImplementedError

class ActivationHandler(ABC):
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
    _epoch_means    :   defaultdict(list)
                        Dictionary keeping per-module average activations for each epoch.
    _fig    :   plt.Figure
                Figure for plotting
    _ax :   plt.AxesSubplot
    _plots  :   dict
                Per-module `Lines2D` object representing the module's plot.
    _counter    :   int
                    Total umber of activations seen. `_counter / len(_accumulator.keys())` is the
                    number of propagations seen.
    _monitor_testing    :   bool
    '''

    def __init__(self, monitor_testing=False):
        '''
        Parameters
        ----------
        monitor_testing :   bool
                            Whether or not test rungs (where the module's `training` property is
                            `False`) should also be plotted. If `True`, no distinctino is being made
                            and the test activations also increase the counter for averaging.
                            Default is `False`.
        '''
        super().__init__()
        self._accumulator   = defaultdict(float)
        self._epoch_means   = defaultdict(list)
        self._fig, self._ax = plt.subplots()
        self._plots         = {}
        self._ax.set_autoscaley_on(True)
        self._fig.suptitle('Mean layer activations per epoch')
        self._ax.set_xlabel('Epoch')
        self._ax.set_ylabel('Mean activation')
        self._ax.set_yscale('symlog')
        self._counter = 0
        self._ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._monitor_testing = monitor_testing

    def process_activations(self, module, activations):
        if module.training or self._monitor_testing:
            self._accumulator[module] += activations[0].mean().item()
            self._counter += 1
        else:
            pass

    def on_epoch_started(self):
        pass

    def on_epoch_finished(self):

        iterations = self._counter / len(self._accumulator.keys())
        for module, mean_activation in self._accumulator.items():
            # reset epoch accumulator
            self._accumulator[module] = 0
            self._epoch_means[module].append(mean_activation / iterations)

        self._counter = 0

        for idx, (module, mean_activations) in enumerate(self._epoch_means.items()):

            if module not in self._plots:
                self._plots[module] = self._ax.plot([], [],
                                                    label=f'{module.__class__.__name__}-{idx}')[0]

            # set the extended data for the plots
            epoch = len(self._epoch_means[module])
            x = np.arange(1, epoch+1)
            self._plots[module].set_xdata(x)
            self._plots[module].set_ydata(mean_activations)

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

import torch
from torch.utils.data import DataLoader

class EpochLossHandler(Handler):

    '''Handler for plotting the test loss after each epoch.

    Attributes
    ----------
    _dataloader    :    torch.utils.data.DataLoaderj
                        The loaded dataset (test) instance to load from.
    _loss_function  :   torch.nn._Loss
                        Loss function to use
    _fig    :   plt.Figure
    _ax_acc :   plt.AxesSubplot
                Subplot in which the accuracy is displayed
    _ax_loss    :   plt.AxesSubplot
                    Subplot in which the loss is displayed
    _plot_acc   :   plt.Lines2D
    _plot_loss  :   plt.Lines2D
    _label_iter :   iterator
                    Iterator for the label batches
    _forward_counter    :   int
                            Counter for the number of forward passes seen.
    _cum_loss   :   float
                    Accumulator for the network loss
    _cum_accuracy   :   float
                        Accumulator for the network accuracy
    '''

    def __init__(self, test_loader, loss_function):
        super().__init__()
        self._dataloader      = test_loader
        self._loss_function   = loss_function

        ###########################################################################################
        #                                       Plot setup                                        #
        ###########################################################################################
        self._fig, (self._ax_loss, self._ax_acc) = plt.subplots(1, 2)
        self._fig.suptitle('Test loss + accuracy per epoch')

        # set autoscaling because we update the plots
        self._ax_acc.set_autoscaley_on(True)
        self._ax_loss.set_autoscaley_on(True)
        self._ax_acc.set_xlabel('Epoch')
        self._ax_loss.set_xlabel('Epoch')
        self._ax_acc.set_ylabel('Accuracy')
        self._ax_loss.set_ylabel('Loss')
        self._ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._plot_acc, = self._ax_acc.plot([], [])
        self._plot_loss, = self._ax_loss.plot([], [])
        self._reset()

    def _reset(self):
        '''Prepare the Handler for a new test run by resetting all counters and reinitialising the
        label iterator.'''
        # loader yields data, labels; we only need labels
        def second(t):
            return t[1]
        data_label_iter       = iter(self._dataloader)
        self._label_iter      = map(second, data_label_iter)
        self._forward_counter = 0
        self._cum_loss        = 0
        self._cum_accuracy    = 0

    def process_outputs(self, module, out_):
        if module.training:
            # ignore training passes
            pass
        else:
            labels = next(self._label_iter)
            self._forward_counter += 1
            # loss
            self._cum_loss += self._loss_function(out_, labels.cuda())

            # accuracy
            correct_predictions    = torch.eq(labels.cuda(), torch.argmax(out_, dim=1))
            n_predictions          = 100 * self._dataloader.batch_size
            self._cum_accuracy    += correct_predictions.sum().to(torch.float) / n_predictions


    def on_epoch_finished(self):
        '''Update plot data on each epoch.'''
        x = self._plot_acc.get_xdata()
        # append the current epoch number to x data
        x = np.append(x, x[-1]+1 if x.size > 0 else 1)
        self._plot_acc.set_xdata(x)
        self._plot_loss.set_xdata(x)

        accuracies = self._plot_acc.get_ydata()
        accuracies = np.append(accuracies, self._cum_accuracy)
        self._plot_acc.set_ydata(accuracies)

        losses = self._plot_loss.get_ydata()
        losses = np.append(losses, self._cum_loss.item() / self._forward_counter)
        self._plot_loss.set_ydata(losses)

        self._ax_acc.relim()
        self._ax_acc.autoscale_view()
        self._ax_loss.relim()
        self._ax_loss.autoscale_view()

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        self._fig.show()

        self._reset()

    def on_epoch_started(self):
        pass

    def on_training_started(self):
        pass
