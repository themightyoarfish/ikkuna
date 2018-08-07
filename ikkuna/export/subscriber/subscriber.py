'''
.. moduleauthor:: Rasmus Diederichsen

.. module:: subscriber

This module contains the base definition for subscriber functionality. The :class:`Subscriber` class
should be subclassed for adding new metrics.

'''
import abc
from collections import defaultdict
from ikkuna.visualization import TBBackend, MPLBackend


class ModuleData(object):

    '''Data object for holding a set of artifacts for a module at one point during training.
    This data type can be used to buffer different kinds and check whether all expected kinds have
    been received for a module.

    Attributes
    ----------
    _module :   str
                Name of the layer
    _expected_kinds :   list(str)
                        The expected kinds of messages per iteration
    _data   :   dict(str, torch.Tensor)
                The tensors received for each kind
    _seq    :   int
                Sequence number of this object (incremented whenever a :class:`ModuleData` is
                created)
    _step   :   int
                Sequence number (training step) of the received messages (should match across all
                msgs in one iteration)
    _epoch  :   int
                Epoch of the received messages (should match across all msgs in one
                iteration)
    '''

    global_seq = 0

    def __init__(self, module, kinds):
        self._module         = module
        self._expected_kinds = kinds
        self._data           = {kind: None for kind in kinds}
        self._seq            = ModuleData.global_seq
        self._step           = None
        self._epoch          = None
        ModuleData.global_seq  += 1

    @property
    def module(self):
        return self._module

    @property
    def expected_kinds(self):
        return self._expected_kinds

    @property
    def data(self):
        return self._data

    @property
    def seq(self):
        return self._seq

    @property
    def step(self):
        return self._step

    def epoch(self):
        return self._epoch

    def complete(self):
        '''Check i all expected messages have been received. This means the message can be released
        to subscribers.

        Returns
        -------
        bool
        '''
        # check if any _data entry is still None
        return all(map(lambda val: val is not None, self._data.values()))

    def _check_step(self, step):
        '''Check step consistency or set current step, if not set.

        Parameters
        ----------
        step    :   int
                    Step number to check

        Raises
        ------
        ValueError
            If ``step`` does not match the current step.
        '''
        if not self._step:
            self._step = step

        if step != self._step:
            raise ValueError(f'Attempting to add message with step {step} to bundle with '
                             'initial step {self._step}')

    def _check_epoch(self, epoch):
        '''Check epoch consistency or set current epoch, if not set.

        Parameters
        ----------
        epoch    :   int
                    Step number to check

        Raises
        ------
        ValueError
            If ``epoch`` does not match the current epoch.
        '''
        if not self._epoch:
            self._epoch = epoch

        if epoch != self._epoch:
            raise ValueError(f'Attempting to add message from epoch {epoch} to bundle with '
                             'initial epoch {self._epoch}')

    def add_message(self, message):
        '''Add a new message to this object. Will fail if the new messsage does not have the same
        sequence number and epoch.

        Parameters
        ----------
        message :   ikkuna.export.messages.NetworkData

        Raises
        ------
        ValueError
            If the message is for a different module (layer) or a message of this kind was already
            received.
        '''
        self._check_step(message.step)
        self._check_epoch(message.epoch)

        if self._module != message.module:
            raise ValueError(f'Unexpected module "{message.module}" (expected "{self._module}")')

        if self._data[message.kind] is not None:
            raise ValueError(f'Got duplicate value for kind "{message.kind}".')

        self._data[message.kind] = message.payload

    def __getattr__(self, name):
        '''Override to mimick a property for each kind of message in this data (e.g.
        ``activations``)'''
        if name in self._expected_kinds:
            return self._data[name]
        else:
            return self.__getattribute__(name)

    def __str__(self):
        mod   = self._module
        step  = self._step
        kinds = list(self._data.keys())
        return f'<ModuleData: module={mod}, kinds={kinds}, step={step}>'

    def __repr__(self):
        return str(self)


class Subscription(object):

    '''Specification for a subscription that can span multiple kinds and a tag.

    Attributes
    ----------
    _tag    :   str
                Tag for filtering the processed messages
    _subscriber :   Subscriber
                    The subscriber associated with the subscription
    '''

    def __init__(self, subscriber, tag=None):
        '''
        Parameters
        ----------
        subscriber  :   Subscriber
                        Object that wants to receive the messages
        tag :   str or None
                Optional tag for filtering messages. If ``None`` is passed, all messages will be
                relayed
        '''
        self._tag        = tag
        self._subscriber = subscriber

    def _process_message(self, network_data):
        '''Process a newly arrived message. Subclasses should override this method for any special
        treatment.

        Parameters
        ----------
        network_data    :   ikkuna.messages.NetworkData
        '''
        data = ModuleData(network_data.module, network_data.kind)
        data.add_message(network_data)
        self._subscriber(data)

    def receive_message(self, network_data):
        '''Callback for receiving an incoming message.

        Parameters
        ----------
        network_data    :   ikkuna.messages.NetworkData
        '''
        if network_data.kind not in self._subscriber.kinds:
            return

        if self._tag is None or self._tag == network_data.tag:
            self._process_message(network_data)


class SynchronizedSubscription(Subscription):
    '''A subscription which buffers messages and publishes a set of messages, each of a different
    kind, when one round (a train step) is over. This is useful for receiving several kinds of
    messages in each train step and always have them be processed together.'''

    def __init__(self, subscriber, tag=None):
        super().__init__(subscriber, tag)
        self._current_seq = None
        self._modules     = {}
        self._step        = None

    def _new_round(self, seq):
        '''Start a new round of buffering, clearing the previous cache and resetting the record for
        which kinds were received in this round.

        Parameters
        ----------
        seq :   int
                Sequence number for the new round

        Raises
        ------
        RuntimeError
            If not all desired kinds have been received yet in the current round
        '''
        for bundle in self._modules.values():
            if not bundle.complete():
                raise ValueError(f'Bundle for module {bundle._module} not yet complete.')
        self._current_seq = seq
        self._modules     = {}

    def _process_message(self, network_data):
        '''Start a new round if a new sequence number is seen.'''

        # if we get a new sequence number, a new train step must have begun
        if self._current_seq is None or self._current_seq != network_data.seq:
            self._new_round(network_data.seq)

        # module not seen -> init data
        module = network_data.module
        if module not in self._modules:
            self._modules[module] = ModuleData(module, self._subscriber.kinds)
        self._modules[module].add_message(network_data)

        delete_these = []
        # all full? publish
        for module, data in self._modules.items():
            if data.complete():
                self._subscriber.process_data(data)
                delete_these.append(module)

        for module in delete_these:
            del self._modules[module]


class Subscriber(abc.ABC):

    '''Base class for receiving and processing activations, gradients and other stuff into
    insightful metrics.

    Attributes
    ----------
    _counter    :   dict(str, int)
                    Number of times the subscriber was called for each module label
    kinds   :   list(str)
                List of string identifiers for different message kinds. These are all the
                message kinds the subscriber wishes to receive
    '''

    def __init__(self, kinds, subscription, tag=None, subsample=1):
        self._counter       = defaultdict(int)
        self._subsample     = subsample
        self._kinds         = kinds
        self._current_epoch = 0
        self._subscription  = subscription

    @property
    def kinds(self):
        return self._kinds

    @kinds.setter
    def kinds(self, kinds):
        self._kinds = kinds

    @abc.abstractmethod
    def _metric(self, module_data):
        pass

    def receive_message(self, network_data):
        self._subscription.receive_message(network_data)

    def process_data(self, module_data):
        '''Callback for processing a :class:`ModuleData` object. The exact nature of this
        package is determined by the :class:`Subscription` attached to this :class:`Subscriber`.

        Parameters
        ----------
        module_data    :   ModuleData

        Raises
        ------
        ValueError
            If the received :class:`ModuleData` object is not :meth:`ModuleData.complete()`
        '''
        if not module_data.complete():
            raise ValueError(f'Data received for "{module_data._module}" is not complete.')

        module = module_data.module
        # only do work for subsample of messages
        if (self._counter[module] + 1) % self._subsample == 0:
            self._metric(module_data)
        self._counter[module] += 1

    @abc.abstractmethod
    def epoch_finished(self, epoch):
        '''Called automatically by the :class:`ikkuna.export.Exporter` object when an epoch has just
        finished.

        Parameters
        ----------
        epoch   :   int
                    0-based epoch index
        '''
        self._current_epoch = epoch


from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import sys


class PlotSubscriber(Subscriber):

    '''Base class for subscribers that output scalar or histogram values per time and module

    Attributes
    ----------
    _metric_values :    dict(str, list)
                        Per-module record of the scalar metric values for each batch
    _figure :   plt.Figure
                Figure to plot metric values in (will update continuously)
    _ax     :   plt.AxesSubplot
                Axes containing the plots
    _batches_per_epoch  :   int
                            Inferred number of batches per epoch. This relies on each epoch being
                            full-sized (no smaller last batch)
    '''

    def __init__(self, kinds, subscription,  plot_config, tag=None, subsample=1, backend='tb'):
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
        super().__init__(kinds, subscription, tag, subsample)

        if backend not in ('tb', 'mpl'):
            raise ValueError('Backend must be "tb" or "mpl"')
        if backend == 'tb':
            self._backend = TBBackend(**plot_config)
        else:
            self._backend = MPLBackend(**plot_config)

    @abc.abstractmethod
    def _metric(self, module_data):
        pass

    def epoch_finished(self, epoch):
        '''The plot is updated, respecting the ``average`` parameter set. Successive metric values
        resolution is ``batches_per_epoch / subsample / average``.'''
        super().epoch_finished(epoch)
