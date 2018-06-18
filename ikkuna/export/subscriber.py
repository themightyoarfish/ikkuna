'''
.. moduleauthor:: Rasmus Diederichsen

.. module:: subscriber

This module contains the base definition for subscriber functionality. The :class:`Subscriber` class
should be subclassed for adding new metrics.

'''
import abc
import numpy as np
from collections import defaultdict


class ModuleData(object):

    step = 0

    def __init__(self, module, kinds):
        self._module = module
        self._data = {kind: None
                      for kind in kinds}
        self._seq  = ModuleData.step
        ModuleData.step += 1
        self._step = None
        self._epoch = None

    def complete(self):
        return all(map(lambda val: val is not None, self._data.values()))

    def _check_step(self, step):
        if not self._step:
            self._step = step

        if step != self._step:
            raise ValueError(f'Attempting to add message with step {step} to bundle with '
                             'initial step {self._step}')

    def _check_epoch(self, epoch):
        if not self._epoch:
            self._epoch = epoch

        if epoch != self._epoch:
            raise ValueError(f'Attempting to add message from epoch {epoch} to bundle with '
                             'initial epoch {self._epoch}')

    def add_message(self, message):
        # print(f'{message.seq}, {message.module}, {message.kind}')
        self._check_step(message.step)
        self._check_epoch(message.epoch)
        if self._module != message.module:
            raise ValueError(f'Unexpected module "{message.module}" (expected "{self._module}")')
        if self._data[message.kind] is not None:
            raise ValueError(f'Got duplicate value for kind "{message.kind}".')
        self._data[message.kind] = message.payload

    def getattr(self, name):
        if name in self._expected_kinds.value():
            return self._data[name]
        else:
            return self.__getattribute__(name)


class Subscription(object):

    '''Specification for a subscription that can span multiple kinds and a tag.'''

    def __init__(self, subscriber, kinds, tag=None):
        '''
        Parameters
        ----------
        subscriber  :   Subscriber
                        Object that wants to receive the messages
        kinds   :   list(str)
                    List of string identifiers for different message kinds. These are all the
                    message kinds the subscriber wishes to receive
        tag :   str or None
                Optional tag for filtering messages. If ``None`` is passed, all messages will be
                relayed
        '''
        self._tag        = tag
        self._subscriber = subscriber
        self._kinds = kinds

    def epoch_finished(self, epoch):
        self._subscriber.epoch_finished(epoch)

    def _new_message(self, network_data):
        '''Process a newly arrived message. Subclasses should override this method for any special
        treatment.

        Parameters
        ----------
        network_data    :   ikkuna.messages.NetworkData
        '''
        data = ModuleData(network_data.module, network_data.kind)
        data.add_message(network_data)
        self._subscriber(data)

    def __call__(self, network_data):
        '''Callback for receiving an incoming message.

        Parameters
        ----------
        network_data    :   ikkuna.messages.NetworkData
        '''
        if network_data.kind not in self._kinds:
            return

        if self._tag is None or self._tag == network_data.tag:
            self._new_message(network_data)


class SynchronizedSubscription(Subscription):
    '''A subscription which buffers messages and publishes a set of messages, each of a different
    kind, when one round (a train step) is over. This is useful for receiving several kinds of
    messages in each train step and always have them be processed together.'''

    def __init__(self, subscriber, kinds, tag=None):
        super().__init__(subscriber, kinds, tag)
        self._current_seq = None
        self._modules = {}
        self._step = None

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

    def _new_message(self, network_data):
        '''Start a new round if a new sequence number is seen. If the buffer is full, all is
        published.'''
        print(f'New messge {network_data.seq}, {network_data.module}, {network_data.kind}')

        # if we get a new sequence number, a new train step must have begun
        if self._current_seq is None or self._current_seq != network_data.seq:
            self._new_round(network_data.seq)

        # module not seen -> init data
        module = network_data.module
        if module not in self._modules:
            self._modules[module] = ModuleData(module, self._kinds)
        self._modules[module].add_message(network_data)

        # all full? publish
        # NOTE: This assumes that the messages for a set of modules are interleaved, otherwise,
        # we'll have finished the first module before others are filled. A cleaner way to do this
        # would be to have a list of modules to expect and not publish before all have been
        # completed.
        if all(map(lambda d: d.complete(), self._modules.values())):
            self._subscriber(self._modules.values())


class Subscriber(abc.ABC):

    '''Base class for receiving and processing activations, gradients and other stuff into
    insightful metrics.

    Attributes
    ----------
    _counter    :   int
                    Number of times the subscriber was called
    '''

    def __init__(self):
        self._counter = 0

    @abc.abstractmethod
    def __call__(self, module_data):
        for m in module_data:
            if not m.complete():
                raise ValueError(f'Data received for "{m._module}" is not complete.')
        self._counter += 1

    @abc.abstractmethod
    def epoch_finished(self, epoch):
        pass


class HistogramSubscriber(Subscriber):

    def __init__(self, clip_min, clip_max, step):
        super().__init__()
        if clip_max <= clip_min:
            raise ValueError(f'`clip_min` must be smaller than'
                             ' `clip_max` (was {clip_min} and {clip_max})')
        n_bins = (clip_min - clip_max) // step
        self._gradient_hist = defaultdict(lambda: np.zeros(n_bins, dtype=np.int64))

    def __call__(self, module_data):
        super().__call__(module_data)

    def epoch_finished(self, epoch):
        super().epoch_finished(epoch)
