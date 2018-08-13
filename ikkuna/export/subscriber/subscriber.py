'''
.. moduleauthor:: Rasmus Diederichsen

.. module:: subscriber

This module contains the base definition for subscriber functionality. The
:class:`ikkuna.export.subscriber.Subscriber` class should be subclassed for adding new metrics.

'''
import abc
from collections import defaultdict
from ikkuna.visualization import TBBackend, MPLBackend
from ikkuna.export.messages import ModuleData, TrainingMessage


class Subscription(object):
    '''Specification for a subscription that can span multiple kinds and a tag.

    Attributes
    ----------
    _tag    :   str
                Tag for filtering the processed messages
    _subscriber :   ikkuna.export.subscriber.Subscriber
                    The subscriber associated with the subscription
    counter    :   dict(ikkuna.utils.NamedModule or str, int)
                    Number of times the subscriber was called for each module label or meta data
                    identifier. Since one :class:`ikkuna.export.subscriber.Subscriber` is associated
                    with only one configuration of :class:`ikkuna.export.messages.ModuleData`, this
                    will enable proper subsampling of message streams.
    kinds   :   list(str)
                List of string identifiers for different message kinds. These are all the
                message kinds the subscriber wishes to receive
    _subsample  :   int
                    Factor for subsampling incoming messages. Only every ``subsample``-th
                    message will be processed.
    '''

    def __init__(self, subscriber, kinds, tag=None, subsample=1):
        '''
        Parameters
        ----------
        subscriber  :   ikkuna.export.subscriber.Subscriber
                        Object that wants to receive the messages
        tag :   str or None
                Optional tag for filtering messages. If ``None``, all messages will be
                relayed
        '''
        self._tag        = tag
        self._subscriber = subscriber
        self._kinds      = kinds
        self._counter    = defaultdict(int)
        self._subsample  = subsample

    @property
    def counter(self):
        # caution: if you alter this dict, you're on your own
        return self._counter

    @property
    def kinds(self):
        return self._kinds

    def _handle_message(self, message):
        '''Process a newly arrived message. Subclasses should override this method for any special
        treatment.

        Parameters
        ----------
        message    :   ikkuna.export.messages.Message
        '''
        if isinstance(message, TrainingMessage):
            data = ModuleData(message.module, message.kind)
            data.add_message(message)
            self._subscriber.process_data(data)
        else:
            self._subscriber.process_meta(message)

    def handle_message(self, message):
        '''Callback for receiving an incoming message.

        Parameters
        ----------
        message    :   ikkuna.export.messages.TrainingMessage
        '''
        if not (self._tag is None or self._tag == message.tag):
            return

        if message.kind not in self.kinds:
            return

        key = message.module if isinstance(message, TrainingMessage) else message.kind
        if self._counter[key] % self._subsample == 0:
            self._handle_message(message)
        self._counter[key] += 1


class SynchronizedSubscription(Subscription):
    '''A subscription which buffers messages and publishes a set of messages, each of a different
    kind, when one round (a train step) is over. This is useful for receiving several kinds of
    messages in each train step and always have them be processed together.'''

    def __init__(self, subscriber, kinds, tag=None, subsample=1):
        super().__init__(subscriber, kinds, tag, subsample)
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

    def _handle_message(self, message):
        '''Start a new round if a new sequence number is seen.'''

        # if we get a new sequence number, a new train step must have begun
        if self._current_seq is None or self._current_seq != message.seq:
            self._new_round(message.seq)

        if isinstance(message, TrainingMessage):
            # module not seen -> init data
            module = message.module
            if module not in self._modules:
                self._modules[module] = ModuleData(module, self.kinds)
            self._modules[module].add_message(message)

            delete_these = []
            # any full? publish
            for module, data in self._modules.items():
                if data.complete():
                    self._subscriber.process_data(data)
                    delete_these.append(module)

            for module in delete_these:
                del self._modules[module]
        else:
            self._subscriber.process_meta(message)


class Subscriber(abc.ABC):

    '''Base class for receiving and processing activations, gradients and other stuff into
    insightful metrics.

    '''

    def __init__(self, subscription, tag=None):
        self._current_epoch = 0
        self._subscription  = subscription

    @abc.abstractmethod
    def _metric(self, message_or_data):
        '''This is where the magic happens. Subclasses should override this method so that they can
        compute their metric upon reception of their desired messages. They should then use their
        ``backend`` to publish the metric.

        Parameters
        ----------
        message_or_data :   ikkuna.export.messages.Message
                            Can either be :class:`ikkuna.export.messages.MetaMessage` if the
                            Subscriber is not interested in actual training artifacts, or
                            :class:`ikkuna.export.messages.TrainingMessage`
        '''
        pass

    def receive_message(self, message):
        self._subscription.handle_message(message)

    def process_meta(self, message):
        self._metric(message)

    def process_data(self, module_data):
        '''Callback for processing a :class:`ikkuna.export.messages.ModuleData` object.

        Parameters
        ----------
        module_data    :   ikkuna.export.messages.ModuleData
                            The exact nature of this package is determined by the
                            :class:`ikkuna.export.subscriber.Subscription` attached to this
                            :class:`ikkuna.export.subscriber.Subscriber`.

        Raises
        ------
        ValueError
            If the received :class:`ikkuna.export.messages.ModuleData` object is not
            :meth:`ikkuna.export.messages.ModuleData.complete()`
        '''
        if not module_data.complete():
            raise ValueError(f'Data received for "{module_data._module}" is not complete.')

        self._metric(module_data)

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

    def __init__(self, subscription,  plot_config, tag=None, backend='tb'):
        '''
        Parameters
        ----------
        average :   int
                    Inverse resolution of the plot. For plotting ``average`` ratios will be averaged
                    for each module to remove noise.
        ylims   :   tuple(int, int)
                    Optional Y-axis limits
        '''
        super().__init__(subscription, tag)

        if backend not in ('tb', 'mpl'):
            raise ValueError('Backend must be "tb" or "mpl"')
        if backend == 'tb':
            self._backend = TBBackend(**plot_config)
        else:
            self._backend = MPLBackend(**plot_config)

    @abc.abstractmethod
    def _metric(self, message_or_data):
        pass

    def epoch_finished(self, epoch):
        '''The plot is updated, respecting the ``average`` parameter set. Successive metric values
        resolution is ``batches_per_epoch / subsample / average``.'''
        super().epoch_finished(epoch)
