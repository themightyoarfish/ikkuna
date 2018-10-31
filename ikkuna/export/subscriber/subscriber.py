'''
.. moduleauthor:: Rasmus Diederichsen

This module contains the base definition for subscriber functionality. The
:class:`~ikkuna.export.subscriber.Subscriber` class should be subclassed for adding new metrics.

'''
import abc
from collections import defaultdict
import ikkuna.visualization
from ikkuna.export.messages import MessageBundle, TrainingMessage, ALLOWED_KINDS, get_default_bus


class Subscription(object):
    '''Specification for a subscription that can span multiple kinds and a tag.

    Attributes
    ----------
    _tag    :   str
                Tag for filtering the processed messages
    _subscriber :   ikkuna.export.subscriber.Subscriber
                    The subscriber associated with the subscription
    counter :   dict(ikkuna.utils.NamedModule or str, int)
                Number of times the subscriber was called for each module label or meta data
                identifier. Since one :class:`ikkuna.export.subscriber.Subscriber` is associated
                with only one configuration of :class:`ikkuna.export.messages.MessageBundle`, this
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
                Optional tag for filtering messages. If ``None``, all messages will be relayed
        '''
        self._tag        = tag
        self._subscriber = subscriber

        for k in kinds:
            if k not in ALLOWED_KINDS:
                raise ValueError(f'Unknown message kind "{k}" encountered.')

        self._kinds      = kinds
        self._counter    = defaultdict(int)
        self._subsample  = subsample

    @property
    def counter(self):
        return dict(self._counter)

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
        data = MessageBundle(message.key, message.kind)
        data.add_message(message)
        self._subscriber.process_message_bundle(data)

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

        if isinstance(message, TrainingMessage):
            key = (message.module, message.kind)
        else:
            key = message.kind
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
        self._identifiers = {}
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
            If not all desired kinds have been received for all identifiers yet in the current round
        '''
        for bundle in self._identifiers.values():
            if not bundle.complete():
                raise RuntimeError(f'Bundle for module {bundle._module} not yet complete.')
        self._current_seq = seq
        self._identifiers = {}

    def _publish_complete(self):
        delete_these = []
        # any full? publish
        for identifier, message_bundle in self._identifiers.items():
            if message_bundle.complete():
                self._subscriber.process_message_bundle(message_bundle)
                delete_these.append(identifier)

        # purge published data
        for identifier in delete_these:
            del self._identifiers[identifier]

    def _handle_message(self, message):
        '''Start a new round if a new sequence number is seen.'''

        # if we get a new sequence number, a new train step must have begun
        if self._current_seq is None or self._current_seq != message.seq:
            self._new_round(message.seq)

        # module not seen -> init data
        key = message.key
        if key not in self._identifiers:
            self._identifiers[key] = MessageBundle(key, self.kinds)
        self._identifiers[key].add_message(message)

        self._publish_complete()


class Subscriber(abc.ABC):
    '''Base class for receiving and processing activations, gradients and other stuff into
    insightful metrics.'''

    def __init__(self, subscription, message_bus):
        '''
        Parameters
        ----------
        subscription    :   Subscription
        '''
        self._subscription = subscription
        self._msg_bus      = message_bus

    @property
    def subscription(self):
        return self._subscription

    @property
    def message_bus(self):
        return self._msg_bus

    @abc.abstractmethod
    def compute(self, message_or_data):
        '''This is where the magic happens. Subclasses should override this method so that they can
        compute their metric upon reception of their desired messages or do whatever else they want.
        If interested in plotting, they should then use their
        :attr:`~ikkuna.export.subscriber.PlotSubscriber.backend` property to plot the metric (if
        they display line plots) and their ``message_bus`` to publish a new message with the metric.

        Parameters
        ----------
        message_or_data :   ikkuna.export.messages.Message
                            Can either be :class:`~ikkuna.export.messages.MetaMessage` if the
                            Subscriber is not interested in actual training artifacts, or
                            :class:`~ikkuna.export.messages.TrainingMessage`
        '''
        pass

    def receive_message(self, message):
        '''Process a message received from an :class:`~ikkuna.export.Exporter`.'''
        self._subscription.handle_message(message)

    def process_message_bundle(self, message_bundle):
        '''Callback for processing a :class:`~ikkuna.export.messages.MessageBundle` object with
        :class:`~ikkuna.export.messages.MetaMessage`\ s or
        :class:`~ikkuna.export.messages.TrainingMessage`\ s in it.

        Parameters
        ----------
        message_bundle    :   ikkuna.export.messages.MessageBundle
                            The exact nature of this package is determined by the
                            :class:`ikkuna.export.subscriber.Subscription` attached to this
                            :class:`ikkuna.export.subscriber.Subscriber`.

        Raises
        ------
        ValueError
            If the received :class:`~ikkuna.export.messages.MessageBundle` object is not
            :meth:`~ikkuna.export.messages.MessageBundle.complete()`
        '''
        if not message_bundle.complete():
            raise ValueError(f'Data received for "{message_bundle._module}" is not complete.')

        self.compute(message_bundle)


class PlotSubscriber(Subscriber):
    '''Base class for subscribers that output scalar or histogram values per time and module

    Attributes
    ----------
    _backend    :   ikkuna.visualization.Backend
                    Plotting backend
    '''

    def __init__(self, subscription, message_bus, plot_config, backend='tb', **tbx_params):
        '''
        Parameters
        ----------
        ylims   :   tuple(int, int)
                    Optional Y-axis limits
        plot_config :   dict
                        Configuration parameters for plotting. Relevant keys are ``title``,
                        ``xlabel``, ``ylabel`` and ``ylims``. Which of them are actually used
                        depends on the :class:`~ikkuna.visualization.Backend`
        **tbx_params    :   dict
                            Keywords for the :class:`tensorboardX.SummaryWriter`
        '''
        super().__init__(subscription, message_bus)

        self._backend = ikkuna.visualization.get_backend(backend, plot_config, **tbx_params)

    @property
    def backend(self):
        '''ikkuna.visualization.Backend: The backend to use for plotting'''
        return self._backend

    @abc.abstractmethod
    def compute(self, message_or_data):
        pass
