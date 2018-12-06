'''
.. moduleauthor:: Rasmus Diederichsen

This module contains the base definition for subscriber functionality. The
:class:`~ikkuna.export.subscriber.Subscriber` class should be subclassed for adding new metrics.

'''
import abc
from collections import defaultdict
import ikkuna.visualization
from ikkuna.export.messages import MessageBundle, ModuleMessage, get_default_bus


class Subscription(object):
    '''Specification for a subscription that can span multiple kinds and a tag.

    Attributes
    ----------
    _tag    :   str
                Tag for filtering the processed messages
    _subscriber :   ikkuna.export.subscriber.Subscriber
                    The subscriber associated with the subscription
    counter :   dict(ikkuna.utils.NamedModule or str, int)
                Number of times the Subscription was called for each module label or meta data
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
        kinds   :   list
                    List of message kinds to process
        tag :   str
                tag for filtering messages
        subsample   :   int
                        Number of messages to ignore before processing one. Note that this number if
                        applied to every kind, regardless of frequency. So if ``subsample = 10``,
                        every tenth ``weights`` message would be processed, but also only every
                        tenth ``epoch_finished`` message.
        tag :   str or None
                Optional tag for filtering messages. If ``None``, all messages will be relayed
        '''
        if not isinstance(kinds, list):
            raise ValueError(f'Expected list of kinds, got {type(kinds)}')
        self._tag        = tag
        self._subscriber = subscriber

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
        self._subscriber.process_messages(message)

    def handle_message(self, message):
        '''Callback for receiving an incoming message.

        Parameters
        ----------
        message    :   ikkuna.export.messages.ModuleMessage
        '''
        if not (self._tag is None or self._tag != message.tag or message.kind not in self.kinds):
            return

        if isinstance(message, ModuleMessage):
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
        self._current_global_step = None
        self._open_bundles        = {}

    def _new_round(self, round_idx):
        '''Start a new round of buffering, clearing the previous cache and resetting the record for
        which kinds were received in this round.

        Parameters
        ----------
        round_idx : int
                    Global train step of the new round

        Raises
        ------
        RuntimeError
            If not all desired kinds have been received for all identifiers yet in the current round
        '''
        for bundle in self._open_bundles.values():
            if not bundle.complete():
                raise RuntimeError(f'Bundle with id={bundle.key} not yet complete.')
        self._current_global_step = round_idx
        self._open_bundles = {}

    def _publish_complete(self):
        delete_these = []
        # any full? publish
        for module_and_kind, message_bundle in self._open_bundles.items():
            if message_bundle.complete():
                self._subscriber.process_messages(message_bundle)
                delete_these.append(module_and_kind)

        # purge published data
        for module_and_kind in delete_these:
            del self._open_bundles[module_and_kind]

    def _handle_message(self, message):
        '''Start a new round if a new sequence number is seen.'''

        # if we get a new sequence number, a new train step must have begun
        if self._current_global_step is None or self._current_global_step != message.global_step:
            self._new_round(message.global_step)

        # module not seen -> init data
        key = message.key
        if key not in self._open_bundles:
            self._open_bundles[key] = MessageBundle(self.kinds)
        self._open_bundles[key].add_message(message)

        self._publish_complete()


class Subscriber(abc.ABC):
    '''Base class for receiving and processing activations, gradients and other stuff into
    insightful metrics.'''

    def __init__(self, subscriptions, message_bus):
        '''
        Parameters
        ----------
        subscriptions    :   list(Subscription)
        '''
        self._subscriptions    = {kind: subscription
                                  for subscription in subscriptions for kind in subscription.kinds}
        self._msg_bus          = message_bus
        message_bus.register_subscriber(self)
        self._published_topics = dict()

    def _add_publication(self, topic, type='DATA'):
        '''
        Parameters
        ----------
        topic   :   str
                    Topic name
        type    :   str
                    ``DATA`` or ``META``
        '''
        if type not in ('DATA', 'META'):
            raise ValueError(f'Unknown message type "{type}"')

        self._published_topics[type] = topic
        if type == 'DATA':
            self._msg_bus.register_data_topic(topic)
        else:
            self._msg_bus.register_meta_topic(topic)

    def __del__(self):
        '''If for whatever reason a subscriber ceases to exist before the interpreter ends, delete
        the registered topics'''
        for type, topic in self._published_topics.items():
            if type == 'DATA':
                self._msg_bus.deregister_data_topic(topic)
            else:
                self._msg_bus.deregister_meta_topic(topic)

    @property
    def publications(self):
        return {k: v for k, v in self._published_topics.items()}

    @property
    def kinds(self):
        return list(set(self.subscriptions.keys()))

    @property
    def subscriptions(self):
        return self._subscriptions

    @property
    def message_bus(self):
        return self._msg_bus

    @abc.abstractmethod
    def compute(self, message_or_bundle):
        '''This is where the magic happens. Subclasses should override this method so that they can
        compute their metric upon reception of their desired messages or do whatever else they want.
        If interested in plotting, they should then use their
        :attr:`~ikkuna.export.subscriber.PlotSubscriber.backend` property to plot the metric (if
        they display line plots) and their ``message_bus`` to publish a new message with the metric.

        Parameters
        ----------
        message_or_bundle : ikkuna.export.messages.Message or ikkuna.export.messages.MessageBundle
                            If the subscriber uses a :class:`SynchronizedSubscription`, a bundle is
                            received in each call, otherwise just a
                            :class:`~ikkuna.export.messages.Message`.

                            Messages can either be :class:`~ikkuna.export.messages.NetworkMessage`
                            if the Subscriber is not interested in actual training artifacts, or
                            :class:`~ikkuna.export.messages.ModuleMessage`
        '''
        pass

    def receive_message(self, message):
        '''Process a single message received from an :class:`~ikkuna.export.messages.MessageBus`.'''

        if message.kind in self._subscriptions:
            self._subscriptions[message.kind].handle_message(message)

    def process_messages(self, message_or_bundle):
        '''Callback for processing a single :class:`~ikkuna.export.messages.Message` or a
        :class:`~ikkuna.export.messages.MessageBundle` object with
        :class:`~ikkuna.export.messages.NetworkMessage`\ s or
        :class:`~ikkuna.export.messages.ModuleMessage`\ s in it.

        Parameters
        ----------
        message_or_bundle : ikkuna.export.messages.Message or ikkuna.export.messages.MessageBundle
                            The exact nature of this package is determined by the
                            :class:`ikkuna.export.subscriber.Subscription` attached to this
                            :class:`ikkuna.export.subscriber.Subscriber`.

        Raises
        ------
        ValueError
            If the received :class:`~ikkuna.export.messages.MessageBundle` object is not
            :meth:`~ikkuna.export.messages.MessageBundle.complete()`
        '''
        if (isinstance(message_or_bundle, ikkuna.export.messages.MessageBundle)
            and not message_or_bundle.complete()):
            raise ValueError(f'Data received for "{message_or_bundle._module}" is not complete.')

        self.compute(message_or_bundle)


class PlotSubscriber(Subscriber):
    '''Base class for subscribers that output scalar or histogram values per time and module

    Attributes
    ----------
    _backend    :   ikkuna.visualization.Backend
                    Plotting backend
    '''

    def __init__(self, subscriptions, message_bus, plot_config, backend='tb'):
        '''
        Parameters
        ----------
        ylims   :   tuple(int, int)
                    Optional Y-axis limits
        plot_config :   dict
                        Configuration parameters for plotting. Relevant keys are ``title``,
                        ``xlabel``, ``ylabel`` and ``ylims``. Which of them are actually used
                        depends on the :class:`~ikkuna.visualization.Backend`
        '''
        super().__init__(subscriptions, message_bus)

        self._backend = ikkuna.visualization.get_backend(backend, plot_config)

    @property
    def backend(self):
        '''ikkuna.visualization.Backend: The backend to use for plotting'''
        return self._backend

    @abc.abstractmethod
    def compute(self, message_or_bundle):
        pass


class CallbackSubscriber(Subscriber):
    '''Subscriber class for subscribing to :class:`~ikkuna.export.messages.ModuleMessage`\ s and
    running a callback with them.'''

    def __init__(self, kinds, callback, message_bus=get_default_bus(), tag=None, subsample=1):
        '''
        Parameters
        ----------
        subscription    :   Subscription
        message_bus :   ikkuna.export.messages.MessageBus
        callback    :   function
                        A function that accepts as many parameters as the ``Subscription`` delivers
                        messages at once.
        '''
        subscription = SynchronizedSubscription(self, kinds, tag, subsample)
        super().__init__([subscription], message_bus)
        self._callback = callback

    def compute(self, message_or_bundle):
        kinds = self._subscriptions[0].kinds
        args  = (message_or_bundle.data[kind] for kind in kinds)

        id_ = message_or_bundle.key
        self._callback(*args, message_or_bundle.global_step, message_or_bundle.train_step,
                       message_or_bundle.epoch, id_)
