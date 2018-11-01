'''
.. _meta_kinds:
.. data:: META_KINDS

    Message kinds which are not tied to any specific module.

.. _data_kinds:
.. data:: DATA_KINDS

    Message kinds which are tied to a specific module and always carry data

.. _allowed_kinds:
.. data:: ALLOWED_KINDS

    Simply the union of ``META_KINDS`` and ``DATA_KINDS``
'''

import abc

from ikkuna.utils import NamedModule

META_KINDS = {
    'batch_started', 'batch_finished', 'epoch_started', 'epoch_finished', 'input_data',
    'input_labels', 'network_output'
}

DATA_KINDS = {
    'weights', 'weight_gradients', 'weight_updates', 'biases', 'bias_gradients', 'bias_updates',
    'activations', 'layer_gradients'
}

ALLOWED_KINDS = set.union(META_KINDS, DATA_KINDS)


class Message(abc.ABC):
    '''Base class for messages emitted from the :class:`~ikkuna.export.Exporter`.

    These messages are assembled into :class:`MessageBundle` objects in the
    :class:`~ikkuna.export.subscriber.Subscription`.
    '''
    def __init__(self, tag, global_step, train_step, epoch, kind):
        '''
        Parameters
        ----------
        tag :   str
                Tag for this message
        global_step :   int
                Global train step
        train_step    :   int
                    Epoch-local train step
        epoch   :   int
                    Epoch index
        kind    :   str
                    Message topic. Must be in :ref:`ALLOWED_KINDS <allowed_kinds>` or ``None`` in
                    which case no checking is performed. This guards against misspellings or
                    otherwise incorrect topics.
        '''
        self._tag = tag
        self._global_step = global_step

        # check train_step
        if train_step < 0:
            raise ValueError('Step cannot be negative.')
        else:
            self._train_step = train_step

        # check epoch
        if epoch < 0:
            raise ValueError('Epoch cannot be negative')
        else:
            self._epoch = epoch

        # check kind
        if kind is not None and kind not in ALLOWED_KINDS:
            raise ValueError(f'Invalid message kind "{kind}"')
        else:
            self._kind = kind

        self._data = None

    @property
    def tag(self):
        ''' str: The tag associated with this message '''
        return self._tag

    @property
    def global_step(self):
        '''int: Global sequence number. This counter should not reset after each epoch.'''
        return self._global_step

    @property
    def train_step(self):
        '''int: Epoch-local sequence number (the current batch index)'''
        return self._train_step

    @property
    def epoch(self):
        '''int: Current epoch number'''
        return self._epoch

    @property
    def kind(self):
        '''str: Message kind'''
        return self._kind

    @property
    def data(self):
        '''torch.Tensor, tuple(torch.Tensor) or None:  This field is optional for
        :class:`NetworkMessage`, but mandatory for :class:`ModuleMessage`'''
        return self._data

    @abc.abstractproperty
    def key(self):
        '''object: A key used for grouping messages into :class:`MessageBundle` s'''
        pass

    def __str__(self):
        return (f'<{self.__class__.__name__}: global_step={self.global_step}, '
                f'train_step={self.train_step}, epoch={self.epoch}, kind={self.kind}>')

    def __repr__(self):
        return str(self)


class NetworkMessage(Message):
    '''A message with meta information not tied to any specific module. Can still carry tensor data,
    if necessary.'''
    def __init__(self, tag, global_step, train_step, epoch, kind, data=None):
        if kind not in META_KINDS:
            raise ValueError(f'Invalid message kind "{kind}"')
        super().__init__(tag, global_step, train_step, epoch, kind)
        self._data = data

    @property
    def data(self):
        '''torch.Tensor, tuple, float, int or None: Optional data. Can be used e.g. for input to the
        network, labels or network output'''
        return self._data

    @property
    def key(self):
        return 'META'


class ModuleMessage(Message):
    '''A message tied to a specific module, with tensor data attached.'''

    def __init__(self, tag, global_step, train_step, epoch, kind, named_module, data):
        super().__init__(tag, global_step, train_step, epoch, kind)
        self._module  = named_module
        if data is None:
            raise ValueError('Data cannot be `None` for `ModuleMessage`')
        self._data = data

    @property
    def module(self):
        '''torch.nn.Module: Module emitting this data'''
        return self._module

    @property
    def key(self):
        return self.module


class SubscriberMessage(Message):
    '''A message published by subscribers. There's no whitelist of allowed ``kind``\ s as there is
    for messages originating from the model.'''

    def __init__(self, tag, global_step, train_step, epoch, kind, identifier, data):
        # we pass None for kind so as to explicitly avoid the validity check
        super().__init__(tag, global_step, train_step, epoch, None)

        if isinstance(identifier, NamedModule):
            self._identifier = identifier.name
        else:
            self._identifier = identifier

        self._kind = kind

        if data is None:
            raise ValueError('Data cannot be `None` for `SubscriberMessage`')
        self._data = data

    @property
    def key(self):
        # man I really need to rethink this key business
        return self._identifier


class MessageBundle(object):

    '''Data object for holding a set of artifacts for a module (or meta information) at one point
    during training. This data type can be used to buffer different kinds and check whether all
    expected kinds have been received for a module or meta information. The collection is enforced
    to be homogeneous with respect to global step, train step, epoch, and identifier
    (:attr:`Message.key`)'''

    def __init__(self, kinds):
        '''
        Parameters
        ----------
        kinds   :   str or list
                    Single kind when a :class:`~ikkuna.export.subscriber.Subscription` is used, or a
                    list of Message kinds contained in this bundle for use with
                    :class:`~ikkuna.export.subscriber.SynchronizedSubscription`
        '''
        if isinstance(kinds, str):
            # this has bitten me before. base `Subscription`s don't use multiple kinds
            kinds = [kinds]
        self._key            = None
        self._expected_kinds = kinds
        self._received       = {kind: False for kind in kinds}
        self._data           = {kind: None for kind in kinds}
        self._global_step    = None
        self._train_step     = None
        self._epoch          = None

    @property
    def key(self):
        ''' str: An object denoting the common aspect of the collected messages (besides the steps).
        This can be the :class:`~ikkuna.utils.NamedModule` emitting the data or a string such as
        ``'META'`` or other denoting these are messages which do not belong to a module.'''
        return self._key

    @property
    def kinds(self):
        '''list(str): Alias to :attr:`expected_kinds`'''
        return self.expected_kinds

    @property
    def expected_kinds(self):
        '''list(str): The expected kinds of messages per iteration '''
        return self._expected_kinds

    @property
    def data(self):
        '''dict(str, torch.Tensor): The tensors received for each kind'''
        return self._data

    @property
    def global_step(self):
        '''int: Global sequence number of this class'''
        return self._global_step

    @property
    def train_step(self):
        '''int: Sequence number (training step) of the received messages (should match across all
        msgs in one iteration)'''
        return self._train_step

    @property
    def epoch(self):
        '''int: Epoch of the received messages (should match across all msgs in one iteration)'''
        return self._epoch

    def complete(self):
        '''Check if all expected messages have been received. This means the bundle can be released
        to subscribers.

        Returns
        -------
        bool
        '''
        return all(self._received.values())

    def _check_message(self, message):
        '''Check consistency of sequence number, step and epoch or set if not set yet. Check
        consistency of identifier and check for duplication.

        Parameters
        ----------
        message :   ikkuna.export.messages.Message

        Raises
        ------
        ValueError
            If ``message.(global_step|step|epoch|identifier)`` does not match the current
            ``(global_step|step|epoch|identifier)`` or in case a message of ``message.kind`` has
            already been received
        '''
        #################
        #  global_step  #
        #################
        if self._global_step is None:
            self._global_step = message.global_step
        elif message.global_step != self._global_step:
            raise ValueError('Attempting to add message with global_step '
                             f'{message.global_step} to bundle with '
                             'initial global_step {self._global_step}')

        ################
        #  train_step  #
        ################
        if self._train_step is None:
            self._train_step = message.train_step
        elif message.train_step != self._train_step:
            raise ValueError(f'Attempting to add message with step {message.step} to bundle with '
                             'initial step {self._train_step}')
        #############
        #  epoch    #
        #############
        if self._epoch is None:
            self._epoch = message.epoch
        elif message.epoch != self._epoch:
            raise ValueError(f'Attempting to add message from epoch {message.epoch} to bundle with '
                             'initial epoch {self._epoch}')

        ################
        #  key         #
        ################
        if self._key is None:
            self._key = message.key
        elif self._key != message.key:
            raise ValueError(f'Unexpected message key "{message.key}" '
                             f'(expected "{self._key}")')

        #################
        #  Duplication  #
        #################
        if self._received[message.kind]:
            raise ValueError(f'Got duplicate value for kind "{message.kind}".')

    def add_message(self, message):
        '''Add a new message to this object. Will fail if the new messsage does not have the same
        sequence number and epoch.

        Parameters
        ----------
        message :   ikkuna.export.messages.Message

        Raises
        ------
        ValueError
            see :meth:`~MessageBundle._check_message()`
        '''
        self._check_message(message)
        self._received[message.kind] = True

        if message.data is not None:
            self._data[message.kind] = message.data

    def __getattr__(self, name):
        '''Override to mimick a property for each kind of message in this data (e.g.
        ``message_bundle.activations`` instead of ``message_bundle.data['activations']``)
        '''
        if name in self._expected_kinds:
            return self._data[name]
        else:
            return self.__getattribute__(name)

    def __str__(self):
        mod   = self._key
        step  = self._train_step
        kinds = list(self._data.keys())
        return f'<MessageBundle: identifier={mod}, kinds={kinds}, step={step}>'

    def __repr__(self):
        return str(self)


class MessageBus(object):
    '''A class which receives messages, registers subscribers and relays the former to the
    latter.'''

    def __init__(self, name):
        '''
        Parameters
        ----------
        name    :   str
                    Identifier for this bus
        '''
        self._name = name
        self._subscribers = set()

    @property
    def name(self):
        '''str: The name of this bus'''
        return self._name

    def register_subscriber(self, sub):
        '''Add a new subscriber to the set. Adding subscribers mutliple times will still only call
        them once per message.

        Parameters
        ----------
        sub :   ikkuna.export.subscriber.Subscriber
        '''
        self._subscribers.add(sub)

    def publish_subscriber_message(self, global_step, train_step, epoch, kind, identifier,
                                   data=None):
        '''Publish an update of type :class:`~ikkuna.export.messages.SubscriberMessage` to all
        registered subscribers.

        Parameters
        ----------
        global_step :   int
                        Global training step
        train_step  :   int
                        Epoch-relative training step
        epoch   :   int
                    Epoch index
        kind    :   str
                    Identifier chosen by the publishing subscriber
        identifier  :   str
                        Identifier for this message. Usually the module name, if it is tied to a
                        module, or 'META' for other messages. The string is used for collection in
                        :class:`MessageBundle`\ s which must be uniform in this respect.
        data    :   torch.Tensor, tuple(torch.Tensor), float, int or None
                    Payload, if necessary
        '''
        msg = SubscriberMessage(global_step=global_step, tag=None, kind=kind, identifier=identifier,
                                train_step=train_step, epoch=epoch, data=data)
        for sub in self._subscribers:
            sub.receive_message(msg)

    def publish_network_message(self, global_step, train_step, epoch, kind, data=None):
        '''Publish an update of type :class:`~ikkuna.export.messages.NetworkMessage` to all
        registered subscribers.

        Parameters
        ----------
        global_step :   int
                        Global training step
        train_step  :   int
                        Epoch-relative training step
        epoch   :   int
                    Epoch index
        kind    :   str
                    Kind of message
        data    :   torch.Tensor or None
                    Payload, if necessary
        '''
        msg = NetworkMessage(global_step=global_step, tag=None, kind=kind, train_step=train_step,
                             epoch=epoch, data=data)
        for sub in self._subscribers:
            sub.receive_message(msg)

    def publish_module_message(self, global_step, train_step, epoch, kind, named_module, data):
        '''Publish an update of type :class:`~ikkuna.export.messages.ModuleMessage` to all
        registered subscribers.

        Parameters
        ----------
        global_step :   int
                        Global training step
        train_step  :   int
                        Epoch-relative training step
        epoch   :   int
                    Epoch index
        kind    :   str
                    Kind of message
        named_module  :   ikkuna.utils.NamedModule
                    The module in question
        data    :   torch.Tensor
                    Payload
        '''
        msg = ModuleMessage(global_step=global_step, tag=None, kind=kind, named_module=named_module,
                            train_step=train_step, epoch=epoch, data=data)

        for sub in self._subscribers:
            sub.receive_message(msg)


__default_bus = MessageBus('default')


def get_default_bus():
    '''Get the default message bus which is always created when this module is loaded.

    Returns
    -------
    MessageBus
    '''
    global __default_bus
    return __default_bus
