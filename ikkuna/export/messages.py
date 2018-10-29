'''
.. data:: META_KINDS

    Message kinds which are not tied to any specific module.

.. data:: DATA_KINDS

    Message kinds which are tied to a specific module and always carry data

.. data:: ALLOWED_KINDS

    Simply the union of ``META_KINDS`` and ``DATA_KINDS``
'''

import abc
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
    def __init__(self, tag, seq, step, epoch, kind):
        self._tag  = tag
        self._seq  = seq

        # check step
        if step < 0:
            raise ValueError('Step cannot be negative.')
        else:
            self._step = step

        # check epoch
        if epoch < 0:
            raise ValueError('Epoch cannot be negative')
        else:
            self._epoch = epoch

        # check kind
        if kind not in ALLOWED_KINDS:
            raise ValueError(f'Invalid message kind "{kind}"')
        else:
            self._kind = kind
        self._data = None

    @property
    def tag(self):
        ''' str: The tag associated with this message '''
        return self._tag

    @property
    def seq(self):
        '''int: Global sequence number. This counter should not reset after each epoch.'''
        return self._seq

    @property
    def step(self):
        '''int: Epoch-local sequence number (the current batch index)'''
        return self._step

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
        :class:`MetaMessage`, but mandatory for :class:`TrainingMessage`'''
        return self._data

    @abc.abstractproperty
    def key(self):
        '''object: A key used for grouping messages into :class:`MessageBundle` s'''
        pass

    def __str__(self):
        return (f'<{self.__class__.__name__}: seq={self.seq}, '
                f'step={self.step}, epoch={self.epoch}, kind={self.kind}>')

    def __repr__(self):
        return str(self)


class MetaMessage(Message):
    '''A message with meta information not tied to any specific module. Can still carry tensor data,
    if necessary.'''
    def __init__(self, tag, seq, step, epoch, kind, data=None):
        if kind not in META_KINDS:
            raise ValueError(f'Invalid message kind "{kind}"')
        super().__init__(tag, seq, step, epoch, kind)
        self._data = data

    @property
    def data(self):
        '''torch.Tensor, tuple or None: Optional data. Can be used e.g. for input to the network,
        labels or network output'''
        return self._data

    @property
    def key(self):
        return 'META'


class TrainingMessage(Message):
    '''A message tied to a specific module, with tensor data attached.'''

    def __init__(self, tag, seq, step, epoch, kind, module, data):
        super().__init__(tag, seq, step, epoch, kind)
        self._module  = module
        if data is None:
            raise ValueError('Data cannot be `None` for `TrainingMessage`')
        self._data = data

    @property
    def module(self):
        '''torch.nn.Module: Module emitting this data'''
        return self._module

    @property
    def key(self):
        return self.module.name


class MessageBundle(object):

    '''Data object for holding a set of artifacts for a module (or meta information) at one point
    during training. This data type can be used to buffer different kinds and check whether all
    expected kinds have been received for a module or meta information.

    .. note::

        This docstring doesn't make sense yet.

    '''
    def __init__(self, identifier, kinds):
        '''
        Parameters
        ----------
        identifier  :   str
                        String to uniquely identify the bundle. Will usually be either the module
                        name or `'META'` for a collection of meta messages.
        kinds   :   str or list
                    Single kind when a :class:`~ikkuna.export.subscriber.Subscription` is used, or a
                    list of Message kinds contained in this bundle for use with
                    :class:`~ikkuna.export.subscriber.SynchronizedSubscription`
        '''
        if isinstance(kinds, str):
            # this has bitten me before. base `Subscription`s don't use multiple kinds
            kinds = [kinds]
        self._identifier     = identifier
        self._expected_kinds = kinds
        self._received       = {kind: False for kind in kinds}
        self._data           = {kind: None for kind in kinds}
        self._seq            = None
        self._step           = None
        self._epoch          = None

    @property
    def identifier(self):
        ''' str: A string denoting the common aspect of the collected messages (besides the step).
        This can be the module name or a string such as ``META`` or other  denoting these are
        messages which do not belong to a module.'''
        return self._identifier

    @property
    def expected_kinds(self):
        '''list(str): The expected kinds of messages per iteration '''
        return self._expected_kinds

    @property
    def data(self):
        '''dict(str, torch.Tensor): The tensors received for each kind'''
        return self._data

    @property
    def seq(self):
        '''int: Global sequence number of this class'''
        return self._seq

    @property
    def step(self):
        '''int: Sequence number (training step) of the received messages (should match across all
        msgs in one iteration)'''
        return self._step

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
            If ``message.(seq|step|epoch|identifier)`` does not match the current
            ``(seq|step|epoch|identifier)`` or in case a message of ``message.kind`` has already
            been received
        '''
        ############
        #  seqnum  #
        ############
        if not self._seq:
            self._seq = message.seq

        if message.seq != self._seq:
            raise ValueError(f'Attempting to add message with seq {message.seq} to bundle with '
                             'initial seq {self._seq}')

        ##########
        #  step  #
        ##########
        if not self._step:
            self._step = message.step

        if message.step != self._step:
            raise ValueError(f'Attempting to add message with step {message.step} to bundle with '
                             'initial step {self._step}')
        #############
        #  content  #
        #############

        if not self._epoch:
            self._epoch = message.epoch

        if message.epoch != self._epoch:
            raise ValueError(f'Attempting to add message from epoch {message.epoch} to bundle with '
                             'initial epoch {self._epoch}')

        ################
        #  Identifier  #
        ################
        if self._identifier != message.key:
            raise ValueError(f'Unexpected identifier "{message.key}" '
                             '(expected "{self._identifier}")')

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
        mod   = self._identifier
        step  = self._step
        kinds = list(self._data.keys())
        return f'<MessageBundle: identifier={mod}, kinds={kinds}, step={step}>'

    def __repr__(self):
        return str(self)
