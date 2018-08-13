meta_kinds = {
    'batch_started', 'batch_finished', 'epoch_started', 'epoch_finished'
}

data_kinds = {
    'weights', 'weight_updates', 'biases', 'bias_updates', 'activations', 'gradients',
}

allowed_kinds = set.union(meta_kinds, data_kinds)


class Message(object):
    '''
    Base class for messages emitted from the :class:`ikkuna.export.Exporter`.

    Attributes
    ----------
    tag :   str
            The tag associated with this message
    seq :   int
            Global sequence number. This counter should not reset after each epoch.
    step    :   int
                Epoch-local sequence number (the current batch index)
    epoch   :   int
                Current epoch number
    kind    :   str
                Message kind
    '''
    def __init__(self, tag, seq, step, epoch, kind):
        self._tag  = tag
        self._seq  = seq
        self.step  = step
        self.epoch = epoch
        self.kind  = kind

    @property
    def tag(self):
        return self._tag

    @property
    def seq(self):
        return self._seq

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if value < 0:
            raise ValueError
        else:
            self._step = value

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        if value < 0:
            raise ValueError
        else:
            self._epoch = value

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value):
        if value not in allowed_kinds:
            raise ValueError(f'Invalid message kind "{value}"')
        else:
            self._kind = value


class MetaMessage(Message):
    @Message.kind.setter
    def kind(self, value):
        if value not in meta_kinds:
            raise ValueError(f'Invalid message kind "{value}"')
        else:
            self._kind = value


class TrainingMessage(Message):
    '''
    These messages are assembled
    into :class:`ModuleData` objects in the :class:`ikkuna.export.subscriber.Subscription`.

    module  :   ikkuna.utils.NamedModule
    payload :   torch.Tensor
    '''
    def __init__(self, tag, seq, step, epoch, kind, module, payload):
        super().__init__(tag, seq, step, epoch, kind)
        self._module  = module
        self._payload = payload

    @property
    def module(self):
        return self._module

    @property
    def payload(self):
        return self._payload

    @Message.kind.setter
    def kind(self, value):
        if value not in data_kinds:
            raise ValueError(f'Invalid message kind "{value}"')
        else:
            self._kind = value


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
                Global sequence number of this class (incremented whenever a :class:`ModuleData` is
                created)
    _step   :   int
                Sequence number (training step) of the received messages (should match across all
                msgs in one iteration)
    _epoch  :   int
                Epoch of the received messages (should match across all msgs in one
                iteration)
    '''

    def __init__(self, module, kinds):
        self._module            = module
        if isinstance(kinds, str):  # this has bitten me before. base subscriptions don't use multiple kinds
            kinds = [kinds]
        self._expected_kinds    = kinds
        self._data              = {kind: None for kind in kinds}
        self._seq               = None
        self._step              = None
        self._epoch             = None

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
        if all(map(lambda k: k in meta_kinds, self._expected_kinds)):
            # only meta kinds -> don't wait for payload
            return True
        else:
            return all(map(lambda val: val is not None, self._data.values()))

    def _check_message(self, message):
        '''Check consistency of sequence number, step and epocg or set if not set yet

        Parameters
        ----------
        message :   ikkuna.export.messages.TrainingMessage

        Raises
        ------
        ValueError
            If ``message.(seq|step|epoch)`` does not match the current ``(seq|step|epoch)``
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

    def add_message(self, message):
        '''Add a new message to this object. Will fail if the new messsage does not have the same
        sequence number and epoch.

        Parameters
        ----------
        message :   ikkuna.export.messages.TrainingMessage

        Raises
        ------
        ValueError
            If the message is for a different module (layer) or a message of this kind was already
            received.
        '''
        self._check_message(message)

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
