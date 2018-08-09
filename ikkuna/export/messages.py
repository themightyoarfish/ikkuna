from collections import namedtuple


allowed_kinds = {
    'weights', 'weight_updates', 'biases', 'bias_updates', 'activations', 'gradients'
}


class NetworkData(object):
    '''
    Primitive data emitted from the :class:`ikkun.export.Exporter`. These messages are assembled
    into :class:`ModuleData` objects in the :class:`ikkuna.export.subscriber.Subscription`. Perhaps
    these classes are not perfectly named.

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
    module  :   ikkuna.utils.NamedModule
    payload :   torch.Tensor
    '''
    def __init__(self, tag, seq, step, epoch, kind, module, payload):
        self._tag     = tag
        self._seq     = seq
        self.step     = step
        self.epoch    = epoch
        self.kind     = kind
        self._module  = module
        self._payload = payload

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
            raise ValueError
        else:
            self._kind = value

    @property
    def module(self):
        return self._module

    @property
    def payload(self):
        return self._payload


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

    global_seq = 0

    def __init__(self, module, kinds):
        self._module            = module
        self._expected_kinds    = kinds
        self._data              = {kind: None for kind in kinds}
        self._seq               = ModuleData.global_seq
        self._step              = None
        self._epoch             = None
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
