import torch
from collections import defaultdict
from ikkuna.export.messages import NetworkData


class Exporter(object):
    '''Class for managing publishing of data from model code.

    An :class:`Exporter` is used in the model code by either explicitly registering modules for
    tracking with :meth:`Exporter.add_modules()` or by calling it with newly constructed modules
    which will then be returned as-is, but be registered in the process.

    .. code-block:: python

        e = Exporter(...)
        features = nn.Sequential([
            nn.Linear(...),
            e(nn.Conv2d(...)),
            nn.ReLU()
        ])

    No further changes to the model code are necessary, but for certain visualizations, the
    exporter requires access to the model in its entirety, so :meth:`Exporter.add_model()` should be
    used.

    Attributes
    ----------
    _modules    :   list
                    All tracked modules
    _layer_counter  :   defaultdict
                        Dictionary tracking number and kind of added modules for
                        generating a name for each.
    _layer_names    :   list
                        List which parallels _modules and contains the label for each
    _weight_cache   :   dict
                        Cache for keeping the previous weights for computing differences
    _bias_cache :   dict
    '''

    def __init__(self, frequency=1):
        '''Create a new ``Exporter``.

        Parameters
        ----------
        frequency   :   int
                        Number of training steps to pass by before publishing a new update.
        '''
        self._modules            = []
        self._frequency          = frequency
        self._layer_counter      = defaultdict(int)
        self._layer_names        = []
        self._weight_cache       = {}     # expensive :(
        self._bias_cache         = {}
        self._subscribers        = defaultdict(list)
        self._activation_counter = defaultdict(int)
        self._gradient_counter   = defaultdict(int)
        self._model              = None
        self._global_step        = 0
        self._epoch              = 0

    def subscribe(self, kind, fn):
        '''Add a subscriber function or a callable for a certain event. The signature should be

        .. py:function:: callback(data: NetworkData)

        Parameters
        ----------
        kind    :   str
                    Kind of update to receive. Valid choices are ``gradients``, ``activations``,
                    ``weight_updates``, ``bias_updates`` ``weights``, ``biases``
        fn  :   function
                Callable to register

        Raises
        ------
        ValueError
            If ``kind`` is invalid
        '''
        kinds = ['gradients', 'activations', 'weight_updates', 'bias_updates' 'weights', 'biases']
        if kind not in kinds:
            raise ValueError(f'Cannot subscribe to "{kind}"')
        self._subscribers[kind].append(fn)

    def get_or_make_label(self, module):
        '''Create or retrieve the label for a module. If the module is already tracked, its label
        is returned, else a new one is created.

        Parameters
        ----------
        module  :   torch.nn.Module

        Returns
        -------
        str
        '''
        if module not in self._modules:
            layer_kind = module.__class__.__name__
            number     = self._layer_counter[layer_kind]
            layer_name = f'{layer_kind}-{number}'
        else:
            index      = self._modules.index(module)
            layer_name = self._layer_names[index]
        return layer_name

    def add_modules(self, module):
        '''Add modules to supervise. Currently, only activations and gradients are tracked. If the
        module has ``weight`` and/or ``bias`` members, updates to those will be tracked.

        .. note::
            In the future, this method should automagically discover which parts of a compound
            module or list of modules need to be tracked and return a fused module so it still works
            in :class:`torch.nn.Sequential` and the like.

        Parameters
        ----------
        module  :   torch.nn.Module
                    In a future version, a list of modules should be supported.
        '''
        if isinstance(module, tuple):   # name already given -> use that
            layer_name, module = module
        else:
            layer_name = self.get_or_make_label(module)
            self._layer_counter[module.__class__] += 1
        if module not in self._modules:
            self._layer_names.append(layer_name)
            self._modules.append(module)
            module.register_forward_hook(self.new_activations)
            module.register_backward_hook(self.new_gradients)
        return module

    def __call__(self, *args):
        # TODO: Handle initialization methods torch.nn.Sequential with named modules
        if all(map(lambda o: isinstance(o, torch.nn.Module), args)):
            return self.add_modules(*args)
        elif len(args) == 1 and isinstance(args[0], torch.optim.Optimizer):
            return self.add_optimizer(args[0])

    def publish(self, module, kind, data):
        '''Publish an update to all registered subscribers.

        Parameters
        ----------
        module  :   torch.nn.Module
                    The module in question
        kind    :   str
                    Kind of subscriber to notify
        data    :   torch.Tensor
                    Payload
        '''
        for sub in self._subscribers[kind]:
            msg = NetworkData(kind=kind, module=self.get_or_make_label(module),
                              step=self._global_step, epoch=self._epoch, payload=data)
            sub(msg)

    def export(self, kind, module, data):
        '''Publish new data to any subscribers.

        Parameters
        ----------
        kind    :   str
                    Kind of subscription
        module  :   torch.nn.Module
        data    :   torch.Tensor
                    Payload to publish
        '''
        if not self._subscribers[kind]:
            pass
        else:
            self.publish(module, kind, data)

    def new_activations(self, module, in_, out_):
        '''Callback for newly arriving activations. Registered as a hook to the tracked modules.
        Will trigger exportation of all new activation and weight/bias data.

        Parameters
        ----------
        module  :   torch.nn.Module
        in_ :   torch.Tensor
                Dunno what this is
        out_    :   torch.Tensor
                    The new activations
        '''
        self._activation_counter[module] += 1
        if hasattr(module, 'weight'):
            if module in self._weight_cache:
                self.export('weight_updates', module, module.weight - self._weight_cache[module])
            self.export('weights', module, module.weight)
            self._weight_cache[module] = torch.tensor(module.weight)
        if hasattr(module, 'bias'):
            if module in self._bias_cache:
                self.export('bias_updates', module, module.bias - self._bias_cache[module])
            self.export('biases', module, module.bias)
            self._bias_cache[module] = torch.tensor(module.bias)
        self.export('activations', module, out_)

    def new_gradients(self, module, in_, out_):
        '''Callback for newly arriving gradients. Registered as a hook to the tracked modules.
        Will trigger exportation of all new gradient data.

        Parameters
        ----------
        module  :   torch.nn.Module
        in_ :   torch.Tensor
                Dunno what this is
        out_    :   torch.Tensor
                    The new activations
        '''
        self._gradient_counter[module] += 1
        if isinstance(out_, tuple):
            if len(out_) > 1:
                raise RuntimeError(f'Not sure what to do with tuple gradients.')
            else:
                out_, = out_
        self.export('gradients', module, out_)

    def set_model(self, model):
        '''Set the model for direct access for some metrics.

        Parameters
        ----------
        model   :   torch.nn.Module
        '''
        self._model = model

    def step(self):
        '''Increase batch counter.'''
        self._global_step += 1

    def epoch_finished(self):
        '''Increase the epoch counter.'''
        self._epoch      += 1
        self._global_step = 0
