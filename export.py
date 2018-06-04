import torch
from collections import defaultdict
from types import MethodType


def record(tensor, label, method_name, *method_args):
    '''Record changes to a tensor.'''
    names = ['add_', 'mul_', 'addcdiv_']
    if method_name not in names:
        raise ValueError(f'Can only record {",".join(names)}')
    else:
        print(f'Recording {method_name} on {label}')


def patch_mutator(method):
    '''Patch mutating :class:`torch.Tensor` methods used by builtin optimizers so that the changes
    made to the tensor can be recorded. Permissible methods must be bound (retrieved via Tensor
    object, not class) and are ``add_``, ``addcdiv_``, and ``mul_`` at this time. The
    :class:`torch.nn.Parameter` or :class:`torch.Tensor` object to which the method is bound must
    posses a ``_label`` attribute (must be patched externally) in order to identify the layer which
    the weights belong to.

    .. note::
        I tried subclassing :class:`torch.nn.Parameter` and :class:`torch.Tensor` in order to simply
        trap calls to the mutating methods, and replace the values in the :class:`torch.nn.Module`s
        containing them, but this fails either because a new :class:`torch.Tensor` object is created
        on each access or because there's weird error hinting at device mismatch between my wrapper
        class and the update tensors. Forwarding each and every property access to the wrapped
        tensor was not sufficient.

    Parameters
    ----------
    method  :   MethodType
                Bound method to wrap

    Raises
    ------
    ValueError
        If method is unbound or bound to object without a ``_label`` attribute.
    '''
    if not hasattr(method, '__self__'):
        raise ValueError(f'Got unbound method {method}')

    this = method.__self__  # get bound object

    if not hasattr(this, '_label'):
        raise ValueError(f'{this.__class__.__name__} has not been given a `_label` attribute.')

    label = this._label     # get the layer name

    def replacement(tensor, *args):
        record(tensor, label, method.__name__, *args)
        method(*args)

    this.__dict__[method.__name__] = MethodType(replacement, this)


class WatchedParameter(torch.nn.Parameter):
    '''Wrapper class in order to trap mutations to a tensor inside a parameter. This is necesssary
    to retrieve updates to weights as applied by the optimizer, not just the raw gradients.

   We cannot simply set the ``param.data`` field directly since whenever it is accessed, torch
   creates a new python object referencing the data, which is then returned, meaning our wrapper
   will not actually get notified when stuff happens since the updates are being done on a new
   object with a reference to the binary data. Thus we need to start one level above, at the
   Parameter.

   Attributes
   ----------
   _name    :   str
                Name for the parameter, usually the layer identifier
   '''

    def __new__(cls, *args, name=None):
        return super().__new__(cls, *args)

    def __init__(self, baseObject, name=None):
        self.__class__ = type('Watched' + baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self._name = name

    def __getattribute__(self, name):
        if name == 'data':
            data = torch.nn.Parameter.__getattribute__(self, 'data')
            # set the label so it can be retrieved during recording
            data.__dict__['_label'] = self._name
            for method in [data.add_, data.addcdiv_, data.mul_]:
                patch_mutator(method)
            return data
        else:
            return torch.nn.Parameter.__getattribute__(self, name)


class Exporter(object):
    '''Class for managing publishing of data from model code.

    Usage
    --------
    An :class:`Exporter` is used in the model code by either explicitly registering modules for
    tracking with :meth:`Exporter.add_modules()` or by calling it with newly constructed modules
    which will then be returned as-is, but be registered in the process.

    .. code-block::
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
        self._modules = []
        self._frequency = frequency
        self._layer_counter = defaultdict(int)
        self._layer_names = []
        self._weight_cache = {}     # expensive :(
        self._bias_cache = {}
        self._subscribers = defaultdict(list)
        self._activation_counter = defaultdict(int)
        self._gradient_counter = defaultdict(int)

    def subscribe(self, kind, fn):
        '''Add a subscriber function for a certain event. The signature should be

        .. py:function:: callback(step: int, label: str, tensor: torch.Tensor)
        '''
        kinds = ['gradients', 'activations', 'weight_updates', 'bias_updates' 'weights', 'biases']
        if kind not in kinds:
            raise ValueError(f'Cannot subscribe to "{kind}"')
        self._subscribers[kind].append(fn)

    def get_or_make_label(self, module):
        '''Create or retrieve the label for a module. If the module is already tracked, its label
        is returned, else a new one is created.

        Paramters
        ---------
        module  :   torch.nn.Module

        Returns
        -------
        str
        '''
        if module not in self._modules:
            layer_kind = module.__class__.__name__
            number = self._layer_counter[layer_kind]
            layer_name = f'{layer_kind}-{number}'
        else:
            index = self._modules.index(module)
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

    def publish(self, module, step, kind, data):
        '''Publish an update to all registered subscribers.

        Paramters
        ---------
        module  :   torch.nn.Module
                    The module in question
        step    :   int
                    Step number (depends on the frequency configured)
        kind    :   str
                    Kind of subscriber to notify
        data    :   torch.Tensor
                    Payload
        '''
        for sub in self._subscribers[kind]:
            sub(step, self.get_or_make_label(module), data)

    def export_activations(self, module, activations):
        if not self._subscribers['activations']:
            pass
        else:
            print(f'New activations for {self.get_or_make_label(module)}')

    def export_gradients(self, module, gradients):
        if not self._subscribers['gradients']:
            pass
        else:
            print(f'New gradients for {self.get_or_make_label(module)}')

    def export_weights(self, module, weights):
        if not self._subscribers['weights']:
            pass
        else:
            print(f'New weights for {self.get_or_make_label(module)}')

    def export_biases(self, module, biases):
        if not self._subscribers['biases']:
            pass
        else:
            print(f'New biases for {self.get_or_make_label(module)}')

    def export_weight_updates(self, module, old, new):
        if not self._subscribers['weight_updates']:
            pass
        else:
            print(f'New weight updates for {self.get_or_make_label(module)}')

    def export_bias_updates(self, module, old, new):
        if not self._subscribers['bias_updates']:
            pass
        else:
            print(f'New bias updates for {self.get_or_make_label(module)}')

    def new_activations(self, module, in_, out_):
        self._activation_counter[module] += 1
        if hasattr(module, 'weight'):
            if module in self._weight_cache:
                self.export_weight_updates(module, self._weight_cache[module], module.weight)
            self.export_weights(module, module.weight)
            self._weight_cache[module] = torch.tensor(module.weight)
        if hasattr(module, 'bias'):
            if module in self._bias_cache:
                self.export_bias_updates(module, self._bias_cache[module], module.weight)
            self.export_biases(module, module.bias)
            self._bias_cache[module] = torch.tensor(module.bias)
        self.export_activations(module, out_)

    def new_gradients(self, module, in_, out_):
        self._gradient_counter[module] += 1
        self.export_gradients(module, out_)
