from types import MethodType
import torch


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
        trap calls to the mutating methods, and replace the values in the :class:`torch.nn.Module` s
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
