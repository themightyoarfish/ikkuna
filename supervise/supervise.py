_supervisor_stack = []

import torch.nn as nn   # noqa
from patches import nn_module   # noqa


def supervisors():
    '''Get the stack of all supervisors'''
    global _supervisor_stack
    return _supervisor_stack


def current_supervisor():
    '''Get topmost supervisor.'''
    global _supervisor_stack
    return None if not _supervisor_stack else _supervisor_stack[-1]


def capture_modules(*modules, allow_subclass=False):
    '''Create a new supervisor for capturing modules.

    Parameters
    ----------
    modules :   list(type)
                Arbitrary types to register on creation
    '''
    return Supervisor(modules, allow_subclass=allow_subclass)


class Supervisor():
    '''A context manager for tracking the creation of nn modules.

    Attributes
    ----------
    _modules    :   list
                    List of modules created in this context manager
    _valid_predicate    :   function
                            Predicate based on the ``allow_subclass`` parameter for determining
                            whether to accept a newly created module for supervision.
    _activation_observers :   set
                            Set of observers for activation updates
    _gradient_observers :   set
                            Set of observers for gradient updates
    '''

    def __init__(self, *allowed_modules, allow_subclass: bool=False):
        '''
        Parameters
        ----------
        allowed_modules :   list(type)
                            List of module classes for supervision
        allow_subclass  :   bool
                            Allow calling :meth:`add_module` with subclass instances of any
                            ``allowed_modules`` member

        '''
        self._modules = []

        if allow_subclass:
            self._valid_predicate = lambda m: m.__class__ in allowed_modules
        else:
            self._valid_predicate = lambda m: any(isinstance(m, cls) for cls in allowed_modules)

        self._gradient_observers   = set()
        self._activation_observers = set()

    def _check_module_is_supervised(self, module: nn.Module):
        '''Check if a module is tracked.

        Raises
        ------
        ValueError
            If ``module`` is not currently supervised.
        '''
        if module not in self._modules:
            raise ValueError(f'Module {module} not registered with this Supervisor')

    def register_activation_observer(self, observer):
        '''Register and observer for handling new activations.

        Parameters
        ----------
        observer    :   object
        '''
        for module in self._modules:
            observer.add_module(module)
        self._activation_observers.add(observer)

    def register_gradient_observer(self, observer):
        '''Register and observer for handling new gradients.

        Parameters
        ----------
        observer    :   object
        '''
        for module in self._modules:
            observer.add_module(module)
        self._gradient_observers.add(observer)

    def _process_activations(self, module, in_, out_):
        '''Hook to register on modules for receiving activation updates. See
        :meth:`nn.Module.register_forward_hook`.

        Raises
        ------
        ValueError
            If ``module`` was not previously added with :meth:`add_module`.
        '''
        self._check_module_is_supervised(module)
        for o in self._activation_observers:
            o.process_activations(module, out_)

    def _process_gradients(self, module, grad_in_, grad_out_):
        '''Hook to register on modules for receiving gradient updates. See
        :meth:`nn.Module.register_backward_hook`

        Raises
        ------
        ValueError
            If ``module`` was not previously added with :meth:`add_module`.
        '''
        self._check_module_is_supervised(module)
        for o in self._gradient_observers:
            o.process_gradients(module, grad_out_)

    def add_module(self, module: nn.Module):
        '''Add a module to supervise.'''
        # attention: This call can be executed before the module is fully constructed, in case
        # nn.Module.__init__ is patched to call this method. In that case, the concrete module's
        # initializer will call into the super class and we land here. So do not reference any
        # object attributes here, only class ones.
        # TODO: Think about the safety of this injection some more for production purposes
        if self._valid_predicate(module):
            self._modules.append(module)
            module.register_forward_hook(self._process_activations)
            module.register_backward_hook(self._process_gradients)
            import itertools
            for handler in itertools.chain(self._gradient_observers, self._activation_observers):
                handler.add_module(module)

    def __enter__(self):
        global _supervisor_stack
        _supervisor_stack.append(self)

    def __exit__(self, exc_type, exc_val, traceback):
        global _supervisor_stack
        assert self == _supervisor_stack[-1]    # sanity check
        _supervisor_stack.pop()
