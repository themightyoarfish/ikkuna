'''
.. module:: supervise
.. moduleauthor:: Rasmus Diederichsen

This module gives access to supervision functionality, so that declared modules in a network can
automatically be monitored by a supervisor with minimal code adjustment.

.. highlight:: python
    import torch, supervise

    with supervise.capture_modules(torch.nn.Conv2d, torch.nn.ReLU):
        features = nn.Sequential(
            nn.Conv2d(...),
            nn.ReLU(...),
            nn.MaxPool2d(...),
            ...
        )

This will create a supervisor which then captures the modules whose classes are named.

The module maintains a stack of supervisors so that `with` blocks can be nested (for whatever
reason).

.. warning::
    This module is not threadsafe.
'''
_supervisor_stack = []

from patches import nn_module

def supervisors():
    '''Get the stack of all supervisors'''
    global _supervisor_stack
    return _supervisor_stack

def current_supervisor():
    '''Get topmost supervisor.'''
    global _supervisor_stack
    return None if not _supervisor_stack else _supervisor_stack[-1]

def capture_modules(*modules):
    '''Create a new supervisor for capturing modules.

    Parameters
    ----------
    modules :   list(type)
                Arbitrary types to register on creation
    '''
    return Supervisor(allowed_modules=modules)

class Supervisor():

    def __init__(self, allowed_modules=[]):
        self._modules = []
        self._prev_supervisor = None
        self._allowed_modules = allowed_modules

    def add_module(self, module):
        # attention: This call can be executed before the module is fully constructed, in case
        # nn.Module is patched. In that case, the concrete module's initializer will call into the
        # super class and we land here. So do not reference any object attributes here, only class
        # ones.
        # TODO: Think about the safety of this injection some more for production purposes
        if module.__class__ in self._allowed_modules:
            print(f'Adding {module.__class__.__name__}')
            self._modules.append(module)

    def __enter__(self):
        global _supervisor_stack
        _supervisor_stack.append(self)

    def __exit__(self, exc_type, exc_val, traceback):
        global _supervisor_stack
        _supervisor_stack.pop()
