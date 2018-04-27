'''
.. module:: nn_module
.. moduleauthor:: Rasmus Diederichsen

Importing this module causes the initializer of :class:`torch.nn.Module` to be wrapped such that it
checks whether a supervisor is registered and add itself to its list of supervised modules. This
will allow the supervisor to monitor the variables inside.

.. warning::
    This module is not threadsafe. Since :mod:`torch.nn.Module`s `__init__` method is patched,
    creation of objects anywhere will be affected.
'''
import torch.nn as nn
_patch_history = {}

def _unwrap_init(cls):
    global _patch_history
    cls.__init__ = _patch_history.pop(cls)

def _wrap_init(cls):
    global _patch_history
    _patch_history[cls] = cls.__init__
    old_init = _patch_history[cls]
    def init(self, *args, **kwargs):
        # this needs to be in the innermost scope, otherwise we'll get a circular import with
        # supervise/supervise.py
        from supervise import current_supervisor
        old_init(self, *args, **kwargs)
        sup = current_supervisor()
        if sup:
            sup.add_module(self)
    cls.__init__ = init

_wrap_init(nn.Module)
