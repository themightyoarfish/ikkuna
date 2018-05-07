'''
.. module:: supervise
.. moduleauthor:: Rasmus Diederichsen

This module gives access to supervision functionality, so that declared modules in a network can
automatically be monitored by a supervisor with minimal code adjustment.

.. code-block: python

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
from .supervise import *
