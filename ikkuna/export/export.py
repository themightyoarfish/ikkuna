import sys
import torch
from collections import defaultdict

from ikkuna.export.messages import TrainingMessage, MetaMessage
from ikkuna.utils import ModuleTree


class Exporter(object):
    '''Class for managing publishing of data from model code.

    An :class:`Exporter` is used in the model code by either explicitly registering modules for
    tracking with :meth:`~Exporter.add_modules()` or by calling it with newly constructed modules
    which will then be returned as-is, but be registered in the process.

    .. code-block:: python

        e = Exporter(...)
        features = nn.Sequential([
            nn.Linear(...),
            e(nn.Conv2d(...)),
            nn.ReLU()
        ])

    Modules will be tracked recursive unless specified otherwise, meaning the following is possible:

    .. code-block:: python

        e = Exporter(...)
        e.add_modules(extremely_complex_model)
        # e will now track all layers of extremely_complex_model

    No further changes to the model code are necessary, but for a call to
    :meth:`~Exporter.set_model()` to have the :class:`Exporter` wire up the appropriate callbacks.

    Attributes
    ----------
    _modules    :   list(ikkuna.utils.NamedModule)
                    All tracked modules
    _weight_cache   :   dict
                        Cache for keeping the previous weights for computing differences
    _bias_cache :   dict
                    see ``_weight_cache``
    _subscribers    :   set(ikkuna.export.subscriber.Subscriber)
    _model          :   torch.nn.Module
    _train_step :   int
                    Current batch index
    _global_step    :   int
                        Global step accross all epochs
    _epoch  :   int
                Current epoch
    _is_training    :   bool
                        Flag enabling/disabling some messages during testing
    _did_publish_grads  :   defaultdict(bool)
                            Record for whether gradients have already been published at this train
                            step.  This is necessary since tensor hooks are called mutliple times.
    '''

    def __init__(self):
        self._modules           = []
        self._weight_cache      = {}     # potential memory hog
        self._bias_cache        = {}
        self._subscribers       = set()
        self._model             = None
        self._epoch             = 0
        # for gradient and activation to have the same step number, we need to increase it before
        # propagation or after backpropagation. but we don't know when the backprop finishes, while
        # we do know when the forward prop starts. So we step before and thus initialize the
        # counters with -1 to start at 0
        self._train_step        = -1
        self._global_step       = -1
        self._is_training       = True
        self._did_publish_grads = defaultdict(bool)

    def _check_model(self):
        if not self._model:
            import warnings
            warnings.warn('Warning: No model set. This will either do nothing or crash.')

    def subscribe(self, subscriber):
        '''Add a subscriber.

        Parameters
        ----------
        subscription    :   ikkuna.export.subscriber.Subscriber
                            Subscriber to register
        '''
        self._subscribers.add(subscriber)

    def _add_module_by_name(self, named_module):
        module = named_module.module
        self._modules.append(named_module)
        module.register_forward_hook(self.new_activations)


        has_bias = hasattr(module, 'bias') and module.bias is not None
        has_weight = hasattr(module, 'weight') and module.weight is not None
        if not has_weight and not has_bias:
            return

        # For some reason, registered tensor hooks are called twicein my setup. Maybe this means
        # that the gradient is computed twice, because the grad tensors are identical. Not sure why
        # this is so.
        # cache weight and bias gradients and only call new_gradients when both are received
        grad_cache = {'weight': None, 'bias': None}

        def weight_hook(grad):
            if self._did_publish_grads[module]:
                return
            else:
                grad_cache['weight'] = grad
                if not has_bias or grad_cache['bias'] is not None:
                    self.new_gradients(module, (grad_cache['weight'], grad_cache['bias']))
                    self._did_publish_grads[module] = True

        def bias_hook(grad):
            if self._did_publish_grads[module]:
                return
            else:
                grad_cache['bias'] = grad
                if not has_bias or grad_cache['weight'] is not None:
                    self.new_gradients(module, (grad_cache['weight'], grad_cache['bias']))
                    self._did_publish_grads[module] = True

        if has_bias:
            module.bias.register_hook(bias_hook)
        module.weight.register_hook(weight_hook)

    def add_modules(self, module, recursive=True, depth=-1):
        '''Add modules to supervise. If the module has ``weight`` and/or ``bias`` members, updates
        to those will be tracked.

        Parameters
        ----------
        module  :   tuple(str, torch.nn.Module) or torch.nn.Module
        recursive   :   bool
                        Descend recursively into the module tree
        depth   :   int
                    Depth to which to traverse the tree. Modules below this level will be ignored
        Raises
        ------
        ValueError
            If ``module`` is neither a tuple, nor a (subclass of) :class:`torch.nn.Module`
        '''
        if isinstance(module, tuple):   # name already given -> use that
            name, module = module
        else:
            name = module.__class__.__name__.lower()

        if isinstance(module, torch.nn.Module):
            module_tree = ModuleTree(module, name=name, recursive=recursive, drop_name=recursive)
            for named_module in module_tree.preorder(depth):
                module, name = named_module
                self._add_module_by_name(named_module)
        else:
            raise ValueError(f'Don\'t know how to handle {module.__class__.__name__}')

    def __call__(self, module, recursive=True, depth=-1):
        '''Shorthand for :meth:`~Exporter.add_modules()` which returns its input unmodified.

        Parameters
        ----------
        see :meth:`Exporter.add_modules()`

        Returns
        -------
        torch.nn.Module
            The input ``module``
        '''
        self.add_modules(module, recursive, depth)
        return module

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
        self._check_model()
        try:
            # TODO: Can I save the NamedModule instead of having to searh for it?
            index = next(i for i, m in enumerate(self._modules) if m.module == module)
        except StopIteration:
            raise RuntimeError(f'Received message for unknown module {module.name}')
        msg = TrainingMessage(seq=self._global_step, tag=None, kind=kind,
                                module=self._modules[index], step=self._train_step,
                                epoch=self._epoch, payload=data)
        for sub in self._subscribers:
            sub.receive_message(msg)

    def train(self, train=True):
        self._is_training = train

    def test(self, test=True):
        self.train(not test)

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
        if len(self._subscribers) == 0:
            pass
        else:
            self.publish(module, kind, data)

    def new_activations(self, module, in_, out_):
        '''Callback for newly arriving activations. Registered as a hook to the tracked modules.
        Will trigger export of all new activation and weight/bias data.

        Parameters
        ----------
        module  :   torch.nn.Module
        in_ :   torch.Tensor
                Dunno what this is
        out_    :   torch.Tensor
                    The new activations
        '''
        if not self._is_training:
            return
        if self._train_step == 0:
            msg_epoch = MetaMessage(seq=self._global_step, tag=None, kind='epoch_started',
                                        step=self._train_step, epoch=self._epoch)
            msg_batch = MetaMessage(seq=self._global_step, tag=None, kind='batch_started',
                                        step=self._train_step, epoch=self._epoch)
            for sub in self._subscribers:
                sub.receive_message(msg_epoch)
                sub.receive_message(msg_batch)

        if hasattr(module, 'weight'):
            if module in self._weight_cache:
                self.export('weight_updates', module, module.weight - self._weight_cache[module])
            else:
                self.export('weight_updates', module, torch.zeros_like(module.weight))
            self.export('weights', module, module.weight)
            self._weight_cache[module] = torch.tensor(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:   # bias can be present, but be None
            # in the first train step, there can be no updates, so we just publish zeros, otherwise
            # clients would error out since they don't receive the expected messages
            if module in self._bias_cache:
                self.export('bias_updates', module, module.bias - self._bias_cache[module])
            else:
                self.export('bias_updates', module, torch.zeros_like(module.bias))
            self.export('biases', module, module.bias)
            self._bias_cache[module] = torch.tensor(module.bias)
        self.export('activations', module, out_)

    def new_gradients(self, module, gradients):
        '''Callback for newly arriving gradients. Registered as a hook to the tracked modules.
        Will trigger export of all new gradient data.

        Parameters
        ----------
        module  :   torch.nn.Module
        gradients    :   tuple(torch.Tensor, torch.Tensor)
                        The gradients w.r.t weight and bias.
        '''
        # For some reason, the grad hooks get called twice per step, so only export the first time
        self.export('weight_gradients', module, gradients[0])
        self.export('bias_gradients', module, gradients[1])

    def set_model(self, model):
        '''Set the model for direct access for some metrics.

        Parameters
        ----------
        model   :   torch.nn.Module
        '''
        self._model = model
        from types import MethodType
        #############################################
        #  Patch the train function to notify self  #
        #############################################
        train_fn = model.train

        def new_train_fn(this, mode=True):
            train_fn(mode=mode)
            self._is_training = mode
        model.train = MethodType(new_train_fn, model)

        #########################################################
        #  Patch forward function to step() self automatically  #
        #########################################################
        forward_fn = model.forward

        def new_forward_fn(this, *args, should_train=True):
            # In order for accuracy subscribers to not need the model access, we add a secret
            # parameter which they can use to temporarily set the training to False and have it
            # revert automatically. TODO: Check if this is inefficient
            was_training = this.training        # store old value
            this.train(should_train)            # disable/enable training
            if this.training:
                # we need to step before forward pass, else act and grads get different steps
                self.step()
            ret = forward_fn(*args)             # do forward pass w/o messages spawning
            this.train(was_training)            # restore previous state
            return ret
        model.forward = MethodType(new_forward_fn, model)

    def step(self):
        '''Increase batch counter (per epoch) and the global step counter.'''
        self._train_step  += 1
        self._global_step += 1

        msg = MetaMessage(seq=self._global_step, tag=None, kind='batch_finished',
                          step=self._train_step, epoch=self._epoch)
        for sub in self._subscribers:
            sub.receive_message(msg)

        self._did_publish_grads.clear()

    def epoch_finished(self):
        '''Increase the epoch counter and reset the batch counter.'''
        msg = MetaMessage(seq=self._global_step, tag=None, kind='epoch_finished',
                          step=self._train_step, epoch=self._epoch)
        for sub in self._subscribers:
            sub.receive_message(msg)
        self._epoch      += 1
        self._train_step = 0
