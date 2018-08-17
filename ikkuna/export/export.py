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

    Modules will be tracked recursively unless specified otherwise, meaning the following is
    possible:

    .. code-block:: python

        e = Exporter(...)
        e.add_modules(extremely_complex_model)
        # e will now track all layers of extremely_complex_model

    Three further changes to the training code are necessary

        #. :meth:`~Exporter.set_model()` to have the :class:`Exporter` wire up the appropriate
           callbacks.
        #. :meth:`~Exporter.set_loss()` should be called with the loss function so that
           labels can be extracted during training.
        #. :meth:`~Exporter.epoch_finished()` should be called if any
           :class:`~ikkuna.export.subscriber.Subscriber`\ s rely on the ``'epoch_finished'`` signal

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
    _depth  :   int
                Depth to which to traverse the module tree

    '''

    def __init__(self, depth):
        self._modules           = {}
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
        self._depth             = depth
        self._frozen            = set()

    @property
    def modules(self):
        '''list(torch.nn.Module) - Modules tracked by this :class:`Exporter`'''
        return list(self._modules.keys())

    @property
    def named_modules(self):
        '''list(ikkuna.utils.NamedModule) - Named modules tracked by this :class:`Exporter`'''
        return list(self._modules.values())

    def _check_model(self):
        '''Check if a model is set, issue a warning otherwise. Some things may still work and not
        crash (???).'''
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
        '''Register a module with a name attached.

        Parameters
        ----------
        named_module    :   ikkuna.utils.NamedModule
        '''
        module                = named_module.module
        self._modules[module] = named_module
        module.register_forward_hook(self.new_activations)

        has_bias   = hasattr(module, 'bias') and module.bias is not None
        has_weight = hasattr(module, 'weight') and module.weight is not None
        if not has_weight and not has_bias:
            return

        # For some reason, registered tensor hooks are called twicein my setup. Maybe this means
        # that the gradient is computed twice, because the grad tensors are identical. Not sure why
        # this is so.
        # cache weight and bias gradients and only call new_gradients when both are received
        grad_cache = {'weight': None, 'bias': None}

        # the hooks will check whether both weight and bias have been received and if so, trigger
        # publication. If the module has no bias, then ``None`` is published for the bias component.
        # They also check whether we grads were already published at this train step and do nothing
        # in that case.
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

    def add_modules(self, module, recursive=True):
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
            for named_module in module_tree.preorder(depth=self._depth):
                module, name = named_module
                self._add_module_by_name(named_module)
        else:
            raise ValueError(f'Don\'t know how to handle {module.__class__.__name__}')

    def __call__(self, module, recursive=True):
        '''Shorthand for :meth:`~Exporter.add_modules()` which returns its input unmodified.

        Parameters
        ----------
        see :meth:`Exporter.add_modules()`

        Returns
        -------
        torch.nn.Module
            The input ``module``
        '''
        self.add_modules(module, recursive)
        return module

    def publish_meta(self, kind, data=None):
        '''Publish an update of type :class:`~ikkuna.export.messages.MetaMessage` to all
        registered subscribers.

        Parameters
        ----------
        kind    :   str
                    Kind of message
        data    :   torch.Tensor or None
                    Payload, if necessary
        '''
        self._check_model()
        msg = MetaMessage(seq=self._global_step, tag=None, kind=kind, step=self._train_step,
                          epoch=self._epoch, data=data)
        for sub in self._subscribers:
            sub.receive_message(msg)

    def publish_training(self, kind, module, data):
        '''Publish an update of type :class:`~ikkuna.export.messages.TrainingMessage` to all
        registered subscribers.

        Parameters
        ----------
        kind    :   str
                    Kind of message
        module  :   torch.nn.Module
                    The module in question
        data    :   torch.Tensor
                    Payload
        '''
        self._check_model()

        named_module = self._modules[module]
        msg          = TrainingMessage(seq=self._global_step, tag=None, kind=kind,
                                       module=named_module, step=self._train_step,
                                       epoch=self._epoch, data=data)
        for sub in self._subscribers:
            sub.receive_message(msg)

    def train(self, train=True):
        '''Switch to training mode. This will ensure all data is published.'''
        self._is_training = train

    def test(self, test=True):
        '''Switch to testing mode. This will turn off all publishing.'''
        self.train(not test)

    def new_input(self, *args):
        '''Callback for new training input to the network.

        Parameters
        ----------
        *args   :   tuple
                    Network inputs
        '''
        if len(args) == 1:
            input_data = args[0]
        else:
            input_data = args

        self.publish_meta('input_data', input_data)

    def new_output_and_labels(self, network_output, labels):
        '''Callback for final network output.

        Parameters
        ----------
        data    :   torch.Tensor
                    The final layer's output
        '''
        self.publish_meta('network_output', network_output)
        self.publish_meta('input_labels', labels)

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
            self.publish_meta('epoch_started')
            self.publish_meta('batch_started')

        if hasattr(module, 'weight'):
            if module in self._weight_cache:
                self.publish_training('weight_updates', module, module.weight -
                                      self._weight_cache[module])
            else:
                self.publish_training('weight_updates', module, torch.zeros_like(module.weight))
            self.publish_training('weights', module, module.weight)
            self._weight_cache[module] = torch.tensor(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:   # bias can be present, but be None
            # in the first train step, there can be no updates, so we just publish_training zeros,
            # otherwise clients would error out since they don't receive the expected messages
            if module in self._bias_cache:
                self.publish_training('bias_updates', module,
                                      module.bias - self._bias_cache[module])
            else:
                self.publish_training('bias_updates', module, torch.zeros_like(module.bias))
            self.publish_training('biases', module, module.bias)
            self._bias_cache[module] = torch.tensor(module.bias)
        self.publish_training('activations', module, out_)

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
        self.publish_training('weight_gradients', module, gradients[0])

        if gradients[1] is not None:
            self.publish_training('bias_gradients', module, gradients[1])

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
            self.new_input(*args)               # do this after stepping
            ret = forward_fn(*args)             # do forward pass w/o messages spawning
            this.train(was_training)            # restore previous state
            return ret
        model.forward = MethodType(new_forward_fn, model)

    def set_loss(self, loss_function):
        '''Add hook to loss function to extract labels.

        Parameters
        ----------
        loss_function   :   torch.nn._Loss
        '''

        def hook(mod, output_and_labels, loss):
            network_output, labels = output_and_labels
            self.new_output_and_labels(network_output, labels)

        loss_function.register_forward_hook(hook)

    def step(self):
        '''Increase batch counter (per epoch) and the global step counter.'''
        self._train_step  += 1
        self._global_step += 1

        self.publish_meta('batch_finished')

        self._did_publish_grads.clear()

    def _freeze_module(self, named_module):
        '''Convenience method for freezing training for a module.

        Parameters
        ----------
        named_module    :   ikkuna.utils.NamedModule
                            Module to freeze
        '''

        def freeze(mod):
            for p in mod.parameters():
                p.requires_grad = False

        if named_module not in self._frozen:
            self._frozen.add(named_module)
            print(f'Freezing {named_module.name}')
            named_module.module.apply(freeze)

    def epoch_finished(self):
        '''Increase the epoch counter and reset the batch counter.'''

        self.publish_meta('epoch_finished')
        self._epoch     += 1
        self._train_step = 0
