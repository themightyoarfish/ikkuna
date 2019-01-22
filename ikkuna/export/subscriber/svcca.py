from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import svcca

from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus
from ikkuna.utils import freeze_module


class ChunkedDict(object):
    '''A dictionary-like class that can be used to incrementally fill tensors of predetermined size
    by accumulating data. The dictionary keeps track of what has already been received and kan thus
    be asked if all data for some module was received.  This can be used e.g. for stitching together
    an activation matrix of (n_neurons, n_datapoints) when the activations arrive in batches.'''

    def __init__(self, expected_n):
        '''
        Parameters
        ----------
        expected_n  :   int
                        The number of datapoints to expect for all modules
        '''
        self._expected_n = expected_n
        self._received_n = defaultdict(int)
        self._data = dict()

    def clear(self):
        '''Remove all entries.'''
        self._received_n.clear()
        self._data.clear()

    def append(self, module, data):
        '''Add data for a module. Creates new record if one is not already open for this module.

        Parameters
        ----------
        module  :   torch.nn.Module or str
                    Some unique key for this module
        data    :   torch.Tensor
                    Data tensor to append
        '''
        if module not in self._data:
            self._data[module] = torch.zeros(self._expected_n, *data.shape[1:], device=data.device)

        n = data.shape[0]
        received = self._received_n[module]
        self._data[module][received:received + n, ...] = data
        self._received_n[module] += n

    def keys(self):
        '''Get all known modules, completed or otherwise.

        Returns
        -------
        dict_keys
            View of the data dictionary's keys
        '''
        return self._data.keys()

    def bytesize(self):
        '''Compute occupied bytes of the data (without the keys)

        Returns
        -------
        int
            Bytes consumed by all the tensors
        '''
        size = 0
        for module in self._data:
            size += self._data[module].numel() * 4
        return size

    def __setitem__(self, module, data):
        '''Set the data for a module, overwriting any previous data.'''
        if module in self._data:
            del self._data[module]
            del self._received_n[module]
        self.append(module, data)

    def __getitem__(self, key):
        '''Get data for a module.

        Returns
        -------
        torch.Tensor

        Raises
        ------
        KeyError
            If the module's data is not yet complete
        '''
        if not self.complete(key):
            raise KeyError(f'Data for {key} not yet complete.')
        return self._data[key]

    def __contains__(self, module):
        return module in self._data

    def complete(self, module):
        '''Check if expected number of datapoints have been received for a module.

        Returns
        -------
        bool
        '''
        return self._received_n[module] == self._expected_n

    def pop(self, module):
        '''Remove and return data for module'''
        data = self[module]
        del self._data[module]
        del self._received_n[module]
        return data


class BatchedSVCCASubscriber(PlotSubscriber):
    '''A subscriber which at some interval halts training, propagates some dataset through the net,
    records activations for all modules and stores them. At the next checkpoint, SVCCA similarity
    is computed between every layer at time step t and the same layer at t-1.

    .. note::
        It seems that despite the fact that SVCCA runs at least twice as fast on the GPU, this
        subscriber is not faster than on the CPU. I'm not sure why that is, since I'm even caching
        input data to avoid repeatedly copying to the GPU

    .. warning::
        Currently, this subscriber uses Numpy for SVCAA. For some reason, the activations produced
        by networks lead to highly ill-conditioned covariance matrices and PyTorch's SVD handles
        them differently than Numpy does. SVD does not converge in that case, and
        ``robust_cca_similarity`` must be used, which defeats the whole idea of speeding up the
        process.
    '''

    def __init__(self, dataset_meta, n, forward_fn, freeze_at=10, batch_size=256,
                 message_bus=get_default_bus(), tag='default', subsample=1, ylims=None,
                 backend='tb'):
        '''
        Parameters
        ----------
        dataset_meta    :   train.DatasetMeta
                            Dataset to load data from for retrieving activations
        n   :   int
                Number of datapoints to randomly sample
        freeze_at   :   float
                        Similarity threshold after which a layer is frozen. Values >= 1 will turn
                        off freezing
        batch_size  :   int
                        Batch size to use for forward passes. Can be set to some value if the entire
                        data at once would be too larger. Otherwise use :class:`SVCCASubscriber`
        '''

        self._forward_fn     = forward_fn
        self._previous_acts  = ChunkedDict(n)
        self._current_acts   = ChunkedDict(n)
        indices              = np.random.randint(0, dataset_meta.size, size=n)
        dataset              = Subset(dataset_meta.dataset, indices)
        self._loader         = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                          pin_memory=True)
        # cache tensors so we don't repeatedly deserialize and copy
        self._input_cache    = []

        self._freeze_at      = freeze_at
        self._ignore_modules = set()

        title        = f'self_similarity'
        ylabel       = 'Similarity'
        xlabel       = 'Train step'
        subscription1 = Subscription(self, ['batch_finished'], tag=tag, subsample=subsample)
        subscription2 = Subscription(self, ['activations'], tag='svcca_testing',
                                     subsample=1)
        super().__init__([subscription1, subscription2],
                         message_bus,
                         {'title': title,
                          'ylabel': ylabel,
                          'ylims': ylims,
                          'xlabel': xlabel},
                         backend=backend)
        self._add_publication(f'self_similarity', type='DATA')

    def _module_complete_previous(self, module_name):
        '''Check if activations for module are completely buffered from the previous step.'''
        return not (module_name not in self._previous_acts
                    or not self._previous_acts.complete(module_name))

    def _module_complete_current(self, module_name):
        '''Check if activations for module are completely buffered from the current step.'''
        return not (module_name not in self._current_acts
                    or not self._current_acts.complete(module_name))

    def _record_activations_previous(self, module_name, data):
        '''Record activations into the ``previous`` buffer'''
        self._previous_acts.append(module_name, data)

    def _record_activations_current(self, module_name, data):
        '''Record activations into the ``current`` buffer'''
        self._current_acts.append(module_name, data)

    def _do_forward_pass(self):
        # cache inputs initially
        if not self._input_cache:
            loader = iter(self._loader)
        else:
            loader = iter(self._input_cache)

        for i, (X, labels) in enumerate(loader):
            X = X.cuda()
            if len(self._input_cache) < i + 1:
                self._input_cache.append((X, labels))
            self._forward_fn(X, should_train=False, tag='svcca_testing')

    def _compute_similarity(self, name):
        # now both current and previous acts are complete and we can compute
        # print(f'Computing similarity for {name}')
        previous_acts = self._previous_acts.pop(name)
        current_acts = self._current_acts.pop(name)

        self._previous_acts[name] = current_acts

        if previous_acts.ndimension() > 2:
            c = previous_acts.shape[1]  # channel dim
            previous_acts = previous_acts.permute([0, 2, 3, 1]).reshape(-1, c)

        if current_acts.ndimension() > 2:
            c = current_acts.shape[1]  # channel dim
            current_acts = current_acts.permute([0, 2, 3, 1]).reshape(-1, c)

        previous_acts = previous_acts.detach().cpu().numpy()
        current_acts = current_acts.detach().cpu().numpy()
        result_dict = svcca.cca_core.robust_cca_similarity(previous_acts.T,
                                                           current_acts.T,
                                                           epsilon=1e-8,
                                                           threshold=0.98,
                                                           verbose=False,
                                                           compute_dirns=False,
                                                           rescale=True)
        return result_dict['mean'][0]

    def compute(self, message):
        '''A :class:`~ikkuna.export.messages.NetworkMessage` with the identifier ``self_similarity``
        will be published.'''

        if message.tag == 'default' and message.kind == 'batch_finished':
            self._do_forward_pass()

        elif message.tag == 'svcca_testing':
            module, name = message.key
            if module in self._ignore_modules:
                return

            if not self._module_complete_previous(name):
                self._record_activations_previous(name, message.data)
            elif not self._module_complete_current(name):
                self._record_activations_current(name, message.data)

            if self._module_complete_current(name) and self._module_complete_previous(name):
                mean = self._compute_similarity(name)
                self._backend.add_data(name, mean, message.global_step)
                self.message_bus.publish_module_message(message.global_step,
                                                        message.train_step,
                                                        message.epoch,
                                                        'self_similarity',
                                                        message.key,
                                                        data=mean)

                if mean > self._freeze_at:
                    freeze_module(module)
                    self._ignore_modules.add(module)
                    if name in self._previous_acts:
                        self._previous_acts.pop(name)
                    if name in self._current_acts:
                        self._current_acts.pop(name)


class SVCCASubscriber(BatchedSVCCASubscriber):
    '''Simplified :class:`BatchedSVCCASubscriber` subclass which does not use a
    ``ChunkedDict`` but instead regular dicts and thus assumes that the entire test data can be
    propagated through the model in one go and batching is unnecessary.'''

    def __init__(self, dataset_meta, n,  *args, **kwargs):
        super().__init__(dataset_meta, n, *args, **kwargs, batch_size=n)
        self._previous_acts = dict()
        self._current_acts = dict()

    def _module_complete_previous(self, module_name):
        return module_name in self._previous_acts

    def _module_complete_current(self, module_name):
        return module_name in self._current_acts

    def _record_activations_previous(self, module_name, data):
        self._previous_acts[module_name] = data

    def _record_activations_current(self, module_name, data):
        self._current_acts[module_name] = data
