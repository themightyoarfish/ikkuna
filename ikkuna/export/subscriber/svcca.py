from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import svcca

from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class ChunkedDict(object):

    def __init__(self, expected_n):
        self._expected_n = expected_n
        self._received_n = defaultdict(int)
        self._data = dict()

    def clear(self):
        self._received_n.clear()
        self._data.clear()

    def append(self, module, data):
        if module not in self._data:
            self._data[module] = torch.zeros(self._expected_n, *data.shape[1:], device=data.device)

        n = data.shape[0]
        received = self._received_n[module]
        self._data[module][received:received + n, ...] = data
        self._received_n[module] += n

    def keys(self):
        return self._data.keys()

    def bytesize(self):
        size = 0
        for module in self._data:
            size += self._data[module].numel() * 4
        return size

    def __setitem__(self, module, data):
        if module in self._data:
            del self._data[module]
            del self._received_n[module]
        self.append(module, data)

    def __getitem__(self, key):
        if not self.complete(key):
            raise KeyError(f'Data for {key} not yet complete.')
        return self._data[key]

    def __contains__(self, module):
        return module in self._data

    def complete(self, module):
        return self._received_n[module] == self._expected_n

    def pop(self, module):
        data = self._data[module]
        del self._data[module]
        del self._received_n[module]
        return data


class BatchedSVCCASubscriber(PlotSubscriber):

    def __init__(self, dataset_meta, n, forward_fn,
                 message_bus=get_default_bus(), tag='default', subsample=1, ylims=None,
                 backend='tb', freeze_at=10, batch_size=256):

        self._forward_fn = forward_fn
        self._previous_acts = ChunkedDict(n)
        self._current_acts = ChunkedDict(n)
        indices = np.random.randint(0, dataset_meta.size, size=n)
        dataset = Subset(dataset_meta.dataset, indices)
        self._loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        self._freeze_at = freeze_at
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
        # TODO: Publish messages in compute()

    def _module_complete_previous(self, module_name):
        return not (module_name not in self._previous_acts or not self._previous_acts.complete(module_name))

    def _module_complete_current(self, module_name):
        return not (module_name not in self._current_acts or not self._current_acts.complete(module_name))

    def _record_activations_previous(self, module_name, data):
        self._previous_acts.append(module_name, data)

    def _record_activations_current(self, module_name, data):
        self._current_acts.append(module_name, data)

    def compute(self, message):
        if message.tag == 'default' and message.kind == 'batch_finished':
            loader = iter(self._loader)
            try:
                while True:
                    X, _ = next(loader)
                    X = X.cuda()
                    self._forward_fn(X, should_train=False, tag='svcca_testing')
            except StopIteration:
                pass

        elif message.tag == 'svcca_testing':
            module, name = message.key
            if module in self._ignore_modules:
                return

            # if name == '/features/conv2d0':
            #     __import__('ipdb').set_trace()
            if not self._module_complete_previous(name):
                self._record_activations_previous(name, message.data)
            elif not self._module_complete_current(name):
                self._record_activations_current(name, message.data)

            if self._module_complete_current(name) and self._module_complete_previous(name):
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
                result_dict = svcca.cca_core.get_cca_similarity(previous_acts.T,
                                                                current_acts.T,
                                                                epsilon=1e-8,
                                                                threshold=0.98,
                                                                verbose=False,
                                                                compute_dirns=False)
                mean = result_dict['mean'][0]
                self._backend.add_data(name, mean, message.global_step)

                if mean > self._freeze_at:
                    def freeze(mod):
                        for p in mod.parameters():
                            p.requires_grad = False

                    print(f'Freezing {name}')
                    module.apply(freeze)
                    self._ignore_modules.add(name)
                    if name in self._previous_acts:
                        self._previous_acts.pop(name)
                    if name in self._current_acts:
                        self._current_acts.pop(name)


class SVCCASubscriber(BatchedSVCCASubscriber):

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
