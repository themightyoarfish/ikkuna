from ikkuna.export.subscriber import PlotSubscriber, Subscription
from torch.utils.data import DataLoader


class AccuracySubscriber(PlotSubscriber):

    def __init__(self, dataset_meta, forward_fn, frequency=100, ylims=None, tag=None, subsample=1,
                 backend='tb'):
        kinds = ['batch_finished']
        title = f'Train/test accuracy'
        ylabel = 'Accuracy'
        xlabel = 'Train step'
        subscription = Subscription(self, tag=tag)
        super().__init__(kinds, subscription, {'title': title, 'ylabel': ylabel, 'ylims': ylims,
                                               xlabel: xlabel},
                         tag=tag, subsample=subsample, backend=backend)
        self._dataset = dataset_meta
        self._data_loader = DataLoader(dataset_meta.dataset, batch_size=dataset_meta.size,
                                       shuffle=False, pin_memory=True)
        self._frequency = frequency
        self._forward_fn = forward_fn

    def _metric(self, module_data):
        if self._counter[None] % self._frequency == 0:
            X, labels   = next(iter(self._data_loader))
            outputs     = self._forward_fn(X.cuda(), should_train=False)
            predictions = outputs.argmax(1)
            n_correct   = (predictions.cpu() == labels).sum().item()
            accuracy    = n_correct / self._dataset.size
            self._backend.add_data('accuracy', accuracy, module_data.step)
