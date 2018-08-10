from ikkuna.export.subscriber import PlotSubscriber, Subscription
from torch.utils.data import DataLoader


class AccuracySubscriber(PlotSubscriber):
    '''
    Subscriber which can compute the accuracy on a given data set (generally the test/validation
    set).

    Attributes
    ----------
    _dataset_meta    :   train.DatasetMeta
                         Dataset to compute accuracy over
    _data_loader    :   torch.utils.data.DataLoader
                        Loader on the dataset
    _frequency  :   int
                    Number of batches to ignore before computing accuracy once
    _forward_fn :   function
                    Bound method on the model to push data through and get predictions
    '''

    def __init__(self, dataset_meta, forward_fn, frequency=100, ylims=None, tag=None, subsample=1,
                 backend='tb'):
        '''
        Parameters
        ----------
        dataset_meta    :   train.DatasetMeta
                            Test dataset
        forward_fn  :   function
                        Bound version of :meth:`torch.nn.Module.forward()`. This could be obtained
                        from the :attr:`train.Trainer.model` property.
        frequency   :   int
                        Inverse of the frequency with which to compute the accuracy
        '''
        kinds  = ['batch_finished']
        title  = f'Test accuracy'
        ylabel = 'Accuracy'
        xlabel = 'Train step'
        subscription = Subscription(self, tag=tag)
        super().__init__(kinds, subscription, {'title': title, 'ylabel': ylabel, 'ylims': ylims,
                                               xlabel: xlabel},
                         tag=tag, subsample=subsample, backend=backend)

        self._dataset_meta = dataset_meta
        self._data_loader  = DataLoader(dataset_meta.dataset, batch_size=dataset_meta.size,
                                        shuffle=False, pin_memory=True)
        self._frequency    = frequency
        self._forward_fn   = forward_fn

    def _metric(self, module_data):
        if self._counter[None] % self._frequency == 0:
            X, labels   = next(iter(self._data_loader))
            outputs     = self._forward_fn(X.cuda(), should_train=False)
            predictions = outputs.argmax(1)
            n_correct   = (predictions.cpu() == labels).sum().item()
            accuracy    = n_correct / self._dataset_meta.size
            self._backend.add_data('test accuracy', accuracy, module_data.step)
