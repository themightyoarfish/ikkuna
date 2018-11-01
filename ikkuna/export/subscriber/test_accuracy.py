from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus
from torch.utils.data import DataLoader


class TestAccuracySubscriber(PlotSubscriber):
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

    def __init__(self, dataset_meta, forward_fn, batch_size, message_bus=get_default_bus(),
                 frequency=100, ylims=None, tag=None, subsample=1, backend='tb', **tbx_params):
        '''
        Parameters
        ----------
        dataset_meta    :   train.DatasetMeta
                            Test dataset
        forward_fn  :   function
                        Bound version of :meth:`torch.nn.Module.forward()`. This could be obtained
                        from the :attr:`train.Trainer.model` property.
        batch_size  :   int
                        Batch size to use for pushing the test data through the model. This doesn't
                        have to be the training batch size, but must be selected so that the forward
                        pass fits into memory.
        frequency   :   int
                        Inverse of the frequency with which to compute the accuracy
        '''
        kinds  = ['batch_finished']
        title  = f'Test accuracy'
        ylabel = 'Accuracy'
        xlabel = 'Train step'
        subscription = Subscription(self, kinds, tag, subsample)
        super().__init__(subscription,
                         message_bus,
                         {'title': title, 'ylabel': ylabel, 'ylims': ylims, 'xlabel': xlabel},
                         backend=backend, **tbx_params)

        self._dataset_meta = dataset_meta
        self._data_loader  = DataLoader(dataset_meta.dataset, batch_size=batch_size, shuffle=False,
                                        pin_memory=True)
        self._frequency    = frequency
        self._forward_fn   = forward_fn

    def compute(self, message_or_data):
        '''Compute accuracy over the entire test set.

        A :class:`~ikkuna.export.messages.SubscriberMessage` with the identifier
        ``test_accuracy`` is published. '''
        if self._subscription.counter['batch_finished'] % self._frequency == 0:
            n_batches = 0
            accuracy  = 0
            loader    = iter(self._data_loader)
            try:
                while True:
                    X, labels   = next(loader)
                    outputs     = self._forward_fn(X.cuda(), should_train=False)
                    predictions = outputs.argmax(1)
                    n_correct   = (predictions.cpu() == labels).sum().item()
                    accuracy   += n_correct / X.shape[0]
                    n_batches  += 1
            except StopIteration:
                accuracy /= n_batches
                self._backend.add_data('test accuracy', accuracy, message_or_data.global_step)

                kind = 'test_accuracy'
                self.message_bus.publish_subscriber_message(message_or_data.global_step,
                                                            message_or_data.train_step,
                                                            message_or_data.epoch, kind,
                                                            message_or_data.key, accuracy)
