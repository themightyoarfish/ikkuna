from ikkuna.export.subscriber import PlotSubscriber, SynchronizedSubscription


class TrainAccuracySubscriber(PlotSubscriber):

    def __init__(self, tag=None, subsample=1, ylims=None, backend='tb'):
        '''
        Parameters
        ----------
        see :class:`~ikkuna.export.subscriber.PlotSubscriber`
        '''
        subscription = SynchronizedSubscription(self, ['network_output', 'input_labels'], tag,
                                                subsample)

        title  = f'Train batch accuracy'
        xlabel = 'Step'
        ylabel = 'Accuracy'
        super().__init__(subscription,
                         {'title': title, 'xlabel': xlabel, 'ylims': ylims, 'ylabel': ylabel},
                         tag=tag, backend=backend)

    def _metric(self, message_bundle):
        __import__('ipdb').set_trace()
