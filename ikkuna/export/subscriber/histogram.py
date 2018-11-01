from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class HistogramSubscriber(PlotSubscriber):

    '''A :class:`~ikkuna.export.subscriber.Subscriber` which subsamples training artifacts and
    computes histograms per epoch.  Histograms are non-normalized.
    '''

    def __init__(self, kind, message_bus=get_default_bus(), tag=None, subsample=1, backend='tb'):

        if not isinstance(kind, str):
            raise ValueError('HistogramSubscriber only accepts 1 kind')

        subscription = Subscription(self, [kind], tag, subsample)
        title        = f'{kind} histogram'
        ylabel       = 'Frequency'
        super().__init__(subscription, message_bus, {'title': title, 'ylabel': ylabel},
                         backend=backend)

    def compute(self, message_bundle):
        '''
        .. note::
            Since the histogram is computed by the visualization backend (there's no practical way
            around it), this subscriber does *not* publish a
            :class:`~ikkuna.export.messages.SubscriberMessage`
        '''

        module, name = message_bundle.key
        data         = message_bundle.data[self._subscription.kinds[0]]
        self._backend.add_histogram(name, data, message_bundle.global_step)
