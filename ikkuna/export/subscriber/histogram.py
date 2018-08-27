from ikkuna.export.subscriber import PlotSubscriber, SynchronizedSubscription
from collections import defaultdict


class HistogramSubscriber(PlotSubscriber):

    '''A :class:`~ikkuna.export.subscriber.Subscriber` which subsamples training artifacts and
    computes histograms per epoch.  Histograms are non-normalized.

    Attributes
    ----------
    _buffer :   dict(list)
                Per-module buffer of values to compute more robust histogram
    '''

    def __init__(self, kinds, tag=None, subsample=1, backend='tb'):
        subscription = SynchronizedSubscription(self, kinds, tag, subsample)
        title        = f'{kinds[0]} histogram'
        ylabel       = 'Frequency'
        super().__init__(subscription, {'title': title, 'ylabel': ylabel}, backend=backend)
        self._buffer = defaultdict(list)

    def _metric(self, message_bundle):

        module_name = message_bundle.identifier
        data   = message_bundle.data[self._subscription.kinds[0]]
        self._backend.add_histogram(module_name, data, message_bundle.seq)
