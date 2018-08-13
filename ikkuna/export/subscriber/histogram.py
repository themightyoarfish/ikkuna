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
        super().__init__(subscription, {'title': title, 'ylabel': ylabel}, tag=tag, backend=backend)
        self._buffer = defaultdict(list)

    def _metric(self, module_data):

        module = module_data._module
        data   = module_data.data[self._subscription.kinds[0]]
        self._backend.add_histogram(module, data, module_data.seq)
