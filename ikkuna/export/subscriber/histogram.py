from ikkuna.export.subscriber import PlotSubscriber, SynchronizedSubscription
from collections import defaultdict


class HistogramSubscriber(PlotSubscriber):

    '''A :class:`Subscriber` which subsamples training artifacts and computes histograms per epoch.
    Histograms are non-normalized.

    Attributes
    ----------
    _buffer :   dict(list)
                Per-module buffer of values to compute more robust histogram
    '''

    def __init__(self, kinds, tag=None, subsample=1, backend='tb'):
        subscription = SynchronizedSubscription(self, tag)
        title = f'{kinds[0]} histogram'
        ylabel = 'Frequency'
        super().__init__(kinds, subscription, {'title': title, 'ylabel': ylabel}, tag=tag,
                         subsample=1, backend=backend)
        self._buffer = defaultdict(list)

    def _metric(self, module_data):

        module = module_data._module
        data = module_data.data[self.kinds[0]]
        self._backend.add_histogram(module, data, module_data.step)

    def epoch_finished(self, epoch):
        super().epoch_finished(epoch)
        # modules    = list(self._hist.keys())
        # histograms = list(self._hist.values())
        # n_modules  = len(modules)
        # h, w       = (int(np.floor(np.sqrt(n_modules))), int(np.ceil(np.sqrt(n_modules))))

        # figure, axarr = plt.subplots(h, w)
        # figure.suptitle(f'Gradient Histograms for epoch {epoch}')

        # for i in range(h):
        #     for j in range(w):
        #         index = h * i + j

        #         ax = axarr[i][j]
        #         ax.clear()

        #         ax.set_title(str(modules[index]))
        #         ax.set_yscale('log')
        #         ax.plot(self._bin_edges, histograms[index], linewidth=1)
        #         ax.grid(True)
        #         ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

        # figure.tight_layout()
        # figure.subplots_adjust(hspace=1, wspace=1)
        # figure.show()
        # self._clear()
