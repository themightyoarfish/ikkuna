from ikkuna.export.subscriber import Subscriber, SynchronizedSubscription
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict


class HistogramSubscriber(Subscriber):

    '''A :class:`Subscriber` which subsamples training artifacts and computes histograms per epoch.
    Histograms are non-normalized.

    Attributes
    ----------
    _clip_max   :   float
                    Maximum bin edge in the histogram
    _clip_min   :   float
                    Minimum bin edge in the histogram
    _bin_edges  :   np.ndarray
                    Edges of the histogram bins computed from `_clip_max`, `_clip_min`, and the step
                    size
    _buffer_size    :   int
                        number of artifacts to buffer before computing a histogram. This makes sense
                        only in the context :class:`SynchronizedSubscription` (i think)
    _buffer :   list
                List for keeping indirect references to gpu tensors
    _update_counter :   int
                        Counter for subsampling the calls made to this :class:`Subscriber`
    _hist  :   dict
                        Per-module cumulative histogram
    '''

    def __init__(self, kinds, clip_min, clip_max, step, buffer_size=10, tag=None):
        '''
        Parameters
        ----------
        clip_min    :   float
                        Lower limit
        clip_min    :   float
                        Upper limit
        step    :   float
                    Step between histogram bins
        buffer_size :   int
                        Number of calls to :meth:`HistogramSubscriber.__call__()` to buffer

        Raises
        ------
        ValueError
            If `clip_max` is not greater than `clip_min`
        '''
        super().__init__(kinds, tag=tag)
        if clip_max <= clip_min:
            raise ValueError(f'`clip_min` must be smaller than'
                             ' `clip_max` (was {clip_min} and {clip_max})')
        n_bins = int((clip_max - clip_min) // step)
        assert n_bins > 0, 'Bin number must be strictly positive.'

        self._subscription   = SynchronizedSubscription(self, tag)
        self._bin_edges      = np.linspace(clip_min, clip_max, num=n_bins)
        self._hist           = defaultdict(lambda: np.zeros(n_bins, dtype=np.int64))
        self._nbins          = n_bins
        self._clip_max       = clip_max
        self._clip_min       = clip_min
        self._buffer_size    = buffer_size
        self._buffer         = defaultdict(list)
        self._update_counter = defaultdict(int)

    def update_histogram(self, module):
        '''Update the histogram of a module from the current buffer.

        Parameters
        ----------
        module  :   str
                    Layer abel for which the histogram should be updated from the buffer
        '''
        for module_data in self._buffer[module]:
            module                        = module_data._module
            kind                          = self.kinds[0]
            data                          = module_data._data[kind].cpu()
            hist                          = data.histc(self._nbins, self._clip_min, self._clip_max)
            if hist.requires_grad:
                hist = hist.detach()
            self._hist[module]           += hist.numpy().astype(np.int64)

    def _clear(self):
        self._buffer.clear()
        self._hist.clear()

    def _metric(self, module_data):

        module = module_data._module
        if (self._update_counter[module] + 1) % self._buffer_size == 0:
            self.update_histogram(module)
            self._buffer[module] = []
        self._buffer[module].append(module_data)
        self._update_counter[module] += 1

    def epoch_finished(self, epoch):
        super().epoch_finished(epoch)
        modules    = list(self._hist.keys())
        histograms = list(self._hist.values())
        n_modules  = len(modules)
        h, w       = (int(np.floor(np.sqrt(n_modules))), int(np.ceil(np.sqrt(n_modules))))

        figure, axarr = plt.subplots(h, w)
        figure.suptitle(f'Gradient Histograms for epoch {epoch}')

        for i in range(h):
            for j in range(w):
                index = h*i+j

                ax = axarr[i][j]
                ax.clear()

                ax.set_title(str(modules[index]))
                ax.set_yscale('log')
                ax.plot(self._bin_edges, histograms[index], linewidth=1)
                ax.grid(True)

        figure.tight_layout()
        figure.subplots_adjust(hspace=0.5, wspace=0.5)
        figure.show()
        self._clear()
