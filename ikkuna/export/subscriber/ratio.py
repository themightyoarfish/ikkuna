import torch
import numpy as np

from ikkuna.export.subscriber import PlotSubscriber, SynchronizedSubscription

ZERO_TENSOR = torch.tensor(0.0).cuda()


class RatioSubscriber(PlotSubscriber):

    '''A :class:`Subscriber` which computes the average ratio between two quantities.  The dividend
    will be the first element of the :attr:`Subscriber.kinds` property, the divisor the second.
    Therefore it is vital to pass the message kinds to the
    :class:`ikkuna.export.subscriber.Subscription` object in the correct order.'''

    def __init__(self, kinds, tag=None, subsample=1, ylims=None, backend='tb', absolute=True):
        '''
        Parameters
        ----------
        absolute :  bool
                    Whether to use absolute ratio
        '''
        title = f'{kinds[0]}/{kinds[1]} ratios per layer'
        ylabel = 'Ratio'
        xlabel = 'Train step'
        subscription = SynchronizedSubscription(self, tag)
        super().__init__(kinds, subscription, {'title': title, 'ylabel': ylabel, 'ylims': ylims,
                                               xlabel: xlabel},
                         tag=tag, subsample=subsample,
                         backend=backend)
        if absolute:
            self._metric_postprocess = torch.abs
        else:
            self._metric_postprocess = lambda x: x

    def _metric(self, module_data):
        '''The ratio between the two kinds is computed over the subset of not-NaN values and added
        to the record.'''

        module  = module_data.module.name

        dividend = module_data._data[self.kinds[0]]
        divisor  = module_data._data[self.kinds[1]]

        ######################################################################################
        #  We need to see how many NaNs we have and compute the mean only over the non-nans  #
        ######################################################################################
        ratio_tensor = self._metric_postprocess(dividend.div(divisor))
        n            = float(divisor.numel())
        nan_tensor   = torch.isnan(ratio_tensor)
        n_nans       = nan_tensor.sum().to(torch.float32)
        if n_nans > 0:
            ratio_sum = torch.where(1 - nan_tensor, ratio_tensor, ZERO_TENSOR).sum()
        else:
            ratio_sum = ratio_tensor.sum()
        ratio = (ratio_sum / (n - n_nans)).item()

        if np.isnan(ratio):
            raise ValueError(f'NaN value ratio for {module}')

        self._backend.add_data(module, ratio, module_data.step)
