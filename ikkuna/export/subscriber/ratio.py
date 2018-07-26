import torch
import numpy as np

from ikkuna.export.subscriber import LinePlotSubscriber, SynchronizedSubscription

ZERO_TENSOR = torch.tensor(0.0).cuda()


class RatioSubscriber(LinePlotSubscriber):

    '''A :class:`Subscriber` which computes the average ratio between two quantities.  The dividend
    will be the first element of th :attr:`Subscriber.kinds` property, the divisor the second.
    Therefore it is vital to pass the message kinds to the
    :class:`ikkuna.export.subsciber.Subscription` object in the correct order.'''

    def __init__(self, kinds, tag=None, subsample=1, average=1, ylims=None, absolute=True):
        '''
        Parameters
        ----------
        absolute :  bool
                    Whether to use absolute ratio
        '''
        super().__init__(kinds, tag=tag, subsample=subsample, average=average, ylims=ylims)
        self._subscription      = SynchronizedSubscription(self, tag)
        if absolute:
            self._fn = torch.abs
        else:
            self._fn = lambda x: x

        self._ax.set_title(f'{self.kinds[0]}/{self.kinds[1]} ratios per layer '
                           f'(average of {self._average} batches)')
        self._ax.set_xlabel('Ratio')

    def _metric(self, module_data):
        '''The ratio between the two kinds is computed over the subset of not-NaN values and added
        to the record.'''

        module  = module_data.module.name

        dividend = module_data._data[self.kinds[0]]
        divisor  = module_data._data[self.kinds[1]]
        numel    = divisor.numel()

        ######################################################################################
        #  We need to see how many NaNs we have and compute the mean only over the non-nans  #
        ######################################################################################
        ratio_tensor = self._fn(dividend.div(divisor))
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

        ratios = self._metric_values[module]
        ratios.append(ratio)

        # every self._average calls, we replace the last self._average elements with their mean
        # possibly a running average would be more efficient, but who's counting
        n_past_ratios = self._counter[module] + 1
        if n_past_ratios % self._average == 0:
            ratios[-self._average:] = (np.mean(ratios[-self._average:]),)
