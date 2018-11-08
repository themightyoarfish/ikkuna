from ikkuna.export.subscriber import PlotSubscriber, Subscription, Subscriber
from ikkuna.export.messages import get_default_bus
from collections import defaultdict


class RatioLRSubscriber(PlotSubscriber):
    '''A subscriber that computes learning rate mutliplicatives simply based on the average
    update/weight ratio of the previous batches. It uses exponential smoothing on the ratios and
    averages them over all layers. It outputs a factor to get the ratio of change towards a target
    (default is 1e-3).'''

    def __init__(self, base_lr, smoothing=0.9, target=1e-3, max_factor=500,
                 ylims=None, backend='tb', **tbx_params):
        subscription = Subscription(self, ['weight_updates_weights_ratio', 'batch_started'],
                                    tag=None, subsample=1)
        super().__init__([subscription], get_default_bus(),
                         {'title': 'learning_rate',
                          'ylabel': 'Adjusted learning rate',
                          'ylims': ylims,
                          'xlabel': 'Train step'}, backend=backend, **tbx_params)
        self._ratios     = defaultdict(float)
        self._max_factor = max_factor
        self._smoothing  = smoothing
        self._target     = target
        self._factor     = 1
        self._base_lr    = base_lr
        self._add_publication('learning_rate', type='META')

    def _compute_lr_multiplier(self):
        '''Compute learning rate multiplicative. Will output 1 for the first batch since no layer
        ratios have been recorded yet. Will also output 1 if the average ratio is close to 0.
        Will clip the factor to some max limit'''
        n_layers = len(self._ratios)
        if n_layers == 0:   # before first batch
            return 1
        else:
            mean_ratio = sum(ratio for ratio in self._ratios.values()) / n_layers
            if mean_ratio <= 1e-9:
                return 1
            else:
                factor = self._target / mean_ratio
                if factor > self._max_factor:
                    return self._max_factor
                else:
                    return factor

    def compute(self, message):
        if message.kind == 'weight_updates_weights_ratio':
            self._ratios[message.key] = (self._smoothing * message.data
                                         + (1 - self._smoothing) * self._ratios[message.key])
        else:
            self._factor = self._compute_lr_multiplier()
            self._backend.add_data('learning_rate', self._base_lr * self._factor,
                                   message.global_step)
            self.message_bus.publish_network_message(message.global_step,
                                                     message.train_step,
                                                     message.epoch, 'learning_rate',
                                                     self._factor * self._base_lr)

    def __call__(self, epoch):
        return self._factor


class SacredLoggingSubscriber(Subscriber):
    '''Subscriber which logs its subscribed values with sacred's metrics API'''

    def __init__(self, experiment, kinds):
        self._experiment = experiment
        subscription = Subscription(self, kinds)
        super().__init__([subscription], get_default_bus())

    def compute(self, message):
        if message.key != 'META':
            raise ValueError('Logging `ModuleMessage`s doesn\'t make sense')
        self._experiment.log_scalar(message.kind, message.data, message.global_step)
