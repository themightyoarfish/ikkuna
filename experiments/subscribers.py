from ikkuna.export.subscriber import PlotSubscriber, Subscription, Subscriber
from ikkuna.export.messages import get_default_bus
from collections import defaultdict


class RatioLRSubscriber(PlotSubscriber):
    '''A subscriber that computes learning rate mutliplicatives simply based on the average
    update/weight ratio of the previous batches. It uses exponential smoothing on the ratios and
    averages them over all layers. It outputs a factor to get the ratio of change towards a target
    (default is 1e-3).'''

    def __init__(self, base_lr, smoothing=0.9, target=1e-3, ylims=None, backend='tb'):
        subscription = Subscription(self, ['weight_updates_weights_ratio', 'batch_started'],
                                    tag=None, subsample=1)
        super().__init__([subscription], get_default_bus(),
                         {'title': 'learning_rate',
                          'ylabel': 'Adjusted learning rate',
                          'ylims': ylims,
                          'xlabel': 'Train step'}, backend=backend)
        self._ratios     = defaultdict(float)
        self._smoothing  = smoothing
        self._target     = target
        self._factor     = 1
        self._base_lr    = base_lr
        self._add_publication('learning_rate', type='META')

    def _compute_lr_multiplier(self):
        '''Compute learning rate multiplicative. Will output 1 for the first batch since no layer
        ratios have been recorded yet. Will also output 1 if the average ratio is close to 0.'''
        n_layers = len(self._ratios)
        if n_layers == 0:   # before first batch
            return 1
        else:
            mean_ratio = sum(ratio for ratio in self._ratios.values()) / n_layers
            if mean_ratio <= 1e-9:
                return 1
            else:
                return self._target / mean_ratio

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
            name = f'{message.kind}.{message.key.name}'
        else:
            name = message.kind
        self._experiment.log_scalar(name, float(message.data), message.global_step)


class ExponentialRatioLRSubscriber(RatioLRSubscriber):
    '''A subscriber that computes learning rate mutliplicatives based on an exponential decay and
    the average update/weight ratio of the previous batches. It uses exponential smoothing on the
    ratios and averages them over all layers. It outputs a factor to get the ratio of change towards
    a target (default is 1e-3).'''

    def __init__(self, base_lr, decay=0.98, smoothing=0.9, target=1e-3, max_factor=500,
                 ylims=None, backend='tb'):
        super().__init__(base_lr, smoothing, target, max_factor, ylims, backend)
        self._decay = decay

    def compute(self, message):
        if message.kind == 'weight_updates_weights_ratio':
            self._ratios[message.key] = (self._smoothing * message.data
                                         + (1 - self._smoothing) * self._ratios[message.key])
        else:
            self._factor = 0.5 * self._compute_lr_multiplier() + 0.5 * self._decay ** message.epoch
            self._backend.add_data('learning_rate', self._base_lr * self._factor,
                                   message.global_step)
            self.message_bus.publish_network_message(message.global_step,
                                                     message.train_step,
                                                     message.epoch, 'learning_rate',
                                                     self._factor * self._base_lr)

    def __call__(self, epoch):
        return self._factor
