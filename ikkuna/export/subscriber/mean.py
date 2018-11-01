from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class MeanSubscriber(PlotSubscriber):

    def __init__(self, kind, message_bus=get_default_bus(), tag=None, subsample=1, ylims=None,
                 backend='tb', **tbx_params):

        if not isinstance(kind, str):
            raise ValueError('MeanSubscriber only accepts 1 kind')

        title        = f'{kind} mean'
        ylabel       = 'Mean'
        xlabel       = 'Train step'
        subscription = Subscription(self, [kind], tag, subsample)
        super().__init__(subscription, message_bus, {'title': title,
                                                     'ylabel': ylabel,
                                                     'ylims': ylims,
                                                     'xlabel': xlabel},
                         backend=backend, **tbx_params)
        self._add_publication(f'{kind}_mean', type='DATA')

    def compute(self, message_bundle):
        '''Compute the average of a quantity. A :class:`~ikkuna.export.messages.ModuleMessage`
        with the identifier ``{kind}_mean`` is published. '''

        module, module_name  = message_bundle.key

        data = message_bundle.data[self._subscription.kinds[0]]
        mean = data.mean()
        self._backend.add_data(module_name, mean, message_bundle.global_step)

        kind = f'{self._subscription.kinds[0]}_mean'
        self.message_bus.publish_module_message(message_bundle.global_step,
                                                message_bundle.train_step,
                                                message_bundle.epoch, kind,
                                                message_bundle.key, mean)
