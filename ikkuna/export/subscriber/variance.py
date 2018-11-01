from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class VarianceSubscriber(PlotSubscriber):
    '''A :class:`~ikkuna.export.subscriber.Subscriber` which computes the
    variance of quantity for the current batch.
    '''

    def __init__(self, kind, message_bus=get_default_bus(), tag=None, subsample=1, ylims=None,
                 backend='tb', **tbx_params):

        if not isinstance(kind, str):
            raise ValueError('VarianceSubscriber only accepts 1 kind')

        title        = f'variance of {kind}'
        ylabel       = 'σ^2'
        xlabel       = 'Train step'
        subscription = Subscription(self, [kind], tag, subsample)
        super().__init__(subscription, message_bus,
                         {'title': title,
                          'ylabel': ylabel,
                          'ylims': ylims,
                          'xlabel': xlabel},
                         backend=backend, **tbx_params)

    def compute(self, message_bundle):
        '''Compute the variance of a quantity. A :class:`~ikkuna.export.messages.ModuleMessage`
        with the identifier ``{kind}_variance`` is published. '''

        module, module_name = message_bundle.key
        data                = message_bundle.data[self._subscription.kinds[0]]
        var                 = data.var()
        self._backend.add_data(module_name, var, message_bundle.global_step)

        kind = f'{self._subscription.kinds[0]}_variance'
        self.message_bus.publish_module_message(message_bundle.global_step,
                                                message_bundle.train_step,
                                                message_bundle.epoch, kind,
                                                message_bundle.key, var)
