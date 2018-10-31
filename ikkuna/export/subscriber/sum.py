from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class SumSubscriber(PlotSubscriber):

    def __init__(self, kind, message_bus=get_default_bus(), tag=None, subsample=1, ylims=None,
                 backend='tb', **tbx_params):
        if not isinstance(kind, str):
            raise ValueError('SumSubscriber only accepts 1 kind')
        title        = f'{kinds[0]} sum'
        ylabel       = 'Sum'
        xlabel       = 'Train step'
        subscription = Subscription(self, [kind], tag, subsample)
        super().__init__(subscription, message_bus,
                         {'title': title,
                          'ylabel': ylabel,
                          'ylims': ylims,
                          'xlabel': xlabel},
                         backend=backend, **tbx_params)

    def compute(self, message_bundle):
        '''Compute the sum of a quantity. A :class:`~ikkuna.export.messages.SubscriberMessage`
        with the identifier ``{kind}_sum`` is published. '''

        module_name  = message_bundle.identifier

        data = message_bundle.data[self._subscription.kinds[0]]
        sum = data.sum()
        self._backend.add_data(module_name, sum, message_bundle.seq)

        kind = f'{self._subscription.kinds[0]}_sum'
        self.message_bus.publish_subscriber_message(message_bundle.seq, message_bundle.step,
                                                    message_bundle.epoch, kind,
                                                    message_bundle.identifier, sum)
