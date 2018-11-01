from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class NormSubscriber(PlotSubscriber):

    def __init__(self, kind, message_bus=get_default_bus(), tag=None, subsample=1, ylims=None,
                 backend='tb', order=2, **tbx_params):

        if not isinstance(kind, str):
            raise ValueError('NormSubscriber only accepts 1 kind')

        title        = f'{kind} norm'
        ylabel       = f'L{order} Norm'
        xlabel       = 'Train step'
        subscription = Subscription(self, [kind], tag, subsample)
        super().__init__(subscription, message_bus, {'title': title,
                                                     'ylabel': ylabel,
                                                     'ylims': ylims,
                                                     'xlabel': xlabel},
                         backend=backend, **tbx_params)
        self._order  = order

    def compute(self, message_bundle):
        '''Compute the norm of a quantity. A :class:`~ikkuna.export.messages.SubscriberMessage`
        with the identifier ``{kind}_norm{order}`` is published. '''

        module, module_name  = message_bundle.key

        data = message_bundle.data[self._subscription.kinds[0]]
        norm = data.norm(p=self._order)
        self._backend.add_data(module_name, norm, message_bundle.global_step)

        kind = f'{self._subscription.kinds[0]}_norm{self._order}'
        self.message_bus.publish_subscriber_message(message_bundle.global_step,
                                                    message_bundle.train_step,
                                                    message_bundle.epoch, kind,
                                                    message_bundle.key, norm)
