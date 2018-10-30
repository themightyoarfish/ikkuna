from ikkuna.export.subscriber import PlotSubscriber, SynchronizedSubscription
from ikkuna.export.messages import get_default_bus


class SumSubscriber(PlotSubscriber):

    def __init__(self, kinds, message_bus=get_default_bus(), tag=None, subsample=1, ylims=None,
                 backend='tb', **tbx_params):
        title        = f'{kinds[0]} sum'
        ylabel       = 'Sum'
        xlabel       = 'Train step'
        subscription = SynchronizedSubscription(self, kinds, tag, subsample)
        super().__init__(subscription, message_bus,
                         {'title': title,
                          'ylabel': ylabel,
                          'ylims': ylims,
                          'xlabel': xlabel},
                         backend=backend, **tbx_params)

    def compute(self, message_bundle):

        module_name  = message_bundle.identifier

        data = message_bundle.data[self._subscription.kinds[0]]
        self._backend.add_data(module_name, data.sum(), message_bundle.seq)
