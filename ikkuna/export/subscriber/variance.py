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

        module_name  = message_bundle.identifier

        data = message_bundle.data[self._subscription.kinds[0]]
        var = data.var()
        self._backend.add_data(module_name, var, message_bundle.seq)
