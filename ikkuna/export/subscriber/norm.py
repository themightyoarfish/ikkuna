from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class NormSubscriber(PlotSubscriber):

    def __init__(self, kind, message_bus=get_default_bus(), tag=None, subsample=1, ylims=None,
                 backend='tb', order=2, **tbx_params):

        if not isinstance(kind, str):
            raise ValueError('NormSubscriber only accepts 1 kind')

        title        = f'{kind}_norm{order}'
        ylabel       = f'L{order} Norm'
        xlabel       = 'Train step'
        subscription = Subscription(self, [kind], tag, subsample)
        super().__init__([subscription], message_bus, {'title': title,
                                                     'ylabel': ylabel,
                                                     'ylims': ylims,
                                                     'xlabel': xlabel},
                         backend=backend, **tbx_params)
        self._order  = order
        self._add_publication(f'{kind}_norm{order}', type='DATA')

    def compute(self, message):
        '''Compute the norm of a quantity. A :class:`~ikkuna.export.messages.ModuleMessage`
        with the identifier ``{kind}_norm{order}`` is published. '''

        module, module_name  = message.key

        data = message.data
        norm = data.norm(p=self._order)
        self._backend.add_data(module_name, norm, message.global_step)

        kind = f'{message.kind}_norm{self._order}'
        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, kind,
                                                message.key, norm)
