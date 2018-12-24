from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class LossSubscriber(PlotSubscriber):
    '''A Subscriber that plots the training loss at every time step.'''

    def __init__(self, message_bus=get_default_bus(), subsample=1, ylims=None,
                 backend='tb'):

        title        = 'Loss'
        ylabel       = 'loss'
        xlabel       = 'Train step'
        subscription = Subscription(self, ['loss'], None, subsample)
        super().__init__([subscription], message_bus, {'title': title,
                                                       'ylabel': ylabel,
                                                       'ylims': ylims,
                                                       'xlabel': xlabel},
                         backend=backend)

    def compute(self, message):
        data = message.data
        self._backend.add_data('loss', data, message.global_step)
