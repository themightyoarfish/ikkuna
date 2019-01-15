from collections import defaultdict
from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class MessageMeanSubscriber(PlotSubscriber):
    '''Compute the mean over all messages with scalar data of a given kind in one train step. This
    is useful for displaying summary statistics.'''

    def __init__(self, kind, message_bus=get_default_bus(), tag='default', subsample=1, ylims=None,
                 backend='tb'):

        if not isinstance(kind, str):
            raise ValueError('MessageMeanSubscriber only accepts 1 kind')

        title        = f'message_means'
        ylabel       = 'Mean'
        xlabel       = 'Train step'
        subscription = Subscription(self, [kind, 'batch_finished'], tag=tag, subsample=subsample)
        super().__init__([subscription], message_bus,
                         {'title': title,
                          'ylabel': ylabel,
                          'ylims': ylims,
                          'xlabel': xlabel},
                         backend=backend)
        self._add_publication(f'{kind}_message_mean', type='META')
        # kind -> list
        self._buffer = defaultdict(list)

    def compute(self, message):
        '''A :class:`~ikkuna.export.messages.ModuleMessage` with the identifier
        ``{kind}_message_mean`` is published. '''

        if message.kind == 'batch_finished':
            for kind, values in self._buffer.items():
                mean = sum(values) / len(values)
                topic = f'{kind}_message_mean'
                self._backend.add_data(topic, mean, message.global_step)

                self.message_bus.publish_network_message(message.global_step,
                                                         message.train_step,
                                                         message.epoch, topic, data=mean)
                self._buffer[kind] = []
        else:
            self._buffer[message.kind].append(message.data)
