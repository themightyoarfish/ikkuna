import torch
from torch.nn.functional import normalize

from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class SpectralNormSubscriber(PlotSubscriber):

    def __init__(self, kind, message_bus=get_default_bus(), tag=None, subsample=1, ylims=None,
                 backend='tb', **tbx_params):
        '''
        Parameters
        ----------
        kind    :   str
                    Message kind to compute spectral norm on. Doesn't make sense with kinds of
                    non-matrix type.


        For other parameters, see :class:`~ikkuna.export.subscriber.PlotSubscriber`
        '''

        if not isinstance(kind, str):
            raise ValueError('SpectralNormSubscriber only accepts 1 kind')

        subscription = Subscription(self, [kind], tag, subsample)

        title = f'Spectral norms of {kind}'
        xlabel = 'Step'
        ylabel = 'Spectral norm'
        super().__init__(subscription, message_bus,
                         {'title': title, 'xlabel': xlabel, 'ylims': ylims, 'ylabel': ylabel},
                         backend=backend, **tbx_params)
        self.u = dict()

    def compute(self, message_bundle):
        '''The spectral norm computation is taken from the `Pytorch implementation of spectral norm
        <https://pytorch.org/docs/master/_modules/torch/nn/utils/spectral_norm.html>`_. It's
        possible to use SVD instead, but we are not interested in the full matrix decomposition,
        merely in the singular values.

        A :class:`~ikkuna.export.messages.SubscriberMessage`
        with the identifier ``{kind}_spectral_norm`` is published. '''

        module_name = message_bundle.identifier
        # get and reshape the weight tensor to 2d
        weights     = message_bundle.data[self._subscription.kinds[0]]
        height      = weights.size(0)
        weights2d   = weights.reshape(height, -1)

        # buffer for power iteration (don't know what the mahematical purpose is)
        if module_name not in self.u:
            self.u[module_name] = normalize(weights2d.new_empty(height).normal_(0, 1), dim=0)

        # estimate singular values
        with torch.no_grad():
            for _ in range(3):      # TODO: Make niter parameter
                v = normalize(torch.matmul(weights2d.t(), self.u[module_name]), dim=0)
                self.u[module_name] = normalize(torch.matmul(weights2d, v), dim=0)

        norm = torch.dot(self.u[module_name], torch.matmul(weights2d, v)).item()

        self._backend.add_data(module_name, norm, message_bundle.global_step)

        kind = f'{self._subscription.kinds[0]}_spectral_norm'
        self.message_bus.publish_subscriber_message(message_bundle.global_step,
                                                    message_bundle.train_step,
                                                    message_bundle.epoch, kind,
                                                    message_bundle.identifier, norm)
