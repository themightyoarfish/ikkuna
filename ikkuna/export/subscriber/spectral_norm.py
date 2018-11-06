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

        title = f'{kind}_spectral_norm'
        xlabel = 'Step'
        ylabel = 'Spectral norm'
        super().__init__([subscription], message_bus,
                         {'title': title, 'xlabel': xlabel, 'ylims': ylims, 'ylabel': ylabel},
                         backend=backend, **tbx_params)
        self.u = dict()
        self._add_publication(f'{kind}_spectral_norm', type='DATA')

    def compute(self, message):
        '''The spectral norm computation is taken from the `Pytorch implementation of spectral norm
        <https://pytorch.org/docs/master/_modules/torch/nn/utils/spectral_norm.html>`_. It's
        possible to use SVD instead, but we are not interested in the full matrix decomposition,
        merely in the singular values.

        A :class:`~ikkuna.export.messages.ModuleMessage`
        with the identifier ``{kind}_spectral_norm`` is published. '''

        module, module_name = message.key
        # get and reshape the weight tensor to 2d
        weights             = message.data
        height              = weights.size(0)
        weights2d           = weights.reshape(height, -1)

        # buffer for power iteration (don't know what the mahematical purpose is)
        if module_name not in self.u:
            self.u[module_name] = normalize(weights2d.new_empty(height).normal_(0, 1), dim=0)

        # estimate singular values
        with torch.no_grad():
            for _ in range(3):      # TODO: Make niter parameter
                v = normalize(torch.matmul(weights2d.t(), self.u[module_name]), dim=0)
                self.u[module_name] = normalize(torch.matmul(weights2d, v), dim=0)

        norm = torch.dot(self.u[module_name], torch.matmul(weights2d, v)).item()

        self._backend.add_data(module_name, norm, message.global_step)

        kind = f'{message.kind}_spectral_norm'
        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, kind,
                                                message.key, norm)
