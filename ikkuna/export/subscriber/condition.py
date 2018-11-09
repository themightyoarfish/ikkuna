import torch
from torch.nn.functional import normalize

from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class ConditionNumberSubscriber(PlotSubscriber):

    def __init__(self, kind, message_bus=get_default_bus(), tag=None, subsample=1, ylims=None,
                 backend='tb', **tbx_params):
        '''
        Parameters
        ----------
        kind    :   str
                    Message kind to compute condition number norm on. Not sure if it makes sense
                    for non-2d matrices, which have to be reshaped to 2-d
                    non-matrix type.


        For other parameters, see :class:`~ikkuna.export.subscriber.PlotSubscriber`
        '''

        if not isinstance(kind, str):
            raise ValueError(f'{self.__class__.__name__} only accepts 1 kind')

        subscription = Subscription(self, [kind], tag, subsample)

        title = f'{kind}_condition_number'
        xlabel = 'Step'
        ylabel = 'Condition number'
        super().__init__([subscription], message_bus,
                         {'title': title, 'xlabel': xlabel, 'ylims': ylims, 'ylabel': ylabel},
                         backend=backend, **tbx_params)
        self._add_publication(f'{kind}_condition_number', type='DATA')
        self.u = dict()
        self.u_inv = dict()

    def compute(self, message):
        '''A :class:`~ikkuna.export.messages.ModuleMessage`
        with the identifier ``{kind}_condition_number`` is published. '''

        module, module_name = message.key
        # get and reshape the weight tensor to 2d
        weights             = message.data
        height              = weights.size(0)
        weights2d           = weights.reshape(height, -1)
        width               = weights2d.size(1)
        weights2d_inv       = torch.pinverse(weights2d)

        # buffer for power iteration (don't know what the mahematical purpose is)
        if module_name not in self.u:
            self.u[module_name] = normalize(weights2d.new_empty(height).normal_(0, 1), dim=0)
            self.u_inv[module_name] = normalize(weights2d_inv.new_empty(width).normal_(0, 1), dim=0)

        # estimate singular values
        with torch.no_grad():
            for _ in range(3):      # TODO: Make niter parameter
                v = normalize(torch.matmul(weights2d.t(), self.u[module_name]), dim=0)
                self.u[module_name] = normalize(torch.matmul(weights2d, v), dim=0)

                v_inv = normalize(torch.matmul(weights2d_inv.t(), self.u_inv[module_name]), dim=0)
                self.u_inv[module_name] = normalize(torch.matmul(weights2d_inv, v_inv), dim=0)

        norm = torch.dot(self.u[module_name], torch.matmul(weights2d, v)).item()
        norm_inv = torch.dot(self.u_inv[module_name], torch.matmul(weights2d_inv, v_inv)).item()

        condition = norm / norm_inv

        self._backend.add_data(module_name, condition, message.global_step)

        kind = f'{message.kind}_condition_number'
        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, kind,
                                                message.key, condition)
