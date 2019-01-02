import torch
from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class BiasCorrectedMomentsSubscriber(PlotSubscriber):

    def __init__(self, beta1, beta2, eps, message_bus=get_default_bus(), tag=None, subsample=1,
                 ylims=None, backend='tb'):

        title        = f'bias-corrected_running_gradient_moments'
        ylabel       = f'Gradient Mean'
        xlabel       = 'Train step'
        subscription = Subscription(self, ['weight_gradients'], tag, subsample)
        super().__init__([subscription], message_bus, {'title': title,
                                                       'ylabel': ylabel,
                                                       'ylims': ylims,
                                                       'xlabel': xlabel},
                         backend=backend)

        self._beta1 = beta1
        self._beta2 = beta2
        self._eps   = eps
        self._means = dict()
        self._vars  = dict()

        self._add_publication('biased_grad_mean_estimate_mean', type='DATA')
        self._add_publication('biased_grad_mean_estimate_var', type='DATA')
        self._add_publication('biased_grad_var_estimate_mean', type='DATA')
        self._add_publication('biased_grad_var_estimate_var', type='DATA')
        self._add_publication('grad_mean_estimate_mean', type='DATA')
        self._add_publication('grad_mean_estimate_var', type='DATA')
        self._add_publication('grad_var_estimate_mean', type='DATA')
        self._add_publication('grad_var_estimate_var', type='DATA')
        self._add_publication('lr_multiplier_mean', type='DATA')
        self._add_publication('lr_multiplier_var', type='DATA')

        self._add_publication('biased_grad_mean_estimate_norm', type='DATA')
        self._add_publication('biased_grad_var_estimate_norm', type='DATA')
        self._add_publication('grad_mean_estimate_norm', type='DATA')
        self._add_publication('grad_var_estimate_norm', type='DATA')

    def compute(self, message):

        named_module  = message.key

        g_t = message.data
        t   = self._subscriptions['weight_gradients'].counter[(named_module, message.kind)] + 1
        β1  = self._beta1
        β2  = self._beta2

        if named_module not in self._means:
            self._means[named_module] = torch.zeros_like(g_t)
        if named_module not in self._vars:
            self._vars[named_module] = torch.zeros_like(g_t)

        m_t = self._means[named_module]
        v_t = self._vars[named_module]

        self._means[named_module] = (β1 * m_t + (1 - β1) * g_t) / (1 - β1 ** t)
        m_t_corrected = self._means[named_module] / (1 - β1 ** t)

        self._vars[named_module] = (β2 * v_t  + (1 - β2) * g_t.pow(2))
        v_t_corrected = self._vars[named_module] / (1 - β2 ** t)

        lr_multiplier = m_t_corrected / (v_t_corrected.sqrt() + self._eps)

        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'biased_grad_mean_estimate_mean',
                                                message.key, m_t.mean())
        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'biased_grad_mean_estimate_var',
                                                message.key, m_t.var())

        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'biased_grad_var_estimate_mean',
                                                message.key, v_t.mean())
        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'biased_grad_var_estimate_var',
                                                message.key, v_t.var())

        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'grad_mean_estimate_mean',
                                                message.key, m_t_corrected.mean())
        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'grad_mean_estimate_var',
                                                message.key, m_t_corrected.var())

        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'grad_var_estimate_mean',
                                                message.key, v_t_corrected.mean())
        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'grad_var_estimate_var',
                                                message.key, v_t_corrected.var())

        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'lr_multiplier_mean',
                                                message.key, lr_multiplier.mean())
        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'lr_multiplier_var',
                                                message.key, lr_multiplier.var())

        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'biased_grad_mean_estimate_norm',
                                                message.key, m_t.norm())
        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'biased_grad_var_estimate_norm',
                                                message.key, v_t.norm())

        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'grad_mean_estimate_norm',
                                                message.key, m_t_corrected.norm())
        self.message_bus.publish_module_message(message.global_step,
                                                message.train_step,
                                                message.epoch, 'grad_var_estimate_norm',
                                                message.key, v_t_corrected.norm())

        self._backend.add_data(f'{named_module.name}/mean', m_t_corrected.mean(),
                               message.global_step)
        self._backend.add_data(f'{named_module.name}/var', v_t_corrected.mean(),
                               message.global_step)

        self._backend.add_data(f'{named_module.name}/mean_biased', m_t.mean(),
                               message.global_step)
        self._backend.add_data(f'{named_module.name}/var_biased', v_t.mean(),
                               message.global_step)

        self._backend.add_data(f'{named_module.name}/multiplier', lr_multiplier.mean(),
                               message.global_step)
