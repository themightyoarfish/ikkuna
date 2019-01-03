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

        for pub_name in {
            'biased_grad_mean_estimate_mean',
            'biased_grad_mean_estimate_var',
            'biased_grad_var_estimate_mean',
            'biased_grad_var_estimate_var',
            'biased_grad_mean_estimate_norm',
            'biased_grad_var_estimate_norm',
            'grad_mean_estimate_mean',
            'grad_mean_estimate_var',
            'grad_var_estimate_mean',
            'grad_var_estimate_var',
            'grad_mean_estimate_norm',
            'grad_var_estimate_norm',
            'lr_multiplier_mean',
            'lr_multiplier_var'}:
            self._add_publication(pub_name, type='DATA')

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

        # instead of repeating the call to publish_module_message for each topic, look at
        # all topic names and infer the local variable from the topic name
        for topic in self._published_topics:
            if topic.startswith('biased_grad_mean'):
                data = m_t
            elif topic.startswith('biased_grad_var'):
                data = v_t
            elif topic.startswith('grad_mean'):
                data = m_t_corrected
            elif topic.startswith('grad_var'):
                data = v_t_corrected
            elif topic.startswith('lr_multiplier'):
                data = lr_multiplier
            else:
                raise ValueError(f'Unexpected topic "{topic}"')

            if topic.endswith('norm'):
                data = data.norm()
            elif topic.endswith('mean'):
                data = data.mean()
            elif topic.endswith('var'):
                data = data.var()
            else:
                raise ValueError(f'Unexpected topic "{topic}"')

            self.message_bus.publish_module_message(message.global_step,
                                                    message.train_step,
                                                    message.epoch, topic,
                                                    message.key, data)

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
