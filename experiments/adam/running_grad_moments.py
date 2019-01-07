import math
import torch
from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus


class BiasCorrectedMomentsSubscriber(PlotSubscriber):

    def __init__(self, lr, beta1, beta2, eps, message_bus=get_default_bus(), tag=None, subsample=40,
                 ylims=None, backend='tb'):

        title        = 'gradient_moments'
        ylabel       = 'Gradient Moments'
        xlabel       = 'Train step'
        subscription = Subscription(self, ['weight_gradients'], tag, subsample)
        super().__init__([subscription], message_bus, {'title': title,
                                                       'ylabel': ylabel,
                                                       'ylims': ylims,
                                                       'xlabel': xlabel},
                         backend=backend)

        self._lr    = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps   = eps
        self._means = dict()
        self._vars  = dict()

        for pub_name in {
            'biased_grad_mean_estimate_mean',
            'biased_grad_mean_estimate_median',
            'biased_grad_mean_estimate_var',
            'biased_grad_var_estimate_mean',
            'biased_grad_var_estimate_median',
            'biased_grad_var_estimate_var',
            'biased_grad_mean_estimate_norm',
            'biased_grad_var_estimate_norm',
            'grad_mean_estimate_mean',
            'grad_mean_estimate_median',
            'grad_mean_estimate_var',
            'grad_var_estimate_mean',
            'grad_var_estimate_median',
            'grad_var_estimate_var',
            'grad_mean_estimate_norm',
            'grad_var_estimate_norm',
            'effective_lr_mean',
            'effective_lr_median',
            'effective_lr_var',
            'effective_lr_norm',
        }:
            self._add_publication(pub_name, type='DATA')

    def compute(self, message):

        named_module  = message.key

        grad = message.data
        t    = message.global_step + 1

        # init moving avgs if not present
        if named_module not in self._means:
            self._means[named_module] = torch.zeros_like(grad)
        if named_module not in self._vars:
            self._vars[named_module] = torch.zeros_like(grad)

        exp_avg, exp_avg_sq = self._means[named_module], self._vars[named_module]
        beta1, beta2 = self._beta1, self._beta2

        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        bias_correction1     = 1 - beta1 ** t
        bias_correction2     = 1 - beta2 ** t
        unbiased_exp_avg    = exp_avg / bias_correction1
        unbiased_exp_avg_sq = exp_avg_sq / bias_correction2
        step_size            = self._lr * math.sqrt(bias_correction2) / bias_correction1
        denom                = exp_avg_sq.sqrt().add_(self._eps)
        update               = step_size * exp_avg / denom
        update.div_(grad)
        nan_tensor           = torch.isnan(update)
        inf_tensor           = torch.isinf(update)
        effective_lr         = update[(1 - nan_tensor) & (1 - inf_tensor)]
        if grad.sum() == torch.tensor(0.0).cuda():
            # this would mean all entries are nan or inf because the current gradient was
            # zero
            effective_lr = torch.tensor(0.0).cuda()

        # instead of repeating the call to publish_module_message for each topic, look at
        # all topic names and infer the local variable from the topic name
        for topic in self.publications['DATA']:

            if topic.startswith('biased_grad_mean'):
                data = exp_avg
            elif topic.startswith('biased_grad_var'):
                data = exp_avg_sq
            elif topic.startswith('grad_mean'):
                data = unbiased_exp_avg
            elif topic.startswith('grad_var'):
                data = unbiased_exp_avg_sq
            elif topic.startswith('effective_lr'):
                data = effective_lr
            else:
                raise ValueError(f'Unexpected topic "{topic}"')

            if topic.endswith('norm'):
                data = data.norm()
            elif topic.endswith('mean'):
                data = data.mean()
            elif topic.endswith('median'):
                data = data.median()
            elif topic.endswith('var'):
                data = data.var()
            else:
                raise ValueError(f'Unexpected topic "{topic}"')

            self.message_bus.publish_module_message(message.global_step,
                                                    message.train_step,
                                                    message.epoch, topic,
                                                    message.key, data)

        self._backend.add_data(f'{named_module.name}/grad_mean', unbiased_exp_avg.median(),
                               message.global_step)
        self._backend.add_data(f'{named_module.name}/grad_var', unbiased_exp_avg_sq.median(),
                               message.global_step)

        self._backend.add_data(f'{named_module.name}/grad_mean_biased', exp_avg.median(),
                               message.global_step)
        self._backend.add_data(f'{named_module.name}/grad_var_biased', exp_avg_sq.median(),
                               message.global_step)

        if not grad.sum() == torch.tensor(0.0).cuda():
            self._backend.add_data(f'{named_module.name}/lr', effective_lr.median(),
                                   message.global_step)
