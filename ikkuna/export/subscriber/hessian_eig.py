from ikkuna.export.subscriber import PlotSubscriber, Subscription
from ikkuna.export.messages import get_default_bus

from hessian_eigenthings import compute_hessian_eigenthings


class HessianEigenSubscriber(PlotSubscriber):
    '''
    .. note::
        This subscriber is only installed if `hessian_eigenthings` can be imported at install-time.
        See `https://github.com/noahgolmant/pytorch-hessian-eigenthings/`.

    A subscriber to compute the top-k eigenvalues of the hessian of the loss w.r.t. the weights.
    This is done by using defalted power iteration from Noah Golmant's ``hessian_eigenthings``
    module, which has to be installed. This operation is _very_ expensive, since it involves
    differentiating twice over a subset of the training data. Since the weights must be fixed,
    gradients from training cannot be reused.

    .. warning::
        In my experiments, the power iteration from ``hessian_eigenthings`` does not really work;
        the estimates do not converge to a single value when increasing the number of steps, but
        rather are all over the place. Investigation needed.
    '''

    def __init__(self, forward_fn, loss_fn, data_loader, batch_size, frequency=1, num_eig=1,
                 power_steps=20, tag='default', message_bus=get_default_bus(), ylims=None,
                 backend='tb'):
        '''
        Parameters
        ----------
        forward_fn  :   function
                        Function to obtain predictions. You probably want to pass the model's
                        ``forward()`` routine here
        loss_fn     :   torch.nn.Module
                        Loss function (such as :class:`torch.nn.CrossEntropyLoss`)
        data_loader :   torch.utils.data.DataLoader
                        Loader for the dataset to compute gradients over
        batch_finished  :   int
                            Number of samples to compute gradients for in one step of power iteration
                            More should lead to a better estimate
        frequency   :   int
                        How often to compute the eigenvalues (after every nth batch)
        num_eig :   int
                    Number of top eigenvalues to compute
        power_iter_steps    :   int
                                Number of steps in the power iteration for computing a singular value.
                                The total number of batches read is then ``power_iter_steps * num_eig``
                                More steps should lead to a better estimate.
        '''
        title  = f'Top hessian Eigenvalues'
        ylabel = 'tbd'
        xlabel = 'Train step'
        subscription = Subscription(self, ['batch_finished', 'activations'], tag=tag)
        super().__init__([subscription],
                         message_bus,
                         {'title': title,
                          'ylabel': ylabel,
                          'ylims': ylims,
                          'xlabel': xlabel},
                         backend=backend)

        self._forward_fn = forward_fn
        self._loss_fn = loss_fn
        self._power_steps = power_steps
        self._dataloader = data_loader
        self._parameters = set()
        self._num_eig = num_eig
        self._frequency = frequency

    def compute(self, message):
        if message.kind == 'activations':
            module = message.key.module
            for p in module.parameters():
                self._parameters.add(p)
        else:
            if self.subscriptions['batch_finished'].counter['batch_finished'] % self._frequency == 0:
                evals, evecs = compute_hessian_eigenthings(self._forward_fn,
                                                           self._parameters,
                                                           self._dataloader,
                                                           self._loss_fn,
                                                           power_iter_steps=self._power_steps,
                                                           num_eigenthings=self._num_eig)
                for i, val in enumerate(sorted(evals)):
                    self.backend.add_data(f'Eigenvalue {i}', evals[i], message.global_step)
