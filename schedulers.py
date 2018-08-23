from torch.optim import Optimizer


class FunctionScheduler(object):
    '''Scheduler class. This is a ripoff from :class:`torch.optim.lr_scheduler._LRScheduler` because
    PyTorch learning rate schedulers only support a resolution of epochs. The need for changing the
    learning rate on a batch basis came about when reproducing the ResNet paper, as they change the
    learning rate at specific time steps.'''

    def __init__(self, optimizer, lr_fn):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer  = optimizer
        self.train_step = 0
        self.batch      = 0
        self.epoch      = 0
        self.lr_fn      = lr_fn
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step()

    def state_dict(self):
        '''See :meth:`torch.optim.lr_scheduler._LRScheduler.state_dict()`'''
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        '''See :meth:`torch.optim.lr_scheduler._LRScheduler.load_state_dict()`'''
        self.__dict__.update(state_dict)

    def get_lr(self):
        '''Obtain the new learning rate.

        Returns
        -------
        list
            list of learning rates, one for each parameter group in the optimizer
        '''
        return self.lr_fn(self.base_lrs, self.batch, self.train_step, self.epoch)

    def step(self, epoch_finished=False):
        '''Do one train step. Calling this does *NOT* call ``step()`` on the associated optimizer.

        Parameters
        ----------
        epoch_finished  :   bool
                            Should be set to ``True`` after an epoch has ended.
        '''
        self.train_step  += 1
        self.batch       += 1
        if epoch_finished:
            self.batch  = 0
            self.epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
