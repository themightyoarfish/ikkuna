from ikkuna.export.subscriber import Subscriber
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict


class RatioSubscriber(Subscriber):

    '''A :class:`Subscriber` which computes the average ratio between weights and updates per epoch

    Attributes
    ----------
    '''

    def __init__(self):
        super().__init__()
        self._ratios = defaultdict(list)

    def __call__(self, module_datas):
        super().__call__(module_datas)

        for module_data in module_datas:
            module                        = module_data._module
            weights                       = module_data._data['weights']
            updates                       = module_data._data['weight_updates']
            ratio                         = updates.div(weights).mean().item()
            # moving average of ratios
            # self._ratios[module] += ratio + self._counter * self._ratios[module] / self._counter + 1
            self._ratios[module].append(ratio)

    def epoch_finished(self, epoch):
        super().epoch_finished(epoch)
        modules = list(self._ratios.keys())
        ratioses  = list(self._ratios.values())
        # figure, ax = plt.subplots(111)
        # for module, ratios in zip(modules, ratioses):
        #     ax.plot(np.arange(self._counter), ratios, label=module)
        # ax.set_title('Average update ratio')
        # ax.set_ylabel('Batch step')
        # ax.set_xlabel('Mean update ratio')
        # ax.grid(True)
        # figure.show()
