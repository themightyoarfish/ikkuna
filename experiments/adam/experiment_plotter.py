import pymongo
import numpy as np
from experiments.sacred_utils import get_metric_for_ids
from experiments.utils import unify_limits
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 0.8
from colors import Color
import scipy.ndimage
import os

try:
    pwd = os.environ['MONGOPWD']
except KeyError:
    print('You need to set the MONGOPWD variable to connect to the database.')
    import sys
    sys.exit(1)

# obtain runs collection created by sacred
db_client = pymongo.MongoClient(f'mongodb://rasmus:{pwd}@35.189.247.219/sacred')
sacred_db = db_client.sacred
runs      = sacred_db.runs
metrics   = sacred_db.metrics


def median_pool_array(array, ksize, stride):
    from scipy.ndimage.filters import median_filter
    return median_filter(array, size=(ksize,), mode='reflect')[::stride]


def get_layer_metric_map(metric_regex, ids):
    from collections import defaultdict
    import re

    name_metric_map = defaultdict(list)
    for trace in get_metric_for_ids(metric_regex, ids, per_module=True):
        name_metric_map[trace['name']].append(trace['values'])

    return {
        name[re.match(metric_regex, name).span()[1]+1:]: np.mean(arrays, axis=0)
        for name, arrays in name_metric_map.items()
    }


def plot_moments(models, optimizers, learning_rates, **kwargs):
    '''Plot percentage in loss decrease vs current ratio, plus other stuffs.'''

    conditions = [
        # get only experiments which returned 0
        {'$match': {'result': 0}},
        # all exps were run for 45 epochs.
        {'$match': {'config.n_epochs': 45}},
        {'$match': {'config.batch_size': 128}},
        # filter models
        {'$match': {'config.model': {'$in': models}}},
        # filter opts
        {'$match': {'config.optimizer': {'$in': optimizers}}},
        # filter lrs
        {'$match': {'config.base_lr': {'$in': learning_rates}}},
    ]

    pipeline = conditions + [
        # group into groups keyed by (model, optimizer, lr)
        {'$group': {
            '_id': {
                'model': '$config.model',
                'optimizer': '$config.optimizer',
                'base_lr': '$config.base_lr',
            },
            # add field with all run ids belonging to this group
            '_member_ids': {'$addToSet': '$_id'}
        }},
        {'$sort': {'_id.base_lr': 1}}
    ]

    groups = list(sacred_db.runs.aggregate(pipeline))

    if kwargs.get('save', False):
        matplotlib.use('cairo')
        save = True
    else:
        save = False

    import matplotlib.pyplot as plt

    figures = dict()
    start = kwargs.get('start', 0)
    end = kwargs.get('end', -1)
    steps = None

    for group in groups:
        model     = group['_id']['model']
        optimizer = group['_id']['optimizer']
        base_lr   = group['_id']['base_lr']

        # create figure for group
        f       = plt.figure(figsize=kwargs.get('figsize', (9, 6)))
        f.suptitle(f'{model}, {optimizer}, {base_lr}')
        ax_loss = f.add_subplot(326)
        ax_legend = f.add_subplot(222)
        ax_acc  = ax_loss.twinx()
        ax_mean = f.add_subplot(321)
        ax_var  = f.add_subplot(323, sharex=ax_mean)
        ax_lr   = f.add_subplot(325, sharex=ax_mean)

        # set title and labels
        ax_mean.set_title('Bias-Corrected Running Mean estimate')
        ax_var.set_title('Bias-Corrected Running Variance estimate')
        ax_lr.set_title('LR multiplier')
        ax_lr.set_xlabel('Train step')
        ax_loss.set_title('Train Loss & Validation Accuracy')

        ids = group['_member_ids']

        layer_mean_map = get_layer_metric_map('^grad_mean_estimate_mean', ids)
        layer_var_map = get_layer_metric_map('^grad_var_estimate_mean', ids)
        layer_lr_multiplier_map = get_layer_metric_map('^lr_multiplier_mean', ids)

        loss_trace  = np.array([
            trace['values'] for trace in get_metric_for_ids('loss', ids, per_module=False)
        ]).mean(axis=0)

        test_accuracy_steps = None
        values = []
        for trace in get_metric_for_ids('test_accuracy', ids, per_module=False):
            test_accuracy_steps = test_accuracy_steps or trace['steps']
            values.append(trace['values'])

        test_accuracy_steps = np.array(test_accuracy_steps)
        test_accuracy_trace = np.mean(values, axis=0)

        ksize = 50
        for layer_name, mean_trace in layer_mean_map.items():
            data = mean_trace[start:end]

            # if we're here the first time, determine the xtick limits
            n = len(data)
            if end == -1:
                end = start + n
            if steps is None:
                steps = np.arange(start, end)[::ksize]

            ax_mean.plot(steps, median_pool_array(data, ksize, ksize), label=layer_name)

        for layer_name, var_trace in layer_var_map.items():
            data = median_pool_array(var_trace[start:end], ksize, ksize)
            ax_var.plot(steps, data, label=layer_name)

        for layer_name, lr_multiplier_trace in layer_lr_multiplier_map.items():
            data = median_pool_array(lr_multiplier_trace[start:end], ksize, ksize)
            ax_lr.plot(steps, data, label=layer_name)

        # ax_mean.set_yscale('log')
        # ax_var.set_yscale('log')
        # ax_lr.set_yscale('log')

        ax_loss.plot(steps, median_pool_array(loss_trace[start:end], ksize, ksize), label='loss')

        valid_test_accuracy_indices = np.logical_and(test_accuracy_steps >= start,
                                                     test_accuracy_steps < end)
        ax_acc.plot(test_accuracy_steps[valid_test_accuracy_indices],
                    test_accuracy_trace[valid_test_accuracy_indices], label='test-accuracy', c='red')

        handles, labels = ax_mean.get_legend_handles_labels()
        ax_legend.legend(handles, labels, loc='upper right')
        ax_legend.spines['top'].set_visible(False)
        ax_legend.spines['right'].set_visible(False)
        ax_legend.spines['bottom'].set_visible(False)
        ax_legend.spines['left'].set_visible(False)
        ax_legend.xaxis.set_visible(False)
        ax_legend.yaxis.set_visible(False)

        for label in ax_var.get_xticklabels():
            label.set_visible(False)
        for label in ax_mean.get_xticklabels():
            label.set_visible(False)

        m_str        = model.lower()
        o_str        = optimizer.lower()
        lr_str       = str(base_lr).replace('.', '')
        key          = f'{m_str}_{o_str}_{lr_str}_{start}_{end}.pdf'
        figures[key] = f

    if not save:
        plt.show()
    else:
        for name, f in figures.items():
            f.savefig(name)


if __name__ == '__main__':
    plot_moments(['FullyConnectedModel'], ['Adam'], [0.0005, 0.001, 0.05], start=300)
