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


def between(steps, start, end):
    if end == -1:
        end = max(steps)
    steps = np.array(steps)
    return np.logical_and(steps >= start, steps <= end)


def median_pool_array(array, ksize, stride):
    from scipy.ndimage.filters import median_filter
    return median_filter(array, size=(ksize,), mode='reflect')[::stride]


def get_layer_metric_map(metric_regex, ids):
    from collections import defaultdict
    import re

    name_metric_map = defaultdict(list)
    for trace in get_metric_for_ids(metric_regex, ids, per_module=True):
        name_metric_map[trace['name']].append(trace['values'])
        steps = trace['steps']

    return {
        name[re.match(metric_regex, name).span()[1]+1:]: np.mean(arrays, axis=0)
        for name, arrays in name_metric_map.items()
    }, np.array(steps)


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

    for group in groups:
        model     = group['_id']['model']
        optimizer = group['_id']['optimizer']
        base_lr   = group['_id']['base_lr']

        # create figure for group
        f       = plt.figure(figsize=kwargs.get('figsize', (9, 6)))
        f.suptitle(f'{model}, {optimizer}, {base_lr}')
        ax_loss = f.add_subplot(326)
        ax_loss.locator_params(nbins=8, axis='y')
        ax_legend = f.add_subplot(222)
        ax_acc  = ax_loss.twinx()
        ax_mean = f.add_subplot(321)
        ax_var  = f.add_subplot(323, sharex=ax_mean)
        ax_lr   = f.add_subplot(325, sharex=ax_mean)

        # set title and labels
        ax_mean.set_title('Bias-Corrected Running Mean estimate')
        ax_var.set_title('Bias-Corrected Running Variance estimate')
        ax_lr.set_title('Effective LR')
        ax_lr.set_xlabel('Train step')
        ax_loss.set_title('Train Loss & Validation Accuracy')

        ids = group['_member_ids']

        layer_mean_map, steps_mean = get_layer_metric_map('^grad_mean_estimate_median', ids)
        layer_var_map, steps_var = get_layer_metric_map('^grad_var_estimate_median', ids)
        layer_effective_lr_map, steps_lr = get_layer_metric_map('^effective_lr_median', ids)
        valid_idx_mean = between(steps_mean, start, end)
        valid_idx_var = between(steps_var, start, end)
        valid_idx_lr = between(steps_lr, start, end)

        loss_traces = get_metric_for_ids('loss', ids, per_module=False)
        loss = []
        for trace in loss_traces:
            loss.append(trace['values'])
            steps_loss = trace['steps']
        loss_trace  = np.array(loss).mean(axis=0)
        steps_loss = np.array(steps_loss)
        valid_idx_loss = between(steps_loss, start, end)

        test_accuracy_steps = None
        values = []
        for trace in get_metric_for_ids('test_accuracy', ids, per_module=False):
            test_accuracy_steps = test_accuracy_steps or trace['steps']
            values.append(trace['values'])

        test_accuracy_steps = np.array(test_accuracy_steps)
        test_accuracy_trace = np.mean(values, axis=0)
        valid_idx_test_acc = between(test_accuracy_steps, start, end)

        k_full = 50     # filter size for metrics computed on each train step
        k_sparse = 5    # filter size for metrics at every 40th (in this case) step

        for layer_name, mean_trace in layer_mean_map.items():
            data = mean_trace[valid_idx_mean]
            ax_mean.plot(steps_mean[valid_idx_mean], data, label=layer_name)

        for layer_name, var_trace in layer_var_map.items():
            data = var_trace[valid_idx_var]
            ax_var.plot(steps_var[valid_idx_var], data, label=layer_name)

        for layer_name, effective_lr_trace in layer_effective_lr_map.items():
            data = median_pool_array(effective_lr_trace[valid_idx_lr], k_sparse, k_sparse)
            ax_lr.plot(steps_lr[valid_idx_lr][::k_sparse], data, label=layer_name)

        # ax_mean.set_yscale('log')
        ax_var.set_yscale('log')
        ax_lr.set_yscale('log')

        ax_loss.plot(steps_loss[valid_idx_loss][::k_full],
                     median_pool_array(loss_trace[valid_idx_loss], k_full, k_full), label='loss')

        ax_acc.plot(test_accuracy_steps[valid_idx_test_acc],
                    test_accuracy_trace[valid_idx_test_acc], label='test-accuracy', c='red')

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
    plot_moments(['AdamModel'], ['Adam'], [0.001], start=0, end=-1, save=True)
