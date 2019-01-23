import itertools
import pymongo
import numpy as np
from experiments.sacred_utils import get_metric_for_ids
from experiments.utils import unify_limits
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 0.8
from colors import Color
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


def get_layer_metric_map(metric_regex, ids):
    from collections import defaultdict

    name_metric_map = defaultdict(list)
    for trace in get_metric_for_ids(metric_regex, ids, per_module=True):
        name_metric_map[trace['name']].append((trace['steps'], trace['values']))

    return name_metric_map


def plot_accuracy(models, optimizers, learning_rates, **kwargs):

    conditions = [
        # get only experiments which returned 0
        {'$match': {'config.identifier': 'experiments/saturation/experiment'}},
        {'$match': {'result': 0}},
        {'$match': {'config.n_epochs': 30}},
        {'$match': {'config.batch_size': 512}},
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
            '_freeze_points': {'$addToSet': '$config.freeze_at'},
            '_member_ids': {'$addToSet': '$_id'}
        }},
    ]

    groups = list(sacred_db.runs.aggregate(pipeline))
    if not groups:
        raise RuntimeError('No data found. Did you mistype something?')

    if kwargs.get('save', False):
        matplotlib.use('cairo')
        save = True
    else:
        save = False

    import matplotlib.pyplot as plt
    figures = dict()

    for group in groups:
        model     = group['_id']['model']
        optimizer = group['_id']['optimizer']
        base_lr   = group['_id']['base_lr']
        ids       = group['_member_ids']
        freeze_points       = group['_freeze_points']

        # create figure for group
        f       = plt.figure(figsize=kwargs.get('figsize', (9, 6)))
        f.suptitle(f'{model}, {optimizer}, {base_lr}')
        ax_acc   = f.add_subplot(121)
        ax_similarity   = f.add_subplot(122)

        accuracy_traces = get_metric_for_ids('test_accuracy', ids, per_module=False)
        for i, trace in enumerate(accuracy_traces):
            steps = trace['steps']
            values = trace['values']
            ax_acc.plot(steps, values, label=f'Freeze at {freeze_points[i]}')

        similarity_traces = get_layer_metric_map('self_similarity', ids)
        colors = iter(itertools.cycle(plt.get_cmap('Set2').colors))

        for layer, records in similarity_traces.items():
            color = next(colors)
            for steps, values in records:
                ax_similarity.plot(steps, values, c=color, label=layer)

        # set title and labels
        ax_acc.set_title('Accuracy')
        ax_acc.set_xlabel('Trains tep')
        ax_acc.legend()

        new_labels = []
        new_handles = []
        for h, l in zip(*ax_similarity.get_legend_handles_labels()):
            if l not in new_labels:
                new_labels.append(l)
                new_handles.append(h)
        ax_similarity.legend(new_handles, new_labels)

        m_str        = model.lower()
        o_str        = optimizer.lower()
        lr_str       = str(base_lr).replace('.', '')
        key          = f'{m_str}_{o_str}_{lr_str}.pdf'
        figures[key] = f

    if not save:
        plt.show()
    else:
        for name, f in figures.items():
            f.savefig(name)


def plots_for_thesis():
    plot_accuracy(['VGG', 'AlexNetMini'], ['SGD'], [0.01, 0.1, 0.5], save=False)


if __name__ == '__main__':
    plots_for_thesis()
    # plot_moments(['AdamModel'], ['Adam'], [0.001], start=0, end=-1, save=True)
