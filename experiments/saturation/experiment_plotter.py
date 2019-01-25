import itertools
import numpy as np
from experiments.sacred_utils import get_metric_for_ids, get_client
from experiments.utils import unify_limits
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 0.8
from colors import Color
import os


batches_per_epoch = 50000 // 512


def prune_labels(ax, location='best'):
    new_handles_labels = dict()
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in new_handles_labels:
            new_handles_labels[l] = h
    sorted_labels = sorted(new_handles_labels)
    sorted_handles = [new_handles_labels[l] for l in sorted_labels]
    ax.legend(sorted_handles, sorted_labels, loc=location)


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
        {
            '$group':
            {
                '_id':
                {
                    'optimizer': '$config.optimizer',
                    'base_lr': '$config.base_lr',
                },
                '_freeze_points': {'$push': '$config.freeze_at'},
                '_member_ids': {'$addToSet': '$_id'},
                '_models': {'$addToSet': '$config.model'},
            },
        },
        {'$sort': {'_id.base_lr': 1}},
        {'$addFields': {'_num_models': {'$size': '$_models'}}}
    ]
    sacred_db = get_client().sacred

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
        models         = group['_models']
        optimizer     = group['_id']['optimizer']
        base_lr       = group['_id']['base_lr']
        ids           = sorted(group['_member_ids'])
        freeze_points = group['_freeze_points']

        # create figure for group
        f             = plt.figure(figsize=kwargs.get('figsize', (9, 6)))
        f.suptitle(f'{optimizer}, {base_lr}')

        axes = dict()
        first = None
        for i, model in enumerate(models):
            if not first:
                axes[model] = f.add_subplot(1, len(models), i + 1)
                first = axes[model]
            else:
                axes[model] = f.add_subplot(1, len(models), i + 1, sharey=first)

        first.set_ylabel('Accuracy')

        colors = {
            0.99: Color.RED.value,
            0.995: Color.DARKBLUE.value,
            10:  Color.SLATE.value
        }

        accuracy_traces = get_metric_for_ids('test_accuracy', ids, per_module=False)
        models = sacred_db.runs.aggregate([{'$match': {'_id': {'$in': ids}}},
                                           {'$sort': {'_id': 1}},
                                           {'$project' : {'model': '$config.model'}}])
        models = map(lambda doc: doc['model'], models)
        for i, (model, trace) in enumerate(zip(models, accuracy_traces)):
            steps = np.array(trace['steps']) / batches_per_epoch
            values = trace['values']
            axes[model].plot(steps, values, label=f'Freeze at {freeze_points[i]}',
                        c=colors[freeze_points[i]])
            axes[model].set_title(f'{model}')
            axes[model].set_xlabel('Epoch')

        # similarity_traces = get_layer_metric_map('self_similarity', ids)
        # colors = iter(itertools.cycle(plt.get_cmap('Set2').colors[:len(similarity_traces)]))

        # for layer, records in similarity_traces.items():
        #     layer = layer[len('self_similarity/.')-1:]
        #     color = next(colors)
        #     for steps, values in records:
        #         steps = np.array(steps) / batches_per_epoch
        #         ax_similarity.plot(steps, values, c=color, label=layer)

        # ax_similarity.set_title('SVCCA coefficient')
        # ax_similarity.set_xlabel('Epoch')

        # prune_labels(ax_acc, location='lower right')
        # prune_labels(ax_similarity, location='lower right')

        # m_str        = model.lower()
        # o_str        = optimizer.lower()
        # lr_str       = str(base_lr).replace('.', '')
        # key          = f'{m_str}_{o_str}_{lr_str}.pdf'
        # figures[key] = f

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
