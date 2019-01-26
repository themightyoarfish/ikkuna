import itertools
import numpy as np
from experiments.sacred_utils import get_metric_for_ids, get_client
from experiments.utils import prune_labels
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 1
from colors import Color
import os


batches_per_epoch = 50000 // 512


def get_layer_metric_map(metric_regex, ids):
    from collections import defaultdict

    name_metric_map = defaultdict(list)
    for trace in get_metric_for_ids(metric_regex, ids, per_module=True):
        name_metric_map[trace['name']].append((trace['steps'], trace['values']))

    return name_metric_map


def plot_accuracy(models, optimizers, learning_rates, **kwargs):

    conditions = [
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
        {
            '$group':
            {
                '_id':
                {
                    'optimizer': '$config.optimizer',
                    'base_lr': '$config.base_lr',
                    'model': '$config.model',
                    'freeze_at': '$config.freeze_at',
                },
                '_member_ids': {'$addToSet': '$_id'},
                '_models': {'$addToSet': '$config.model'},
            },
        },
        {'$sort': {'_id.base_lr': -1}},
        {'$unwind': '$_models'},
        {
            '$group':
            {
                '_id':
                {
                    'optimizer': '$_id.optimizer',
                    'base_lr': '$_id.base_lr',
                },
                '_models': {'$addToSet': '$_models'},
                '_groups':
                {
                    '$push': {'model': '$_id.model', 'freeze_at': '$_id.freeze_at', '_member_ids':
                              '$_member_ids'},
                },
            }
        }
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
        models        = sorted(group['_models'])
        optimizer     = group['_id']['optimizer']
        base_lr       = group['_id']['base_lr']
        subgroups     = group['_groups']

        # create figure for group
        f             = plt.figure(figsize=kwargs.get('figsize', (9, 6)))
        f.suptitle(f'{optimizer}, {base_lr}')

        axes_acc = dict()
        axes_sim = dict()

        first_acc = None
        first_sim = None
        for i, model in enumerate(models):
            if not first_acc:
                axes_acc[model] = f.add_subplot(2, len(models), i + 1)
                axes_sim[model] = f.add_subplot(2, len(models), i + 1 + len(models))
                first_acc = axes_acc[model]
                first_sim = axes_sim[model]
            else:
                axes_acc[model] = f.add_subplot(2, len(models), i + 1, sharey=first_acc)
                axes_sim[model] = f.add_subplot(2, len(models), i + 1 + len(models), sharey=first_sim)

        first_acc.set_ylabel('Accuracy')
        first_sim.set_ylabel('SVCCA coef')

        colors = {
            0.99: Color.RED.value,
            0.995: Color.DARKBLUE.value,
            10:  Color.SLATE.value
        }
        for subgroup in subgroups:
            model     = subgroup['model']
            freeze_at = subgroup['freeze_at']
            ids       = subgroup['_member_ids']

            # get all accuracy traces for this model and freeze point into np array and mean over
            # runs
            steps  = None
            values = []
            for trace in get_metric_for_ids('test_accuracy', ids, per_module=False):
                steps = steps or trace['steps']
                values.append(trace['values'])

            values = np.array(values).mean(axis=0)
            steps  = np.array(steps) / batches_per_epoch

            # plot data and set labels + title
            axes_acc[model].plot(steps, values, label=f'Freeze at {freeze_at}', c=colors[freeze_at])
            axes_acc[model].set_title(f'{model}')
            axes_acc[model].set_xlabel('Epoch')

            similarity_traces = get_layer_metric_map('self_similarity', ids)
            accuracy_steps = batches_per_epoch * np.arange(1, 30)
            for layer in similarity_traces:
                data = np.zeros([len(similarity_traces[layer]), len(accuracy_steps)])
                for i, (steps, values) in enumerate(similarity_traces[layer]):
                    data[i, :len(steps)] = values
                    data[i, len(steps):] = values[-1]
                axes_sim[model].plot(accuracy_steps, data.mean(axis=0), label=layer)

                # create array of right size (n_runs, n_steps)
                # put each trace in there by filling with last value
                # mean over runs
        # colors = iter(itertools.cycle(plt.get_cmap('Set2').colors[:len(similarity_traces)]))

        # for layer, records in similarity_traces.items():
        #     layer = layer[len('self_similarity/.')-1:]
        #     color = next(colors)
        #     for steps, values in records:
        #         steps = np.array(steps) / batches_per_epoch
        #         ax_similarity.plot(steps, values, c=color, label=layer)

        # ax_similarity.set_title('SVCCA coefficient')
        # ax_similarity.set_xlabel('Epoch')

        for ax in axes_acc.values():
            prune_labels(ax, location='lower right')

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
