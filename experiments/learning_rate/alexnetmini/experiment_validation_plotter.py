from math import sqrt
import pymongo
import matplotlib
import numpy as np
from itertools import product

# obtain runs collection created by sacred
db_client = pymongo.MongoClient('mongodb://rasmus:rasmus@35.189.247.219/sacred')
sacred_db = db_client.sacred
runs      = sacred_db.runs
metrics   = sacred_db.metrics
colors    = {'ratio_adaptive_schedule_fn': '#254167',
            'exponential_schedule_fn': '#ee5679',
            'identity_schedule_fn': '#d9d874'}


def uniquify_list(seq):
    unique_data = [list(x) for x in set(tuple(x) for x in seq)]
    return unique_data


def get_metric_for_ids(name, ids):
    metric = sacred_db.metrics.aggregate([
        {'$match': {'name': name}},
        {'$match': {'run_id': {'$in': ids}}},
        {'$project': {'steps': True,
                      'values': True,
                      '_id': False,
                      }
         }
    ]
    )
    return list(metric)


def accuracy_traces(save=False):
    # get set of all ids for which we have metrics logged
    ids_with_metrics = metrics.aggregate([{'$project': {'run_id': True, '_id': False}}])
    ids_with_metrics = map(lambda e: e['run_id'], ids_with_metrics)
    ids_with_metrics = list(set(ids_with_metrics))

    # this pipeline outputs documents groups for each combo of batch size and learning rate,
    # containing a list of schedules and a list of lists of ids belonging to each schedule
    pipeline = [
        # filter broken experiments
        {'$match': {'result': {'$ne': None}}},
        # filter experiments for which metrics have not been recorded
        {'$match': {'_id': {'$in': ids_with_metrics}}},
        # filter for schedules we're interested in
        {'$match': {'config.schedule': {'$in': ['ratio_adaptive_schedule_fn',
                                                'exponential_schedule_fn',
                                                'identity_schedule_fn']
                                        }
                    }
         },
        # group into sets identified by the key (schedule, batch_size, base_lr) and collect all
        # experiment _ids
        {'$group': {'_id': {'schedule': '$config.schedule',
                            'batch_size': '$config.batch_size',
                            'base_lr': '$config.base_lr'
                            },
                    '_ids': {'$addToSet': '$_id'}
                    },
         },
        # transform into sets identified by key (batch_size, base_lr) with a list of schedules and a
        # list of lists of ids for each schedule
        {'$group': {'_id': {'batch_size': '$_id.batch_size',
                            'base_lr': '$_id.base_lr'},
                    'schedules': {'$push': '$_id.schedule'},
                    '_ids': {'$push': '$_ids'}}}
    ]
    groups = list(sacred_db.runs.aggregate(pipeline))

    batch_sizes  = sorted(sacred_db.runs.distinct('config.batch_size'))
    lrs          = sorted(sacred_db.runs.distinct('config.base_lr'))
    combinations = list(product(batch_sizes, lrs))
    n            = len(combinations)
    h            = int(n // sqrt(n))
    w            = int(round(n / h + 0.5))
    assert h*w >= n, 'Too few plots!'

    if save:
        matplotlib.use('cairo')

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    f, axarr   = plt.subplots(h, w, sharey=True)
    f.set_size_inches(7, 10)
    axarr_flat = axarr.flat
    ax_map   = {}
    for (b, l) in combinations:
        ax_map[(b, l)] = next(axarr_flat)

    patches = [mpatches.Patch(color=c, label=schedule) for schedule, c in colors.items()]
    f.legend(handles=patches, loc='upper center')

    for group in groups:
        batch_size = group['_id']['batch_size']
        base_lr    = group['_id']['base_lr']
        schedules  = group['schedules']
        idss       = group['_ids']      # lists of ideas for each s in schedules
        ax = ax_map[(batch_size, base_lr)]
        ax.set_title(f'b={batch_size} lr={base_lr}')

        for schedule, ids in zip(schedules, idss):
            accuracies      = get_metric_for_ids('test_accuracy', ids)
            stepss, valuess = zip(*[(element['steps'], element['values']) for element in accuracies])
            stepss_unique = uniquify_list(stepss)
            if len(stepss_unique) != 1:
                raise ValueError(f'Inconsistent step values among ids {ids}')
            steps           = stepss[0]   # all steps vectors should be identical
            values          = np.array(list(valuess))
            mean            = values.mean(axis=0)
            error           = [np.abs(values.min(axis=0) - mean), np.abs(values.max(axis=0) - mean)]
            color           = colors[schedule]
            for run in accuracies:
                ax.errorbar(steps, mean, yerr=error, label=schedule, color=color,
                            errorevery=len(steps) // 20)
    if save:
        plt.savefig('accuracies.pdf')
    else:
        plt.show()


def accuracy_boxplots(lr, batch_size, save=False):
    # pipeline to get all accuracies with `_id`s
    accuracies_pipeline = [
        {'$match': {'result': {'$ne': None}}},                   # filter broken experiments
        {'$match': {'config.schedule': {'$in': ['ratio_adaptive_schedule_fn',
                                                'exponential_schedule_fn',
                                                'identity_schedule_fn']}}},
        {'$match': {'config.batch_size': batch_size}},
        {'$match': {'config.base_lr': lr}},
        {'$group': {'_id': '$config.schedule',                   # group schedule fn
                    'accuracies': {'$addToSet': '$result'}}}     # make array from all accuracies
    ]
    # make list, since the iterator is exhausted after one traversal
    grouped_runs = list(sacred_db.runs.aggregate(accuracies_pipeline))
    #
    # run over the list of records/dicts once to group the values of each key (schedule_fn and
    # accuracies) into separate lists so they can be used for boxplotting.
    labels     = []
    accuracies = []
    for d in grouped_runs:
        labels.append(d['_id'])
        accuracies.append(d['accuracies'])

    if save:
        matplotlib.use('cairo')

    import matplotlib.pyplot as plt
    # show boxplots
    f  = plt.figure()
    ax = f.gca()
    ax.boxplot(accuracies, labels=labels)
    ax.set_title(f'Final accuracies for lr={lr}, bs={batch_size}')

    # plot the samples as dots with a random normal yoffset
    for i in range(len(labels)):
        y = accuracies[i]
        # tick locations are [1, 2, 3, ...] for boxplots
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, '.', alpha=0.3, markersize=20, color=colors[labels[i]])

    if save:
        plt.savefig(f'accuracy_boxplots_{lr}_{batch_size}.pdf')
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()
    accuracy_boxplots(lr=0.2, batch_size=128, save=args.save)
