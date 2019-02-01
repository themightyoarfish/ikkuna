from collections import defaultdict
import itertools
import numpy as np
from experiments.sacred_utils import get_metric_for_ids, get_client
from experiments.utils import prune_labels
import matplotlib
import matplotlib.lines as mlines
matplotlib.rcParams['lines.linewidth'] = 1
from colors import Color


batches_per_epoch = 50000 // 512


def get_layer_metric_map(metric_regex, ids):
    '''Get a mapping for layer -> runs for metric where the value for a layer is a list of tuples,
    each tuple consisting of steps and values for each step'''
    name_metric_map = defaultdict(list)
    for trace in get_metric_for_ids(metric_regex, ids, per_module=True):
        name_metric_map[trace['name']].append((trace['steps'], trace['values']))

    return name_metric_map


def plot_freeze_points(models, optimizers, learning_rates, freeze_points):
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
        {'$match': {'config.freeze_at': {'$in': freeze_points}}},
    ]

    pipeline = conditions + [
        {'$lookup': {'from': 'metrics',
                     'localField': '_id',
                     'foreignField': 'run_id',
                     'as': 'metrics'}},
        {'$project': {'_id': True,
                      'base_lr': '$config.base_lr',
                      'model': '$config.model',
                      'optimizer': '$config.optimizer',
                      'freeze_at': '$config.freeze_at',
                      'metrics': {
                          '$filter':
                                  {'input': {
                                      '$map':
                                      {
                                          'input': '$metrics',
                                          'in': {'name': '$$this.name',
                                                 'n_steps': {'$size': '$$this.steps'}}
                                      }
                                  },
                                      'cond':
                                      {
                                          '$eq':
                                          [
                                              {'$substr': ['$$this.name', 0, len('self_similarity')]},
                                              'self_similarity'
                                          ]
                                      }
                                  }
                                  }
                      }
         },
        {'$group':
         {
             '_id': {'base_lr': '$base_lr',
                     'model': '$model',
                     'optimizer': '$optimizer'},
             'freeze_points': {'$push': {'value': '$freeze_at', 'metrics': '$metrics'}}
         }
         }
    ]
    sacred_db = get_client().sacred

    groups = list(sacred_db.runs.aggregate(pipeline))
    __import__('ipdb').set_trace()


def plot_accuracy_similarity(models, optimizers, learning_rates, **kwargs):

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
            # the idea here is that we first form groups of all runs for a given set of hyper
            # parameters, recording which ids are in this set and which models were seen ...
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
        # ... after sorting so we iterate over groups according to the varying parameter – the
        # learning rate – we merge groups so that each group contains the runs for all models and we
        # have in each group a _group property which contains the actual groups per-model and
        # per-freeze-point. We also record the ids for each of theses subgroups by accessing the
        # _member_ids from the original groups.
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

        ###########################################################################################
        # Step 0: Create figure array so each model gets an accuracy, similarity and legend plot  #
        ###########################################################################################
        f             = plt.figure(figsize=kwargs.get('figsize', (9, 12)))
        f.suptitle(f'{optimizer}, {base_lr}')

        axes_acc    = dict()
        axes_sim    = dict()
        axes_legend = dict()

        first_acc   = None
        first_sim   = None
        # this creates a 3xn_models array. First row accuracy (one line per freeze_point), second
        # row selfsimilarity (varying line style per freeze_point) and last row for larger legends
        for i, model in enumerate(models):
            if not first_acc:
                # first plot: don't share axes
                axes_acc[model] = f.add_subplot(3, len(models), i+1)
                axes_sim[model] = f.add_subplot(3, len(models), i+1+len(models))
                first_acc       = axes_acc[model]
                first_sim       = axes_sim[model]
            else:
                # first plot: share yaxis of first plot
                axes_acc[model] = f.add_subplot(3, len(models), i+1, sharey=first_acc)
                axes_sim[model] = f.add_subplot(3, len(models), i+1+len(models))

            axes_legend[model] = f.add_subplot(3, len(models), i+1+2*len(models))

        first_acc.set_ylabel('Accuracy')
        first_sim.set_ylabel('SVCCA coef')

        # we have only 3 freeze points rn, so hardcode color and line style for disambiguation
        colors = {
            0.99: Color.RED.value,
            0.995: Color.DARKBLUE.value,
            'never':  Color.SLATE.value,
            'percentage': Color.YELLOW.value
        }
        linestyles = {
            0.99: '--',
            0.995: '-.',
            'percentage': '-'
        }

        ###########################################################################################
        #                                  Step 1: Plot the data                                  #
        ###########################################################################################
        for subgroup in subgroups:
            model     = subgroup['model']
            freeze_at = subgroup['freeze_at']
            ids       = subgroup['_member_ids']

            ######################################################################
            #  Step 1.1: Average accuracy traces for each freeze point and plot  #
            ######################################################################
            steps  = None
            values = []
            # get all accuracy traces for this model and freeze point into np array and mean over
            # runs
            for trace in get_metric_for_ids('test_accuracy', ids, per_module=False):
                steps = steps or trace['steps']
                values.append(trace['values'])

            nruns  = len(values)
            values = np.array(values).mean(axis=0)
            steps  = np.array(steps) / batches_per_epoch

            # plot data and set labels + title
            axes_acc[model].plot(steps, values, label=f'{freeze_at} ({nruns})', c=colors[freeze_at])
            axes_acc[model].set_title(f'{model}')
            axes_acc[model].set_xlabel('Epoch')

            ########################################################################################
            #  Step 1.2: Extend frozen similarity traces w last value, average over runs and plot  #
            ########################################################################################
            similarity_traces = get_layer_metric_map('self_similarity', ids)
            # Number of epochs hardcoded
            accuracy_steps    = np.arange(1, 30)
            # inifinite color map iteration; each layer has one color
            cmap              = plt.get_cmap('Set2')
            color_cycler      = iter(itertools.cycle(cmap.colors[:len(similarity_traces)]))
            # do not plot the non-frozen runs in the similarity plots as it's not very informative
            if freeze_at != 'never':
                for layer in similarity_traces:
                    data = np.zeros([len(similarity_traces[layer]), len(accuracy_steps)])
                    for i, (steps, values) in enumerate(similarity_traces[layer]):
                        # quick n dirty ways of getting percentage marker lines. they are drawn
                        # multiple times and this should be done separately after this loop
                        if freeze_at == 'percentage':
                            axes_sim[model].axvline(x=len(steps), linewidth=0.5,
                                                    linestyle=':', color='gray',
                                                    zorder=0)
                        data[i, :len(steps)] = values
                        data[i, len(steps):] = values[-1]
                    # sometimes there are correlation coefs > 1 which doesn't make sense. i have not
                    # yet investigated where those come from.
                    data[data > 1] = 1
                    label = layer.split('.')[1]
                    axes_sim[model].plot(accuracy_steps, data.mean(axis=0), label=label,
                                        c=next(color_cycler), linestyle=linestyles[freeze_at])

        ###########################################################################################
        #                                 Step 2: Create legends                                  #
        ###########################################################################################
        for ax in axes_acc.values():
            prune_labels(ax, loc='best')

        # create a few dummy entries for labeling the different line styles in the similarity plot
        legend_lines = [mlines.Line2D([], [], color='black', linestyle=style, markersize=15,
                                      label=f'{freeze_at}')
                        for freeze_at, style in linestyles.items()]
        for model, ax in axes_sim.items():
            handles, labels = prune_labels(ax, loc='lower right')
            axes_sim[model].get_legend().remove()
            axes_legend[model].legend(handles, labels, loc='upper center')
            axes_legend[model].spines['top'].set_visible(False)
            axes_legend[model].spines['right'].set_visible(False)
            axes_legend[model].spines['bottom'].set_visible(False)
            axes_legend[model].spines['left'].set_visible(False)
            axes_legend[model].xaxis.set_visible(False)
            axes_legend[model].yaxis.set_visible(False)
            axes_sim[model].legend(handles=legend_lines)

        m_str        = '_'.join(map(str.lower, models))
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
    # plot_accuracy_similarity(['VGG', 'AlexNetMini'], ['SGD'], [0.01, 0.1, 0.5], save=False)
    plot_freeze_points(['VGG', 'AlexNetMini'], ['SGD'], [0.01, 0.1, 0.5], [0.99, 0.995, 'percentage'])


if __name__ == '__main__':
    plots_for_thesis()
