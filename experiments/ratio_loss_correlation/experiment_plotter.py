import pymongo
import numpy as np
from experiments.sacred_utils import get_metric_for_ids
from experiments.utils import unify_limits
import matplotlib
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


def scatter_ratio_v_loss_decrease(models, optimizers, learning_rates, **kwargs):
    '''Plot percentage in loss decrease vs current ratio, plus other stuffs.'''

    conditions = [
        # get only experiments which returned 0
        {'$match': {'result': 0}},
        {'$match': {'config.n_epochs': kwargs.get('n_epochs', 30)}},
        {'$match': {'config.batch_size': 128}},
        # filter models
        {'$match': {'config.model': {'$in': models}}},
        # filter opts
        {'$match': {'config.optimizer': {'$in': optimizers}}},
        # filter lrs
        {'$match': {'config.base_lr': {'$in': learning_rates}}},
    ]

    if 'schedule' in kwargs:
        conditions.append({'$match': {'config.schedule': kwargs['schedule']}})

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

    from mpl_toolkits.mplot3d import Axes3D     # noqa
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ax_corrs   = []
    ax_losses  = []
    ax_losses2 = []
    ax_ratios  = []
    ax_lrs     = []
    figures = dict()
    samples = kwargs.get('samples', 3000)

    for group in groups:
        # create figure for group
        f         = plt.figure(figsize=kwargs.get('figsize', (0.8 * 12.80, 0.8 * 8.00)))
        ax_corr   = f.add_subplot(121, projection='3d')
        ax_corrs.append(ax_corr)
        ax_loss   = f.add_subplot(222)
        ax_losses.append(ax_loss)
        ax_loss2  = ax_loss.twinx()
        ax_losses2.append(ax_loss2)
        ax_ratio  = f.add_subplot(224, sharex=ax_loss)
        ax_ratios.append(ax_ratio)
        model     = group['_id']['model']
        optimizer = group['_id']['optimizer']
        base_lr   = group['_id']['base_lr']

        # set title and labels
        ax_corr.set_title('Correlation of UW-Ratio averaged\nover layers v loss decrease')
        ax_corr.set_xlabel('Update-Weight-Ratio')
        ax_corr.set_ylabel(r'$loss_{t+1} - loss_t$')
        ax_loss.set_title('Loss and inverse derivative')
        ax_ratio.set_title('UW-Ratio')
        ax_ratio.set_xlabel('Train step')
        ax_loss2.spines['right'].set_position(('outward', 0))
        ax_loss2.set_ylabel(r'$loss_{t+1} - loss_t$')

        ######################################################################################
        #  We get the data for each run, but average over the runs to smooth out some noise  #
        ######################################################################################
        # now get the data
        ids = group['_member_ids']
        # there is one loss trace for each id …
        loss_trace  = np.array([
            trace['values'] for trace in get_metric_for_ids('loss', ids)
        ]).mean(axis=0)
        # we filter out the batchnorm2d\d traces because I'm unsure if their 'weights' should be
        # included
        #               begin of string             dot slash   anything not eqal to batchnorm2d\d
        #                    |                            |-||-----------------|
        layer_ratio_regex = '^weight_updates_weights_ratio\./((?!batchnorm2d).)*$'
        # … but n_layers ratio traces for each id … so we have to average them elementwise during
        # aggregation or do it with numpy. For now, obtain a 3d-array of (nruns, nlayers, nsteps)
        # and average over the second axis
        ratio_traces = [
            [trace['values'] for trace in get_metric_for_ids(layer_ratio_regex, [_id])]
            for _id in ids
        ]
        ratio_array = np.array(ratio_traces)
        # second axis is the layer axis over which we want to average for now
        ratio_trace = ratio_array.mean(axis=(0, 1))

        # get the learning rates
        lr_traces = np.array([
            trace['values'] for _id in ids for trace in get_metric_for_ids('learning_rate', [_id])
        ])
        if lr_traces.size > 0:
            lr_trace = lr_traces.mean(axis=0)
            plot_lr  = True
            ax_lr    = ax_ratio.twinx()
            ax_lrs.append(ax_lr)
            ax_ratio.set_title('UW-Ratio and LR')
            ax_lr.spines['right'].set_position(('outward', 0))
            ax_lr.set_ylabel('Learning Rate')
        else:
            plot_lr = False

        #####################################################################
        #  Apply gaussian smoothing. Not sure if this introduces artifacts  #
        #####################################################################
        if kwargs.get('filter', False):
            filter_size  = kwargs.get('filter_size', 20)
            loss_trace   = scipy.ndimage.filters.gaussian_filter1d(loss_trace, filter_size)
            ratio_trace  = scipy.ndimage.filters.gaussian_filter1d(ratio_trace, filter_size)
            if plot_lr:
                lr_trace = scipy.ndimage.filters.gaussian_filter1d(lr_trace, filter_size)

        # determine size of time slice and set boundaries
        max_step = len(loss_trace)
        start    = kwargs.get('start', 0)
        end      = kwargs.get('end', max_step)
        steps    = end - start

        # create a color sequence interpolating steps-1 values between two colors
        factors        = np.linspace(0, 1, steps-1)
        color_sequence = (Color.RED - Color.DARKBLUE) * factors + Color.DARKBLUE
        color_sequence = color_sequence.T

        # inverse gradient of loss; absolute decrease in value
        loss_trace     = loss_trace[start:end]
        loss_trace_inv = -np.ediff1d(loss_trace)
        # we use start:end-1 here since the ratio at time step t influences the loss at t+1, so
        # the final ratio does not have an associated value
        ratio_trace    = ratio_trace[start:end-1]
        if plot_lr:
            lr_trace   = lr_trace[start:end-1]

        # this didn't end up working
        # p = inverse_probability_for_sequence(ratio_trace)
        # indices = np.random.choice(np.arange(len(ratio_trace)), size=samples, replace=False, p=p)
        # indices.sort()
        # since I know where the density is higher (at least for vgg), I can generate log spaced
        # indices instead. this will fail on arbitrary distributions, of course
        if kwargs.get('subsample', False):
            subsample = kwargs['subsample']
            if subsample == 'log':
                indices = np.unique(np.geomspace(1, steps-1, num=samples).astype(int)) - 1
            else:
                indices = np.unique(np.linspace(0, steps-2, num=samples).astype(int))

        else:
            indices = np.arange(0, steps-1)

        all_x          = np.arange(start, end)[indices]
        all_x_but_last = np.arange(start, end-1)[indices]
        ratio_trace    = ratio_trace[indices]
        loss_trace     = loss_trace[indices]
        loss_trace_inv = loss_trace_inv[indices]
        if plot_lr:
            lr_trace   = lr_trace[indices]

        blue   = np.array(Color.DARKBLUE).squeeze()
        yellow = np.array(Color.YELLOW).squeeze()
        slate  = np.array(Color.SLATE).squeeze()
        red    = np.array(Color.RED).squeeze()

        # 3d scatter with time step as z height
        ax_corr.scatter(ratio_trace,
                        loss_trace_inv,
                        all_x_but_last,
                        s=0.5,
                        c=color_sequence[indices, :],
                        depthshade=False)
        # plot loss and decrease
        ax_loss.plot(all_x, loss_trace, color=yellow)
        ax_loss2.plot(all_x_but_last, loss_trace_inv, color=slate, linewidth=0.8)

        if plot_lr:
            ax_lr.plot(all_x_but_last, lr_trace, color=red, linewidth=0.8)
        # plot update ratios
        ax_ratio.plot(all_x_but_last, ratio_trace, color=blue, linewidth=0.8)

        # we could instead use label='…' in the plot calls and fetch the labels via
        # get_handles_and_labels, but for now this is easier.
        # make legend for loss plot
        loss_patch          = mpatches.Patch(color=Color.YELLOW.hex(), label='Loss')
        loss_decrease_patch = mpatches.Patch(color=Color.SLATE.hex(), label='Loss decrease')
        ax_loss.legend(handles=[loss_patch, loss_decrease_patch], loc='upper right')
        ax_loss2.yaxis.label.set_color(Color.SLATE.hex())

        # make legend for ratio/lr plot
        ratio_patch = mpatches.Patch(color=Color.DARKBLUE.hex(), label='UW-Ratio')
        handles = [ratio_patch]
        if plot_lr:
            lr_patch    = mpatches.Patch(color=Color.RED.hex(), label='LR')
            handles.append(lr_patch)
            ax_lr.yaxis.label.set_color(Color.RED.hex())
        ax_ratio.legend(handles=handles, loc='upper right')

        # save figure for serialization
        m_str        = model.lower()
        o_str        = optimizer.lower()
        lr_str       = str(base_lr).replace('.', '')
        key          = f'{m_str}_{o_str}_{lr_str}_{start}_{end}'
        key         += '_' + kwargs['schedule'] if 'schedule' in kwargs else ''
        figures[key] = f + '.pdf'

    # unify_limits(ax_corrs)
    unify_limits(ax_losses, x=False)
    unify_limits(ax_losses2, x=False)
    # unify_limits(ax_ratios, x=False)
    # if ax_lrs:
    #     unify_limits(ax_lrs, x=False)

    if not save:
        plt.show()
    else:
        for name, f in figures.items():
            f.savefig(name)


if __name__ == '__main__':
    scatter_ratio_v_loss_decrease(['VGG'], ['SGD'], [0.001, 0.01, 0.05, 0.1], n_epochs=30,
                                  filter=True, save=True, subsample='linear', samples=5000,
                                  schedule='ratio_adaptive_schedule_fn')
