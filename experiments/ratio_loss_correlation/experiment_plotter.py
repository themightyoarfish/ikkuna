import pymongo
import numpy as np
from experiments.sacred_utils import get_metric_for_ids
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from colors import Color
import scipy.ndimage

# obtain runs collection created by sacred
db_client = pymongo.MongoClient('mongodb://rasmus:rasmus@35.189.247.219/sacred')
sacred_db = db_client.sacred
runs      = sacred_db.runs
metrics   = sacred_db.metrics


def scatter_ratio_v_loss_decrease(models, optimizers, learning_rates, **kwargs):
    '''Plot percentage in loss decrease vs current ratio.'''
    pipeline = [
        # get only experiments which returned 0
        {'$match': {'result': 0}},
        # I know I didn't run more than 75
        {'$match': {'config.n_epochs': 75}},
        # filter models
        {'$match': {'config.model': {'$in': models}}},
        # filter opts
        {'$match': {'config.optimizer': {'$in': optimizers}}},
        # filter lrs
        {'$match': {'config.base_lr': {'$in': learning_rates}}},
        # group into groups keyed by (model, optimizer, lr)
        {'$group': {
            '_id': {
                'model': '$config.model',
                'optimizer': '$config.optimizer',
                'base_lr': '$config.base_lr',
            },
            # add field with all run ids belonging to this group
            '_member_ids': {'$addToSet': '$_id'}
        }}
    ]

    groups = list(sacred_db.runs.aggregate(pipeline))

    for group in groups:
        # create figure for group
        f         = plt.figure(figsize=kwargs.get('figsize', (12.80, 8.00)))
        ax_corr   = f.add_subplot(121, projection='3d')
        ax_loss   = f.add_subplot(222)
        ax_loss2  = ax_loss.twinx()
        ax_ratio  = f.add_subplot(224, sharex=ax_loss)
        model     = group['_id']['model']
        optimizer = group['_id']['optimizer']
        base_lr   = group['_id']['base_lr']

        # set title and labels
        f.suptitle(f'{model}: {optimizer} with lr={base_lr}')
        ax_corr.set_title('Correlation of UW-Ratio averaged over layers v loss decrease')
        ax_corr.set_xlabel('Update-Weight-Ratio')
        ax_corr.set_ylabel(r'$loss_{t+1} - loss_t$')
        ax_loss.set_title('Loss and its derivative')
        ax_ratio.set_title('UW-Ratio')
        ax_ratio.set_xlabel('Train step')
        ax_loss2.spines['right'].set_position(('outward', 0))
        ax_loss2.set_ylabel(r'$loss_{t+1} - loss_t$')

        # now get the data
        ids = group['_member_ids']
        # there is one loss trace for each id …
        loss_traces  = np.array([trace['values'] for trace in get_metric_for_ids('loss', ids)])
        # we filter out the batchnorm2d\d traces because I'm unsure if their 'weights' should be
        # included
        #               begin of string             dot slash   anything not eqal to batchnorm2d\d
        #                    |                            |-||-----------------|
        layer_ratio_regex = '^weight_updates_weights_ratio\./((?!batchnorm2d).)*$'
        # … but n_layers ratio traces for each id … so we have to average them elementwise during
        # aggregation or do it with numpy. For now, obtain a 3d-array of (nruns, nlayers, nsteps)
        # and average over the second axis
        ratio_traces = np.array([
            [trace['values'] for trace in get_metric_for_ids(layer_ratio_regex, [_id])]
            for _id in ids
        ]).mean(axis=1)     # second axis is the layer axis over which we want to average for now

        #####################################################################
        #  Apply gaussian smoothing. Not sure if this introduces artifacts  #
        #####################################################################
        if kwargs.get('filter', False):
            filter_size = kwargs.get('filter_size', 20)
            loss_traces = scipy.ndimage.filters.gaussian_filter1d(loss_traces, filter_size)
            ratio_traces = scipy.ndimage.filters.gaussian_filter1d(ratio_traces, filter_size)

        # determine size of time slice and set boundaries
        n_runs, max_step = loss_traces.shape
        start            = kwargs.get('start', 0)
        end              = kwargs.get('end', max_step)
        steps            = end - start

        # create a color sequence interpolating steps-1 values between two colors
        factors        = np.linspace(0, 1, steps-1)
        color_sequence = (Color.RED - Color.DARKBLUE) * factors + Color.DARKBLUE
        color_sequence = color_sequence.T

        # make individual plot for each group
        for i in range(n_runs):
            # inverse gradient of loss; absolute decrease in value
            loss_trace    = -np.ediff1d(loss_traces[i, start:end])
            # we use start:end-1 here since the ratio at time step t influences the loss at t+1, so
            # the final ratio does not have an associated value
            ratio_trace   = ratio_traces[i, start:end-1]
            # compute color with alpha proportional to run index, for visual clarity
            shaded_blue   = np.array(Color.DARKBLUE).squeeze() * [1, 1, 1, 0.7**i]
            shaded_yellow = np.array(Color.YELLOW).squeeze() * [1, 1, 1, 0.7**i]
            shaded_slate  = np.array(Color.SLATE).squeeze() * [1, 1, 1, 0.7**i]

            # 3d scatter with time step as z height
            ax_corr.scatter(ratio_trace, loss_trace, np.arange(start, end-1), s=0.33,
                            c=color_sequence)
            # plot loss and decrease
            ax_loss.plot(np.arange(start, end), loss_traces[i, start:end], color=shaded_yellow)
            ax_loss2.plot(np.arange(start, end-1), loss_trace, color=shaded_slate, linewidth=0.8)

            # plot update ratios
            ax_ratio.plot(np.arange(start, end-1), ratio_trace, color=shaded_blue, linewidth=0.8)

    plt.show()


if __name__ == '__main__':
    scatter_ratio_v_loss_decrease(['VGG'], ['SGD'], [0.01, 0.05, 0.1], end=15000)
