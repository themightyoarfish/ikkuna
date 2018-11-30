import pymongo
import numpy as np
from experiments.sacred_utils import get_metric_for_ids
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


def inverse_probability_for_sequence(ary, bins=50):
    '''Compute a probability vector from the histogram for subsampling a sequence according to
    density.'''
    # create a probability distribution from the histogram for subsampling the 3d plot. we
    # sample from the indices 0…steps-1 according to the inverse probability given by the ratio
    # histogram. this means more points are removed where the density is higher.
    count, bin_edges = np.histogram(ary, bins=bins)
    # get index of each bin. -1 because the leftmost one somehow doesn't count. I don't understand
    # np.digitize
    bin_indices = np.digitize(ary, bin_edges, right=False) - 1
    # put everything beyond the last bin into last. I think this happens for ary.max() because
    # np.hist has a right-closed last bin whereas all the others are half-open on the right
    bin_indices[bin_indices == bins] = bins - 1
    # probability distribution / frquency
    p = count[bin_indices].astype('float')
    # normalize
    p /= p.sum()
    # invert
    p = p[::-1]
    return p


def scatter_ratio_v_loss_decrease(models, optimizers, learning_rates, **kwargs):
    '''Plot percentage in loss decrease vs current ratio.'''
    pipeline = [
        # get only experiments which returned 0
        {'$match': {'result': 0}},
        # I know I didn't run more than 75
        {'$match': {'config.n_epochs': kwargs.get('n_epochs', 75)}},
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
    figures = dict()
    samples = kwargs.get('samples', 3000)

    for group in groups:
        # create figure for group
        f         = plt.figure(figsize=kwargs.get('figsize', (12.80, 8.00)))
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
            filter_size  = kwargs.get('filter_size', 20)
            loss_traces  = scipy.ndimage.filters.gaussian_filter1d(loss_traces, filter_size)
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
            loss_trace     = loss_traces[i, start:end]
            loss_trace_inv = -np.ediff1d(loss_trace)
            # we use start:end-1 here since the ratio at time step t influences the loss at t+1, so
            # the final ratio does not have an associated value
            ratio_trace    = ratio_traces[i, start:end-1]

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

            all_x = np.arange(start, end)[indices]
            all_x_but_last = np.arange(start, end-1)[indices]
            ratio_trace = ratio_trace[indices]
            loss_trace = loss_trace[indices]
            loss_trace_inv = loss_trace_inv[indices]

            # compute color with alpha proportional to run index, for visual clarity
            multiplier    = [1, 1, 1, 0.7**i]
            shaded_blue   = np.array(Color.DARKBLUE).squeeze() * multiplier
            shaded_yellow = np.array(Color.YELLOW).squeeze() * multiplier
            shaded_slate  = np.array(Color.SLATE).squeeze() * multiplier

            # 3d scatter with time step as z height
            ax_corr.scatter(ratio_trace,
                            loss_trace_inv,
                            all_x_but_last,
                            s=0.5,
                            c=color_sequence[indices, :],
                            depthshade=False)
            # plot loss and decrease
            ax_loss.plot(all_x, loss_trace, color=shaded_yellow)
            ax_loss2.plot(all_x_but_last, loss_trace_inv, color=shaded_slate, linewidth=0.8)

            # plot update ratios
            ax_ratio.plot(all_x_but_last, ratio_trace, color=shaded_blue, linewidth=0.8)

        loss_patch          = mpatches.Patch(color=Color.YELLOW.hex(), label='Loss')
        loss_decrease_patch = mpatches.Patch(color=Color.SLATE.hex(), label='Loss decrease')
        ax_loss.legend(handles=[loss_patch, loss_decrease_patch], loc='upper right')
        ax_loss2.yaxis.label.set_color(Color.SLATE.hex())

        figures[f'{model.lower()}_{optimizer.lower()}_{str(base_lr).replace(".","")}.pdf'] = f

    def unify_limits(axes, x=True, y=True):
        # Iterate over all limits in the plots to give all the same axis limits
        if x:
            xlims   = [ax.get_xlim() for ax in axes]
            lower_x = min(x[0] for x in xlims)
            upper_x = max(x[1] for x in xlims)
        if y:
            ylims   = [ax.get_ylim() for ax in axes]
            lower_y = min(y[0] for y in ylims)
            upper_y = max(y[1] for y in ylims)
        for ax in axes:
            if x:
                ax.set_xlim((lower_x, upper_x))
            if y:
                ax.set_ylim((lower_y, upper_y))

    unify_limits(ax_corrs)
    unify_limits(ax_losses, x=False)
    unify_limits(ax_losses2, x=False)
    unify_limits(ax_ratios, x=False)

    if not save:
        plt.show()
    else:
        for name, f in figures.items():
            f.savefig(name)


if __name__ == '__main__':
    scatter_ratio_v_loss_decrease(['VGG'], ['SGD'], [0.01, 0.05, 0.1], n_epochs=75, filter=True,
                                  save=True, end=10000, samples=5000, subsample='log')
