import numpy as np


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
