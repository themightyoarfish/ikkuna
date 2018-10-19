from tensorboardX import SummaryWriter
import abc
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D     # noqa
from ikkuna.utils import make_fill_polygons


class Backend(abc.ABC):
    '''Base class for visualiation backends. :class:`~ikkuna.export.subscriber.Subscriber`\ s use
    this class to dispatch their metrics to have them visualised.

    Attributes
    ----------
    title   :   str
                The figure title
    '''

    def __init__(self, title):
        '''
        Parameters
        ----------
        title   :   str
                    Title to use for the figure.
        '''
        self._title = title

    @abc.abstractmethod
    def add_data(self, module_name, datum, step):
        '''Display scalar data (i.e. a line plot)

        Parameters
        ----------
        module_name  :  str
                        Name of module which emitted the data
        datum   :   torch.Tensor
                    Payload
        step    :   int
                    Global step
        '''
        pass

    @abc.abstractmethod
    def add_histogram(self, module_name, datum, step):
        '''Display histogram data (i.e. a line plot)

        Parameters
        ----------
        module_name  :  str
                        Name of module which emitted the data
        datum   :   torch.Tensor
                    Payload, not the histogram itself
        step    :   int
                    Global step
        '''
        pass

    @property
    def title(self):
        return self._title


class UpdatableHistogram(object):
    '''A utility class to wrap a single histogram subplot of a figure which can display up to a
    number of histograms. New data displaces old one. This emulates tensorboard histogram traces
    with matplotlib.

    Attributes
    ----------
    _max_hists  :   int
                    Max number of histograms to display before discarding old ones
    _unit   :   float
                Base length used for spacing successive histograms along the y axis
    _fig    :   matplotlib.figure.Figure
                Figure to put subplot in
    _ax :   mpl_toolkits.mplot3d.axes3d.Axes3D
            3d axes object to plot in
    _base_r :   float
                start red value
    _base_g :   float
                start green value
    _base_b :   float
                start blue value
    _hists  :   list
                Stack of currently displayed histograms
    _title  :   str
                Plot title
    '''

    def __init__(self, figure, subplot_conf=111, title='', max_hists=10,
                 basecolor=(199/255, 64/255, 24/255)):
        '''
        Parameters
        ----------
        figure  :   matplotlib.figure.Figure
                    Figure to add subplot to
        subplot_conf    :   int or tuple
                            Subplot configuration. See :func:`matplotlib.pyplot.subplots`
        title   :   str
                    Subplot title
        max_hists   :   int
                        Number of histograms to keep
        basecolor   :   tuple
                        RGB tuple denoting the color of the most recent histogram.
        '''
        if isinstance(subplot_conf, int):
            subplot_conf = (subplot_conf,)
        self._max_hists = max_hists
        self._unit      = 1 / max_hists
        self._fig       = figure
        self._ax        = self._fig.add_subplot(*subplot_conf, projection='3d')
        self._hists     = []
        self._title     = title
        self._base_r, self._base_g, self._base_b = basecolor
        self._prepare_plot()

    def _prepare_plot(self):
        '''Prepare the axes object for (re)plotting. This removes all unnecessary elements and
        assigns labels.'''

        transparent = (1, 1, 1, 0)
        self._ax.set_title(self._title)
        self._ax.set_xlabel('')
        self._ax.set_zlabel('frequency')
        self._ax.get_yaxis().line.set_color(transparent)
        self._ax.set_yticks([])
        self._ax.grid(False)
        self._ax.w_xaxis.set_pane_color(transparent)
        self._ax.w_yaxis.set_pane_color(transparent)
        self._ax.w_zaxis.set_pane_color(transparent)

    @property
    def figure(self):
        return self._fig

    @property
    def ax(self):
        return self._ax

    def replot(self):
        '''Recompute all visible plots. This will remove all current plots and simply replot
        ``self._hists``.

        .. note::

            This is not the most efficient way to update the plot maybe; the proper way to do this
            could involve moving all current histograms one step backwards, unless the x axis
            changes.
        '''
        self._ax.clear()
        self._prepare_plot()
        for i, (hist, edges) in enumerate(reversed(self._hists)):   # last is most recent
            xpos   = edges[:-1]     # take start edge of every bin for x position
            # TODO: For uniform bins, use mean of bin for xpos
            ypos   = i * self._unit * np.ones_like(edges[:-1])      # move according to recency
            height = hist

            # color gets faded for older hists
            percent = i / len(self._hists)
            color = (self._base_r + percent * (1.0 - self._base_r),
                     self._base_g + percent * (1.0 - self._base_g),
                     self._base_b + percent * (1.0 - self._base_b))
            self._ax.plot(xpos, ypos, height, color='black', linewidth=0)

            # fill below the plot with polygons
            collection = make_fill_polygons(xpos, ypos, height)
            collection.set_facecolor(color)
            collection.set_edgecolor(color)
            self._ax.add_collection3d(collection)

    def add_data(self, X):
        '''Add data for a new histogram. Currently, the number of bins is fixed at 50. Old data is
        deleted.

        Parameters
        ----------
        X   :   list
                Arbitrary sequence of tensors to merge for a histogram
        '''
        import torch
        X           = torch.cat(X).detach().cpu().numpy()
        hist, edges = np.histogram(X, bins=50, density=True)
        self._hists.append((hist, edges))
        if len(self._hists) > self._max_hists:
            self._hists.pop(0)


class MPLBackend(Backend):
    '''Matplotlib backend (use in Jupyter with ``%matplotlib inline`` or via X-forwarding over ssh
    [barely useable])

    Attributes
    ----------
    _xlabel :   str
                X axis label for line plots
    _ylabel :   str
                Y axis label for line plots
    _ylims  :   tuple
                Limits of the y axis for line plots
    _redraw_counter :   int
                        Number of datapoints to consume before redrawing the figure
    _plots  :   dict
                module-plot mapping
    _axes   :   dict
                module-UpdatableHistogram mapping (this should be refactored)
    _buffer :   dict
                Per-module buffer of tensors for more reliable histograms
    _buffer_lim :   int
                    Size of the buffer
    '''

    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        xlabel  :   str
        ylabel  :   str
        ylims   :   tuple
        buffer_lim  :   int
                        Buffer size for more reliable histograms
        '''
        super().__init__(kwargs.get('title'))

        ############################
        #  Line plot/general args  #
        ############################
        self._xlabel         = kwargs.get('xlabel')
        self._ylabel         = kwargs.get('ylabel')
        self._ylims          = kwargs.get('ylims')
        self._redraw_counter = 0
        self._figure         = self._ax = None
        self._plots          = None
        self._axes           = None

        ####################
        #  Histogram args  #
        ####################
        self._buffer      = defaultdict(list)
        self._buffer_lim  = kwargs.get('buffer_size', 20)

    def _prepare_axis(self, ax):
        '''Prepare the line plot axis with labels and scaling.'''
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_autoscaley_on(True)

    @Backend.title.setter
    def title(self, title):
        assert len(title) > 0
        self._title = title

    @property
    def xlabel(self):
        return self._xlabel

    @xlabel.setter
    def xlabel(self, label):
        self._xlabel = label

    @property
    def ylabel(self):
        return self._ylabel

    @ylabel.setter
    def ylabel(self, label):
        self._ylabel = label

    def _reflow_plots(self):
        '''Reqorganize the histogram subplots into a rectangular shape. For now, the sublots are
        arranged on a grid twice as high as it is wide, since vertical space is often in
        abundance.

        .. note::

            This does not recompute the plots
        '''
        nplots = len(self._axes)
        h = int(np.floor(np.sqrt(nplots) * 2 + 0.5))    # remove the factor 2 for square grid
        w = int(np.ceil(np.sqrt(nplots) / 2))
        assert h * w >= nplots
        for i, (module, axis) in enumerate(self._axes.items()):
            if isinstance(axis, UpdatableHistogram):
                axis = axis.ax      # TODO: make less ugly
            axis.change_geometry(h, w, i + 1)

    def add_histogram(self, module_name, datum, step):

        if not self._figure:
            # first time? initialise
            self._figure = plt.figure(figsize=(8, 20))
            self._figure.suptitle(self.title)
            self._axes = {}

        if module_name not in self._axes:
            # haven't seen this module_name before? make new plot for it
            nplots = len(self._axes)
            self._axes[module_name] = UpdatableHistogram(self._figure,
                                                         subplot_conf=(nplots + 1, 1, 1),
                                                         title=module_name)

            self._reflow_plots()

        self._buffer[module_name].append(datum)

        if len(self._buffer[module_name]) == self._buffer_lim:   # buffer full
            self._axes[module_name].add_data(self._buffer[module_name])
            self._axes[module_name].replot()
            self._buffer[module_name] = []

            self._figure.canvas.draw()
            self._figure.canvas.flush_events()
            self._figure.show()

    def add_data(self, module_name, datum, step):

        if not self._figure:
            self._figure, self._ax = plt.subplots()
            self._plots            = {}
            self._prepare_axis(self._ax)

        if module_name not in self._plots:
            # empty plot so we can simply set the data later
            self._plots[module_name] = self._ax.plot([], [], label=f'{module_name}')[0]

        xdata = self._plots[module_name].get_xdata()
        ydata = self._plots[module_name].get_ydata()
        xdata = np.append(xdata, 1 if len(xdata) == 0 else xdata[-1] + 1)
        ydata = np.append(ydata, datum)

        self._plots[module_name].set_xdata(xdata)
        self._plots[module_name].set_ydata(ydata)
        self._ax.legend(ncol=2)

        if self._redraw_counter % 50 == 0:      # TODO: Make modulus a parameter
            self._ax.relim()
            if self._ylims:
                self._ax.set_ylim(self._ylims)
            self._ax.autoscale_view(scaley=True)

            # redraw the figure
            self._figure.canvas.draw()
            self._figure.canvas.flush_events()
            self._figure.show()
            self._redraw_counter = 0

        self._redraw_counter += 1


import functools


# use lru cache to generate nwe result only once per experiment run
# TODO: remove this hack
@functools.lru_cache(None)
def determine_run_index(log_dir):
    import os
    if not os.path.exists(log_dir):
        return 0
    else:
        subdirs = os.listdir(log_dir)
        new_index = len(subdirs)
        return new_index


prefix = None


def configure_prefix(p):
    '''Set a prefix to the log directory for Tensorboard.

    Parameters
    ----------
    p   :   str
            Prefix to the directory name. ``_runs`` will be appended by :class:`TBBackend`
    '''
    global prefix
    prefix = p

def set_run_info(info):
    TBBackend.info = info


class TBBackend(Backend):
    '''Tensorboard backend.

    .. note::

        Whitespace and punctuation in the ``title`` will be replaced
        with underscores due to the fact that it becomes part of a file name.

    Attributes
    ----------
    _writer :   tensorboardX.SummaryWriter
    hist_bins   :   int
                    Number of bins to use for histograms
    '''

    # TODO: make printing metadata non-hacky
    info : str = ''

    def __init__(self, **kwargs):
        super().__init__(kwargs.pop('title'))
        # remove args incompatible with TB
        kwargs.pop('xlabel', None)
        kwargs.pop('ylabel', None)
        kwargs.pop('ylims', None)
        self._hist_bins = kwargs.pop('bins', 50)
        log_dir         = kwargs.pop('log_dir', 'runs' if not prefix else prefix)
        index           = determine_run_index(log_dir)
        log_dir         = f'{log_dir}/run{index}'
        self._writer    = SummaryWriter(log_dir, **kwargs)
        self._writer.add_text('run_conf', TBBackend.info)

    def add_data(self, module_name, datum, step):
        # Unfortunately, xlabels, ylabels and plot titles are not supported
        self._writer.add_scalars(f'{self.title}', {module_name: datum}, global_step=step)

    def add_histogram(self, module_name, datum, step):
        self._writer.add_histogram(f'{self.title}: {module_name}', datum, global_step=step,
                                   bins=self._hist_bins)
