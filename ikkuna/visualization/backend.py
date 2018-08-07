from tensorboardX import SummaryWriter
import abc
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class Backend(abc.ABC):

    def __init__(self, title=None):
        self._title = title

    @abc.abstractmethod
    def add_data(self, module, datum, step):
        pass

    @abc.abstractmethod
    def add_histogram(self, module, datum, step):
        pass

    @property
    def title(self):
        return self._title


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # noqa
import numpy as np
from ikkuna.utils import make_fill_polygons


class UpdatableHistogram(object):

    def __init__(self, figure, subplot_conf=111, title='', max_hists=10,
                 basecolor=(199/255, 64/255, 24/255)):
        if isinstance(subplot_conf, int):
            subplot_conf = (subplot_conf,)
        self._max_hists = max_hists
        self._unit = 1 / max_hists
        self._fig = figure
        self._ax = self._fig.add_subplot(*subplot_conf, projection='3d')
        self._base_r, self._base_g, self._base_b = basecolor
        self._hists = []
        self._title = title
        self._xlabel = ''
        self._zlabel = 'frequency'
        self._prepare_plot()

    def _prepare_plot(self):
        transparent = (1, 1, 1, 0)
        self._ax.set_title(self._title)
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_zlabel(self._zlabel)
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
        self._ax.clear()
        self._prepare_plot()
        for i, (hist, edges) in enumerate(reversed(self._hists)):
            xpos = edges[:-1]
            ypos = i * self._unit * np.ones_like(edges[:-1])
            height = hist

            percent = i / len(self._hists)
            color = (self._base_r + percent * (1.0 - self._base_r),
                     self._base_g + percent * (1.0 - self._base_g),
                     self._base_b + percent * (1.0 - self._base_b))
            self._ax.plot(xpos, ypos, height, color='black', linewidth=0)
            collection = make_fill_polygons(xpos, ypos, height)
            collection.set_facecolor(color)
            collection.set_edgecolor(color)
            self._ax.add_collection3d(collection)

    def add_data(self, X):
        import torch
        X = torch.cat(X).detach().cpu().numpy()
        hist, edges = np.histogram(X, bins=50, density=True)
        self._hists.append((hist, edges))
        if len(self._hists) > self._max_hists:
            self._hists.pop(0)


class MPLBackend(Backend):

    def __init__(self, **kwargs):
        super().__init__(kwargs.get('title'))

        ############################
        #  Line plot/general args  #
        ############################
        self._xlabel        = kwargs.get('xlabel')
        self._ylabel        = kwargs.get('ylabel')
        self._data_name     = kwargs.get('data_name')
        self._metric_values = {}
        self._ylims         = kwargs.get('ylims')
        self._counter       = 0
        self._figure        = self._ax = None
        self._plots         = None
        self._axes          = None

        ####################
        #  Histogram args  #
        ####################
        # self._nbins       = kwargs.get('nbins')
        # self._min         = kwargs.get('min')
        # self._max         = kwargs.get('max')
        # assert max > min, 'Fuck you'
        self._buffer      = defaultdict(list)
        self._buffer_lim  = kwargs.get('buffer_size', 100)
        # default           = (0, np.zeros(self._nbins, dtype=np.int64))    # counter + hist
        # self._hist        = defaultdict(lambda: default)

    def _prepare_axis(self, ax):
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

    @property
    def data_name(self):
        return self._data_name

    @data_name.setter
    def data_name(self, descr):
        assert len(descr) > 0
        self._data_name = descr

    def _reflow_plots(self):
        nplots = len(self._axes)
        h = int(np.floor(np.sqrt(nplots) * 2 + 0.5))
        w = int(np.ceil(np.sqrt(nplots) / 2))
        assert h * w >= nplots
        for i, (module, axis) in enumerate(self._axes.items()):
            if isinstance(axis, UpdatableHistogram):
                axis = axis.ax
            axis.change_geometry(h, w, i + 1)

    def add_histogram(self, module, datum, step):

        if not self._figure:
            self._figure = plt.figure(figsize=(8, 20))
            self._figure.suptitle(self.title)
            self._axes = {}

        if module not in self._axes:
            nplots = len(self._axes)
            self._axes[module] = UpdatableHistogram(self._figure, subplot_conf=(nplots + 1, 1, 1),
                                                    title=module.name)

            self._reflow_plots()

        self._buffer[module].append(datum)

        if len(self._buffer[module]) == self._buffer_lim:
            self._axes[module].add_data(self._buffer[module])
            self._axes[module].replot()
            self._buffer[module] = []

            self._figure.canvas.draw()
            self._figure.canvas.flush_events()
            self._figure.show()

        # h, w = (int(np.floor(np.sqrt(n_modules))), int(np.ceil(np.sqrt(n_modules))))

        # figure, axarr = plt.subplots(h, w)
        # figure.suptitle(f'{self.title}')

        # for i in range(h):
        #     for j in range(w):
        #         index = h * i + j

        #         ax = axarr[i][j]
        #         ax.clear()

        #         ax.set_title(str(module))
        #         ax.set_yscale('log')
        #         ax.hist()
        #         ax.grid(True)
        #         ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

        # figure.subplots_adjust(hspace=1, wspace=1)
        # figure.show()

    def add_data(self, module, datum, step):

        if not self._figure:
            self._figure, self._ax  = plt.subplots()
            self._plots = {}
            self._prepare_axis(self._ax)

        if module not in self._plots:
            self._plots[module] = self._ax.plot([], [], label=f'{module}')[0]

        xdata = self._plots[module].get_xdata()
        ydata = self._plots[module].get_ydata()
        xdata = np.append(xdata, 1 if len(xdata) == 0 else xdata[-1] + 1)
        ydata = np.append(ydata, datum)

        self._plots[module].set_xdata(xdata)
        self._plots[module].set_ydata(ydata)
        self._ax.legend(ncol=2)

        if self._counter % 50 == 0:
            self._ax.relim()
            if self._ylims:
                self._ax.set_ylim(self._ylims)
            self._ax.autoscale_view(scaley=True)

            # redraw the figure
            self._figure.canvas.draw()
            self._figure.canvas.flush_events()
            self._figure.show()
            self._counter = 0

        self._counter += 1


class TBBackend(Backend):

    def __init__(self, **kwargs):
        super().__init__(kwargs.get('title'))
        self._writer = SummaryWriter()

    def add_data(self, module, datum, step):
        # Unfortunately, xlabels, ylabels and plot titles are not supported
        self._writer.add_scalars(f'{self.title}', {module: datum}, global_step=step)

    def add_histogram(self, module, datum, step):
        self._writer.add_histogram(f'{self.title}: {module.name}', datum, global_step=step)
