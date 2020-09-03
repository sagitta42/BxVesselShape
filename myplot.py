''' Class for convenient and pretty plotting '''

import matplotlib
import matplotlib.pyplot as plt
import sys
# import numpy as np
import pandas as pd

# import seaborn as sns
# sns.set_palette('colorblind')

save = 'save' in sys.argv

class Plot:
    def __init__(self,figsize,n=1,sharex=True):
        '''
        figsize [(w,h)]: tuple with figure width x height
        n [int]: number of subplots

        '''
        fig, ax = plt.subplots(n,figsize=figsize, sharex=sharex)
        self.fig = fig
         # convenient if only have one subplot (not ty type axes[0] all the time)
        self.ax = ax
        # array of all our axes; new axes get added here too (e.g. two subplots, one has 2 y axes, means in total 5 axes in the list)
        self.axes = ax if n > 1 else [ax]


    def legend(self, out=False, ncol=1, pos=None, title=None, labels=None, axis=None):
        '''
        Create a customized legend.

        out [True|False]: legend outside of the plot in the bottom (default False)
        ncol [int]: number of columns in the legend (default 1)
        pos [list of int|string]: position of the legend (default "best").
            Important for plotting with a second Y axis (see below)
        '''

        if out:
            # draw legend at the bottom
            self.axes[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),\
                fancybox=True, shadow=False, ncol=ncol,fontsize=15)
            # remove legends from all other subplots
            for ax in self.axes[:-1]:
                legend = ax.legend()
                legend.remove()
        else:
            # draw legend on each axis
            # in case these are not subplots, but one plot with two y axes, legends might crash
            # -> choose given location
            irange = range(len(self.axes)) if axis == None else [axis]
            for i in irange:
                loc = 'best'
                if not pos == None:
                    loc = pos[i] if type(pos) == list else pos
                self.axes[i].legend(ncol=ncol, loc=loc, title=title, labels=labels)


    def pretty(self, large=3, stretch = None, grid='major'):
        '''
        Make the plot pretty.

        large [int]: controls fontsizes (default 3)
        stretch [None|"float"|"year"] change x axis range (default None).
            Useful for cases when the automatic ranges makes the leftmost and
        rightmost point fall right at the frame and get "eaten up"
            None: do not do anything
            "float": increase the range by 10%
            "year": increase the range by 1

        grid ["major"|"minor"]: only major grid, or also minor;
            same as the grid argument of the function ax.grid()
        '''

        for ax in self.axes:
            ## make better x and y limits
            if stretch:
                xlim = list(ax.get_xlim())

                ylim = list(ax.get_ylim())
                if stretch == 'float':
                    strength = 0.7
                    xlim_orig = xlim[0]
                    xlim[0] = - xlim[1] * strength if xlim[0] == 0 \
                        else xlim[0] *(1 - strength)
                    xlim[1] = xlim[1] * (1 - strength) if xlim_orig == 0\
                        else xlim[1] + xlim_orig * strength
                elif stretch == 'year':
                    xlim[0] = xlim[0] - 1
                    xlim[1] = xlim[1] + 1


                ylim[0] = - ylim[1]*0.1 if ylim[0] == 0 else ylim[0]*0.9

                ax.set_xlim(xlim[0], xlim[1])
                ax.set_ylim(ylim[0], ylim[1]*1.1)

            ## increase tick sizes (numbers)
            for t in ax.get_xaxis().get_ticklabels():
                t.set_fontsize(15 + large)
            for t in ax.get_yaxis().get_ticklabels():
                t.set_fontsize(15 + large)

            ## increase title sizes
            ax.get_xaxis().get_label().set_fontsize(17 + large)
            ax.get_yaxis().get_label().set_fontsize(17 + large)
            ax.title.set_fontsize(20 + large)

            ## increase legend font
            legend = ax.get_legend()
            if legend:
                if legend.get_title(): legend.get_title().set_fontsize(19 + large)
                for t in legend.get_texts():
                    t.set_fontsize(17 + large)

            ## add gridlines
            if grid: ax.grid(linestyle='--',zorder=0, which=grid)

        ## increase fontsize of texts on the figure
        for t in self.fig.get_children():
            if type(t) == matplotlib.text.Text:
                t.set_fontsize(20)



        ## remove white borders around the plot
        self.fig.tight_layout(rect=[0,0,1,0.97])


    def figure(self, name=None):
        '''
        Show the plot or save an image.

        name [string]: name of the figure to be saved (with extension)
        '''
        #self.pretty(2)
        print 'Image:', name

        if save:
            if name:
                self.fig.savefig(name)
                print '(saved)'
            else:
                print "You didn't provide an image name!"
        else:
            print '(NOT SAVED)'
            plt.show()


    def add_axis(self, col=None, ax=None):
        '''
        Add a second Y axis.

        col [string]: colour of the axis. Can be useful to differentiate
            the curves and see which one belongs to which Y axis
        ax [None|pyplot Axis]: axis to add. If None, a second axis is created
            automatically as a twin of the first axis. In case of two subplots,
            it will create a second Y axis only for the first subplot.
            So if you want to add two Y axis for both subplots, create them
            "externally" and add by passing them to this function.
        '''

        ax2 = ax if ax else self.ax.twinx()
        if col: ax2.tick_params(axis='y', colors=col)
        self.axes.append(ax2)
