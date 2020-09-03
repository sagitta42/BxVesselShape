import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi
from myplot import *
from scipy.interpolate import spline


class Vessel():
    def __init__(self, bool_spline):
        ''' bool_spline [True|False]: apply spline to the points to make the curve smoother '''

        # read the text file with points (r, theta) corresponding to vessel positions
        self.df = pd.read_csv('radiusVStheta.txt', names = ['theta', 'rad'])

        # apply spline if needed
        if bool_spline:
            df = pd.DataFrame()
            xp, yp = spln(self.df['theta'], self.df['rad'])
            df['theta'] = xp
            df['rad'] = yp

            self.df = df

        self.p = Plot((10,8))

    def vessel_shape(self):
        ''' Draw vessel shape X vs Y points '''

        # calculate x and y points
        self.df['x'] = self.df['rad'] * np.sin(self.df['theta'])
        self.df['y'] = self.df['rad'] * np.cos(self.df['theta'])

        # negative points: same y, but negative x
        dfn = self.df.copy()
        dfn['x'] = -dfn['x']
        dfn['theta'] = -dfn['theta']

        # full info with both pos. and neg. theta (x)
        self.df = pd.concat([self.df, dfn], ignore_index=True)
        self.df = self.df.sort_values('theta')

        label = 'Vessel'
        self.df.plot(x = 'x', y = 'y', style='-', label=label, ax = self.p.ax, linewidth=2.5)


    def fv_shape(self):
        ''' Draw FV shape.
        Two shapes are available, pep and Be7.
        In this code, the Be7 example is commented out, only pep is drawn.
        It is possible to draw Be7 instead, or draw both together'''

        dfv = pd.DataFrame()
        dfv['x'] = np.linspace(-5, 5, 100)
        # fv_pep() is a function that defines pep FV shape
        dfv['z'] = dfv['x'].apply(fv_pep, sign=1)

        ## add negative z points
        dfn = dfv.copy()
        # for pep FV, it's not symmetrical
        dfn['z'] = dfn['x'].apply(fv_pep, sign=-1)
        # Be7 FV, simply make a symmetrical copy for negative z
#        dfn['z'] = -dfn['z']

        # concatenate negative points to existing positive points
        dfv = pd.concat([dfv, dfn], ignore_index=True)

        # this part is need for sorting
        dfv['theta'] = np.arctan(dfv['x']/dfv['z'])
        # tan is between -pi/2 and pi/2, add missing values
        idx = dfv[(dfv['x'] > 0) & (dfv['z'] < 0)].index
        dfv.at[idx, 'theta'] = dfv.loc[idx]['theta'] + pi
        idx = dfv[(dfv['x'] < 0) & (dfv['z'] < 0)].index
        dfv.at[idx, 'theta'] = dfv.loc[idx]['theta'] - pi
        # this is needed for Be7 FV, but not pep
        # in python -0.0 and 0.0 are different, so these are special points
        # dfv.at[ dfv[dfv['x'] == rmax].index, 'theta'] = pi/2
        # dfv.at[ dfv[dfv['x'] == -rmax].index, 'theta'] = -pi/2

        dfv = dfv.sort_values('theta')
        dfv.plot(x = 'x', y = 'z', label = 'FV', ax = self.p.ax, linewidth=2.5)


    def figure(self, name=None):
        ''' Display resulting plot or save image with given name
        if argument 'save' is present when function is called '''

        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        self.p.ax.set_aspect('equal')
        self.p.legend()
        self.p.pretty(large=6)

        self.p.figure(name)



###### FV functions

def fv_be7(x):
    ''' z as a function of x '''

    rmax = 3.021 # m

    zsq = rmax**2 - x**2
    if zsq < 0: return None

    z = sqrt(zsq)
    if z == -0.0: z = 0.0
    zmax = 1.67 # m
    return z if abs(z) < zmax else zmax


def fv_pep(x, sign):
    ''' z as a function of x '''
    rmax = 2.8 # m

    zsq = rmax**2 - x**2
    if zsq < 0: return None

    z = sign * sqrt(zsq)
    zmin = -1.8
    zmax = 2.2 # m
    if z < zmin: return zmin
    if z > zmax: return zmax
    return z


# spline helper function
def spln(xpoints, ypoints):
    '''
    Take x and y points and return 2 splined lists
    '''
    xsmooth = np.linspace(min(xpoints), max(xpoints), 100)
    ysmooth = spline(xpoints, ypoints, xsmooth)
    return xsmooth, ysmooth
