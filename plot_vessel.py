from vessel import *

import seaborn as sns
sns.set_palette('colorblind')

# bool_spline=True means spline will be applied to smoothen the points
v = Vessel(bool_spline=True)

# draw vessel shape
v.vessel_shape()
# draw pep FV shape
v.fv_shape()

# save the figure if argument 'save' is  present when calling this function i.e.
# ~$ python plot_vessel.py save
# otherwise will display the plot

# name = 'vessel_fv_pep.pdf'
name = 'vessel_fv_pep.png'
v.figure(name)
