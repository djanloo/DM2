"""Commons for style of figures and other stuff"""
import matplotlib
from matplotlib import pyplot as plt

font = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 8}

matplotlib.rc('font', **font)
FULLSIZE_FIGURE = (18/2.54, 9/2.54)
HALFSIZE_FIGURE = (9/2.54, 9/2.54)
SHORT_FULLSIZE_FIGURE = (18/2.54, 6/2.54)
SHORT_HALFSIZE_FIGURE = (9/2.54, 6/2.54)
BAND_FULLSIZE_FIGURE = (18/2.54, 4/2.54)

# Sets the default to SHORT_HALFSIZE
plt.rcParams["figure.figsize"] = SHORT_HALFSIZE_FIGURE


# This must be done to put legends outside subplots
# If you want a tight layout you must ask for it

# EDIT: it is possible to set legend to the figure instead of the axis, I didn't remember that
from matplotlib import rcParams
rcParams["figure.autolayout"] = True

SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE, titleweight="bold")     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE, labelweight="bold")    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('legend', titlesize=SMALL_SIZE)    # legend fontsize

plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

print("font sizes set")