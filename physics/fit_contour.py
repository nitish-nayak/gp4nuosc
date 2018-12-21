"""
Script for parameter fitting at every point on the grid.

The four user inputs include index of data set,
normal or inverted hierarchy,
data and output directories.

The data directory must have file realdata_index.pkl
and the output will be single file fit_index.txt.
"""

from math import *
from ROOT import *
import sys
import os
import pickle
from fc_helper import *


data_index = int(sys.argv[1]) - 1  # data set number
fc_type = sys.argv[2]
contour_varstrs = sys.argv[3]
output_dir = sys.argv[4]

contour_vars = contour_varstrs.split("__")  # get list of variables

contour = Contour(fc_type, contour_vars, grid_size=20)
fitter_global, fitter_profile = contour.GetFitters()

data_dir = '/data/users/linggel/fc/physics/realdata/'
with open(os.path.join(data_dir, 'realdata_{}.pkl'.format(data_index)), 'rb') as fd:
    data = pickle.load(fd)

osc_data = {'theta23': asin(sqrt(0.56)), 'dcp': 1.5 * pi, 'dmsq_32': 2.44 * 1e-3}
nuis_init_seed = {'xsec_nue_sigma': 0.0, 'xsec_numu_sigma': 0.0, 'flux_sigma': 0.0}

osc_seed = osc_data.copy()
nuis_seed = nuis_init_seed.copy()

global_fit = fitter_global.Fit(data, osc_seed, nuis_seed, False)
global_results = str(osc_seed) + ' ' + str(nuis_seed)  # extract global fitted values

grid_dim = len(contour_vars)  # grid dimension (1d or 2d contour)
n_total = 20 ** grid_dim

for i in range(n_total):
    osc_seed = osc_data.copy()
    nuis_seed = nuis_init_seed.copy()

    grid_params = contour.GetGridParams(i)
    osc_seed.update(grid_params)
    profile_params = contour.GetProfileParams(osc_data)
    osc_seed.update(profile_params)

    profile_fit = fitter_profile.Fit(data, osc_seed, nuis_seed, False)
    profile_results = str(osc_seed) + ' ' + str(nuis_seed)  # extract profile fitted values

    output_path = os.path.join(output_dir, 'fit_{}.txt'.format(data_index))  # write output to a text file
    with open(output_path, 'a') as f:
        f.write(global_results + ', ' + profile_results + ', ' + '{l1}, {l2}\n'.format(l1=global_fit, l2=profile_fit))
