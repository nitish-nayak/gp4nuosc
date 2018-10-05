"""
Script for simulating reference distribution at a point on the parameter grid.

The three user inputs include grid position index,
normal or inverted hierarchy,
and output directory.

Grid position index ranges from 1 to the total number of points
and the output will be single file contour_index.txt.
"""

from ROOT import *
import random
import sys
import os
from fc_helper import *
from toy_experiment import Generate

index = int(sys.argv[1]) - 1  # grid position index
hierarchy = sys.argv[2]
output_dir = sys.argv[3]

GRID_SIZE = 20
N_MC = 500

# figure out current parameter values
current_theta23, current_dcp = current_params(index, GRID_SIZE)

# current values for mock data
osc_data = {'theta23': current_theta23, 'dcp': current_dcp, 'dmsq_32': 0.}
nuis_data = {'xsec_sigma': 0., 'flux_sigma': 0.}

# seed values for fitting
osc_seed = {'theta23': current_theta23, 'dcp': current_dcp, 'dmsq_32': 0.}
nuis_seed = {'xsec_sigma': 0., 'flux_sigma': 0.}

model = Generate()

# initiate fitters
fitter_global, fitter_profile = initiate_fitters(hierarchy)

for i in range(N_MC):
    # sample from entire allowed range (both normal and inverted hierarchies)
    if random.random() < 0.5:
        current_dmsq_32 = (-random.random() * 4.0) * 1e-3
    else:
        current_dmsq_32 = (random.random() * 4.0) * 1e-3

    osc_data['dmsq_32'] = current_dmsq_32
    osc_seed['dmsq_32'] = current_dmsq_32

    mock_data = model.Data(osc_data, nuis_data)  # throw pseudo experiment

    try:
        global_fit = fitter_global.Fit(mock_data, osc_seed, nuis_seed)
        global_results = str(osc_seed) + ' ' + str(nuis_seed)  # extract global fitted values

        # reset seed values
        osc_seed['theta23'] = current_theta23
        osc_seed['dcp'] = current_dcp
        osc_seed['dmsq_32'] = current_dmsq_32
        nuis_seed['xsec_sigma'] = 0.
        nuis_seed['flux_sigma'] = 0.
        profile_fit = fitter_profile.Fit(mock_data, osc_seed, nuis_seed)
        profile_results = str(osc_seed) + ' ' + str(nuis_seed)  # extract profile fitted values

        # write output to a text file
        path = os.path.join(output_dir, 'contour_{}.txt'.format(index))
        with open(path, 'a') as f:
            f.write(global_results + ', ' + profile_results + ', ' +
                    '{l1}, {l2}\n'.format(l1=global_fit, l2=profile_fit))

    except:
        print('Warning: fitter does not work properly!')

    mock_data.Delete()
