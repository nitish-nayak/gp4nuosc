"""
Script for simulating reference distribution at a point on the parameter grid.

The three user inputs include grid position index,
The profiled category : "NH", "IH", "NHUO", "IHLO" etc
the variable being FC corrected : "dcp", "theta23" or "dmsq_32"
can also pass 2 variables separated by __ for contour corrections like : "dcp__theta23" etc
and Output directory.

Grid position index ranges from 1 to the total number of points
and the output will be single file contour_index.txt.
"""

from math import *
from ROOT import *
import sys
import os
from fc_helper import *
from toy_experiment import Generate

index = int(sys.argv[1]) - 1  # grid position index
fc_type = sys.argv[2]
contour_varstrs = sys.argv[3]
output_dir = sys.argv[4]

GRID_SIZE = 20 # sample GRID_SIZE points per variable
N_MC = 2000 # number of pseudo experiments to throw per point in parameter space

# get list of variables
contour_vars = contour_varstrs.split("__")
contour = FCContour(fc_type, contour_vars, GRID_SIZE)
# figure out current parameter values based on grid position
current_params = contour.GetGridParams(index)

# current values for mock data
osc_data = {'theta23': 0., 'dcp': 0., 'dmsq_32': 0.}
osc_data.update(current_params)
nuis_data = {'xsec_nue_sigma': 0., 'xsec_numu_sigma':0., 'flux_sigma': 0.}

# seed values for fitting
osc_seed = {'theta23': 0., 'dcp': 0., 'dmsq_32': 0.}
osc_seed.update(current_params)
nuis_seed = {'xsec_nue_sigma': 0., 'xsec_numu_sigma':0., 'flux_sigma': 0.}

model = Generate()

# initiate fitters
fitter_global, fitter_profile = contour.GetFitters()
contour.InitiateProfileParamWithPrior()

for i in range(N_MC):
    # sample from entire allowed range
    profile_params, nuis_params = contour.GetProfileParams(index)

    osc_data.update(profile_params)
    nuis_data.update(nuis_params)
    
    osc_seed.update(profile_params)
    nuis_seed.update(nuis_params)
    current_params.update(profile_params)

    mock_data = model.Data(osc_data, nuis_data)  # throw pseudo experiment

    try:
        global_fit = fitter_global.Fit(mock_data, osc_seed, nuis_seed, True)
        global_results = str(osc_seed) + ' ' + str(nuis_seed)  # extract global fitted values
        
        # reset seed values
        osc_seed = current_params.copy()
        nuis_seed = nuis_params.copy()
        
        profile_fit = fitter_profile.Fit(mock_data, osc_seed, nuis_seed, True)
        profile_results = str(osc_seed) + ' ' + str(nuis_seed)  # extract profile fitted values

        # write output to a text file
        path = os.path.join(output_dir, 'contour_{0}.txt'.format(index))
        with open(path, 'a') as f:
            f.write(global_results + ', ' + profile_results + ', ' +
                   '{l1}, {l2}\n'.format(l1=global_fit, l2=profile_fit))

    except:
        print('Warning: fitter does not work properly!')

    mock_data.Delete()
