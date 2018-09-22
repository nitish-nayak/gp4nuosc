from ROOT import *
from array import array
from math import *
import random
import sys
from toy import FitVar, FitDcpInPi, FitConstrainedVar, Fitter, Generate
import pickle

kFitDcpInPi = FitVar('dcp', 'dcp', FitDcpInPi, lambda x: x*pi)

kFitSinSqTheta23 = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                                     lambda x: asin(min(sqrt(max(0, x)), 1)), 0.3, 0.7, False)

kFitDmsq32NH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                                 lambda x: x/1000., 0., 4.)

kFitDmsq32IH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                                 lambda x: x/1000., -4, 0.)

fitter_global = Fitter([kFitSinSqTheta23, kFitDcpInPi, kFitDmsq32NH], ['xsec_sigma', 'flux_sigma'])
fitter_global.InitMinuit()

fitter_profile = Fitter([kFitDmsq32NH], ['xsec_sigma', 'flux_sigma'])
fitter_profile.InitMinuit()

# current values for mock data
osc_data = {}
osc_data['theta23'] = 0.
osc_data['dcp'] = 0.
osc_data['dmsq_32'] = 0.
nuis_data = {}
nuis_data['xsec_sigma'] = 0.
nuis_data['flux_sigma'] = 0.

# seed values for fitting
osc_seed = {}
osc_seed['theta23'] = 0.
osc_seed['dcp'] = 0.
osc_seed['dmsq_32'] = 0.
nuis_seed = {}
nuis_seed['xsec_sigma'] = 0.
nuis_seed['flux_sigma'] = 0.

grid_size = 20

run = int(sys.argv[1]) - 1 # from 0 to 499

for i in range(400):
    j = i / grid_size
    k = i % grid_size

    current_sin2theta23 = j * 1.0 / (grid_size + 1.0) + 1.0 / (grid_size + 1.0) # sin2theta23 ranges from 0 to 1
    current_theta23 = asin(sqrt(current_sin2theta23))
    current_dcp = k * 2.0 * pi / (grid_size + 1.0) + 2.0 * pi / (grid_size + 1.0) # dcp ranges from 0 to 2 pi
  
    current_dmsq_32 = (random.random() * 4.0) * 1e-3

    osc_data['theta23'] = current_theta23
    osc_data['dcp'] = current_dcp
    osc_data['dmsq_32'] = current_dmsq_32
    osc_seed['theta23'] = current_theta23
    osc_seed['dcp'] = current_dcp
    osc_seed['dmsq_32'] = current_dmsq_32

    with open('realdata_{}.pkl'.format(run), 'rb') as f:
        mock_data = pickle.load(f)
        try:
            global_fit = fitter_global.Fit(mock_data, osc_seed, nuis_seed)
            global_results = str(osc_seed) + ' ' + str(nuis_seed) # extract fitted values
            # reset seed values
            osc_seed['theta23'] = current_theta23
            osc_seed['dcp'] = current_dcp
            osc_seed['dmsq_32'] = current_dmsq_32
            nuis_seed['xsec_sigma'] = 0.
            nuis_seed['flux_sigma'] = 0.
            profile_fit = fitter_profile.Fit(mock_data, osc_seed, nuis_seed)
            profile_results = str(osc_seed) + ' ' + str(nuis_seed)
            # write output to a text file
            with open('contour_normal_fit_{}.txt'.format(run), 'a') as myfile:
                myfile.write(global_results + ', ' + profile_results + ', ' +
                             '{l1}, {l2}\n'.format(l1=global_fit, l2=profile_fit))
        except:
            print('Fitter does not work properly!')
        mock_data.Delete()
