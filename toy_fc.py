from ROOT import *
from array import array
from math import *
import random
import sys
from toy import FitVar, FitDcpInPi, FitConstrainedVar, Fitter, Generate

kFitDcpInPi = FitVar('dcp', 'dcp', FitDcpInPi, lambda x: x*pi)

kFitSinSqTheta23 = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                                      lambda x: asin(min(sqrt(max(0, x)), 1)), 0.3, 0.7, False)

kFitDmsq32NH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                                lambda x: x/1000., 0., 4.)

kFitDmsq32IH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                                lambda x: x/1000., -4, 0.)

run = int(sys.argv[1]) - 1 # from 0 to 399
grid_size = 20
j = run / grid_size
k = run % grid_size

current_sin2theta23 = j * 1.0 / (grid_size + 1.0) + 1.0 / (grid_size + 1.0) # sin2theta23 ranges from 0 to 1
current_theta23 = asin(sqrt(current_sin2theta23))
current_dcp = k * 2.0 * pi / (grid_size + 1.0) + 2.0 * pi / (grid_size + 1.0) # dcp ranges from 0 to 2 pi

# current values for mock data
osc_data = {}
osc_data['theta23'] = current_theta23
osc_data['dcp'] = current_dcp
osc_data['dmsq_32'] = 0.
nuis_data = {}
nuis_data['xsec_sigma'] = 0.
nuis_data['flux_sigma'] = 0.

# seed values for fitting
osc_seed = {}
osc_seed['theta23'] = current_theta23
osc_seed['dcp'] = current_dcp
osc_seed['dmsq_32'] = 0.
nuis_seed = {}
nuis_seed['xsec_sigma'] = 0.
nuis_seed['flux_sigma'] = 0.

fitter_global = Fitter([kFitSinSqTheta23, kFitDcpInPi, kFitDmsq32IH],['xsec_sigma', 'flux_sigma'])
fitter_global.InitMinuit()

fitter_profile = Fitter([kFitDmsq32IH],['xsec_sigma', 'flux_sigma'])
fitter_profile.InitMinuit()

model = Generate()

for i in range(1000):
  # sample dmsq_32 from entire allowed range (both normal and inverted hierarchies)
  if random.random() < 0.5:
    current_dmsq_32 = (-random.random() * 4.0) * 1e-3
  else:
    current_dmsq_32 = (random.random() * 4.0) * 1e-3
  osc_data['dmsq_32'] = current_dmsq_32
  osc_seed['dmsq_32'] = current_dmsq_32
  mock_data = model.Data(osc_data, nuis_data) # generate mock data
  try:
    global_fit = fitter_global.Fit(mock_data, osc_seed, nuis_seed)
    global_results = str(osc_seed) + ' ' + str(nuis_seed) # extract global fitted values
    # reset seed values
    osc_seed['theta23'] = current_theta23
    osc_seed['dcp'] = current_dcp
    osc_seed['dmsq_32'] = current_dmsq_32
    nuis_seed['xsec_sigma'] = 0.
    nuis_seed['flux_sigma'] = 0.
    profile_fit = fitter_profile.Fit(mock_data, osc_seed, nuis_seed)
    profile_results = str(osc_seed) + ' ' + str(nuis_seed) # extract profile fitted values
    # write output to a text file
    with open('contour_inverted_{}.txt'.format(run), 'a') as myfile:
      myfile.write(global_results + ', ' + profile_results + ', ' + '{l1}, {l2}\n'.format(l1=global_fit, l2=profile_fit))
  except:
    print('Warning: fitter does not work properly!')
  mock_data.Delete()