from math import *
from ROOT import *
import sys
import os
from fc_helper import *
from toy_experiment import Generate


contour_types = ['theta23__dmsq_32.IH', 'theta23__dmsq_32.NH', 'dcp__theta23.IH', 'dcp__theta23.NH', 
                 'theta23.NH', 'theta23.IH',
                 'dcp.NH', 'dcp.IH', 'dcp.NHUO', 'dcp.IHUO', 'dcp.NHLO', 'dcp.IHLO',
                 'dmsq_32.NH', 'dmsq_32.IH', 'dmsq_32.NHUO', 'dmsq_32.IHUO', 'dmsq_32.NHLO', 'dmsq_32.IHLO']
osc_data = {}
osc_data['theta23'] = asin(sqrt(0.56))
osc_data['dmsq_32'] = 2.44e-3
osc_data['dcp'] = 1.5*pi
nuis_data = {}
Gaus = GenerateRandom(SystLL, -4., 4., "syst")
nuis_data['xsec_nue_sigma'] = Gaus.Random()
nuis_data['xsec_numu_sigma'] = Gaus.Random()
nuis_data['res_nue_sigma'] = Gaus.Random()
nuis_data['res_numu_sigma'] = Gaus.Random()
nuis_data['flux_sigma'] = Gaus.Random()
print nuis_data

nuis_init_seed = {}
nuis_init_seed['xsec_nue_sigma'] = 0.
nuis_init_seed['xsec_numu_sigma'] = 0.
nuis_init_seed['res_nue_sigma'] = 0.
nuis_init_seed['res_numu_sigma'] = 0.
nuis_init_seed['flux_sigma'] = 0.

model = Generate()
data = model.Data(osc_data, nuis_data)

#  save data first
path_data = os.path.join('./data_with_res/', 'toy_data.txt')
with open(path_data, 'w') as f:
  for i in range(data.GetNbinsX()):
    f.write(str(i)+','+str(data.GetBinContent(i+1))+'\n')

GRID_SIZE = 20
for ctypevar in contour_types:
  osc_seed = osc_data.copy()
  nuis_seed = nuis_init_seed.copy()
  ctype = ctypevar.split('.')[1]
  cvars = ctypevar.split('.')[0].split('__')
  contour = Contour(ctype, cvars, GRID_SIZE)
  fitter_global, fitter_profile = contour.GetFitters()

  global_fit = fitter_global.Fit(data, osc_seed, nuis_seed, False)

  path_global = os.path.join('./gaus_results_with_res/', ctypevar+'_global.txt')
  with open(path_global, 'w') as f:
    f.write('{dcp},{dmsq_32},{theta23},{xsec_nue_sigma},{xsec_numu_sigma},{flux_sigma},{res_nue_sigma},{res_numu_sigma},{ll}\n'.format(
             dcp=osc_seed['dcp'], dmsq_32=osc_seed['dmsq_32'], theta23=osc_seed['theta23'],
             xsec_nue_sigma=nuis_seed['xsec_nue_sigma'],xsec_numu_sigma=nuis_seed['xsec_numu_sigma'], 
             flux_sigma=nuis_seed['flux_sigma'],
             res_nue_sigma=nuis_seed['res_nue_sigma'],res_numu_sigma=nuis_seed['res_numu_sigma'], 
             ll=global_fit))

  for index in range(GRID_SIZE**len(cvars)):
    osc_seed = osc_data.copy()
    nuis_seed = nuis_init_seed.copy()

    grid_params = contour.GetGridParams(index)
    osc_seed.update(grid_params)
    profile_params = contour.GetProfileParams(osc_data)
    osc_seed.update(profile_params)
    profile_fit = fitter_profile.Fit(data, osc_seed, nuis_seed, False)

    path_profile = os.path.join('./gaus_results_with_res/', ctypevar+'_profile.txt')
    with open(path_profile, 'a') as f:
      f.write('{dcp},{dmsq_32},{theta23},{xsec_nue_sigma},{xsec_numu_sigma},{flux_sigma},{res_nue_sigma},{res_numu_sigma},{ll}\n'.format(
                dcp=osc_seed['dcp'], dmsq_32=osc_seed['dmsq_32'], theta23=osc_seed['theta23'],
                xsec_nue_sigma=nuis_seed['xsec_nue_sigma'],xsec_numu_sigma=nuis_seed['xsec_numu_sigma'], 
                flux_sigma=nuis_seed['flux_sigma'],
                res_nue_sigma=nuis_seed['res_nue_sigma'],res_numu_sigma=nuis_seed['res_numu_sigma'], 
                ll=profile_fit-global_fit))
