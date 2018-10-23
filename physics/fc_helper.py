from math import *
from ROOT import *
from toy_experiment import FitVar, FitDcpInPi, FitConstrainedVar, Fitter
import sys
import random

def current_grid_params(i, size, key, fc_type):
    """
    Figure out current parameter values on the grid.

    :param i: (int) between 0 and size - 1
    :param size: (int) number of points to sample in parameter space
    :param key: oscillation parameter to use
    :return: (float) values of theta23 or delta cp or dmsq 32
    """
   
    if key == 'theta23':
      if "UO" in fc_type:
        current_sin2theta23 = i * 0.5 / (size + 1.0) + 0.5 + 0.5 / (size + 1.0)
        return asin(sqrt(current_sin2theta23))
      if "LO" in fc_type:
        current_sin2theta23 = i * 0.5 / (size + 1.0) +  0.5 / (size + 1.0)
        return asin(sqrt(current_sin2theta23))
      if "UO" not in fc_type and "LO" not in fc_type:
        current_sin2theta23 = i * 1. / (size + 1.0) + 1. / (size + 1.0)
        return asin(sqrt(current_sin2theta23))

    if key == 'dcp':
      current_dcp = i * 2.0 * pi / (size + 1.0) + 2.0 * pi / (size + 1.0)  # dcp ranges from 0 to 2 pi
      return current_dcp

    if key == 'dmsq_32':
      if "NH" in fc_type:
        current_dmsq32 = i * 3.e-3/ (size + 1.0) + 1.e-3 + 3.e-3/ (size + 1.0)
        return current_dmsq32
      if "IH" in fc_type:
        current_dmsq32 = i * 3.e-3/ (size + 1.0) + 1.e-3 + 3.e-3/ (size + 1.0)
        return -current_dmsq32
      if "NH" not in fc_type and "IH" not in fc_type:
        sys.exit("Please provide hierarchy, either NH or IH")

    return sys.exit("Invalid Key. Please provide the right parameter key")

def get_grid_params(index, grid_size, contour_vars, fc_type):
    """
    Figure out current parameter values on the grid.

    :param i: (int) between 0 and grid_size^(contour_vars.size())- 1
    :param grid_size: (int) the grid is contour_vars.size() dimensional with grid_size for each dimension 
    :return: (float) current values for parameters asked for based on grid position
    """
    contour_indices = [(index / (grid_size**k)) % grid_size for k, v in enumerate(contour_vars)]
    contour_params = {}
    for k, v in enumerate(contour_vars):
        contour_params[v] = current_grid_params(contour_indices[k], grid_size, v, fc_type)
    
    return contour_params 

def current_profile_params(key, fc_type):

  if key == 'theta23':
    if "LO" in fc_type:
      current_sin2theta23 = 0.3 + 0.2*random.random()
      return asin(sqrt(current_sin2theta23))
    if "UO" in fc_type:
      current_sin2theta23 = 0.5 + 0.2*random.random()
      return asin(sqrt(current_sin2theta23))
    if "UO" not in fc_type and "LO" not in fc_type:
      current_sin2theta23 = 0.3 + 0.4*random.random()
      return asin(sqrt(current_sin2theta23))
  if key == 'dcp':
    return random.random()*2.0*pi
  if key == 'dmsq_32':
    if "IH" in fc_type:
      return (-3.e-3 + random.random()*1.e-3)
    if "NH" in fc_type:
      return (2.e-3 + random.random()*1.e-3)
    if "IH" not in fc_type and "NH" not in fc_type:
      sys.exit("Please provide hierarchy, either NH or IH")

    return sys.exit("Invalid Key. Please provide the right parameter key")

def get_profile_params(profile_vars, fc_type):

  profile_params = {}
  for v in profile_vars:
    profile_params[v] = current_profile_params(v, fc_type)
  return profile_params


def initiate_fitters(fc_type, contour_vars):
    """
    Initiate model fitters for global likelihood and profile likelihood.

    :param hierarchy: (string) must be either "normal" or "inverted"
    :return: fitter objects
    """
    kFitDcpInPi = FitVar('dcp', 'dcp', FitDcpInPi, lambda x: x*pi)

    kFitSinSqTheta23 = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                                 lambda x: asin(min(sqrt(max(0, x)), 1)), 0., 1., True)
    kFitSinSqTheta23UO = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                                 lambda x: asin(min(sqrt(max(0, x)), 1)), 0.5, 1., True)
    kFitSinSqTheta23LO = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                                 lambda x: asin(min(sqrt(max(0, x)), 1)), 0., 0.5, True)

    kFitDmsq32NH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                             lambda x: x/1000., 1., 4., True)

    kFitDmsq32IH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                             lambda x: x/1000., -4., -1., True)
    
    kFitDmsq32 = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                             lambda x: x/1000., 1., 4., True, True)

    fitvars_global = [kFitDcpInPi, kFitSinSqTheta23, kFitDmsq32]
    fitvars_select = []
    fitvars_profile = []
    
    if "NH" in fc_type:
        fitvars_select.append(kFitDmsq32NH)
    if "IH" in fc_type:
        fitvars_select.append(kFitDmsq32IH)
    if "UO" in fc_type:
        fitvars_select.append(kFitSinSqTheta23UO)
    if "LO" in fc_type:
        fitvars_select.append(kFitSinSqTheta23LO)
    if "UO" not in fc_type and "LO" not in fc_type:
        fitvars_select.append(kFitSinSqTheta23)
    if "NH" not in fc_type and "IH" not in fc_type:
        sys.exit("Please provide hierarchy, either NH or IH")
    fitvars_select.append(kFitDcpInPi)
   
    # profiled variables are just those that aren't in contour_vars
    fitvars_profile = [fitvar for fitvar in fitvars_select if fitvar.OscKey() not in contour_vars]
    nuis_vars = ['xsec_sigma', 'flux_sigma']

    fitter_global = Fitter(fitvars_global, nuis_vars)
    fitter_profile = Fitter(fitvars_profile, nuis_vars)

    fitter_global.InitMinuit()
    fitter_profile.InitMinuit()

    return fitter_global, fitter_profile
