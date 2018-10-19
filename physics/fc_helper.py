from math import *
from ROOT import *
from toy_experiment import FitVar, FitDcpInPi, FitConstrainedVar, Fitter
import sys

def current_params(i, size, key):
    """
    Figure out current parameter values on the grid.

    :param i: (int) between 0 and size - 1
    :param size: (int) number of points to sample in parameter space
    :param key: oscillation parameter to use
    :return: (float) values of theta23 or delta cp or dmsq 32
    """
   
    if key == 'theta23':
      current_sin2theta23 = i * 1.0 / (size + 1.0) + 1.0 / (size + 1.0)  # sin2theta23 ranges from 0 to 1
      current_theta23 = asin(sqrt(current_sin2theta23))
      return current_theta23

    if key == 'dcp':
      current_dcp = i * 2.0 * pi / (size + 1.0) + 2.0 * pi / (size + 1.0)  # dcp ranges from 0 to 2 pi
      return current_dcp

    if key == 'dmsq_32':
      current_dmsq32 = i * 4.e-3/ ( size + 1.0) + 4.e-3/ (size + 1.0)
      return current_dmsq32

    return sys.exit("Invalid Key. Please provide the right parameter key")

def get_params(index, grid_size, contour_vars):
    """
    Figure out current parameter values on the grid.

    :param i: (int) between 0 and grid_size^(contour_vars.size())- 1
    :param grid_size: (int) the grid is contour_vars.size() dimensional with grid_size for each dimension 
    :return: (float) current values for parameters asked for based on grid position
    """
    contour_indices = [(index / (grid_size**k)) % grid_size for k, v in enumerate(contour_vars)]
    contour_params = {}
    for k, v in enumerate(contour_vars):
        contour_params[v] = current_params(contour_indices[k], grid_size, v)
    
    return contour_params 

def initiate_fitters(fc_type, contour_vars):
    """
    Initiate model fitters for global likelihood and profile likelihood.

    :param hierarchy: (string) must be either "normal" or "inverted"
    :return: fitter objects
    """
    kFitDcpInPi = FitVar('dcp', 'dcp', FitDcpInPi, lambda x: x*pi)

    kFitSinSqTheta23 = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                                 lambda x: asin(min(sqrt(max(0, x)), 1)), 0.3, 0.7, False)
    kFitSinSqTheta23UO = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                                 lambda x: asin(min(sqrt(max(0, x)), 1)), 0.5, 0.7, False)
    kFitSinSqTheta23LO = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                                 lambda x: asin(min(sqrt(max(0, x)), 1)), 0.3, 0.5, False)

    kFitDmsq32NH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                             lambda x: x/1000., 0., 4.)

    kFitDmsq32IH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                             lambda x: x/1000., -4., 0.)

    kFitDmsq32 = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                             lambda x: x/1000., -4., 4.)

    fitvars_global = []
    fitvars_profile = []
    
    if "NH" in fc_type:
        fitvars_global.append(kFitDmsq32NH)
    if "IH" in fc_type:
        fitvars_global.append(kFitDmsq32IH)
    if "UO" in fc_type:
        fitvars_global.append(kFitSinSqTheta23UO)
    if "LO" in fc_type:
        fitvars_global.append(kFitSinSqTheta23LO)
    if "UO" not in fc_type and "LO" not in fc_type:
        fitvars_global.append(kFitSinSqTheta23)
    if "NH" not in fc_type and "IH" not in fc_type:
        sys.exit("Please provide hierarchy, either NH or IH")
    fitvars_global.append(kFitDcpInPi)
   
    # profiled variables are just those that aren't in contour_vars
    fitvars_profile = [fitvar for fitvar in fitvars_global if fitvar.OscKey() not in contour_vars]
    nuis_vars = ['xsec_sigma', 'flux_sigma']

    fitter_global = Fitter(fitvars_global, nuis_vars)
    fitter_profile = Fitter(fitvars_profile, nuis_vars)

    fitter_global.InitMinuit()
    fitter_profile.InitMinuit()

    return fitter_global, fitter_profile
