from math import *
from ROOT import *
from toy_experiment import FitVar, FitDcpInPi, FitConstrainedVar, Fitter


def current_params(i, grid_size):
    """
    Figure out current parameter values on the grid.

    :param i: (int) between 0 and size x size - 1
    :param grid_size: (int) the grid is size x size
    :return: (float) values of theta23 and delta cp
    """
    j = i / grid_size
    k = i % grid_size

    current_sin2theta23 = j * 1.0 / (grid_size + 1.0) + 1.0 / (grid_size + 1.0)  # sin2theta23 ranges from 0 to 1
    current_theta23 = asin(sqrt(current_sin2theta23))

    current_dcp = k * 2.0 * pi / (grid_size + 1.0) + 2.0 * pi / (grid_size + 1.0)  # dcp ranges from 0 to 2 pi

    return current_theta23, current_dcp


def initiate_fitters(hierarchy):
    """
    Initiate model fitters for global likelihood and profile likelihood.

    :param hierarchy: (string) must be either "normal" or "inverted"
    :return: fitter objects
    """
    kFitDcpInPi = FitVar('dcp', 'dcp', FitDcpInPi, lambda x: x*pi)

    kFitSinSqTheta23 = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                                         lambda x: asin(min(sqrt(max(0, x)), 1)), 0.3, 0.7, False)

    kFitDmsq32NH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                                     lambda x: x/1000., 0., 4.)

    kFitDmsq32IH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                                     lambda x: x/1000., -4, 0.)

    if hierarchy == 'normal':
        fitter_global = Fitter([kFitSinSqTheta23, kFitDcpInPi, kFitDmsq32NH],
                               ['xsec_sigma', 'flux_sigma'])  # normal hierarchy
        fitter_profile = Fitter([kFitDmsq32NH], ['xsec_sigma', 'flux_sigma'])
    elif hierarchy == 'inverted':
        fitter_global = Fitter([kFitSinSqTheta23, kFitDcpInPi, kFitDmsq32IH],
                               ['xsec_sigma', 'flux_sigma'])  # normal hierarchy
        fitter_profile = Fitter([kFitDmsq32IH], ['xsec_sigma', 'flux_sigma'])

    fitter_global.InitMinuit()
    fitter_profile.InitMinuit()

    return fitter_global, fitter_profile
