from math import *
from ROOT import *
from toy_experiment import *
import sys
import random
import csv as csv

class Contour:
  def __init__(self, c_type, contour_vars, grid_size):
    self.ctype = c_type
    self.cvars = contour_vars
    self.grid = grid_size
    self.contour = {}
   
    self.nuis_vars = ['xsec_nue_sigma', 'xsec_numu_sigma', 'flux_sigma']
    self.fitglobalvars = [kFitDcpInPi, kFitSinSqTheta23, kFitDmsq32]
    self.fitprofilevars = []

    if "dmsq_32" in self.cvars:
      if "NH" in self.ctype:
          self.contour['dmsq_32'] = (2., 3.)
      if "IH" in self.ctype:
          self.contour['dmsq_32'] = (-3., -2.)
      if "NH" not in self.ctype and "IH" not in self.ctype:
          sys.exit("Please provide hierarchy, either NH or IH")
    else:
      if "NH" in self.ctype:
          self.contour['dmsq_32'] = (2., 3.)
          self.fitprofilevars.append(kFitDmsq32NH)
      if "IH" in self.ctype:
          self.contour['dmsq_32'] = (-3., -2.)
          self.fitprofilevars.append(kFitDmsq32IH)
      if "NH" not in self.ctype and "IH" not in self.ctype:
          sys.exit("Please provide hierarchy, either NH or IH")

    if "theta23" in self.cvars:
      if "UO" in self.ctype:
          self.contour['theta23'] = (0.5, 0.7) 
      if "LO" in self.ctype:
          self.contour['theta23'] = (0.3, 0.5)
      if "UO" not in self.ctype and "LO" not in self.ctype:
          self.contour['theta23'] = (0.3, 0.7)
    else:
      if "UO" in self.ctype:
          self.contour['theta23'] = (0.5, 0.7) 
          self.fitprofilevars.append(kFitSinSqTheta23UO)
      if "LO" in self.ctype:
          self.contour['theta23'] = (0.3, 0.5)
          self.fitprofilevars.append(kFitSinSqTheta23LO)
      if "UO" not in self.ctype and "LO" not in self.ctype:
          self.contour['theta23'] = (0.3, 0.7)
          self.fitprofilevars.append(kFitSinSqTheta23)

    self.contour['dcp'] = (0., 2.)
    if "dcp" not in self.cvars:
      self.fitprofilevars.append(kFitDcpInPi)


  def GetFitters(self):
    fitter_global = Fitter(self.fitglobalvars, self.nuis_vars)
    fitter_profile = Fitter(self.fitprofilevars, self.nuis_vars)
    fitter_global.InitMinuit()
    fitter_profile.InitMinuit()

    return fitter_global, fitter_profile

  def GetGridParams(self, index):

    contour_params = {}
    for k, v in enumerate(self.cvars):
      i = (index/(self.grid**k)) % self.grid
      var_range = self.contour[v][1] - self.contour[v][0]
      current_val = self.contour[v][1] - (var_range*i/self.grid)
      contour_params[v] = [var.GetInvFcn()(current_val) for var in self.fitglobalvars if var.OscKey() == v][0]
    
    return contour_params

  def GetProfileParams(self, data_params):
     profile_params = {}
     if 'dmsq_32' not in self.cvars:
       if "IH" in self.ctype:
         profile_params['dmsq_32'] = -abs(data_params['dmsq_32'])
       if "NH" in self.ctype:
         profile_params['dmsq_32'] = abs(data_params['dmsq_32'])
     return profile_params
     

class FCContour(Contour):
  def __init__(self, c_type, contour_vars, grid_size):
    Contour.__init__(self, c_type, contour_vars, grid_size)
    self.bayesvarpdfs = {}
    self.profvalsfromfile = {}
    self.systvarpdfs = {}

  def InitiateProfileParamFromFile(self, profvar, path, colidx):
    f = open(path, 'rb')
    proflist = csv.reader(f, delimiter=',')
    self.profvalsfromfile[profvar] = []
    for row in proflist:
      self.profvalsfromfile[profvar].append(float(row[colidx]))
    f.close()

  def InitiateProfileParamWithPrior(self):
    for fitvarll in self.fitprofilevars:
      if fitvarll.OscKey() not in self.profvalsfromfile.keys():
        pdf = GenerateRandom(FlatLL, self.contour[fitvarll.OscKey()][0], self.contour[fitvarll.OscKey()][1], fitvarll.OscKey())
        self.bayesvarpdfs[fitvarll] = pdf
    for systvar in self.nuis_vars:
      pdf = GenerateRandom(SystLL, -4., 4., systvar)
      self.systvarpdfs[systvar] = pdf

  def GetProfileParams(self, index):
    profile_params = {}
    syst_params = {}
    for var in self.bayesvarpdfs.keys():
      val = self.bayesvarpdfs[var].Random()
      profile_params[var.OscKey()] = var.GetInvFcn()(val)
    for var in self.profvalsfromfile.keys():
      profile_params[var] = self.profvalsfromfile[var][index]
    for syst in self.systvarpdfs.keys():
      val = self.systvarpdfs[syst].Random()
      syst_params[syst] = val

    return profile_params, syst_params
