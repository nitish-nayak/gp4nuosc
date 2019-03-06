from ROOT import *
from array import array
from math import *
import numpy as np
from scipy.linalg import eigh
import os

#################################### Neutrino oscillation probability ####################################

class Generate:
  def __init__(self):
     
    self.trueENue = TF1("trueENue", self.TrueENue, 0.1, 5, 6)
    self.trueENue.SetParName(0, "xsec_nue_sigma")
    self.trueENue.SetParName(1, "xsec_numu_sigma")
    self.trueENue.SetParName(2, "flux_sigma")
    self.trueENue.SetParName(3, "theta23")
    self.trueENue.SetParName(4, "dmsq_32")
    self.trueENue.SetParName(5, "dcp")

    self.trueENumu = TF1("trueENumu", self.TrueENumu, 0.1, 5, 6)
    self.trueENumu.SetParName(0, "xsec_nue_sigma")
    self.trueENumu.SetParName(1, "xsec_numu_sigma")
    self.trueENumu.SetParName(2, "flux_sigma")
    self.trueENumu.SetParName(3, "theta23")
    self.trueENumu.SetParName(4, "dmsq_32")
    self.trueENumu.SetParName(5, "dcp")
    
    self.resNue = TF1("resNue", self.ResolutionNue, -2, 2, 1)
    self.resNue.SetParName(0, "res_nue_sigma")

    self.resNumu = TF1("resNumu", self.ResolutionNumu, -2, 2, 1)
    self.resNumu.SetParName(0, "res_numu_sigma")
    
    self.dist_convNue = TF1Convolution(self.trueENue, self.resNue, True)
    self.dist_convNue.SetNofPointsFFT(500)
    self.recoENue = TF1("recoENue", self.dist_convNue, 0.1, 5., self.dist_convNue.GetNpar())
    self.recoENue.SetParName(0, "xsec_nue_sigma")
    self.recoENue.SetParName(1, "xsec_numu_sigma")
    self.recoENue.SetParName(2, "flux_sigma")
    self.recoENue.SetParName(3, "theta23")
    self.recoENue.SetParName(4, "dmsq_32")
    self.recoENue.SetParName(5, "dcp")
    self.recoENue.SetParName(6, "res_nue_sigma")
    
    self.dist_convNumu = TF1Convolution(self.trueENumu, self.resNumu, True)
    self.dist_convNumu.SetNofPointsFFT(500)
    self.recoENumu = TF1("recoENumu", self.dist_convNumu, 0.1, 5., self.dist_convNumu.GetNpar())
    self.recoENumu.SetParName(0, "xsec_nue_sigma")
    self.recoENumu.SetParName(1, "xsec_numu_sigma")
    self.recoENumu.SetParName(2, "flux_sigma")
    self.recoENumu.SetParName(3, "theta23")
    self.recoENumu.SetParName(4, "dmsq_32")
    self.recoENumu.SetParName(5, "dcp")
    self.recoENumu.SetParName(6, "res_numu_sigma")
    
    self.km2ev = 5.06773103202e+09
    self.k2 = 4.62711492217e-09
    self.GeV2eV = 1.0e+09
    self.Gf = 1.166371e-5
    self.Ne = 1.42
    self.dm2_31 = 0.

    self.oscH = np.zeros([3,3], dtype=complex)
    self.nustate = np.zeros(3, dtype=complex)
    self.nustate[1] = 1.
    self.initoscH = False

  def XSec(self, E, sigma):
    """
    Simulated NueCC/NumuCC - QE XSec
    sigma represents shift from nominal
    Normalisations taken into account elsewhere
    """
    weight = max(0.6, 1. + sigma*0.1)
    form = 0.
    if E < 1:
      form = form + weight*(TMath.Landau(E, 1, 0.3))
    if E >= 1:
      form = form + weight*(1.79e-01*exp(-2.0e-02*(E-1)))
    return form

  def Flux(self, E, sigma):
    """
    Simulated Flux distribution
    sigma represents shift from nominal
    """
    weight = max(0.3, 0.5 + sigma*0.05)
    form = weight*(TMath.Landau(E, 2, weight))
    return form

  def NumuCalc(self, E, th23, dm2_32):

    return (1.- (sin(2.*th23)**2)*(sin(1.27*dm2_32*810./E)**2))

  def InitOsc(self, th23, dm2_32, dcp, isanti):
    
    th13 = asin(sqrt(2.10e-2))    # reactor mixing angle
    th12 = asin(sqrt(0.305))     
    dm2_21 = 7.53e-5
    if isanti:
        dm2_21 = -7.53e-5
    self.dm2_31 = dm2_32+dm2_21

    h11 = abs(dm2_21)/self.dm2_31
    sij = sin(th12)
    cij = cos(th12)

    h00 = h11*sij*sij - 1.
    h01 = h11*sij*cij
    h11 = h11*cij*cij - 1.

    sij = sin(th13)
    cij = cos(th13)
    expCP = complex(cos(dcp), -sin(dcp))

    h02 = (-h00*sij*cij)*expCP
    h12 = (-h01*sij)*expCP
    h11 -= h00*sij*sij
    h00 *= cij*cij - sij*sij
    h01 *= cij

    sij = sin(th23)
    cij = cos(th23)

    self.oscH[0, 0] = h00 - 0.5*h11
    self.oscH[1, 1] = 0.5*h11*(cij*cij - sij*sij) + 2*h12.real*cij*sij
    self.oscH[2, 2] = -self.oscH[1, 1]

    self.oscH[0, 1] = h02*sij + h01*cij
    self.oscH[0, 2] = h02*cij - h01*sij
    self.oscH[1, 2] = h12 - (h11*cij + 2*h12.real*sij)*sij

    self.oscH[1, 0] = self.oscH[0, 1].conjugate()
    self.oscH[2, 0] = self.oscH[0, 2].conjugate()
    self.oscH[2, 1] = self.oscH[1, 2].conjugate()

    self.initoscH = True
    return 

  def NueCalc(self, E, th23, dm2_32, dcp, isanti):
    """
    Calculator for oscillation probabilities
    Args: 
        theta23 - mixing angle, range is 0 to pi/2
        dcp - CP violating phase, range is 0 to 2pi
        dm2_32 - 3-2 squared mass difference, range is -inf to +inf 
          (Note: 2e-3 < abs(dm2_32) < 3e-3 for realistic situations)
    """
     
    L=810.          # baseline
    self.InitOsc(th23, dm2_32, dcp, isanti)
    lv = 2.*self.GeV2eV*E/self.dm2_31
    kr2GNe = self.k2*sqrt(2)*self.Gf*self.Ne

    A = self.oscH/lv
    A[0, 0] += kr2GNe

    w,v = eigh(A)
    #  nucomp = np.dot(self.nustate, np.matrix(v).getH())
    nucomp = np.dot(np.matrix(v).getH(), self.nustate)
    s = np.sin(-w.real*self.km2ev*L) 
    c = np.cos(-w.real*self.km2ev*L) 
    jpart = np.multiply((c+1j*s), nucomp)
    
    amp_osc = np.dot(v, np.transpose(jpart))
    #  amp_osc = np.dot(jpart, v)

    p_osc = abs(amp_osc[0])**2
    return p_osc

  def TrueENue(self, x, par):
    toy_pot = 3.e4
    unosc = self.XSec(x[0], par[0])*self.Flux(x[0], par[2])
    osc_weight = self.NueCalc(x[0], par[3], par[4], par[5], False)
    return toy_pot*unosc*osc_weight
  
  def TrueENumu(self, x, par):
    toy_pot = 1.2e4
    unosc = self.XSec(x[0], par[1])*self.Flux(x[0], par[2])
    osc_weight = self.NumuCalc(x[0], par[3], par[4])
    return toy_pot*unosc*osc_weight

  def ResolutionNue(self, x, sigma):
    res = 2*(0.11+0.03*sigma[0])
    return TMath.Gaus(x[0], 0., res)/(res*sqrt(2.*pi))
  
  def ResolutionNumu(self, x, sigma):
    res = 2*(0.08+0.02*sigma[0])
    return TMath.Gaus(x[0], 0., res)/(res*sqrt(2.*pi))
 
  def SetParams(self, osc_params, nuis_params):

    for key in osc_params.keys():
      self.trueENue.SetParameter(key, float(osc_params[key]))
      self.trueENumu.SetParameter(key, float(osc_params[key]))
    for key in nuis_params.keys():
      if "res" not in key:
        self.trueENue.SetParameter(key, float(nuis_params[key]))
        self.trueENumu.SetParameter(key, float(nuis_params[key]))
      if key == "res_nue_sigma":
        self.resNue.SetParameter(key, float(nuis_params[key]))
      if key == "res_numu_sigma":
        self.resNumu.SetParameter(key, float(nuis_params[key]))

    self.dist_convNue.Update()
    self.dist_convNumu.Update()

    for key in osc_params.keys():
      self.recoENue.SetParameter(key, float(osc_params[key]))
      self.recoENumu.SetParameter(key, float(osc_params[key]))
    for key in nuis_params.keys():
      if "res" not in key:
        self.recoENue.SetParameter(key, float(nuis_params[key]))
        self.recoENumu.SetParameter(key, float(nuis_params[key]))
      if key == "res_nue_sigma":
        self.recoENue.SetParameter(key, float(nuis_params[key]))
      if key == "res_numu_sigma":
        self.recoENumu.SetParameter(key, float(nuis_params[key]))

    self.sampleInt = 500
    self.xNue = array('d', self.sampleInt*[0.])
    self.wNue = array('d', self.sampleInt*[0.])
    self.recoENue.CalcGaussLegendreSamplingPoints(self.sampleInt, self.xNue, self.wNue, 1e-13)

    self.xNumu = array('d', self.sampleInt*[0.])
    self.wNumu = array('d', self.sampleInt*[0.])
    self.recoENumu.CalcGaussLegendreSamplingPoints(self.sampleInt, self.xNumu, self.wNumu, 1e-13)

  def MC(self, osc_params, nuis_params, hname="recoE_pred"):
    
    self.SetParams(osc_params, nuis_params)
    
    nbins_pred = 24
    recoE_pred = TH1D(hname, hname, nbins_pred, 0.5, 12.5)
    # actually fill prediction in analysis bins
    for binx in range(8):
      #  recoE_pred.SetBinContent(binx+1, self.trueENue.Integral((float(binx)/2.)+0.5, (float(binx)/2.)+1.))
      recoE_pred.SetBinContent(binx+1, self.recoENue.IntegralFast(self.sampleInt, self.xNue, self.wNue, (float(binx)/2.)+0.5, (float(binx)/2.)+1.))
    for binx in range(16):
      #  recoE_pred.SetBinContent(8+binx+1, self.trueENumu.Integral((float(binx)/4.)+0.5, (float(binx)/4.)+0.75))
      recoE_pred.SetBinContent(8+binx+1, self.recoENumu.IntegralFast(self.sampleInt, self.xNumu, self.wNumu, (float(binx)/4.)+0.5, (float(binx)/4.)+0.75))

    return recoE_pred
  
  def MCCurve(self, osc_params, nuis_params):

    self.SetParams(osc_params, nuis_params)
    g = TGraph()
    res = 1000
    totNue = self.trueENue.Integral(0.5, 4.5)
    totNumu = self.trueENumu.Integral(0.5, 4.5)
    for i in range(res):
      g.SetPoint(i, 0.5+(i-1)*4./res, self.trueENue.Integral(0.5+(i-1)*4./res, 0.5+(i)*4./res)*res/8.)
    for i in range(res, 2*res):
      j = i - res
      g.SetPoint(i, 0.5+(i-1)*4./res, self.trueENumu.Integral(0.5+(j-1)*4./res, 0.5+(j)*4./res)*res/8.)

    return g

  def Data(self, osc_params, nuis_params, isfake=False, hname="recoE_data"):


    self.SetParams(osc_params, nuis_params)
    if isfake:
      recoE_data = self.MC(osc_params, nuis_params, hname)
    else: 
      nbins = 24
      #  gRandom.SetSeed(102)
      poisNue = TRandom3(0)
      #  poisNue.SetSeed(102)
      poisNumu = TRandom3(0)
      #  poisNumu.SetSeed(102)
      #  samplesNue = poisNue.Poisson(self.trueENue.Integral(0.1,5))
      #  samplesNumu = poisNumu.Poisson(self.trueENumu.Integral(0.1,5))
      
      samplesNue = poisNue.Poisson(self.recoENue.IntegralFast(self.sampleInt, self.xNue, self.wNue, 0.1,5))
      samplesNumu = poisNumu.Poisson(self.recoENumu.IntegralFast(self.sampleInt, self.xNumu, self.wNumu, 0.1,5))
      recoE_data = TH1D(hname, hname, nbins, 0.5, 12.5)
      
      for i in xrange(samplesNue):
        trueE_i = self.trueENue.GetRandom()
        res_i = self.resNue.GetRandom() + 1
        recoE_i = trueE_i*res_i
        if recoE_i <= 4.5 and recoE_i > 0.5: 
          recoE_data.Fill(recoE_i)
      for i in xrange(samplesNumu):
        trueE_i = self.trueENumu.GetRandom()
        res_i = self.resNumu.GetRandom() + 1
        recoE_i = trueE_i*res_i
        #  recoE_i = self.recoENumu.GetRandom()
        if recoE_i <= 4.5 and recoE_i > 0.5:
          recoE_data.Fill(4.5+(2.*recoE_i-1.))

      poisNue.Delete()
      poisNumu.Delete()
    return recoE_data

#################################### Poisson model likelihood ####################################

class Experiment:
  def __init__(self, mc, data):
    self.mc = mc.Clone()
    self.data = data.Clone()

  def Likelihood(self):
    """
    Poisson Likelihood function with penalty terms for nuisance parameters
    """

    chi = 0.
    for i in xrange(self.data.GetNbinsX()):
      e = self.mc.GetBinContent(i+1)
      o = self.data.GetBinContent(i+1)

      if e < 1e-40:
        if o == 0: chi += 0.
        e = 1e-40
      chi += 2*(e-o)
      if o != 0: chi += 2*o*log(o/e)

    return chi

  def __del__(self):
    self.mc.Delete()
    self.data.Delete()

#################################### Minuit parameter optimization ####################################

class FitVar:
  def __init__(self, key, osc_key, fcn, invfcn, fix = False):
    self.key = key
    self.osc_key = osc_key
    self.fcn = fcn
    self.invfcn = invfcn
    self.fix = fix

  def Penalty(self, value):
    return 0.

  def SetValue(self, *args):
    param = {}
    param[self.osc_key] = self.invfcn(*args)
    return param

  def GetValue(self, *args):
    return self.fcn(*args)

  def Key(self):
    return self.key

  def OscKey(self):
    return self.osc_key

  def IsFixed(self):
    return self.fix

  def GetInvFcn(self):
    return self.invfcn

  def GetFcn(self):
    return self.fcn

class FitConstrainedVar(FitVar):
  def __init__(self, key, osc_key, fcn, invfcn, lowlimit, highlimit, mod=False, fix=False):
    FitVar.__init__(self, key, osc_key, fcn, invfcn, fix)
    self.lo = lowlimit
    self.hi = highlimit
    self.mod = mod

  def Penalty(self, value):
    abs_value = value
    if self.mod:
      abs_value = abs(value)
    mean = (self.lo + self.hi)/2.
    rad = (self.hi - self.lo)/2.
    if(abs_value >= self.lo and abs_value <= self.hi): return 0.
    return (((abs_value-mean)/rad)**2) - 1.

  def Clamp(self, value):
    abs_value = value
    if self.mod:
      abs_value = abs(value)
      if value >= 0:
        return max(self.lo, min(abs_value, self.hi))
      if value < 0:
        return -max(self.lo, min(abs_value, self.hi))
    if not self.mod:
      return max(self.lo, min(abs_value, self.hi))

  def SetValue(self, *args):
    param = {}
    param[self.osc_key] = self.invfcn(self.Clamp(*args))

    return param

  def GetLowLimit(self):
    return self.lo
  
  def GetHighLimit(self):
    return self.hi

class FitVarWithLL(FitConstrainedVar):
  def __init__(self, key, osc_key, fcn, invfcn, llfcn, lowlimit, highlimit, mod=False, fix=False):
    FitConstrainedVar.__init__(self, key, osc_key, fcn, invfcn, lowlimit, highlimit, mod, fix)
    self.llfcn = llfcn
  
  def Penalty(self, value):
    return self.llfcn(value)

  def GetLLFcn(self):
    return self.llfcn
 

def FitDcpInPi(dcp):
  ret = dcp/pi;
  a = long(long(ret/2+1))
  ret -= 2*a
  while ret < 0.: ret += 2
  while ret > 2.: ret -= 2

  return ret

def NumuDmsq32LL(dmsq_32):
  return exp(-((abs(dmsq_32) - 2.44)/0.08)**2/2.)

def NumuSinSqTheta23LL(ssth23):
  ss2th23 = 4*ssth23*(1-ssth23)
  return exp(-((ss2th23 - 0.9856)/0.0192)**2/2.)

def SystLL(nuis):
  return exp(-nuis**2/2.)

def FlatLL(val):
  return 1.

class GenerateRandom():
  def __init__(self, llfcn, lowlimit, highlimit, distname):
    self.fcn = llfcn
    self.lo = lowlimit
    self.hi = highlimit
    self.distname = distname
    self.pdf = TF1(self.distname, self.PDF, self.lo, self.hi)

  def PDF(self, x):
    return self.fcn(x[0])
  
  def GetPDF(self):
    g = TGraph()
    res = 10000
    diff = self.hi-self.lo
    for i in range(res):
      g.SetPoint(i, self.lo+(i-1)*diff/res, self.pdf.Integral(self.lo+(i-1)*diff/res, self.lo+(i)*diff/res)*res/4.)
    return g

  def Random(self):
    gRandom.SetSeed(0)
    return self.pdf.GetRandom() 


class Fitter():
  def __init__(self, fitvars, systs):
    self.fitvars = fitvars
    self.systs = systs
    self.model = Generate()
    self.osc_params = {}
    self.nuis_params = {}
   
  def InitMinuit(self):
    self.gMinuit = TMinuit(len(self.fitvars)+len(self.systs))
    self.gMinuit.SetPrintLevel(-1)
    self.gMinuit.SetFCN(self.fcn)
  
  def fcn(self, npar, gin, f, par, iflag):
   
    data2fit = self.gMinuit.GetObjectFit()
    penalty = 0.
    for varidx in xrange(len(self.fitvars)):
      val = par[varidx]
      penalty += self.fitvars[varidx].Penalty(val)
      self.osc_params.update(self.fitvars[varidx].SetValue(val))
    for systidx in xrange(len(self.systs)):
      val = par[len(self.fitvars)+systidx]
      self.nuis_params.update({self.systs[systidx]:val})
      penalty += val**2
    mc = self.model.MC(self.osc_params, self.nuis_params)
    f[0] = Experiment(mc, data2fit).Likelihood() + penalty

  def RunFitterSeeded(self, data, osc_seed, nuis_seed):

    self.osc_params = osc_seed.copy()
    self.nuis_params = nuis_seed.copy()
    self.gMinuit.SetObjectFit(data)

    arglist = array('d', 10*[0.])
    ierflag = Long(1982)

    arglist[0] = 1
    self.gMinuit.mnexcm("SET ERR", arglist, 1, ierflag)

    count = 0
    for fitvar in self.fitvars:
      val = fitvar.GetValue(osc_seed[fitvar.OscKey()])
      err = 0
      if val: err = val/2.
      else: err = 0.1
      self.gMinuit.mnparm(count, fitvar.Key(), val, 0.001, 0, 0, ierflag)
      if fitvar.IsFixed(): 
          self.gMinuit.FixParameter(count)
      count = count + 1

    for syst in self.systs:
      val = nuis_seed[syst]
      self.gMinuit.mnparm(count, syst, val, 0.001, 0, 0, ierflag)
      count = count + 1

    arglist[0] = 500
    arglist[1] = 1.
    self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflag)

    amin, edm, errdef = Double(), Double(), Double()
    nvpar, nparx, icstat = Long(), Long(), Long()
    self.gMinuit.mnstat( amin, edm, errdef, nvpar, nparx, icstat )

    for varidx in xrange(len(self.fitvars)):
       par, err = Double(), Double()
       self.gMinuit.GetParameter(varidx, par, err)
       osc_seed.update(self.fitvars[varidx].SetValue(par))
    for systidx in xrange(len(self.systs)):
       par, err = Double(), Double()
       self.gMinuit.GetParameter(len(self.fitvars)+systidx, par, err)
       nuis_seed.update({self.systs[systidx]:par})

    return amin
 

  def Fit(self, data, osc_seed, nuis_seed, batch=False):

    if not batch:
      print "Finding best fit for: ",
      for fitvar in self.fitvars:
        print " ", fitvar.Key(),
      for syst in self.systs:
        print " ", syst
      print "...."

    chi = self.RunFitterSeeded(data, osc_seed, nuis_seed)

    if not batch: 
      print "Best fit: ",
      for fitvar in self.fitvars:
        print ", ", fitvar.Key(), " = ", fitvar.GetValue(osc_seed[fitvar.OscKey()]),
      for syst in self.systs:
        print ", ", syst, " = ", nuis_seed[syst],
      print ", LL = ", chi

    return chi

kFitDcpInPi = FitVar('dcp', 'dcp', FitDcpInPi, lambda x: x*pi)

kFitSinSqTheta23 = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                             lambda x: asin(min(sqrt(max(0, x)), 1)), 0., 1.)
kFitSinSqTheta23UO = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                             lambda x: asin(min(sqrt(max(0, x)), 1)), 0.5, 1.)
kFitSinSqTheta23LO = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                             lambda x: asin(min(sqrt(max(0, x)), 1)), 0., 0.5)

kFitDmsq32NH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                         lambda x: x/1000., 1., 4.)

kFitDmsq32IH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                         lambda x: x/1000., -4., -1.)

kFitDmsq32 = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                         lambda x: x/1000., 1., 4., True)

kFitNumuDmsq32 = FitVarWithLL('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                         lambda x: x/1000., NumuDmsq32LL, 1., 4., True)
kFitNumuDmsq32NH = FitVarWithLL('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                         lambda x: x/1000., NumuDmsq32LL, 1., 4.)
kFitNumuDmsq32IH = FitVarWithLL('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                         lambda x: x/1000., NumuDmsq32LL, -4., -1.)
kUnFitNumuDmsq32NH = FitVarWithLL('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                         lambda x: x/1000., NumuDmsq32LL, 1., 4., False, True)
kUnFitNumuDmsq32IH = FitVarWithLL('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                         lambda x: x/1000., NumuDmsq32LL, -4., -1., False, True)
kFitNumuSinSqTheta23 = FitVarWithLL('ssth23', 'theta23', lambda x: sin(x)**2,
                         lambda x:asin(min(sqrt(max(0, x)), 1)), NumuSinSqTheta23LL, 0., 1.)
kFitNumuSinSqTheta23UO = FitVarWithLL('ssth23', 'theta23', lambda x: sin(x)**2,
                         lambda x:asin(min(sqrt(max(0, x)), 1)), NumuSinSqTheta23LL, 0.5, 1.)
kFitNumuSinSqTheta23LO = FitVarWithLL('ssth23', 'theta23', lambda x: sin(x)**2,
                         lambda x:asin(min(sqrt(max(0, x)), 1)), NumuSinSqTheta23LL, 0., 0.5)
kUnFitNumuSinSqTheta23 = FitVarWithLL('ssth23', 'theta23', lambda x: sin(x)**2,
                         lambda x:asin(min(sqrt(max(0, x)), 1)), NumuSinSqTheta23LL, 0., 1., False, True)
kUnFitNumuSinSqTheta23UO = FitVarWithLL('ssth23', 'theta23', lambda x: sin(x)**2,
                         lambda x:asin(min(sqrt(max(0, x)), 1)), NumuSinSqTheta23LL, 0.5, 1., False, True)
kUnFitNumuSinSqTheta23LO = FitVarWithLL('ssth23', 'theta23', lambda x: sin(x)**2,
                         lambda x:asin(min(sqrt(max(0, x)), 1)), NumuSinSqTheta23LL, 0., 0.5, False, True)


#  Dmsq32NHRand = GenerateRandom(SystLL, -4., 4., "dmsq_32")
#  test = TH1D("dmsq_32", "dmsq_32", 8, -1., 1.)
#  g = Dmsq32NHRand.GetPDF()
#  c = TCanvas()
#  test.Draw("hist")
#  #  test.GetYaxis().SetRangeUser(0, 1)
#  g.Draw("ac same")
#  c.Print("test.pdf")

#  for i in xrange(10000):
#    test.Fill(Dmsq32NHRand.Random())

"""
osc_data = {}
osc_data['theta23'] = asin(sqrt(0.56))
osc_data['dmsq_32'] = 2.44e-3
osc_data['dcp'] = 1.5*pi
nuis_data = {}
nuis_data['xsec_nue_sigma'] = 0.
nuis_data['xsec_numu_sigma'] = 0.
nuis_data['res_nue_sigma'] = 1.
nuis_data['res_numu_sigma'] = 1.
nuis_data['flux_sigma'] = 0.

osc_mc = {}
osc_mc['theta23'] = asin(sqrt(0.56))
osc_mc['dmsq_32'] = -2.44e-3
osc_mc['dcp'] = 0.5*pi
nuis_mc = {}
nuis_mc['xsec_nue_sigma'] = 0.
nuis_mc['xsec_numu_sigma'] = 0.
nuis_mc['res_nue_sigma'] = 0.
nuis_mc['res_numu_sigma'] = 0.0
nuis_mc['flux_sigma'] = 0.

model = Generate()
gROOT.SetBatch(True)
"""
#  gXSec = TGraphErrors()
#  gXSecCV = TGraph()
#  gFlux = TGraphErrors()
#  gFluxCV = TGraph()
#  gOsc = TGraph()
#  gPred = TGraph()
#  gOscNH = TGraphErrors()
#  gOscIH = TGraphErrors()
#  gOscShaded = TMultiGraph()
#  for i in range(1000):
#      E = 5.*(i+1)/1000.
#      #  XSeccv = model.XSec(E, 0)
#      #  Fluxcv = model.Flux(E, 0)
#      Osccv = model.NueCalc(E, osc_data['theta23'], osc_data['dmsq_32'], osc_data['dcp'], False)
#      OscNH1 = model.NueCalc(E, asin(sqrt(0.44)), osc_data['dmsq_32'], 1.5*pi, False)
#      OscNH2 = model.NueCalc(E, asin(sqrt(0.44)), osc_data['dmsq_32'], 0.5*pi, False)
#      OscIH1 = model.NueCalc(E, osc_data['theta23'], -osc_data['dmsq_32'], 0.5*pi, False)
#      OscIH2 = model.NueCalc(E, osc_data['theta23'], -osc_data['dmsq_32'], 1.5*pi, False)
#      OscNHCV = 0.5*(OscNH1+OscNH2)
#      OscIHCV = 0.5*(OscIH1+OscIH2)
#      gOscNH.SetPoint(i, E, OscNHCV)
#      gOscNH.SetPointError(i, E, abs(OscNH1-OscNHCV))
#      gOscIH.SetPoint(i, E, OscIHCV)
#      gOscIH.SetPointError(i, E, abs(OscIH1-OscIHCV))
#
#      #  gXSec.SetPoint(i, E, XSeccv)
#      #  gXSecCV.SetPoint(i, E, XSeccv)
#      #  gXSec.SetPointError(i, E, model.XSec(E, 1) - XSeccv)
#      #  gFlux.SetPoint(i, E, Fluxcv)
#      #  gFluxCV.SetPoint(i, E, Fluxcv)
#      #  gFlux.SetPointError(i, E, Fluxcv - model.Flux(E, -1) )
#      #  gOsc.SetPoint(i, E, Osccv)
#      #  gPred.SetPoint(i, E, 3.e4*Fluxcv*XSeccv*Osccv)
#
#  gStyle.SetTitleSize(0.08, "t")
#  c = TCanvas()
#  gXSecCV.SetLineColor(kRed)
#  gXSec.SetLineColor(kRed)
#  gXSec.SetFillColor(kRed-7)
#  gXSec.Draw("a3 same")
#  gXSecCV.Draw("p same")
#  gXSec.GetXaxis().SetRangeUser(0, 5)
#  gXSec.GetYaxis().SetTitle("Arbitrary Units")
#  gXSec.GetYaxis().SetTitleSize(0.05)
#  gXSec.GetYaxis().SetTitleOffset(0.9)
#  gXSec.GetXaxis().SetTitle("Neutrino Energy (GeV)")
#  gXSec.GetXaxis().SetTitleSize(0.05)
#  gXSec.GetXaxis().SetLabelSize(0.04)
#  gXSec.GetXaxis().SetTitleOffset(0.9)
#  gXSec.SetTitle("#nu_{e} Interaction Cross-Section")
#  c.Print("xsec.pdf")
#
#  c2 = TCanvas()
#  gFluxCV.SetLineColor(kRed)
#  gFlux.SetLineColor(kBlue)
#  gFlux.SetFillColor(kBlue-7)
#  gFlux.Draw("a3 same")
#  gFluxCV.Draw("p same")
#  gFlux.GetXaxis().SetRangeUser(0, 5)
#  gFlux.GetYaxis().SetTitle("Arbitrary Units")
#  gFlux.GetYaxis().SetTitleSize(0.05)
#  gFlux.GetYaxis().SetTitleOffset(0.9)
#  gFlux.GetXaxis().SetTitle("Neutrino Energy (GeV)")
#  gFlux.GetXaxis().SetTitleSize(0.05)
#  gFlux.GetXaxis().SetLabelSize(0.04)
#  gFlux.GetXaxis().SetTitleOffset(0.9)
#  gFlux.SetTitle("#nu_{#mu} Flux")
#  c2.Print("flux.pdf")
#
#  c3 = TCanvas()
#  gOsc.SetLineColor(kRed)
#  gOsc.Draw("ac same")
#  gOsc.GetXaxis().SetRangeUser(0, 5)
#  gOsc.GetYaxis().SetRangeUser(0, 0.15)
#  gOsc.GetYaxis().SetTitle("P(#nu_{#mu} #rightarrow #nu_{e})")
#  gOsc.GetYaxis().SetTitleSize(0.05)
#  gOsc.GetYaxis().SetTitleOffset(0.9)
#  gOsc.GetXaxis().SetTitle("Neutrino Energy (GeV)")
#  gOsc.GetXaxis().SetTitleSize(0.05)
#  gOsc.GetXaxis().SetLabelSize(0.04)
#  gOsc.GetXaxis().SetTitleOffset(0.9)
#  gOsc.SetTitle("Oscillation Probability")
#  c3.Print("osc.pdf")
#
#  c4 = TCanvas()
#  gPred.SetLineColor(kRed)
#  gPred.Draw("ac same")
#  gPred.GetXaxis().SetRangeUser(0, 5)
#  gPred.GetYaxis().SetTitle("Events")
#  gPred.GetYaxis().SetTitleSize(0.05)
#  gPred.GetYaxis().SetTitleOffset(0.9)
#  gPred.GetXaxis().SetTitle("Neutrino Energy (GeV)")
#  gPred.GetXaxis().SetTitleSize(0.05)
#  gPred.GetXaxis().SetLabelSize(0.04)
#  gPred.GetXaxis().SetTitleOffset(0.9)
#  gPred.SetTitle("Prediction")
#  c4.Print("pred.pdf")

#  c5 = TCanvas()
#  gOscNH.SetFillColorAlpha(kBlue, 0.57)
#  gOscIH.SetFillColorAlpha(kRed, 0.57)
#  gOscShaded.Add(gOscNH)
#  gOscShaded.Add(gOscIH)
#  gOscShaded.Draw("a3")
#  gOscShaded.GetXaxis().SetRangeUser(0, 5)
#  gOscShaded.GetYaxis().SetRangeUser(0, 0.12)
#  gOscShaded.GetYaxis().SetTitle("P(#nu_{#mu} #rightarrow #nu_{e})")
#  gOscShaded.GetYaxis().SetTitleSize(0.05)
#  gOscShaded.GetYaxis().SetTitleOffset(0.9)
#  gOscShaded.GetXaxis().SetTitle("Neutrino Energy (GeV)")
#  gOscShaded.GetXaxis().SetTitleSize(0.05)
#  gOscShaded.GetXaxis().SetLabelSize(0.04)
#  gOscShaded.GetXaxis().SetTitleOffset(0.9)
#  gOscShaded.GetHistogram().SetTitle("Oscillation Probability")
#
#  leg = TLegend(0.5, 0.5, 0.85, 0.85)
#  leg.AddEntry(gOscNH, "#splitline{NH, sin^{2}#theta_{23} = 0.44}{#delta_{CP} = (0, 2#pi)}", "f")
#  leg.AddEntry(gOscIH, "#splitline{IH, sin^{2}#theta_{23} = 0.56}{#delta_{CP} = (0, 2#pi)}", "f")
#  leg.SetBorderSize(0)
#  leg.SetTextSize(0.04)
#  leg.Draw()
#  c5.Print("osc_shaded.pdf")

#  gStyle.SetOptStat(0)
#
#

#  c = TCanvas()
#  model.SetParams(osc_data, nuis_data)
#  model.trueENumu.Draw()
#  model.recoENumu.SetLineColor(kBlue)
#  model.recoENumu.Draw("same")
#  x = array('d', 1000*[0.])
#  w = array('d', 1000*[0.])
#  model.recoENumu.CalcGaussLegendreSamplingPoints(1000, x, w, 1e-13)
#  print model.recoENumu.IntegralFast(1000, x, w, 1.2, 1.7), model.trueENumu.Integral(1.2, 1.7)
#  c.Print("numu_pred.pdf")
#
#  c2 = TCanvas()
#  model.SetParams(osc_data, nuis_data)
#  model.trueENue.Draw()
#  model.recoENue.SetLineColor(kBlue)
#  model.recoENue.Draw("same")
#  c2.Print("nue_pred.pdf")
#  x = array('d', 1000*[0.])
#  w = array('d', 1000*[0.])
#  model.recoENue.CalcGaussLegendreSamplingPoints(1000, x, w, 1e-13)

"""
data = model.Data(osc_data, nuis_data, False)
mc1 = model.MC(osc_data, nuis_data)
mc2 = model.MC(osc_mc, nuis_mc)
data_numu = TH1D("data_numu", "data_numu", 20, 0, 5)
data_nue = TH1D("data_nue", "data_nue", 10, 0, 5)
mc1_numu = TH1D("mc1_numu", "mc1_numu", 20, 0, 5)
mc1_nue = TH1D("mc1_nue", "mc1_nue", 10, 0, 5)
mc2_numu = TH1D("mc2_numu", "mc2_numu", 20, 0, 5)
mc2_nue = TH1D("mc2_nue", "mc2_nue", 10, 0, 5)
for i in range(1, 25):
    if i <= 8:
        data_nue.SetBinContent(i+1, data.GetBinContent(i))
        mc1_nue.SetBinContent(i+1, mc1.GetBinContent(i))
        mc2_nue.SetBinContent(i+1, mc2.GetBinContent(i))
    else:
        data_numu.SetBinContent(i-6, data.GetBinContent(i))
        mc1_numu.SetBinContent(i-6, mc1.GetBinContent(i))
        mc2_numu.SetBinContent(i-6, mc2.GetBinContent(i))


c5 = TCanvas()
print data_nue.Integral(), mc1_nue.Integral(), mc2_nue.Integral()
data_nue.SetLineColor(kBlack)
mc1_nue.SetLineColor(kBlue)
mc2_nue.SetLineColor(kRed)
mc1_nue.SetLineWidth(2)
mc2_nue.SetLineWidth(2)
data_nue.Draw("ep same")
mc1_nue.Draw("hist same")
mc2_nue.Draw("hist same")
data_nue.GetYaxis().SetRangeUser(0, 1.3*data_nue.GetMaximum())
data_nue.GetXaxis().SetTitle("Neutrino Energy (GeV)")
data_nue.GetYaxis().SetTitle("Number of Events")
data_nue.SetTitle("#nu_{#mu} #rightarrow #nu_{e}")
data_nue.GetYaxis().SetTitleSize(0.05)
data_nue.GetYaxis().SetTitleOffset(0.9)
data_nue.GetXaxis().SetTitleSize(0.05)
data_nue.GetXaxis().SetLabelSize(0.04)
data_nue.GetXaxis().SetTitleOffset(0.9)

leg_nue = TLegend(0.45, 0.5, 0.85, 0.85)
leg_nue.AddEntry(data_nue, "Mock Observation", "le")
leg_nue.AddEntry(mc1_nue, "#splitline{Prediction}{(NH, sin^{2}#theta_{23}= 0.56, #delta_{CP}= 3#pi/2)}", "l")
leg_nue.AddEntry(mc2_nue, "#splitline{Prediction}{(IH, sin^{2}#theta_{23}= 0.56, #delta_{CP}= #pi/2)}", "l")
leg_nue.SetBorderSize(0)
leg_nue.SetTextSize(0.04)
#  leg_nue.Draw()

c5.Print("datamc_nue.pdf")

c6 = TCanvas()
print data_numu.Integral(), mc1_numu.Integral(), mc2_numu.Integral()
data_numu.SetLineColor(kBlack)
mc1_numu.SetLineColor(kBlue)
mc2_numu.SetLineColor(kRed)
mc1_numu.SetLineWidth(2)
mc2_numu.SetLineWidth(2)
data_numu.Draw("ep same")
mc1_numu.Draw("hist same")
mc2_numu.Draw("hist same")
data_numu.GetYaxis().SetRangeUser(0, 1.3*data_numu.GetMaximum())
data_numu.GetXaxis().SetTitle("Neutrino Energy (GeV)")
data_numu.GetYaxis().SetTitle("Number of Events")
data_numu.SetTitle("#nu_{#mu} #rightarrow #nu_{#mu}")
data_numu.GetYaxis().SetTitleSize(0.05)
data_numu.GetYaxis().SetTitleOffset(0.9)
data_numu.GetXaxis().SetTitleSize(0.05)
data_numu.GetXaxis().SetLabelSize(0.04)
data_numu.GetXaxis().SetTitleOffset(0.9)

leg_numu = TLegend(0.45, 0.6, 0.85, 0.85)
leg_numu.AddEntry(data_numu, "Mock Observation", "le")
leg_numu.AddEntry(mc1_numu, "Prediction (NH, sin^{2}#theta_{23}= 0.56, #delta_{CP}= 3#pi/2)", "l")
leg_numu.AddEntry(mc2_numu, "Prediction (IH, sin^{2}#theta_{23}= 0.56, #delta_{CP}= #pi/2)", "l")
leg_numu.SetBorderSize(0)
#  leg_numu.Draw()

c6.Print("datamc_numu.pdf")
"""
