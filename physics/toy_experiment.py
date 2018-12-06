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
    if E > 1:
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

  def SetParams(self, osc_params, nuis_params):

    for key in osc_params.keys():
      self.trueENue.SetParameter(key, float(osc_params[key]))
      self.trueENumu.SetParameter(key, float(osc_params[key]))
    for key in nuis_params.keys():
      self.trueENue.SetParameter(key, float(nuis_params[key]))
      self.trueENumu.SetParameter(key, float(nuis_params[key]))

  def MC(self, osc_params, nuis_params, hname="recoE_pred"):
    
    self.SetParams(osc_params, nuis_params)
    nbins_pred = 24
    recoE_pred = TH1D(hname, hname, nbins_pred, 0.5, 12.5)
    # actually fill prediction in analysis bins
    for binx in range(8):
      recoE_pred.SetBinContent(binx+1, self.trueENue.Integral((float(binx)/2.)+0.5, (float(binx)/2.)+1.))
    for binx in range(16):
      recoE_pred.SetBinContent(8+binx+1, self.trueENumu.Integral((float(binx)/4.)+0.5, (float(binx)/4.)+0.75))

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
      gRandom.SetSeed(0)
      poisNue = TRandom3()
      poisNue.SetSeed(0)
      poisNumu = TRandom3()
      poisNumu.SetSeed(0)
      samplesNue = poisNue.Poisson(self.trueENue.Integral(0.1,5))
      samplesNumu = poisNumu.Poisson(self.trueENumu.Integral(0.1,5))
      recoE_data = TH1D(hname, hname, nbins, 0.5, 12.5)
      for i in xrange(samplesNue):
        trueE_i = self.trueENue.GetRandom()
        res_i = 1.
        if trueE_i <= 4.5 and trueE_i > 0.5: 
          recoE_data.Fill(trueE_i)
      for i in xrange(samplesNumu):
        trueE_i = self.trueENumu.GetRandom()
        if trueE_i <= 4.5 and trueE_i > 0.5:
          recoE_data.Fill(4.5+(2.*trueE_i-1.))

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
  #  test.Fill(Dmsq32NHRand.Random())
#
#  osc_data = {}
#  osc_data['theta23'] = asin(sqrt(0.56))
#  osc_data['dmsq_32'] = 2.44e-3
#  osc_data['dcp'] = 1.5*pi
#  nuis_data = {}
#  nuis_data['xsec_nue_sigma'] = 0.
#  nuis_data['xsec_numu_sigma'] = 0.
#  nuis_data['flux_sigma'] = 0.
#  osc_mc = {}
#
#  osc_mc['theta23'] = asin(sqrt(0.56))
#  osc_mc['dmsq_32'] = -2.44e-3
#  osc_mc['dcp'] = 0.5*pi
#  nuis_mc = {}
#  nuis_mc['xsec_nue_sigma'] = 0.
#  nuis_mc['xsec_numu_sigma'] = 0.
#  nuis_mc['flux_sigma'] = 0.
#
#  model = Generate()
#  gROOT.SetBatch(True)
#  gStyle.SetOptStat(0)
#
#  c = TCanvas()
#
#  data = model.Data(osc_data, nuis_data, False)
#  data.SetLineColor(kBlack)
#  mc1 = model.MC(osc_data, nuis_data)
#  mc1.SetLineColor(kBlue)
#  mc2 = model.MC(osc_mc, nuis_mc)
#  mc2.SetLineColor(kRed)
#  data.Draw("ep same")
#  mc1.Draw("hist same")
#  mc2.Draw("hist same")
#  data.GetYaxis().SetRangeUser(0, 2*data.GetMaximum())
#  data.GetXaxis().SetTitle("Neutrino Energy")
#  data.GetYaxis().SetTitle("Number of Events")
#  data.SetTitle("")
#
#  leg = TLegend(0.45, 0.6, 0.85, 0.85)
#  leg.AddEntry(data, "Mock Observation", "le")
#  leg.AddEntry(mc1, "Prediction (NH, sin^{2}#theta_{23}= 0.5, #delta_{CP}= 3#pi/2)", "l")
#  leg.AddEntry(mc2, "Prediction (IH, sin^{2}#theta_{23}= 0.5, #delta_{CP}= #pi/2)", "l")
#  leg.SetBorderSize(0)
#  leg.Draw()
#
#  c.Print("test.pdf")
