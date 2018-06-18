from ROOT import *
from array import array
from math import *
import numpy as np
from scipy.linalg import eigh

class Generate:
  def __init__(self):
     
    self.trueE = TF1("trueE", self.TrueE, 0.1, 5, 5)
    self.trueE.SetParName(0, "xsec_sigma")
    self.trueE.SetParName(1, "flux_sigma")
    self.trueE.SetParName(2, "theta23")
    self.trueE.SetParName(3, "dmsq_32")
    self.trueE.SetParName(4, "dcp")

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
    Simulated NueCC - QE XSec
    sigma represents shift from nominal
    """
    weight = max(0.84, 1. + sigma*0.04)
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
    weight = max(0.42, 0.5 + sigma*0.02)
    form = weight*(TMath.Landau(E, 2, weight))
    return form

  def InitOsc(self, th23, dm2_32, dcp):
    
    th13 = asin(sqrt(2.10e-2))    # reactor mixing angle
    th12 = asin(sqrt(0.307))      
    dm2_21 = 7.53e-5
    self.dm2_31 = dm2_32+dm2_21

    h11 = dm2_21/self.dm2_31
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

  def Calc(self, E, th23, dm2_32, dcp):
    """
    Calculator for oscillation probabilities
    Args: 
        theta23 - mixing angle, range is 0 to pi/2
        dcp - CP violating phase, range is 0 to 2pi
        dm2_32 - 3-2 squared mass difference, range is -inf to +inf 
          (Note: 2e-3 < abs(dm2_32) < 3e-3 for realistic situations)
    """
     
    L=810.          # baseline
    self.InitOsc(th23, dm2_32, dcp)
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

  def TrueE(self, x, par):
    toy_pot = 4e4
    unosc = self.XSec(x[0], par[0])*self.Flux(x[0], par[1])
    osc_weight = self.Calc(x[0], par[2], par[3], par[4])
    return toy_pot*unosc*osc_weight

  def SetParams(self, osc_params, nuis_params):

    for key in osc_params.keys():
      self.trueE.SetParameter(key, osc_params[key])
    for key in nuis_params.keys():
      self.trueE.SetParameter(key, nuis_params[key])

  def MC(self, osc_params, nuis_params, hname="recoE_pred"):
    
    self.SetParams(osc_params, nuis_params)
    nbins_pred = 8
    recoE_pred = TH1D(hname, hname, nbins_pred, 0.5, 4.5)
    # actually fill prediction in analysis bins
    for binx in range(nbins_pred):
      recoE_pred.SetBinContent(binx+1, self.trueE.Integral((float(binx)/2.)+0.5, (float(binx)/2.)+1.))

    return recoE_pred

  def Data(self, osc_params, nuis_params, isfake=False, hname="recoE_data"):


    self.SetParams(osc_params, nuis_params)
    if isfake:
      recoE_data = self.MC(osc_params, nuis_params, hname)
    else: 
      nbins = 8
      pois = TRandom3()
      pois.SetSeed(0)
      samples = pois.Poisson(self.trueE.Integral(0.1,5))

      recoE_data = TH1D(hname, hname, nbins, 0.5, 4.5)
      for i in xrange(samples+10000):
        trueE_i = self.trueE.GetRandom()
        res_i = 1.
        #  res_i = self.resolution.GetRandom()
        # ROOT probably needs some iterations to truly get random values from TF1. Do that
        if i > 10000:
          recoE_data.Fill(trueE_i*res_i)

      pois.Delete()
    return recoE_data

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

class FitVar:
  def __init__(self, key, osc_key, fcn, invfcn):
    self.key = key
    self.osc_key = osc_key
    self.fcn = fcn
    self.invfcn = invfcn

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

class FitConstrainedVar(FitVar):
  def __init__(self, key, osc_key, fcn, invfcn, lowlimit, highlimit, strong=True):
    FitVar.__init__(self, key, osc_key, fcn, invfcn)
    self.lo = lowlimit
    self.hi = highlimit
    self.constraint = strong

  def Penalty(self, value):
    if(value >= self.lo and value <= self.hi): return 0.
    mean = (self.lo + self.hi)/2.
    rad = (self.hi - self.lo)/2.
    return (((value-mean)/rad)**2) - 1.

  def Clamp(self, value):
    return max(self.lo, min(value, self.hi))

  def SetValue(self, *args):
    param = {}
    if self.constraint:
        param[self.osc_key] = self.invfcn(self.Clamp(*args))
    else:
        param[self.osc_key] = self.invfcn(*args)

    return param


def FitDcpInPi(dcp):
  ret = dcp/pi;
  a = long(long(ret/2+1))
  ret -= 2*a
  while ret < 0.: ret += 2
  while ret > 2.: ret -= 2

  return ret

kFitDcpInPi = FitVar('dcp', 'dcp', FitDcpInPi, lambda x: x*pi)

kFitSinSqTheta23 = FitConstrainedVar('ssth23','theta23', lambda x: sin(x)**2,
                                      lambda x: asin(sqrt(x)), 0.3, 0.7, False)

# For numu disappearance, not relevant right now
kFitSinSq2Theta23 = FitConstrainedVar('ssth23', 'theta23', lambda x: sin(2*x)**2,
                                      lambda x: asin(sqrt(x))/2., 0., 1.)

kFitDmsq32 = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                                lambda x: x/1000., -4, 4)
kFitDmsq32NH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                                lambda x: x/1000., 0., 4.)
kFitDmsq32IH = FitConstrainedVar('dmsq_32', 'dmsq_32', lambda x: x*1000.,
                                lambda x: x/1000., -4, 0.)

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
 

  def Fit(self, data, osc_seed, nuis_seed):

    print "Finding best fit for: ",
    for fitvar in self.fitvars:
      print " ", fitvar.Key(),
    for syst in self.systs:
      print " ", syst
    print "...."

    chi = self.RunFitterSeeded(data, osc_seed, nuis_seed)

    print "Best fit: ",
    for fitvar in self.fitvars:
      print ", ", fitvar.Key(), " = ", fitvar.GetValue(osc_seed[fitvar.OscKey()]),
    for syst in self.systs:
      print ", ", syst, " = ", nuis_seed[syst],
    print ", LL = ", chi

    return chi

osc_data = {}
osc_data['theta23'] = 0.25*pi
osc_data['dmsq_32'] = 2.44e-3
osc_data['dcp'] = 1.5*pi
nuis_data = {}
nuis_data['xsec_sigma'] = 0.
nuis_data['flux_sigma'] = 0.

osc_mc = {}
osc_mc['theta23'] = 0.25*pi
osc_mc['dmsq_32'] = -2.44e-3
osc_mc['dcp'] = 0.5*pi
nuis_mc = {}
nuis_mc['xsec_sigma'] = 0.
nuis_mc['flux_sigma'] = 0.

model = Generate()

#  osc_seed = osc_mc.copy()
#  nuis_seed = nuis_mc.copy()
#  #  fitter = Fitter([kFitSinSqTheta23, kFitDmsq32IH], [])
#  #  #  fitter = Fitter([kFitDcpInPi, kFitSinSqTheta23],['xsec_sigma', 'flux_sigma'])
#  #  fitter.InitMinuit()
#  model = Generate()
#  mock_data = model.Data(osc_data, nuis_data, True)
#  print "---------"
#  mock_mc = model.MC(osc_seed, nuis_seed)
#  osc_seed['theta23']=0.5*pi
#  mock_mc2 = model.MC(osc_seed, nuis_seed)
#  print mock_data.Integral(), mock_mc.Integral(), mock_mc2.Integral()
#  print Experiment(mock_mc, mock_data).Likelihood(), Experiment(mock_mc2, mock_data).Likelihood()
#  #  mock_ratio.Divide(mock_mc)
#  c = TCanvas()
#  mock_data.Draw("hist")
#  mock_mc.SetLineColor(kRed)
#  mock_mc2.SetLineColor(kGreen+2)
#  mock_mc.Draw("hist same")
#  mock_mc2.Draw("hist same")
#  c.Print("test.pdf")
#  #  x_data = array('d')
#  #  y_data = array('d')
#  #  for i in range(50):
#  #    osc_seed = osc_mc.copy()
#  #    nuis_seed = nuis_mc.copy()
#  #    osc_seed['dcp'] = i*pi/25.
#  #    #  print osc_seed
#  #    mock_mc = model.MC(osc_seed, nuis_seed)
#  #    ll = fitter.Fit(mock_data, osc_seed, nuis_seed)
#  #    #  print nuis_seed
#  #    #  print osc_seed['dcp']/pi, ll
#  #    #  print Experiment(mock_mc, mock_data).Likelihood()
#  #    x_data.append(i/25.)
#  #    y_data.append(ll)
#  #    #  mock_mc.Delete()
#  #
#  #  c1 = TCanvas()
#  #  gr = TGraph(len(x_data), x_data, y_data)
#  #  gr.Draw()
#  #  c1.Print("test.pdf")
