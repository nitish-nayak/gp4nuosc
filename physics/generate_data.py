from ROOT import *
from math import *
from toy_experiment import *

import sys
import os
import pickle


output_dir = sys.argv[1]
n_data = int(sys.argv[2])

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

# current best values
osc_data = {'theta23': asin(sqrt(0.56)), 'dcp': 1.5 * pi, 'dmsq_32': 2.44 * 1e-3}

for i in range(n_data):
	Gaus = GenerateRandom(SystLL, -4., 4., "syst")
	nuis_data = {'xsec_nue_sigma': Gaus.Random(), 'xsec_numu_sigma': Gaus.Random(), 'flux_sigma': Gaus.Random()}
	model = Generate()
	realdata = model.Data(osc_data, nuis_data)
	path = os.path.join(output_dir, 'realdata_{}.pkl'.format(i))
	pickle.dump(realdata, open(path, 'wb'))
	realdata.Delete()
