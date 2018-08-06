from ROOT import *
from array import array
from math import *
import random
import sys
from toy import Generate
import pickle

# current best values
osc_data = {}
osc_data['theta23'] = 0.886077124
osc_data['dcp'] = 1.25 * pi
osc_data['dmsq_32'] = 2.52 * 1e-3
nuis_data = {}
nuis_data['xsec_sigma'] = 0.
nuis_data['flux_sigma'] = 0.

for i in range(500):
	# generate and save data
	model = Generate()
	mock_data = model.Data(osc_data, nuis_data)
	pickle.dump(mock_data, open('realdata_{}.pkl'.format(i), 'wb'))
	mock_data.Delete()