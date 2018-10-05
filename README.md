# ToyNuOscCI

Toy Feldman-Cousins Study for Neutrino Oscillations 

## Toy Neutrino Oscillation Experiment

A Toy Experiment modelled on NOvA is mocked up. An input flux of muon neutrinos is oscillated into electron neutrinos using the full 3-flavor PMNS formulation with the MSW effect. A toy cross-section is then used to get the final estimate of oscillated electron neutrinos. A mock data histogram is calculated by a Poisson variation of the prediction for given input PMNS parameters and is fit to various predictions. The oscillation parameters are then estimated by maximising the corresponding likelihood. The toy analysis uses 8 bins from 0.5-4.5 GeV of neutrino energies and no background is assumed. 
* Implemented in _physics/toy_experiment.py_

[!alt text](./pred_vs_data.png)

## Feldman Cousins

Contours/Slices of the likelihood ratio test (LRT) thresholds are made using the Feldman-Cousins method. 
* Implemented in _physics/fc.py, physics/fc_helper.py_

[!alt text](./threshold.png)

## Approximation

Apply Gaussian process to approximate the FC contour and test coverage
* Implemented in _approximation.py_
* _utils.py, helper.py_: utility and helper functions

## Illustrative Notebooks

* _approximation.ipynb_: example of approximation contours
* _compare_normal.ipynb_: compare approximation coverage on many data sets
