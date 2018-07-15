import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from utils import grid_to_data
from approximation import adaptive_search


def load_fitted_contour(real_data_path):
	loglik_global = []
	loglik_profile = []
	with open(real_data_path, 'r') as f:
	    for line in f: 
	        loglik_global.append(float(line.split(',')[8][1:]))
	        loglik_profile.append(float(line.split(',')[9][1:-2]))
	loglik_global = np.asarray(loglik_global)
	loglik_profile = np.asarray(loglik_profile)
	contour_fit = np.zeros((20, 20))
	for i in range(400):
	    j = i / 20
	    k = i % 20
	    contour_fit[j, k] = 2 * (loglik_profile[i] - np.min(loglik_profile))
	return(contour_fit)


def grid_to_points(grid):
    points = np.zeros((20 * 20, 2))
    values = np.zeros(20 * 20)
    n = 0
    for i in range(20):
        for j in range(20):
            points[n, 0] = i * 1.0 / 20
            points[n, 1] = j * 1.0 / 20
            values[n] = grid[i, j]
            n += 1
    return(points, values)


def refine_grid(points, res):
    fine_grid = np.zeros((res * res, 2))
    n = 0
    for i in range(res):
        for j in range(res):
            fine_grid[n, 0] = i * np.max(points[:, 0]) / res
            fine_grid[n, 1] = j * np.max(points[:, 1]) / res
            n += 1
    return(fine_grid)


def smooth_grid(grid, res):
    points, values = grid_to_points(grid)
    fine_grid = refine_grid(points, res)
    return(griddata(points, values, fine_grid, method='linear').reshape((res, res)))


def perform_comparison(real_data_num, post_hoc_smooth=True, verbose=True):
    contour_thres = np.load('/Users/linggeli/monte_carlo/penalty_contour_68.npy')
    contour_stat = load_fitted_contour('/Users/linggeli/Downloads/penalty_inverted/contour_inverted_fit_{}.txt'.format(real_data_num))
    contour_true = contour_stat < contour_thres
    data = grid_to_data(contour_thres)
    all_points = data[:, 0:2]
    target = data[:, 2]
    lr_stats = grid_to_data(contour_stat)[:, 2]
    contour_thres_hat, contour_hat = adaptive_search(all_points, target, lr_stats, 50, 5, 10, verbose=False)
    if post_hoc_smooth:
        contour_true = smooth_grid(contour_true, 40) > 0.5
        contour_hat = smooth_grid(contour_hat, 40) > 0.5
    contour_diff = contour_hat != contour_true
    if verbose:
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(contour_true, cmap='Greens', interpolation='nearest', alpha=0.5)
        plt.imshow(contour_hat, cmap='Blues', interpolation='nearest', alpha=0.5)
        plt.show()
    overlap = 1 - np.mean(contour_diff)
    print('Percentage of pointwise overlap: {}'.format(overlap))
    return(overlap)