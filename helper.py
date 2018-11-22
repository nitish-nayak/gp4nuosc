"""
This script has helper functions to load data into numpy arrays:
load_contour_dist and load_fitted_contour
as well as helper functions to perform numerical experiments.
"""

import matplotlib.pyplot as plt
from utils import *
#from approximation import adaptive_search


def load_contour_dist_2d(file_pattern, grid_size, n_sample):
    """
    Load likelihood ratio test statistic distributions from simulated data files.
    """
    contour_dist = np.zeros((grid_size, grid_size, n_sample))
    for i in range(grid_size * grid_size):
        j = i / grid_size
        k = i % grid_size
        loglik_global = []
        loglik_profile = []
        try:
            # TODO: file_pattern seems like a dumb way of handling files
            with open(file_pattern + '_{a}.{b}_of_1600.txt'.format(a=i, b=i + 1), 'r') as f:
                for line in f:
                    if len(line.split(',')) == 10:
                        loglik_global.append(float(line.split(',')[8][1:]))
                        loglik_profile.append(float(line.split(',')[9][1:-2]))
            loglik_global = np.asarray(loglik_global)
            loglik_profile = np.asarray(loglik_profile)
            lrt = 2 * (loglik_profile - loglik_global)  # distribution of likelihood ratio test statistic
            n = lrt.shape[0]
            if n < n_sample:  # not enough
                contour_dist[j, k, :n] = lrt
                contour_dist[j, k, n:] = -1
            else:
                contour_dist[j, k, :] = lrt[:n_sample]
        except:
            print(i)
            contour_dist[j, k, :] = contour_dist[j, k - 1, :]  # catch occasional missing data files
    return contour_dist


def load_contour_stat_2d(file_path, grid_size):
    """
    Load contour of fitted LRT statistic from data in text file.
    """
    loglik_global = []
    loglik_profile = []
    with open(file_path, 'r') as f:
        for line in f:
            loglik_global.append(float(line.split(',')[8][1:]))
            loglik_profile.append(float(line.split(',')[9][1:-2]))
    loglik_global = np.asarray(loglik_global)
    loglik_profile = np.asarray(loglik_profile)
    contour_stat = np.zeros((grid_size, grid_size))
    # TODO: implement 1d version
    for i in range(grid_size * grid_size):
        j = i / grid_size
        k = i % grid_size
        contour_stat[j, k] = 2 * (loglik_profile[i] - np.min(loglik_profile))
    return contour_stat


def calculate_overlap(hat_grid, contour_tile):
    contour_68_diff = (hat_grid < 0.68) != (contour_tile < 0.68)
    contour_90_diff = (hat_grid < 0.90) != (contour_tile < 0.90)
    return 0.5 * (1 - np.mean(contour_68_diff)) + 0.5 * (1 - np.mean(contour_90_diff))


"""
def perform_comparison(real_data_num, hierarchy, init_size, n_iter, iter_size, post_hoc_smooth=True, verbose=True):
    if hierarchy == 'normal':
        contour_thres = np.load('/Users/linggeli/monte_carlo/data/penalty_normal_contour_thres_68.npy')
        contour_stat = load_fitted_contour('/Users/linggeli/monte_carlo/data/penalty_normal/contour_normal_fit_{}.txt'.format(real_data_num))
    elif hierarchy == 'inverted':
        contour_thres = np.load('/Users/linggeli/monte_carlo/penalty_contour_inverted_68.npy')
        contour_stat = load_fitted_contour('/Users/linggeli/Downloads/penalty_inverted/contour_inverted_fit_{}.txt'.format(real_data_num))

    contour_true = contour_stat < contour_thres
    data = grid_to_data(contour_thres)  # format data
    all_points = data[:, 0:2]
    target = data[:, 2]
    lr_stats = grid_to_data(contour_stat)[:, 2]

    contour_thres_hat, contour_hat = adaptive_search(all_points, target, lr_stats, init_size, n_iter, iter_size)  # approximation

    if post_hoc_smooth:
        contour_true = smooth_grid(contour_true, 40) > 0.5
        contour_hat = smooth_grid(contour_hat, 40) > 0.5

    contour_diff = contour_hat != contour_true
    overlap = 1 - np.mean(contour_diff)

    if verbose:
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(contour_true, cmap='Greens', interpolation='nearest', alpha=0.5)
        plt.imshow(contour_hat, cmap='Blues', interpolation='nearest', alpha=0.5)
        plt.show()
        print('Percentage of pointwise overlap: {}'.format(overlap))

    return overlap
"""