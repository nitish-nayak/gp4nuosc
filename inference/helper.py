"""
This script has helper functions to load data into numpy arrays:
load_contour_dist and load_fitted_contour
as well as helper functions for experiments.
"""

from scipy.stats import norm
from utils import *


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
            with open(file_pattern + '_{a}.{b}_of_400.txt'.format(a=i, b=i + 1), 'r') as f:
                for line in f:
                    if len(line.split(',')) == 12:
                        loglik_global.append(float(line.split(',')[10][1:]))
                        loglik_profile.append(float(line.split(',')[11][1:-2]))
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
            loglik_global.append(float(line.split(',')[10][1:]))
            loglik_profile.append(float(line.split(',')[11][1:-2]))
    loglik_global = np.asarray(loglik_global)
    loglik_profile = np.asarray(loglik_profile)
    contour_stat = np.zeros((grid_size, grid_size))
    for i in range(grid_size * grid_size):
        j = i / grid_size
        k = i % grid_size
        contour_stat[j, k] = 2 * (loglik_profile[i] - np.min(loglik_profile))
    return contour_stat


def load_contour_dist_1d(file_pattern, grid_size, n_sample):
    """
    Load likelihood ratio test statistic distributions from simulated data files.
    """
    contour_dist = np.zeros((grid_size, n_sample))
    for i in range(grid_size):
        loglik_global = []
        loglik_profile = []
        try:
            # TODO: file_pattern seems like a dumb way of handling files
            with open(file_pattern + '_{a}.{b}_of_20.txt'.format(a=i, b=i + 1), 'r') as f:
                for line in f:
                    if len(line.split(',')) == 12:
                        loglik_global.append(float(line.split(',')[10][1:]))
                        loglik_profile.append(float(line.split(',')[11][1:-2]))
            loglik_global = np.asarray(loglik_global)
            loglik_profile = np.asarray(loglik_profile)
            lrt = 2 * (loglik_profile - loglik_global)  # distribution of likelihood ratio test statistic
            n = lrt.shape[0]
            if n < n_sample:  # not enough
                contour_dist[i, :n] = lrt
                contour_dist[i, n:] = -1
            else:
                contour_dist[i, :] = lrt[:n_sample]
        except:
            print(i)
            contour_dist[i, :] = contour_dist[i - 1, :]  # catch occasional missing data files
    return contour_dist


def load_contour_stat_1d(file_path, grid_size):
    """
    Load contour of fitted LRT statistic from data in text file.
    """
    loglik_global = []
    loglik_profile = []
    with open(file_path, 'r') as f:
        for line in f:
            loglik_global.append(float(line.split(',')[10][1:]))
            loglik_profile.append(float(line.split(',')[11][1:-2]))
    loglik_global = np.asarray(loglik_global)
    loglik_profile = np.asarray(loglik_profile)
    contour_stat = np.zeros(grid_size)
    for i in range(grid_size):
        contour_stat[i] = 2 * (loglik_profile[i] - np.min(loglik_global))
    return contour_stat


def calculate_percentile_2d(contour_dist, contour_stat):
    """
    Calculate percentile of statistic in distribution on 2d contour.
    :param contour_dist: (3d numpy array) [grid_size, grid_size, sample_size] reference distribution on 2d grid
    :param contour_stat: (2d numpy array) [grid_size, grid_size] observed statistic on 2d grid
    :return: (2d numpy array) [grid_size, grid_size] percentile between 0 and 1 on 2d grid
    """
    grid_size = contour_dist.shape[0]
    contour_tile = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            stat = contour_stat[i, j]
            reference = np.sort(contour_dist[i, j])
            contour_tile[i, j] = np.searchsorted(reference, stat) * 1.0
    contour_tile = contour_tile / contour_dist.shape[2]
    return contour_tile


def calculate_percentile_1d(contour_dist, contour_stat):
    """
    Similar to above.
    """
    grid_size = contour_dist.shape[0]
    contour_tile = np.zeros(grid_size)
    for i in range(grid_size):
        lrt = contour_dist[i, :]
        stat = contour_stat[i]
        reference = np.sort(lrt[lrt > -1])
        n = reference.shape[0]
        contour_tile[i] = np.searchsorted(reference, stat) * 1.0 / n
    return contour_tile


def convert_to_sigma(norm_pct):
    """
    Convert confidence level to standard deviation.
    :param norm_pct: (1d numpy array) percentile of normal distribution
    :return: (1d numpy array) number of sigma
    """
    sigma = norm.ppf(norm_pct + 0.5 * (1 - norm_pct))
    sigma[sigma > 3.05] = 3.05
    return sigma


def calculate_overlap(hat_grid, contour_tile):
    """
    Calculate overlap of 68% and 90% confidence contours based on percentile on the grid.
    :param hat_grid: (1d or 2d numpy array) approximated percentile
    :param contour_tile: (1d or 2d numpy array) true percentile
    :return:
    """
    contour_68_diff = (hat_grid < 0.68) != (contour_tile < 0.68)
    contour_90_diff = (hat_grid < 0.90) != (contour_tile < 0.90)
    return 0.5 * (1 - np.mean(contour_68_diff)) + 0.5 * (1 - np.mean(contour_90_diff))
