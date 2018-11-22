import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from utils import *
from scipy.stats import binom


def quantile_interval(level, n, p):
    """
    Calculate quantiles for confidence interval endpoints.
    Example: quantile_interval(0.90, 500, 0.90) gives the lower and upper endpoints (indices)
    for the 90th percentile of a probability distribution based on a sample of size 500.

    :param level: (float) confidence interval level between 0 and 1
    :param n: (int) sample size
    :param p: (float) percentile of interest between 0 and 1
    """
    # TODO: need to check corner cases
    l = int(n * p)
    r = l
    dens = 0
    while dens < level:
        prob_l = binom.pmf(l - 1, n, p)
        prob_r = binom.pmf(r + 1, n, p)
        if prob_l > prob_r:
            l += - 1
            if l < 0:
                l = 0
                break
        else:
            r += 1
            if r >= n:
                r = n - 1
                break
        dens = binom.cdf(r, n, p) - binom.cdf(l, n, p)
    return l, r


def percentile_interval(sample, x, level):
    """
    Confidence interval for the percentile of x using sample at level.
    Note: this implementation calls quantile_interval recursively.

    :param sample: (1d numpy array) sample data
    :param x: (float) single observation
    :param level: (float) confidence interval level between 0 and 1
    """
    n = sample.shape[0]
    sample = np.sort(sample)
    q = np.searchsorted(sample, x) * 1.0 / n
    lower = q
    upper = q
    # TODO: check corner cases again
    while upper < 0.99:
        l, r = quantile_interval(level, n, upper + 0.01)
        if l < 0:
            break
        if sample[l] > x:
            break
        upper += 0.01
    while lower > 0.01:
        l, r = quantile_interval(level, n, lower - 0.01)
        if r >= n:
            break
        if sample[r] < x:
            break
        lower -= 0.01
    return lower, upper


def acquisition_function(gp_mean, gp_std):
    """
    Inverse of distance from threshold scaled by standard deviation.
    Want to explore points near the threshold.
    """
    # TODO: different acquisition functions
    return gp_std / np.absolute(gp_mean - 0.68) + gp_std / np.absolute(gp_mean - 0.9)


def initialize_sample(n_total, init_point, size):
    """
    Initialize a sample for training GP approximation.
    :param n_total: (int) total number of points on the grid
    :param init_point: (int) number of initial points to place
    :param size: (int) sample size at each point
    :return: (1d numpy array) vector of sample size
    """
    sample_size = np.zeros(n_total)
    init_indices = np.random.permutation(n_total)[:init_point]
    sample_size[init_indices] = size
    return sample_size


def expand_sample(priority, sample_size, n_delta, size, limit):
    """
    Increase sample size according to priority.

    :param priority: (1d numpy array) acquisition priority
    :param sample_size: (1d numpy array) vector of sample size
    :param n_delta: (int) number of incremental points
    :param size: (int) sample size at each point
    :param limit: (int) maximum sample size allowed
    :return: (1d numpy array) vector of sample size
    """
    indices = np.argsort(priority)[::-1]
    n = 0
    for i in indices:
        if sample_size[i] <= limit - size:
            sample_size[i] += size
            n += 1
        if n >= n_delta:
            break
    return sample_size


def get_current_training(all_points, sample_size):
    """
    Current training points for GP approximation.
    """
    current_indices = np.where(sample_size > 0)[0]
    current_points = all_points[current_indices]
    return current_points


def get_current_target(all_samples, all_obs, sample_size):
    """
    Current target with point and interval estimates.

    :param all_samples: (2d numpy array) each row is the sample at a point
    :param all_obs: (1d numpy array) observed statistic vector
    :param sample_size: (1d numpy array) vector of sample size
    :return:
    """
    current_indices = np.where(sample_size > 0)[0]
    n_training = current_indices.shape[0]
    current_target = np.zeros(n_training)
    current_error = np.zeros((n_training, 2))
    for k, i in enumerate(current_indices):
        n = int(sample_size[i])
        sample = np.sort(all_samples[i, :n])  # only use sample size n
        obs = all_obs[i]
        current_target[k] = np.searchsorted(sample, obs) * 1.0 / sample.shape[0]  # point estimate
        l, r = percentile_interval(sample, obs, 0.68)  # 68% interval estimate (1 standard deviation)
        current_error[k, 0] = l
        current_error[k, 1] = r
    return current_target, current_error


def build_approximation(current_points, current_target, current_error, all_points):
    """
    Train GP approximation and make prediction.
    """
    variance = (current_error[:, 1] - current_error[:, 0]) ** 2 + 0.0001  # (diagonal) variance vector
    gp = GaussianProcessRegressor(kernel=RBF()+WhiteKernel(), alpha=variance, normalize_y=True)
    gp = gp.fit(current_points, current_target)
    return gp.predict(all_points, return_std=True)
