import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from utils import *


def acquisition_function(gp_mean, gp_std, lr_stats):
    """
    Inverse of distance from threshold scaled by standard deviation.
    Want to explore points near the threshold.
    """
    return gp_std / np.absolute(lr_stats - gp_mean)


def acquire_points(scores, selected, n):
    """
    Find unexplored points ranked by acquisition scores.
    """
    valid_indices = np.where(selected == 0)[0]  # indices of unexplored points
    valid_scores = scores[valid_indices]  # scores of unexplored points
    sorted_indices = np.argsort(valid_scores)  # rank points
    new_indices = valid_indices[sorted_indices[-n:]]
    return new_indices


def intialize_model(all_points, target, init_size):
    gp = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(0.1, 10.0))+WhiteKernel(),
                                  normalize_y=True)
    current_indices = np.random.permutation(target.shape[0])[:init_size]
    current_points = all_points[current_indices, :]
    current_target = target[current_indices]
    gp.fit(current_points, current_target)
    return gp, current_indices


def update_model(gp, current_indices, iter_size, all_points, target, lr_stats):
    hat, std = gp.predict(all_points, return_std=True)
    scores = acquisition_function(hat, std, lr_stats)
    selected = np.zeros(target.shape[0])
    selected[current_indices] = 1
    new_indices = acquire_points(scores, selected, iter_size)

    current_indices = np.concatenate((current_indices, new_indices), axis=0)
    current_points = all_points[current_indices, :]
    current_target = target[current_indices]
    gp.fit(current_points, current_target)

    return gp, current_indices, scores, hat


def update_results(hat, current_indices, target, lr_stats):
    n = int(np.sqrt(target.shape[0]))
    hat[current_indices] = target[current_indices]
    mean_grid = data_to_grid(hat, n)
    lr_grid = data_to_grid(lr_stats, n)
    conf_grid = lr_grid < mean_grid
    return mean_grid, conf_grid


def adaptive_search(all_points, target, lr_stats, init_size, n_iter, iter_size):
    n = int(np.sqrt(target.shape[0]))
    mean_grid = np.zeros((n, n))
    conf_grid = np.zeros((n, n))

    gp, current_indices = intialize_model(all_points, target, init_size)
    for i in range(n_iter):
        if i == 0:
            gp, current_indices = intialize_model(all_points, target, init_size)
        gp, current_indices, scores, hat = update_model(gp, current_indices, iter_size, all_points, target, lr_stats)
        mean_grid, conf_grid = update_results(hat, current_indices, all_points, target, lr_stats)

    return mean_grid, conf_grid
