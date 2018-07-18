import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from utils import grid_to_data, data_to_grid


def acquisition_function(gp_mean, gp_std, lr_stats):
    """
    Inverse of distance from threshold scaled by standard deviation.
    Want to explore points "near" the threshold.
    """
    return(gp_std / np.absolute(lr_stats - gp_mean))


def acquire_points(scores, selected, n):
    """
    Find unexplored points ranked by acquisition scores.
    """
    valid_indices = np.where(selected == 0)[0]  # indices of unexplored points
    valid_scores = scores[valid_indices]  # scores of unexplored points
    sorted_indices = np.argsort(valid_scores)  # rank points
    new_indices = valid_indices[sorted_indices[-n:]]
    return(new_indices)


def plot_helper(select_grid, score_grid, mean_grid, conf_grid):
    """
    Helper function for plotting.
    """
    fig = plt.figure(figsize=(16, 4))
    plt.subplot(141)
    im = plt.imshow(select_grid, cmap='binary', interpolation='none')  #
    plt.subplot(142)
    im = plt.imshow(score_grid, cmap='coolwarm', interpolation='none')
    plt.colorbar(im)
    plt.subplot(143)
    im = plt.imshow(mean_grid, cmap='coolwarm', interpolation='none')
    plt.colorbar(im)
    plt.subplot(144)
    #conf_array = grid_to_data(conf_grid)
    #conf_array = conf_array[conf_array[:, 2] == 1, :2]
    #plt.scatter(conf_array[:, 0], conf_array[:, 1], c='g', s=100, alpha=0.5, edgecolor='none')
    im = plt.imshow(conf_grid, cmap='Greens', interpolation='nearest')
    plt.show()
    
    
def adaptive_search(all_points, target, lr_stats, init_size, n_iter, iter_size, verbose=True):
    """
    Explore the parameter space iteratively guided by GP acquisition function.
    """
    # initialize points and train GP
    n = target.shape[0]
    m = int(np.sqrt(n))
    current_indices = np.random.permutation(n)[:init_size]
    current_points = all_points[current_indices, :]
    current_target = target[current_indices]
    gp = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(0.1, 10.0))+WhiteKernel(), normalize_y=True)
    gp.fit(current_points, current_target)
    print(gp.kernel_)
    selected = np.zeros(n)
    selected[current_indices] = 1  # keep track of already explored points
    lr_grid = data_to_grid(lr_stats, m)
    best_approx = np.zeros((m, m))  # best approximation
    final_prod = np.zeros((m, m))  # final product
    for i in range(n_iter):
        hat, std = gp.predict(all_points, return_std=True)  # predict with GP
        scores = acquisition_function(hat, std, lr_stats)
        select_grid = data_to_grid(selected, m)  # calculate grids
        score_grid = data_to_grid(scores, m)
        mean_grid = data_to_grid(hat, m)
        conf_grid = lr_grid < mean_grid
        if verbose:
            plot_helper(select_grid, score_grid, mean_grid, conf_grid)  # make plots
        new_indices = acquire_points(scores, selected, iter_size)  # acquire new points
        current_indices = np.concatenate((current_indices, new_indices), axis=0)
        selected[current_indices] = 1
        current_points = all_points[current_indices, :]  # re-train GP
        current_target = target[current_indices]
        gp.fit(current_points, current_target)
        print(gp.kernel_)
        if i == n_iter - 1:
            best_approx = mean_grid
            final_prod = conf_grid
    return(best_approx, final_prod)
