import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from utils import grid_to_data
from approximation import adaptive_search
from matplotlib import colors


def load_contour_thres(file_pattern, conf_level):
    """
    Load likelihood ratio test threshold contour from simulated data files.
    
    Args
        file_pattern: (string) pattern of data files or directory_path/contour_name_i.txt
        conf_level: (int) confidence level or 1 - alpha (of likelihood ratio test)
    Returns
        contour_thres: (numpy array) likelihood ratio test threshold values on the grid
    """
    contour_thres = np.zeros((20, 20))
    for i in range(400):
        j = i / 20
        k = i % 20
        loglik_global = []
        loglik_profile = []
        try:
            with open(file_pattern + '_{}.txt'.format(i), 'r') as f:
                for line in f: 
                    loglik_global.append(float(line.split(',')[8][1:]))
                    loglik_profile.append(float(line.split(',')[9][1:-2]))
            loglik_global = np.asarray(loglik_global)
            loglik_profile = np.asarray(loglik_profile)
            lrt = 2 * (loglik_profile - loglik_global)  # distribution of likelihood ratio test statistic 
            contour_thres[j, k] = np.percentile(lrt, conf_level)
        except:
            contour_thres[j, k] = contour_thres[j, k - 1]  # catch occasional missing data files
    return contour_thres


def load_fitted_contour(filepath):
    """
    Load contour of fitted LRT statistic from data in text file.
    """
    loglik_global = []
    loglik_profile = []
    with open(filepath, 'r') as f:
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
    return contour_fit


def plot_contour_image(contour, name, fig_size=4):
    """
    Plot contour as an image.
    
    Args
        contour: (numpy array) contour to be plotted
        name: (string) plot title
        fig_size: (int) plot size
    """
    fig = plt.figure(figsize=(fig_size, fig_size))
    im = plt.imshow(np.flip(contour, axis=0), cmap='coolwarm', interpolation='nearest')
    grid_size = contour.shape[0]
    plt.xticks([0, int(grid_size / 2), grid_size], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize=14)
    plt.yticks([0, int(grid_size / 2), grid_size], [r'$1$', r'$0.5$', r'$0$'], fontsize=14)
    plt.xlabel(r'$\delta_{CP}$', fontsize=14)
    plt.ylabel(r'$\sin^2{\theta_{23}}$', fontsize=14)
    plt.title(name, fontsize=14)
    plt.colorbar(im)
    plt.show()


def plot_conf_contour(contours, name, fig_size=4, colour='red'):
    """
    Plot confidence contours.

    Args
        contours: (list of numpy arrays) confidence contours at different levels
        name: (string) plot title
        fig_size: (int) plot size
        colour: (string) contour color
    """
    fig = plt.figure(figsize=(fig_size, fig_size))
    cmap = colors.ListedColormap(['white', colour])
    for c in contours:
        im = plt.imshow(np.flip(c, axis=0), cmap=cmap, interpolation='nearest', alpha=0.4)
    grid_size = contours[0].shape[0]
    plt.xticks([0, int(grid_size / 2), grid_size], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize=14)
    plt.yticks([0, int(grid_size / 2), grid_size], [r'$1$', r'$0.5$', r'$0$'], fontsize=14)
    plt.xlabel(r'$\delta_{CP}$', fontsize=14)
    plt.ylabel(r'$\sin^2{\theta_{23}}$', fontsize=14)
    plt.title(name, fontsize=14)
    plt.show()


def grid_to_points(grid):
    """
    Map grid to points in the unit square for smoothing.
    """
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
    """
    Add more points to the grid.
    """
    fine_grid = np.zeros((res * res, 2))
    n = 0
    for i in range(res):
        for j in range(res):
            fine_grid[n, 0] = i * np.max(points[:, 0]) / res
            fine_grid[n, 1] = j * np.max(points[:, 1]) / res
            n += 1
    return(fine_grid)


def smooth_grid(grid, res):
    """
    Smooth grid with linear interpolation.
    """
    points, values = grid_to_points(grid)
    fine_grid = refine_grid(points, res)
    return(griddata(points, values, fine_grid, method='linear').reshape((res, res)))


def perform_comparison(real_data_num, hierarchy='normal', post_hoc_smooth=True, verbose=True):
    """
    Compare Feldman-Cousins with approximation.
    """
    if hierarchy == 'normal':
        contour_thres = np.load('/Users/linggeli/monte_carlo/penalty_contour_normal_68.npy')
        contour_stat = load_fitted_contour('/Users/linggeli/Downloads/penalty_normal/contour_normal_fit_{}.txt'.format(real_data_num))
    elif hierarchy == 'inverted':
        contour_thres = np.load('/Users/linggeli/monte_carlo/penalty_contour_inverted_68.npy')
        contour_stat = load_fitted_contour('/Users/linggeli/Downloads/penalty_inverted/contour_inverted_fit_{}.txt'.format(real_data_num))
    contour_true = contour_stat < contour_thres
    data = grid_to_data(contour_thres)  # format data
    all_points = data[:, 0:2]
    target = data[:, 2]
    lr_stats = grid_to_data(contour_stat)[:, 2]
    contour_thres_hat, contour_hat = adaptive_search(all_points, target, lr_stats, 50, 5, 10, verbose=False)  # approximation
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
