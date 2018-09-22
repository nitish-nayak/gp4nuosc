import matplotlib.pyplot as plt
from utils import *
from approximation import adaptive_search


def load_contour_thres(file_pattern, conf_level):
    """
    Load likelihood ratio test threshold contour from simulated data files.

    :param file_pattern: (string) pattern of data files or directory_path/contour_name_i.txt
    :param conf_level: (int) confidence level or 1 - alpha (of likelihood ratio test)
    :return: (numpy array) likelihood ratio test threshold values on the grid
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


def perform_comparison(real_data_num, hierarchy, init_size, n_iter, iter_size, post_hoc_smooth=True, verbose=True):
    """
    Compare Feldman-Cousins with approximation.
    """
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


def load_dist(file_pattern, coord_list):
    """
    Load distributions at specific points on the grid.
    :param file_pattern: (string) pattern of data files or directory_path/contour_name_i.txt
    :param coord_list: (list) of tuples that are grid coordinates
    :return: (list) of distributions
    """
    dist = []
    for coord in coord_list:
        j, k = coord
        i = j * 20 + k
        loglik_global = []
        loglik_profile = []
        with open(file_pattern + '_{}.txt'.format(i), 'r') as f:
            for line in f:
                loglik_global.append(float(line.split(',')[8][1:]))
                loglik_profile.append(float(line.split(',')[9][1:-2]))
        loglik_global = np.asarray(loglik_global)
        loglik_profile = np.asarray(loglik_profile)
        lrt = 2 * (loglik_profile - loglik_global)  # distribution of likelihood ratio test statistic
        dist.append(lrt)
    return dist
