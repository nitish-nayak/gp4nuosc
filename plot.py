import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage import measure
from pyefd import elliptic_fourier_descriptors
from utils import *


def plot_contour_image(contour, name='', fig_size=4):
    """
    Plot contour as an image.
    """
    fig = plt.figure(figsize=(fig_size, fig_size))
    im = plt.imshow(np.flip(contour, axis=0), cmap='coolwarm', interpolation='nearest')
    grid_size = contour.shape[0]
    x_min = 0
    x_max = contour.shape[0] - 1
    y_min = 0
    y_max = contour.shape[1] - 1
    plt.xticks([x_min, 0.5 * (x_min + x_max), x_max], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize=14)
    plt.yticks([y_min, 0.5 * (y_min + y_max), y_max], [r'$1$', r'$0.5$', r'$0$'], fontsize=14)
    plt.xlabel(r'$\delta_{CP}$', fontsize=14)
    plt.ylabel(r'$\sin^2{\theta_{23}}$', fontsize=14)
    plt.title(name, fontsize=14)
#    plt.colorbar(im)
    plt.show()


def plot_conf_contours(contours, name, fig_size=4, colour='red'):
    """
    Plot confidence contours.
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


def expand_image(image):
    """
    Expand the image by 10 pixels in each side to include the contour boundary.
    """
    n = image.shape[0]
    large_image = np.zeros((n + 20, n + 20))
    large_image[10:(n + 10), 10:(n + 10)] = image
    return large_image


def efd_curve(coeffs, locus=(0., 0.), n=300):
    """
    Adapted function to return elliptic fourier curves.
    """
    N = coeffs.shape[0]
    N_half = int(np.ceil(N / 2))
    n_rows = 2

    t = np.linspace(0, 1.0, n)
    xt = np.ones((n,)) * locus[0]
    yt = np.ones((n,)) * locus[1]

    for n in range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + \
              (coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t))
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + \
              (coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t))

    return xt, yt


def plot_smooth_contour(cont_stat, cont_thres, n_point):
    conf = smooth_grid(cont_stat, n_point) < smooth_grid(cont_thres, n_point)
    conf = expand_image(conf)
    contours = [x - 10 for x in measure.find_contours(conf, 0.9)]
    for contour in contours:
        coeffs = elliptic_fourier_descriptors(contour, order=10)
        xt, yt = efd_curve(coeffs, np.mean(contour, axis=0))
        plt.plot(yt, xt)
        plt.xlim(0, n_point - 1)
        plt.ylim(0, n_point - 1)


def plot_points(all_points, selected):
    plt.scatter(all_points[selected == 1, 1], all_points[selected == 1, 0], color='black')
    plt.scatter(all_points[:, 1], all_points[:, 0], color='gray', alpha=0.1)
    x_min = np.min(all_points[:, 1])
    x_max = np.max(all_points[:, 1])
    y_min = np.min(all_points[:, 0])
    y_max = np.max(all_points[:, 0])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks([x_min, 0.5 * (x_min + x_max), x_max], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize=14)
    plt.yticks([y_min, 0.5 * (y_min + y_max), y_max], [r'$0$', r'$0.5$', r'$1$'], fontsize=14)
    plt.xlabel(r'$\delta_{CP}$', fontsize=14)
    plt.ylabel(r'$\sin^2{\theta_{23}}$', fontsize=14)
    plt.show()


def plot_priority(scores):
    scores_pct = np.argsort(np.argsort(scores)) * 100.0 / (len(scores) - 1)
    score_grid = data_to_grid(scores_pct, int(np.sqrt(scores.shape[0])))
    plt.imshow(np.flip(score_grid, axis=0), cmap='Greens', interpolation='none', alpha=0.5)
    x_min = 0
    x_max = score_grid.shape[0] - 1
    y_min = 0
    y_max = score_grid.shape[1] - 1
    plt.xticks([x_min, 0.5 * (x_min + x_max), x_max], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize=14)
    plt.yticks([y_min, 0.5 * (y_min + y_max), y_max], [r'$1$', r'$0.5$', r'$0$'], fontsize=14)
    plt.xlabel(r'$\delta_{CP}$', fontsize=14)
    plt.ylabel(r'$\sin^2{\theta_{23}}$', fontsize=14)
    plt.show()
