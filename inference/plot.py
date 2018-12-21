import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage import measure
from pyefd import elliptic_fourier_descriptors
from utils import *


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


def plot_smooth_contour(conf, hierarchy='NH', **kwargs):
    """
    Plot 2d contour after Fourier smoothing.
    """
    n_point = conf.shape[0]
    conf = expand_image(conf)
    contours = [x - 10 for x in measure.find_contours(conf, 0.9)]
    contour = contours[0]
    coeffs = elliptic_fourier_descriptors(contour, order=10)
    xt, yt = efd_curve(coeffs, np.mean(contour, axis=0))
    if hierarchy == 'NH':
        colour = 'dodgerblue'
    elif hierarchy == 'IH':
        colour = 'firebrick'
    plt.plot(yt, xt, c=colour, **kwargs)
    plt.xlim(0, n_point - 1)
    plt.ylim(0, n_point - 1)


def plot_iteration(sample_size, all_points, priority_grid, hat_grid, hierarchy='NH'):
    """
    Make plots of: acquired points, acquisition priority, current approximation, and confidence contours on the grid.
    """
    fig = plt.figure(figsize=(8, 8))

    plt.subplot(221)
    rgba_colors = np.zeros((400, 4))
    rgba_colors[:, 3] = sample_size * 1.0 / 2000
    plt.scatter(all_points[:, 1], all_points[:, 0], color=rgba_colors)

    plt.subplot(222)
    im1 = plt.imshow(np.flip(priority_grid, axis=0), cmap='Greens', interpolation='none', alpha=0.5)

    plt.subplot(223)
    im2 = plt.imshow(np.flip(hat_grid, axis=0), cmap='coolwarm', interpolation='none')
    plt.colorbar(im2, fraction=0.046, pad=0.04)

    plt.subplot(224)
    if hierarchy == 'NH':
        cmap = colors.ListedColormap(['white', 'dodgerblue'])
    elif hierarchy == 'IH':
        cmap = colors.ListedColormap(['white', 'firebrick'])
    im2 = plt.imshow(np.flip(hat_grid, axis=0) < 0.90, cmap=cmap, interpolation='nearest', alpha=0.5)
    im3 = plt.imshow(np.flip(hat_grid, axis=0) < 0.68, cmap=cmap, interpolation='nearest', alpha=0.5)
    plot_smooth_contour(np.flip(hat_grid, axis=0) < 0.68)
    plot_smooth_contour(np.flip(hat_grid, axis=0) < 0.9, alpha=0.5)
    plt.gca().invert_yaxis()


def plot_x_axis(contour_type):
    if contour_type == 'dcp__theta23_NH' or contour_type == 'dcp__theta23_IH':
        plt.xlabel(r'$\delta_{CP}$')
        plt.xticks([0, 5, 10, 15, 19],
                   [r'$0$', r'$0.5\pi$', r'$\pi$', r'$1.5\pi$', r'$2\pi$'])
    elif contour_type == 'theta23__dmsq_32_NH':
        plt.xlabel(r'$\sin^2{\theta_{23}}$')
        plt.xticks([0, 5, 10, 15, 19],
                   [r'$0.3$', r'$0.45$', r'$0.5$', r'$0.6$', r'$0.7$'])
    elif contour_type == 'theta23__dmsq_32_IH':
        plt.xlabel(r'$\sin^2{\theta_{23}}$')
        plt.xticks([0, 5, 10, 15, 19],
                   [r'$0.3$', r'$0.45$', r'$0.5$', r'$0.6$', r'$0.7$'])


def plot_y_axis(contour_type):
    if contour_type == 'dcp__theta23_NH' or contour_type == 'dcp__theta23_IH':
        plt.ylabel(r'$\sin^2{\theta_{23}}$')
        plt.yticks([19, 15, 10, 5, 0],
                   [r'$0.3$', r'$0.4$', r'$0.5$', r'$0.6$', r'$0.7$'])
    elif contour_type == 'theta23__dmsq_32_NH':
        plt.ylabel(r'$\Delta m_{32}^2$')
        plt.yticks([19, 15, 10, 5, 0],
                   [r'$2\times10^{-3}$', r'$2.25\times10^{-3}$', r'$2.5\times10^{-3}$',
                    r'$2.75\times10^{-3}$', r'$3\times10^{-3}$'])
    elif contour_type == 'theta23__dmsq_32_IH':
        plt.ylabel(r'$\Delta m_{32}^2$')
        plt.yticks([19, 15, 10, 5, 0],
                   [r'$-3\times10^{-3}$', r'$-2.75\times10^{-3}$', r'$-2.5\times10^{-3}$',
                    r'$-2.25\times10^{-3}$', r'$-2\times10^{-3}$'])
