import numpy as np


def pos_to_param(i, j):
    """
    Map grid position to parameter space 
    """
    x_low, x_high = x_range
    x = x_low + i * (x_high - x_low) / float(grid_size)
    y_low, y_high = y_range
    y = y_low + j * (y_high - y_low) / float(grid_size)
    return((x, y))


def grid_to_data(grid):
    """
    Map grid (position, value) to data (X, y) for modeling. 
    """
    m = grid.shape[0]
    n = grid.shape[1]
    data = np.zeros((m * n, 3))
    for i in range(m):
        for j in range(n):
            data[i * m + j, 0] = (1.0 / float(m)) * (i + 1)
            data[i * m + j, 1] = (5.0 / float(n)) * (j + 1)
            data[i * m + j, 2] = grid[i, j]
    return(data)


def data_to_grid(data, m):
    """
    Map data onto a grid of size m.
    """
    n = m
    data_grid = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            data_grid[i, j] = data[i * m + j]
    return(data_grid)
