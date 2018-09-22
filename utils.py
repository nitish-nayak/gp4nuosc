import numpy as np
from scipy.interpolate import griddata


# Conversion between grid data format and regular data format

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
    return data


def data_to_grid(data, m):
    """
    Map data onto a grid of size m.
    """
    n = m
    data_grid = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            data_grid[i, j] = data[i * m + j]
    return data_grid


# Smooth data on a finer grid

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
    return points, values


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
    return fine_grid


def smooth_grid(grid, res):
    """
    Smooth grid with linear interpolation.
    """
    points, values = grid_to_points(grid)
    fine_grid = refine_grid(points, res)
    return griddata(points, values, fine_grid, method='linear').reshape((res, res))
