from datetime import time, datetime

from scipy.spatial import Voronoi, KDTree, cKDTree
from shapely import Polygon
import numpy as np
from scipy.interpolate import griddata
from scipy.fftpack import fft2, fftshift
import scipy.spatial
import libpysal as ps
import numpy as np

from pointpats import PointPattern, PoissonPointProcess, as_window, g_test, f_test, j_test, k_test, l_test
    # , Genv, Fenv, Jenv, Kenv, Lenv
import plotHelpers


def calculateOutputs(cells, createGif=False, has_voronoi_analysis=False, has_NN_analysis=False, logger=None):
    sCones = cells.get_sCones_cells()
    sCones_coordinates = np.array([cells.hex_grid.hex_centers[i] for i in sCones])

    if createGif:
        plotHelpers.create_gif(cells.create_string_params())

    if has_voronoi_analysis:
        areas, variance = voronoi_analysis(cells)
        logger.log_final_voronoi_metrics(areas, variance) if logger else None

    if has_NN_analysis:
        distances, indexes = NN_analysis(cells)
        plotHelpers.createNNPlot(sCones_coordinates, distances, indexes, cells.create_string_params())
        logger.log_final_NN_metrics(distances) if logger else None


def voronoi_analysis(cells):
    vor, areas, grid_bounds = calculate_voronoi_areas(cells)
    variance = calculate_area_variance(areas)
    plotHelpers.createVoronoiPlot(vor, grid_bounds, areas, cells.create_string_params())

    return areas, variance


def NN_analysis(cells):
    distances, indexes = nearest_neighbor_distances(cells)
    return distances, indexes

def nearest_neighbor_distances(cells):
    """Return a list of nearest neighbor distances for a given list of points using a k-D tree."""
    sCones = cells.get_sCones_cells()
    points = np.array([cells.hex_grid.hex_centers[i] for i in sCones])

    tree = cKDTree(points)  # Create a k-D tree from points
    distances, indexes = tree.query(points, k=2)  # Query the nearest two neighbors for each point (itself and the nearest neighbor)
    return distances[:, 1], indexes[:, 1]  # Return the distances to the nearest neighbor (excluding itself)

# def NN_analysis(points):
#     pp = PointPattern(points)
#
#     print(f"Started calculating NNA at {datetime.now()}")
#     # Calculate the various Ripley functions
#     ripleyG = g_test(points, support=20, keep_simulations=True, n_simulations=1000, hull='convex') # Ripley's G function in the format of [x, y] pairs
#     ripleyF = f_test(points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
#     ripleyJ = j_test(points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
#     ripleyK = k_test(points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
#     ripleyL = l_test(points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
#
#     print(f"Finished calculating NNA at {datetime.now()}")
#
#     plotHelpers.createRipleyPlots(points, pp.nnd, ripleyG, ripleyF, ripleyJ, ripleyK, ripleyL, self.string_params)
#
#     # Return a tuple of np.arrays in the following format:
#     # ([x, y] pairs for points, [x, y] pairs for ripleyG, [x, y] pairs for ripleyF, ...)
#     # TODO: This is not working
#     return (*zip(self.points, pp.nnd),
#             *zip(ripleyG.support, ripleyG.statistic),
#             *zip(ripleyF.support, ripleyF.statistic),
#             *zip(ripleyJ.support, ripleyJ.statistic),
#             *zip(ripleyK.support, ripleyK.statistic),
#             *zip(ripleyL.support, ripleyL.statistic))
#


# def calculate_FT_transform_frequencies(self):
#     fourier_transform = np.fft.fft2(self.points)
#
#     # Compute magnitude and frequency
#     magnitude = np.abs(fourier_transform) # Magnitude for x and y components
#     total_magnitude = magnitude[:, 0] + magnitude[:, 1]
#     frequency = np.fft.fftfreq(self.points.shape[0])
#
#     createFTPlot(frequency, total_magnitude, self.string_params)
#
#     # Return a np.array of frequencies and magnitudes like the following np.array([freq1, mag1], [freq2, mag2], ...)
#     return np.array([frequency, total_magnitude]).T

#
# def calculate_NNA(self, NNAPlot):
#     pp = PointPattern(self.points)
#
#     print(f"Started calculating NNA at {datetime.now()}")
#     # Calculate the various Ripley functions
#     ripleyG = g_test(self.points, support=20, keep_simulations=True, n_simulations=1000, hull='convex') # Ripley's G function in the format of [x, y] pairs
#     ripleyF = f_test(self.points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
#     ripleyJ = j_test(self.points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
#     ripleyK = k_test(self.points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
#     ripleyL = l_test(self.points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
#
#     print(f"Finished calculating NNA at {datetime.now()}")
#
#     if NNAPlot:
#         createRipleyPlots(self.points, pp.nnd, ripleyG, ripleyF, ripleyJ, ripleyK, ripleyL, self.string_params)
#
#     # Return a tuple of np.arrays in the following format:
#     # ([x, y] pairs for points, [x, y] pairs for ripleyG, [x, y] pairs for ripleyF, ...)
#     # TODO: This is not working
#     return (*zip(self.points, pp.nnd),
#             *zip(ripleyG.support, ripleyG.statistic),
#             *zip(ripleyF.support, ripleyF.statistic),
#             *zip(ripleyJ.support, ripleyJ.statistic),
#             *zip(ripleyK.support, ripleyK.statistic),
#             *zip(ripleyL.support, ripleyL.statistic))
#

def calculate_area_variance(areas):
    return np.var(areas)

def calculate_external_cell_bounds(points):
    # Find the hex_center with the maximum x and y coordinates
    xmax = max([i[0] for i in points])
    ymax = max([i[1] for i in points])
    # Find the hex_center with the minimum x and y coordinates
    xmin = min([i[0] for i in points])
    ymin = min([i[1] for i in points])
    return xmin, xmax, ymin, ymax

def calculate_voronoi_areas(cells):
    sCones = cells.get_sCones_cells()
    points = np.array([cells.hex_grid.hex_centers[i] for i in sCones])
    vor = Voronoi(points)
    xmin, xmax, ymin, ymax = calculate_external_cell_bounds(points)

    areas = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            if all(xmin <= x <= xmax and ymin <= y <= ymax for x, y in polygon):
                areas.append(Polygon(polygon).area)
    return vor, areas, (xmin, xmax, ymin, ymax)

def calculate_VDRI(cells):
    vor, areas, grid_bounds = calculate_voronoi_areas(cells)
    return np.mean(areas) / np.std(areas) # Return the mean divided by the standard deviation of the areas

def calculate_NNRI(cells):
    distances, indexes = nearest_neighbor_distances(cells)
    return np.mean(distances) / np.std(distances) # Return the mean divided by the standard deviation of the distances
