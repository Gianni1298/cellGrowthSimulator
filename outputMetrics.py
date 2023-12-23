from datetime import time, datetime

from scipy.spatial import Voronoi, KDTree
from shapely import Polygon
import numpy as np
from scipy.interpolate import griddata
from scipy.fftpack import fft2, fftshift
import scipy.spatial
import libpysal as ps
import numpy as np

from pointpats import PointPattern, PoissonPointProcess, as_window, g_test, f_test, j_test, k_test, l_test
    # , Genv, Fenv, Jenv, Kenv, Lenv

from plotHelpers import createCDFPlot, createFTPlot, createVoronoiPlot, create_gif, createNNAPlot, createRipleyPlots


class outputMetrics:
    def __init__(self, cells, blue_cells, params):
        self.cells = cells
        self.blue_cells = blue_cells
        self.points = np.array([self.cells.hex_grid.hex_centers[i] for i in self.blue_cells])
        self.params = params
        self.string_params = self.create_string_params()

    def create_gif(self):
        create_gif(self.string_params)

    def create_string_params(self):
        string_params = f"sConesInit={self.params['s_cones_init_count']}_" \
                        f"mConesInit={self.params['m_cones_init_count']}_" \
                        f"sConesFinal={self.params['s_cones_final_count']}_" \
                        f"mConesFinal={self.params['m_cones_final_count']}_" \
                        f"maxProb={self.params['max_probability']}_" \
                        f"gridSize={self.params['grid_size']}"

        return string_params

    def calculate_voronoi_areas(self, createCDF, plotVoronoi):
        vor = Voronoi(self.points)
        cell_perimeter = self.cells.hex_grid.get_perimeter_points()
        areas = calculate_voronoi_areas(vor, self.cells.hex_grid.grid_bounds)
        variance = calculate_area_variance(areas)

        if createCDF:
            createCDFPlot(areas, self.string_params)

        if plotVoronoi:
            createVoronoiPlot(vor, self.cells.hex_grid.grid_bounds, self.string_params)

        return areas, variance

    def calculate_FT_transform_frequencies(self, FTPlot):
        fourier_transform = np.fft.fft2(self.points)

        # Compute magnitude and frequency
        magnitude = np.abs(fourier_transform) # Magnitude for x and y components
        total_magnitude = magnitude[:, 0] + magnitude[:, 1]
        frequency = np.fft.fftfreq(self.points.shape[0])

        if FTPlot:
            createFTPlot(frequency, total_magnitude, self.string_params)

        # Return a np.array of frequencies and magnitudes like the following np.array([freq1, mag1], [freq2, mag2], ...)
        return np.array([frequency, total_magnitude]).T

    def calculate_NNA(self, NNAPlot):
        # Step 1: Create a KDTree for efficient nearest neighbor search
        tree = KDTree(self.points)

        # Step 2: Find the nearest neighbor for each cell
        distances, _ = tree.query(self.points, k=2)  # The nearest neighbor is the point itself, so k=2
        nearest_neighbor_distances = distances[:, 1]  # Ignore the first column which is distance to itself

        # Step 3: Calculate the mean nearest neighbor distance
        mean_distance = np.mean(nearest_neighbor_distances)

        # Expected mean distance for a random distribution (Poisson process) in a unit square
        expected_mean_distance = 1 / (2 * np.sqrt(len(self.points)))

        # NNA ratio (R)
        R = mean_distance / expected_mean_distance

        if NNAPlot:
            createNNAPlot(self.points, nearest_neighbor_distances, R)

        return nearest_neighbor_distances, R

    def calculate_NNA_v2(self, NNAPlot):
        pp = PointPattern(self.points)

        print(f"Started calculating NNA at {datetime.now()}")
        # Calculate the various Ripley functions
        ripleyG = g_test(self.points, support=40, keep_simulations=True, n_simulations=1000, hull='convex') # Ripley's G function in the format of [x, y] pairs
        ripleyF = f_test(self.points, support=40, keep_simulations=True, n_simulations=1000, hull='convex')
        ripleyJ = j_test(self.points, support=40, keep_simulations=True, n_simulations=1000, hull='convex')
        ripleyK = k_test(self.points, support=40, keep_simulations=True, n_simulations=1000, hull='convex')
        ripleyL = l_test(self.points, support=40, keep_simulations=True, n_simulations=1000, hull='convex')

        print(f"Finished calculating NNA at {datetime.now()}")

        if NNAPlot:
            createRipleyPlots(self.points, pp.nnd, ripleyG, ripleyF, ripleyJ, ripleyK, ripleyL, self.string_params)

        return 1, 2


def calculate_voronoi_areas(vor, grid_bounds):
    xmin, xmax, ymin, ymax = grid_bounds
    areas = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            if all(xmin <= x <= xmax and ymin <= y <= ymax for x, y in polygon):
                areas.append(Polygon(polygon).area)
    return areas

def calculate_area_variance(areas):
    return np.var(areas)

