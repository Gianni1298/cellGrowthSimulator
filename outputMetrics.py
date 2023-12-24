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

from plotHelpers import createCDFPlot, createFTPlot, createVoronoiPlot, create_gif, createRipleyPlots


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
                        f"maxProb={self.params['max_probability']}_" \
                        f"gridSize={self.params['grid_size']}"

        return string_params

    def calculate_voronoi_areas(self):
        vor = Voronoi(self.points)
        areas = calculate_voronoi_areas(vor, self.cells.hex_grid.grid_bounds)
        variance = calculate_area_variance(areas)

        createVoronoiPlot(vor, self.cells.hex_grid.grid_bounds, areas, self.string_params)

        return areas, variance

    def calculate_FT_transform_frequencies(self):
        fourier_transform = np.fft.fft2(self.points)

        # Compute magnitude and frequency
        magnitude = np.abs(fourier_transform) # Magnitude for x and y components
        total_magnitude = magnitude[:, 0] + magnitude[:, 1]
        frequency = np.fft.fftfreq(self.points.shape[0])

        createFTPlot(frequency, total_magnitude, self.string_params)

        # Return a np.array of frequencies and magnitudes like the following np.array([freq1, mag1], [freq2, mag2], ...)
        return np.array([frequency, total_magnitude]).T


    def calculate_NNA(self, NNAPlot):
        pp = PointPattern(self.points)

        print(f"Started calculating NNA at {datetime.now()}")
        # Calculate the various Ripley functions
        ripleyG = g_test(self.points, support=20, keep_simulations=True, n_simulations=1000, hull='convex') # Ripley's G function in the format of [x, y] pairs
        ripleyF = f_test(self.points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
        ripleyJ = j_test(self.points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
        ripleyK = k_test(self.points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')
        ripleyL = l_test(self.points, support=20, keep_simulations=True, n_simulations=1000, hull='convex')

        print(f"Finished calculating NNA at {datetime.now()}")

        if NNAPlot:
            createRipleyPlots(self.points, pp.nnd, ripleyG, ripleyF, ripleyJ, ripleyK, ripleyL, self.string_params)

        # Return a tuple of np.arrays in the following format:
        # ([x, y] pairs for points, [x, y] pairs for ripleyG, [x, y] pairs for ripleyF, ...)
        # TODO: This is not working
        return (*zip(self.points, pp.nnd),
                *zip(ripleyG.support, ripleyG.statistic),
                *zip(ripleyF.support, ripleyF.statistic),
                *zip(ripleyJ.support, ripleyJ.statistic),
                *zip(ripleyK.support, ripleyK.statistic),
                *zip(ripleyL.support, ripleyL.statistic))


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

