from scipy.spatial import Voronoi
from shapely import Polygon
import numpy as np
from scipy.interpolate import griddata
from scipy.fftpack import fft2, fftshift

from plotHelpers import createCDFPlot, createFTPlot, createVoronoiPlot, create_gif


class outputMetrics:
    def __init__(self, cells, blue_cells, string_params):
        self.cells = cells
        self.blue_cells = blue_cells
        self.points = np.array([self.cells.hex_grid.hex_centers[i] for i in self.blue_cells])
        self.string_params = string_params

    def create_gif(self):
        create_gif(self.string_params)

    def calculate_voronoi_areas(self, createCDF, plotVoronoi):
        vor = Voronoi(self.points)
        areas = calculate_voronoi_areas(vor)
        variance = calculate_area_variance(areas)

        if createCDF:
            createCDFPlot(areas, self.string_params)

        if plotVoronoi:
            createVoronoiPlot(vor, self.string_params)

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


def calculate_voronoi_areas(vor):
    areas = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            areas.append(Polygon(polygon).area)
    return areas

def calculate_area_variance(areas):
    return np.var(areas)

