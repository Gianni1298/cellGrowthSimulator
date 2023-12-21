from scipy.spatial import Voronoi
from shapely import Polygon
import numpy as np
from scipy.interpolate import griddata
from scipy.fftpack import fft2, fftshift

from plotHelpers import createCDFPlot, createFTPlot


class outputMetrics:
    def __init__(self, cells, blue_cells, string_params):
        self.cells = cells
        self.blue_cells = blue_cells
        self.points = np.array([self.cells.hex_grid.hex_centers[i] for i in self.blue_cells])
        self.string_params = string_params

    def calculate_voronoi_areas(self, createCDF):
        vor = Voronoi(self.points)
        areas = calculate_voronoi_areas(vor)
        variance = calculate_area_variance(areas)

        if createCDF:
            createCDFPlot(areas, self.string_params)

        return areas, variance

    def calculate_FT_transform_frequencies(self, FTPlot):
        # Create a 2D grid for interpolation
        grid_x, grid_y = np.mgrid[min(self.points[:, 0]):max(self.points[:, 0]):100j,
                         min(self.points[:, 1]):max(self.points[:, 1]):100j]

        # Interpolate onto the grid
        grid_z = griddata(self.points, np.ones(len(self.points)), (grid_x, grid_y), method='nearest')

        # Apply 2D Fourier Transform
        ft = fftshift(fft2(grid_z))

        # Calculate the frequencies and magnitudes
        freq_x = np.fft.fftfreq(grid_x.shape[0], d=(grid_x[1,0] - grid_x[0,0])/grid_x.shape[0])
        ft_magnitude = np.abs(ft)
        ft_magnitude_summed = np.sum(ft_magnitude, axis=0)

        if FTPlot:
            createFTPlot(freq_x, ft_magnitude_summed, self.string_params)

        # Return a np.array of frequencies and magnitudes like the following np.array([freq1, mag1], [freq2, mag2], ...)
        return np.array([freq_x, ft_magnitude_summed]).T


def calculate_voronoi_areas(vor):
    areas = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            areas.append(Polygon(polygon).area)
    return areas

def calculate_area_variance(areas):
    return np.var(areas)

