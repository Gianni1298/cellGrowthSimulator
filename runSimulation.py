from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from HexGrid import HexGrid
from Cells import Cells
import numpy as np

from plotHelpers import create_gif, createCDFPlot
from voronoi import calculate_voronoi_areas, calculate_area_variance
import itertools
import logging


# Function to run the simulation with given parameters
def run_simulation(cells_parameters, grid_size, writeLogs=False, createGif=False, createCDF=False):
    if writeLogs:
        logging.basicConfig(filename='parameter_optimization_log_v4.txt', level=logging.INFO)

    grid = HexGrid(size=grid_size)

    cells = Cells(grid, cells_parameters)

    while cells.stopSignal is False:
        cells.move_cell_bfs(savePlot=createGif)

    string_params = f"sConesInit={cells_parameters['s_cones_init_count']}_"\
                    f"mConesInit={cells_parameters['m_cones_init_count']}_" \
                    f"sConesFinal={cells_parameters['s_cones_final_count']}_" \
                    f"mConesFinal={cells_parameters['m_cones_final_count']}_" \
                    f"maxProb={cells_parameters['max_probability']}_" \
                    f"gridSize={grid.size}"

    if createGif:
        create_gif(string_params)

    blue_cells = [i for i in cells.cell_indexes.keys() if cells.cell_indexes[i] == 'b']

    points = [cells.hex_grid.hex_centers[i] for i in blue_cells]
    vor = Voronoi(points)
    areas = calculate_voronoi_areas(vor)
    variance = calculate_area_variance(areas)

    if writeLogs:
        logResults(cells_parameters['s_cones_init_count'], cells_parameters['m_cones_init_count'], cells_parameters['max_probability'], grid_size, variance)

    if createCDF:
        createCDFPlot(areas, string_params)

    return areas, variance


def logResults(s_cones_init, m_cones_init, max_probability, grid_size, variance):
    logging.info(f"Parameters: s_cones_init={s_cones_init} "
                 f"m_cones_init={m_cones_init}, "
                 f"max_probability={max_probability}, "
                 f"grid_size={grid_size}, "
                 f"Variance: {variance}")
