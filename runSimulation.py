from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from HexGrid import HexGrid
from Cells import Cells
from mylogging import myLogger as Logger
import numpy as np

from outputMetrics import outputMetrics
from plotHelpers import create_gif, createCDFPlot
import itertools


# Function to run the simulation with given parameters
def run_simulation(cells_parameters, grid_size, writeLogs=False, createGif=False, plotVoronoi=False, createCDF=False, FTPlot=False):
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

    output_metrics = outputMetrics(cells, blue_cells, string_params)
    voronoi_areas, voronoi_variance = output_metrics.calculate_voronoi_areas(createCDF, plotVoronoi)
    FTFrequencies = output_metrics.calculate_FT_transform_frequencies(FTPlot)

    if writeLogs:
        logger = Logger('logs.csv')
        logger.log_results(cells_parameters, grid_size, cells.cell_indexes, blue_cells, voronoi_areas, voronoi_variance, FTFrequencies)



