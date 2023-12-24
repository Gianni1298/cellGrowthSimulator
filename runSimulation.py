import csv

from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from HexGrid import HexGrid
from Cells import Cells
from mylogging import myLogger as Logger
import numpy as np

from outputMetrics import outputMetrics
import itertools


# Function to run the simulation with given parameters
def run_simulation(params, writeLogs=False, createGif=False, voronoi_analysis=False,
                   FT_analysis=False, NN_analysis=False):
    grid = HexGrid(size=params['grid_size'])

    cells = Cells(grid, params)

    while cells.stopSignal is False:
        cells.move_cell_bfs(savePlot=createGif)

    blue_cells = [i for i in cells.cell_indexes.keys() if cells.cell_indexes[i] == 'b']

    output_metrics = outputMetrics(cells, blue_cells, params)
    if createGif:
        output_metrics.create_gif()

    voronoi_areas, voronoi_variance = None, None
    if voronoi_analysis:
        voronoi_areas, voronoi_variance = output_metrics.calculate_voronoi_areas()

    FTFrequencies = None
    if FT_analysis:
        FTFrequencies = output_metrics.calculate_FT_transform_frequencies()

    nnd, ripleyG, ripleyF, ripleyJ, ripleyK, ripleyL = None, None, None, None, None, None
    if NN_analysis:
        nnd, ripleyG, ripleyF, ripleyJ, ripleyK, ripleyL = output_metrics.calculate_NNA(NN_analysis)

    if writeLogs:
        logger = Logger('logs.csv')
        logger.log_results(params, cells.cell_indexes, blue_cells, voronoi_areas, voronoi_variance,
                           FTFrequencies, nnd, ripleyG, ripleyF, ripleyJ, ripleyK, ripleyL)
