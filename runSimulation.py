from Cells import Cells
from HexGrid import HexGrid
from Logging import myLogger as Logger
from Metrics import calculateOutputs


# Function to run the simulation with given parameters
def run_simulation(params, writeLogs=False, createGif=False, has_voronoi_analysis=False, has_NN_analysis=False):
    grid = HexGrid(size=params['grid_size'])
    cells = Cells(grid, params)

    logger = Logger(cells, params) if writeLogs else None

    while cells.stopSignal is False:
        cells.move_cell_bfs(savePlot=createGif)
        logger.log_running_metrics(cells, has_voronoi_analysis, has_NN_analysis) if writeLogs else None

    calculateOutputs(cells, createGif=createGif, has_voronoi_analysis=has_voronoi_analysis, has_NN_analysis=has_NN_analysis, logger=logger)

    logger.write_logs() if logger else None
