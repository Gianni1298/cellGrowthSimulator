from scipy.spatial import Voronoi
from expandGridV4 import HexGrid, Scones
import numpy as np
from voronoi import calculate_voronoi_areas, calculate_area_variance
import itertools
import mylogging



# Function to run the simulation with given parameters
def run_simulation(s_cones_init, m_cones_init, max_probability, grid_size):
    grid = HexGrid(size=grid_size)
    s_cones_parameters = {
        "s_cones_init_count": s_cones_init,
        "m_cones_init_count": m_cones_init,

        "s_cones_final_count": 80,
        "m_cones_final_count": 920,

        "init_mode": "bfs",
        "max_probability": max_probability
    }

    s_cones = Scones(grid, s_cones_parameters)

    while s_cones.stopSignal is False:
        s_cones.move_cell_bfs()

    blue_cells =  [i for i in s_cones.cell_indexes.keys() if s_cones.cell_indexes[i] == 'b']

    points = [s_cones.hex_grid.hex_centers[i] for i in blue_cells]
    vor = Voronoi(points)
    areas = calculate_voronoi_areas(vor)
    variance = calculate_area_variance(areas)

    return areas, variance


# Define parameter ranges for grid search
s_cones_init = np.linspace(1, 80, 8)
m_cones_init = np.linspace(1, 920, 20)  # Adjust the range and step as needed
max_probability = np.linspace(0.01, 10, 5)  # Adjust the range and step as needed


grid_size = 45

for s, m, p in itertools.product(s_cones_init, m_cones_init, max_probability):
    _, variance = run_simulation(s, m, p, grid_size)

# Create a runnable application


