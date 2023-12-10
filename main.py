from scipy.spatial import Voronoi
from expandGridV3 import HexGrid, Scones
import numpy as np
from voronoi import calculate_voronoi_areas, calculate_area_variance
import itertools
import logging

# Setting up logging
logging.basicConfig(filename='parameter_optimization_log.txt', level=logging.INFO)


# Function to run the simulation with given parameters
def run_simulation(birth_rate, a, m, c, init_mode, grid_size):
    grid = HexGrid(size=grid_size)
    s_cones_parameters = {
        "s_cones_final_count": 160,
        "m_cones_final_count": 1840,
        "m_cones_birth_rate": birth_rate,
        "a": a,
        "m": m,
        "c": c,
        "init_mode": init_mode
    }

    s_cones = Scones(grid, s_cones_parameters)

    # grid.draw(s_cones.blue_indices)

    while len(s_cones.m_cones.get_green_indices()) < s_cones_parameters["m_cones_final_count"]:
        s_cones.move_sCones()

    points = [s_cones.hex_grid.hex_centers[i] for i in s_cones.blue_indices]
    vor = Voronoi(points)
    areas = calculate_voronoi_areas(vor)
    variance = calculate_area_variance(areas)

    return variance


# Define parameter ranges for grid search
birth_rate_values = np.linspace(0.7, 1, 3)
a_values = np.linspace(1, 250, 5)  # Adjust the range and step as needed
m_values = np.linspace(0.001, 30, 10)  # Adjust the range and step as needed
c_values = np.linspace(1, 25, 4)  # Adjust the range and step as needed

# Define the initial mode
init_mode = "random"
grid_size = 45


# Grid search
best_variance = float('inf')


for br, a, m, c in itertools.product(birth_rate_values, a_values, m_values, c_values):
    variance = run_simulation(br, a, m, c, init_mode, grid_size)
    logging.info(f"Parameters: birth_rate={br} a={a}, m={m} c={c}, init_mode={init_mode}, grid_size={grid_size}, Variance: {variance}")

