from scipy.spatial import Voronoi
from expandGridV3 import HexGrid, Scones
import numpy as np
from voronoi import calculate_voronoi_areas, calculate_area_variance
import itertools
import logging

# Setting up logging
logging.basicConfig(filename='parameter_optimization_log.txt', level=logging.INFO)


# Function to run the simulation with given parameters
def run_simulation(birth_rate, a, c, exponential_probability_growth_factor):
    grid = HexGrid(size=45)
    s_cones_parameters = {
        "s_cones_final_count": 160,
        "m_cones_final_count": 1840,
        "m_cones_birth_rate": birth_rate,
        "a": a,
        "c": c,
        "exponential_probability_growth_factor": exponential_probability_growth_factor
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
birth_rate_values = np.linspace(0.4, 1, 6)
a_values = np.linspace(80, 400, 10)  # Adjust the range and step as needed
c_values = np.linspace(5, 20, 10)  # Adjust the range and step as needed
exponential_factors = np.linspace(1.5, 4, 5)  # Adjust the range and step as needed

# Grid search
best_variance = float('inf')
best_params = {}

for br, a, c, exp_factor in itertools.product(birth_rate_values, a_values, c_values, exponential_factors):
    variance = run_simulation(br, a, c, exp_factor)
    logging.info(f"Parameters: birth_rate={br} a={a}, c={c}, exp_factor={exp_factor}, Variance: {variance}")

    if variance < best_variance:
        best_variance = variance
        best_params = {'a': a, 'c': c, 'exponential_probability_growth_factor': exp_factor}

# Output the best parameters
print(f"Best Parameters: {best_params}, Best Variance: {best_variance}")
