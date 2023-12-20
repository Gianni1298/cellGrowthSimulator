import itertools
import numpy as np
from runSimulation import run_simulation

def explore_parameters():
    # Define parameter ranges for grid search
    s_cones_init = np.array([1, 20, 40, 60, 80])
    m_cones_init = np.array([1, 30, 80, 200, 300, 500, 700, 900])
    max_probability = np.array([0.1, 0.3, 0.5, 0.75, 1])  # Adjust the range and step as needed

    grid_size = 45

    for s, m, p in itertools.product(s_cones_init, m_cones_init, max_probability):
        cells_parameters = {
            "s_cones_init_count": s,
            "m_cones_init_count": m,

            "s_cones_final_count": 80,
            "m_cones_final_count": 920,

            "init_mode": "bfs",
            "max_probability": p
        }
        _, variance = run_simulation(cells_parameters, grid_size, writeLogs=True, createGif=True, createCDF=True)


if __name__ == "__main__":
    explore_parameters()