import itertools
import numpy as np
from runSimulation import run_simulation

def explore_parameters():
    # Define parameter ranges for grid search
    s_cones_init = np.array([1, 20, 40, 60, 80])
    m_cones_init = np.array([1, 30, 80, 200, 300, 500, 700, 900])
    max_probability = np.array(0.1)  # Adjust the range and step as needed

    for i in range(10):
        params = {
            "grid_size": 45, # total number of hexagons in the grid = grid_size * grid_size

            "s_cones_init_count": 90,
            "m_cones_init_count": 0,
            "sCones_to_mCones_ratio": 0.03,


            "init_mode": "bfs",
            "max_probability": 0.1
        }
        run_simulation(params, writeLogs=True, createGif=True, voronoi_analysis=True, FT_analysis=True, NN_analysis=False)

        for m in [50, 200, 500, 1000, 2000, 2700]:
            params = {
                "grid_size": 45, # total number of hexagons in the grid = grid_size * grid_size

                "s_cones_init_count": 0,
                "m_cones_init_count": m,


                "init_mode": "bfs",
                "max_probability": 0.1
            }
            run_simulation(params, writeLogs=True, createGif=True, voronoi_analysis=True, FT_analysis=True, NN_analysis=False)




if __name__ == "__main__":
    explore_parameters()