import itertools
import numpy as np
from runSimulation import run_simulation

def explore_parameters():
    max_probability = [0.1, 0.5, 1] # values of max_probability to explore

    for prob in max_probability:
        for i in range(10):
            params = {
                "grid_size": 80, # total number of hexagons in the grid = grid_size * grid_size

                "s_cones_init_count": 50,
                "m_cones_init_count": 0,
                "sCones_to_mCones_ratio": 0.08,

                "init_mode": "bfs",
                "move_mode": "line",
                "max_probability": prob
            }
            run_simulation(params, writeLogs=True, createGif=True, has_voronoi_analysis=True, has_NN_analysis=True)

            for m in [200, 1000, 1690]:
                params = {
                    "grid_size": 80, # total number of hexagons in the grid = grid_size * grid_size

                    "s_cones_init_count": 0,
                    "m_cones_init_count": m,
                    "sCones_to_mCones_ratio": 0.08,


                    "init_mode": "bfs",
                    "move_mode": "line",
                    "max_probability": prob
                }
                run_simulation(params, writeLogs=True, createGif=True, has_voronoi_analysis=True, has_NN_analysis=True)




if __name__ == "__main__":
    explore_parameters()