from runSimulation import run_simulation

def run_single_simulation():

    params = {
        "grid_size": 30, # total number of hexagons in the grid = grid_size * grid_size

        "s_cones_init_count": 1,
        "m_cones_init_count": 0,
        "sCones_to_mCones_ratio": 0.02,

        "init_mode": "bfs",
        "max_probability": 0.1
    }

    run_simulation(params, writeLogs=False, createGif=True, voronoi_analysis=False, FT_analysis=False, NN_analysis=False)


if __name__ == "__main__":
    run_single_simulation()