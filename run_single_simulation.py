from runSimulation import run_simulation

def run_single_simulation():

    params = {
        "grid_size": 70, # total number of hexagons in the grid = grid_size * grid_size

        "s_cones_init_count": 5,
        "m_cones_init_count": 10,
        "sCones_to_mCones_ratio": 0.02,

        "init_mode": "bfs",
        "move_mode": "line",
        "max_probability": 0.1
    }

    run_simulation(params, writeLogs=False, createGif=True, voronoi_analysis=False, FT_analysis=False, NN_analysis=False)


if __name__ == "__main__":
    run_single_simulation()