from runSimulation import run_simulation

def run_single_simulation():

    params = {
        "grid_size": 70, # total number of hexagons in the grid = grid_size * grid_size

        "s_cones_init_count": 1,
        "m_cones_init_count": 700,
        "sCones_to_mCones_ratio": 0.08,

        "init_mode": "bfs",
        "move_mode": "line",
        "max_probability": 0.1
    }

    run_simulation(params, writeLogs=True, createGif=True, has_voronoi_analysis=True, has_NN_analysis=True)


if __name__ == "__main__":
    run_single_simulation()