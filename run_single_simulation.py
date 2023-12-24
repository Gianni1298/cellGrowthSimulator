from runSimulation import run_simulation

def run_single_simulation():

    params = {
        "grid_size": 70, # total number of hexagons in the grid = grid_size * grid_size

        "s_cones_init_count": 1,
        "m_cones_init_count": 500,

        # Only fill 70% of the grid with 92% m-cones and 8% s-cones
        # Total final cell count: grid_size * grid_size * 0.7 = 10000 * 0.7 = 7000
        # "s_cones_final_count": 0.08 * 0.7 * 10000,
        # "m_cones_final_count": 0.92 * 0.7 * 10000,
        "s_cones_final_count": 560,
        "m_cones_final_count": 6440,

        "init_mode": "bfs",
        "max_probability": 0.1
    }

    run_simulation(params, writeLogs=True, createGif=True, voronoi_analysis=True, FT_analysis=True, NN_analysis=False)


if __name__ == "__main__":
    run_single_simulation()