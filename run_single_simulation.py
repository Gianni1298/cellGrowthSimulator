from runSimulation import run_simulation

def run_single_simulation():
    grid_size = 45

    cells_parameters = {
        "s_cones_init_count": 10,
        "m_cones_init_count": 0,

        "s_cones_final_count": 10,
        "m_cones_final_count": 920,

        "init_mode": "bfs",
        "max_probability": 0.1
    }

    run_simulation(cells_parameters, grid_size, writeLogs=False, createGif=True, createCDF=True)


if __name__ == "__main__":
    run_single_simulation()