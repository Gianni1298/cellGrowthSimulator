from runSimulation import run_simulation

def run_single_simulation():

    params = {
        "grid_size": 40,

        "s_cones_init_count": 1,
        "m_cones_init_count": 50,

        "s_cones_final_count": 80,
        "m_cones_final_count": 920,

        "init_mode": "bfs",
        "max_probability": 0.1
    }

    run_simulation(params, writeLogs=True, createGif=True, plotVoronoi=True, createCDF=True, FTPlot=True, NNAPlot=True)


if __name__ == "__main__":
    run_single_simulation()