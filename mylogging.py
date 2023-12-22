import os
import csv


class myLogger:
    def __init__(self, filename):
        # Creates a csv file if it doesn't exist with the required columns
        # Opens it if a csv file with the same name already exists
        self.file = open(filename, 'a')
        self.writer = csv.writer(self.file, delimiter='|')
        if os.stat(filename).st_size == 0:
            self.writer.writerow(['gridSize', 'sConesInit','mConesInit', 'sConesFinal','mConesFinal','maxProb',
                                  'cell_indexes', 'blueHexCenters','voronoi areas','voronoi area variance','FTFreq-Magnitude',
                                  'NNA distances', 'R_NNA'])

    def log_results(self, parameters, cell_indexes, blue_cell_indexes, voronoi_areas, voronoi_variance, FTFrequencies, neareast_neigbour_distances, R_NNA):
        # Convert arrays to strings

        voronoi_areas_str = ','.join(map(str, voronoi_areas))


        self.writer.writerow([f"{parameters['grid_size']}",
                        f"{parameters['s_cones_init_count']}",
                        f"{parameters['m_cones_init_count']}",
                        f"{parameters['s_cones_final_count']}",
                        f"{parameters['m_cones_final_count']}",
                        f"{parameters['max_probability']}",
                        f"{cell_indexes}",
                        f"{blue_cell_indexes}",
                        f"{voronoi_areas_str}",
                        f"{voronoi_variance}",
                        f"{FTFrequencies}",
                        f"{neareast_neigbour_distances}",
                        f"{R_NNA}"])

    def close(self):
        self.file.close()