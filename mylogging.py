import os
import csv

import pandas as pd


class myLogger:
    def __init__(self, filename):
        # Initialize the DataFrame with the required columns
        self.data = pd.DataFrame(columns=['gridSize', 'sConesInit', 'mConesInit', 'sConesFinal', 'mConesFinal',
                                          'maxProb', 'cell_indexes', 'blueHexCenters', 'voronoi areas',
                                          'voronoi area variance', 'FTFreq-Magnitude', 'NN distances',
                                          'ripleyG', 'ripleyF', 'ripleyJ', 'ripleyK', 'ripleyL'])
        self.filename = filename

    def log_results(self, parameters, cell_indexes, blue_cell_indexes, voronoi_areas, voronoi_variance, FTFrequencies,
                    nnd, ripleyG, ripleyF, ripleyJ, ripleyK, ripleyL):
        # Create a new row as a dictionary
        new_row = {'gridSize': parameters['grid_size'],
                   'sConesInit': parameters['s_cones_init_count'],
                   'mConesInit': parameters['m_cones_init_count'],
                   'sConesFinal': int(parameters['grid_size'] * parameters['grid_size'] * 0.7 * 0.08),
                   'mConesFinal': int(parameters['grid_size'] * parameters['grid_size'] * 0.7 * 0.92),
                   'maxProb': parameters['max_probability'],
                   'cell_indexes': cell_indexes,
                   'blueHexCenters': blue_cell_indexes,
                   'voronoi areas': voronoi_areas,
                   'voronoi area variance': voronoi_variance,
                   'FTFreq-Magnitude': FTFrequencies,
                   'NN distances': nnd,
                   'ripleyG': ripleyG,
                   'ripleyF': ripleyF,
                   'ripleyJ': ripleyJ,
                   'ripleyK': ripleyK,
                   'ripleyL': ripleyL}

        # Append the row to the DataFrame
        self.data.loc[len(self.data)] = new_row
        self.save_to_csv()

    def save_to_csv(self):
        # Check if the file exists
        if os.path.exists(self.filename):
            # Append without header
            self.data.to_csv(self.filename, mode='a', header=False, index=False, sep='|')
        else:
            # Write with header
            self.data.to_csv(self.filename, mode='w', header=True, index=False, sep='|')