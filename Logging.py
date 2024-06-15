import os
import csv
from datetime import datetime
import json

import pandas as pd

import Metrics


class myLogger:
    def __init__(self, cells, params):
        # Initialize the DataFrame with the required rows
        self.data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'gridSize': params['grid_size'],
            'totalCells': cells.final_cell_count,
            'sCones_mCones_ratio': params['sCones_to_mCones_ratio'],
            'maxProb': params['max_probability'],
            'sConesInit': params['s_cones_init_count'],
            'mConesInit': params['m_cones_init_count'],
            'move_mode': params['move_mode'],
            'sCones_coordinates': [],  # List of sCones coordinates
            'voronoi areas': [], # List of voronoi areas
            'voronoi area variance': [], # List of voronoi area variance
            'NN-distances': [], # List of NN distances
            'VDRI[# of cells -> VDRI]': [], # List of VDRI values
            'NNRI[# of cells -> NNRI]': [], # List of NNRI values
        }

    def log_sCones_coordinates(self, sCones_coordinates):
        self.data['sCones_coordinates'] = sCones_coordinates.tolist()

    def log_running_metrics(self, cells, has_voronoi_analysis, has_NN_analysis):
        if len(cells.get_sCones_cells()) < 10:
            return
        if len(cells.cell_indexes) % 20 != 0:
            return

        if has_voronoi_analysis:
            self.data['VDRI[# of cells -> VDRI]'].append([
                len(cells.cell_indexes),
                Metrics.calculate_VDRI(cells)
            ])

        if has_NN_analysis:
            self.data['NNRI[# of cells -> NNRI]'].append([
                len(cells.cell_indexes),
                Metrics.calculate_NNRI(cells)
            ])


    def log_final_voronoi_metrics(self, areas, variance):
        self.data['voronoi areas'] = areas
        self.data['voronoi area variance'] = variance

    def log_final_NN_metrics(self, distances):
        self.data['NN-distances'] = list(distances)

    def write_logs(self):
        if not os.path.exists('output/logs.json'):
            with open('output/logs.json', 'w') as f:
                f.write(json.dumps(self.data) + "\n")  # Add a newline to separate entries
        else:
            with open("output/logs.json", "a") as f:
                f.write(json.dumps(self.data) + "\n")  # Add a newline to separate entries


    def log_results(self, parameters, cell_indexes, blue_cell_indexes, voronoi_areas, voronoi_variance, FTFrequencies,
                    nnd, ripleyG, ripleyF, ripleyJ, ripleyK, ripleyL):
        # Create a new row as a dictionary
        new_row = {'gridSize': parameters['grid_size'],
                   'sConesInit': parameters['s_cones_init_count'],
                   'mConesInit': parameters['m_cones_init_count'],
                   'sConesFinal': int(parameters['grid_size'] * parameters['grid_size'] * 0.7 * parameters['sCones_to_mCones_ratio']),
                   'mConesFinal': int(parameters['grid_size'] * parameters['grid_size'] * 0.7 * (1-parameters['sCones_to_mCones_ratio'])),
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
                   'ripleyL': ripleyL,
                   'sCones-mCones ratio': parameters['sCones_to_mCones_ratio']}

        # Append the row to the DataFrame
        self.data.loc[len(self.data)] = new_row
        self.save_to_csv()

    def save_to_csv(self):

        # Check if the file exists
        if os.path.exists(f"logs/{self.filename}"):
            # Append without header
            self.data.to_csv("logs/" + self.filename, mode='a', header=False, index=False, sep='|')
        else:
            # Write with header
            self.data.to_csv("logs/" + self.filename, mode='w', header=True, index=False, sep='|')