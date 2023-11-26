import numpy as np
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *

class HexGrid:
    def __init__(self, size):
        self.size = size
        self.hex_centers, _ = self.generate_hex_centers_with_ix(self.size)
        # self.grid_center = [self.hex_centers[:, 0].mean(), self.hex_centers[:, 1].mean()]

    def generate_hex_centers_with_ix(self, size):
        # Generate hexagon centers
        return create_hex_grid(nx=self.size, ny=self.size, do_plot=False)

    def find_neighbours(self, hex_ix):
        # Find the neighbors of the hexagon at the specified index
        x, y = self.hex_centers[hex_ix]

        neighbors = [] # List of neighbors [center, index]

        if hex_ix+1 < len(self.hex_centers):                                                    # Right
            neighbors.append([self.hex_centers[hex_ix+1], hex_ix+1])
        if hex_ix-1 >= 0:                                                                       # Left
            neighbors.append([self.hex_centers[hex_ix-1], hex_ix-1])
        if y + np.sqrt(3)/2 in self.hex_centers[:, 1] and x-1/2 in self.hex_centers[:, 0]:      # Top Left
            index = np.where(np.all(self.hex_centers == [x-1/2, y+np.sqrt(3)/2], axis=1))[0][0]
            neighbors.append([self.hex_centers[index], index])
        if y + np.sqrt(3)/2 in self.hex_centers[:, 1] and x+1/2 in self.hex_centers[:, 0]:      # Top Right
            index = np.where(np.all(self.hex_centers == [x+1/2, y+np.sqrt(3)/2], axis=1))[0][0]
            neighbors.append([self.hex_centers[index], index])
        if y - np.sqrt(3)/2 in self.hex_centers[:, 1] and x-1/2 in self.hex_centers[:, 0]:      # Bottom Left
            index = np.where(np.all(self.hex_centers == [x-1/2, y-np.sqrt(3)/2], axis=1))[0][0]
            neighbors.append([self.hex_centers[index], index])
        if y - np.sqrt(3)/2 in self.hex_centers[:, 1] and x+1/2 in self.hex_centers[:, 0]:      # Bottom Right
            index = np.where(np.all(self.hex_centers == [x+1/2, y-np.sqrt(3)/2], axis=1))[0][0]
            neighbors.append([self.hex_centers[index], index])

        return neighbors

    def draw(self):
        hex_centers = self.hex_centers

        # Color the selected hexagons blue and the rest white
        blue_indices = []
        colors = ['b' if i in blue_indices else 'w' for i in range(len(hex_centers))]

        # Now plot the hexagonal grid with the specified colors
        fig, ax = plt.subplots()
        plot_single_lattice_custom_colors(hex_centers[:, 0], hex_centers[:, 1],
                                          face_color=colors,
                                          edge_color='k',  # Keep the edges black for visibility
                                          min_diam=0.9,
                                          plotting_gap=0,
                                          rotate_deg=0)

        plt.title(f'Hexagonal Grid Size {self.size}x{self.size}')
        plt.show()

# class s-cones:
#     def __init__(self):
#         continue


# Example usage
grid = HexGrid(size=5)

grid.draw()
neighbours = grid.find_neighbours(6)

