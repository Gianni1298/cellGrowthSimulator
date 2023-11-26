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

    def find_neighbour(self, x, y, dx, dy):
        # Check if a neighbour exists at a specified offset
        neighbour_x, neighbour_y = x + dx, y + dy

        # Use np.isclose to check if the hexagon at these coordinates exists in the grid
        # It compares each element of the hex_centers with the neighbour_x and neighbour_y within a tolerance
        mask = np.isclose(self.hex_centers[:, 0], neighbour_x) & np.isclose(self.hex_centers[:, 1], neighbour_y)

        if np.any(mask):
            index = np.where(mask)[0][0]
            return [self.hex_centers[index], index]
        return None

    def find_neighbours(self, hex_ix):
        # Find the neighbors of the hexagon at the specified index
        x, y = self.hex_centers[hex_ix]

        # Directions for hexagonal grid neighbors
        directions = [(1, 0),  # Right
                      (-1, 0),  # Left
                      (0.5, np.sqrt(3) / 2),  # Top right
                      (-0.5, np.sqrt(3) / 2),  # Top left
                      (0.5, -np.sqrt(3) / 2),  # Bottom right
                      (-0.5, -np.sqrt(3) / 2)]  # Bottom left

        neighbors = []
        for dx, dy in directions:
            neighbour = self.find_neighbour(x, y, dx, dy)
            if neighbour:
                neighbors.append(neighbour)

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
neighbours = grid.find_neighbours(24)
