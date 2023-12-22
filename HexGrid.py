import os

import imageio
from hexalattice.hexalattice import *

from plotHelpers import save_plot


class HexGrid:
    def __init__(self, size):
        self.size = size
        self.hex_centers, _ = self.generate_hex_centers(self.size)
        self.blue_indices = []
        self.x_center, self.y_center = self.hex_centers[:, 0].mean(), self.hex_centers[:, 1].mean()
        self.sorted_distances, self.sorted_indexes = self.get_sorted_distances()

    def generate_hex_centers(self, size):
        # Generate hexagon centers
        return create_hex_grid(nx=self.size, ny=self.size, do_plot=False)

    def find_hexagon_index(self, x, y):
        # Find the index of the hexagon at the specified coordinates, if it exists
        mask = np.isclose(self.hex_centers[:, 0], x) & np.isclose(self.hex_centers[:, 1], y)
        if np.any(mask):
            index = np.where(mask)[0][0]
            return index
        return None

    def find_closest_hexagon(self, x, y):
        # Calculate distances from the point (x, y) to each hex center
        distances = np.sqrt((self.hex_centers[:, 0] - x) ** 2 + (self.hex_centers[:, 1] - y) ** 2)

        # Find the index of the closest hexagon
        closest_index = np.argmin(distances)

        return closest_index

    def find_neighbour(self, x, y, dx, dy):
        # Check if a neighbour exists at a specified offset
        neighbour_x, neighbour_y = x + dx, y + dy

        index = self.find_hexagon_index(neighbour_x, neighbour_y)
        if index:
            return self.hex_centers[index]
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
        indexes = []
        for dx, dy in directions:
            neighbour = self.find_neighbour(x, y, dx, dy)
            if neighbour is not None:
                neighbors.append(neighbour)
                indexes.append(self.find_hexagon_index(neighbour[0], neighbour[1]))

        return neighbors, indexes

    def calculate_distance(self, index):
        # Calculate the radial distance from the center of the grid
        center_x = self.x_center
        center_y = self.y_center
        distance = ((self.hex_centers[index, 0] - center_x) ** 2 + (self.hex_centers[index, 1] - center_y) ** 2) ** 0.5
        return distance

    def get_sorted_distances(self):
        # Calculate the radial distances from the center of the grid
        center_x = self.hex_centers[:, 0].mean()
        center_y = self.hex_centers[:, 1].mean()
        distances = ((self.hex_centers[:, 0] - center_x) ** 2 + (self.hex_centers[:, 1] - center_y) ** 2) ** 0.5

        [distances, indexes] = zip(*sorted(zip(distances, range(len(distances)))))

        return distances, indexes

    def draw(self, cell_indexes, showPlot=False):
        hex_centers = self.hex_centers
        # Color the selected hexagons blue and the rest white
        colors = [cell_indexes.get(i, 'w') for i in range(len(hex_centers))]

        # Now plot the hexagonal grid with the specified colors
        fig, ax = plt.subplots()
        plot_single_lattice_custom_colors(hex_centers[:, 0], hex_centers[:, 1],
                                          face_color=colors,
                                          edge_color='w',  # Keep the edges black for visibility
                                          min_diam=0.9,
                                          plotting_gap=0,
                                          rotate_deg=0)

        total_cells = len(cell_indexes)
        green_cells = sum([1 for color in cell_indexes.values() if color == "aquamarine"])
        blue_cells = sum([1 for color in cell_indexes.values() if color == "b"])
        plt.title(f'Hexagonal Grid Size {total_cells} total cells. Green = {green_cells}, Blue = {blue_cells}')
        if showPlot:
            plt.show()
        else:
            save_plot(fig, total_cells)

    def calculate_hex_grid_bounds(self):
        # Assuming each hexagon has a unit diameter
        diameter = 1
        vertical_distance = 0.75 * diameter

        # Horizontal extent (x-axis)
        xmax = xmax = (self.size - 1) * diameter / 2
        xmin = -xmax

        # Vertical extent (y-axis) - slightly more complex due to hexagonal staggering
        ymax = (self.size - 1) * vertical_distance / 2
        ymin = -ymax

        return xmin, xmax, ymin, ymax
