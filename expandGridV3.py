import random
import numpy as np
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *
from collections import deque
import os
import imageio


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

    def draw(self, blue_indices, green_indices=None):
        if green_indices is None:
            green_indices = set()
        hex_centers = self.hex_centers
        self.blue_indices = blue_indices

        # Color the selected hexagons blue and the rest white
        colors = ['b' if i in blue_indices else ('aquamarine' if i in green_indices else 'w') for i in range(len(hex_centers))]


        # Now plot the hexagonal grid with the specified colors
        fig, ax = plt.subplots()
        plot_single_lattice_custom_colors(hex_centers[:, 0], hex_centers[:, 1],
                                          face_color=colors,
                                          edge_color='w',  # Keep the edges black for visibility
                                          min_diam=0.9,
                                          plotting_gap=0,
                                          rotate_deg=0)

        green_cells = len(green_indices)
        plt.title(f'Hexagonal Grid Size {green_cells} green cells')
        self.save_plot(fig, green_cells)

    def save_plot(self, fig, filename):
        plt.savefig(f'output_plots/{filename}.png')
        plt.close(fig)

    def create_gif(self, gif_name):
        filenames = [f for f in os.listdir('output_plots') if f.endswith('.png')]
        # Sorting files numerically based on the number in the filename
        filenames.sort(key=lambda x: int(x.split('.')[0]))

        images = []
        for filename in filenames:
            images.append(imageio.v3.imread(f'output_plots/{filename}'))
        imageio.mimsave(f'output_plots/{gif_name}.gif', images, duration=0.5)

        for filename in filenames:
            os.remove(f'output_plots/{filename}')


class Scones:
    def __init__(self, hex_grid, s_cone_count):
        self.hex_grid = hex_grid
        self.s_cone_count = s_cone_count
        self.blue_indices = self.init_blue_indices()
        self.m_cones = Mcones(self.hex_grid, birth_rate=0.5)
        self.decay_rate = 0.01

    def init_blue_indices(self):
        if self.s_cone_count == 0:
            return Exception("s_cone_count must be greater than 0")

        blue_indices = set()

        # Select middle hexagon as the first blue hexagon
        start_hex = self.hex_grid.find_hexagon_index(0, 0)
        blue_indices.add(start_hex)

        blue_cells_to_place = self.s_cone_count - 1
        queue = deque([start_hex])  # Using a deque as a queue for BFS
        while blue_cells_to_place > 0:
            cur_cell = queue.popleft()  # Popping from the left side for BFS
            neighbours, indexes = self.hex_grid.find_neighbours(cur_cell)
            for index in indexes:
                if index not in blue_indices:
                    blue_indices.add(index)
                    queue.append(index)  # Adding to the right side (end of the queue)
                    blue_cells_to_place -= 1
                    if blue_cells_to_place == 0:
                        break
        return blue_indices

    def move_sCones(self):
        new_blue_indices = set()

        # Pick a random element from the blue_indices set and remove it from the set
        while self.blue_indices:
            cell_to_move = random.choice(list(self.blue_indices))
            self.blue_indices.remove(cell_to_move)
            neighbours, indexes = self.hex_grid.find_neighbours(cell_to_move)

            allowed_moves = []
            for index in indexes:
                if index not in self.blue_indices and index not in new_blue_indices:
                    allowed_moves.append(index)

            # Pick a random element from the allowed_moves list and add it to the new_blue_indices set
            if allowed_moves:
                move = self.choose_move(cell_to_move, allowed_moves)
                new_blue_indices.add(move)
                if cell_to_move not in self.m_cones.get_green_indices():
                    self.m_cones.add_green_index(cell_to_move)
                else:
                    self.m_cones.add_green_closest_to_center()
            else:
                new_blue_indices.add(cell_to_move)

        self.blue_indices = new_blue_indices
        return self.blue_indices

    def choose_move(self, cell_to_move, allowed_moves):
        max_distance = self.hex_grid.sorted_distances[-1]
        current_distance = self.hex_grid.calculate_distance(cell_to_move)

        probabilities = []

        # Amplification factor: the closer we're to the center, the more likely we are to move outwards
        # If we are at the center, we are guaranteed to move outwards
        # If we are closer to the edge, it is equally likely to move in all directions
        A = np.exp(-self.decay_rate * (current_distance / max_distance))

        # Calculate the probability of moving in each direction
        for move in allowed_moves:
            move_distance = self.hex_grid.calculate_distance(move) - current_distance
            probabilities.append(A * np.exp(move_distance / 2))


        # Normalize probabilities to sum up to 1
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]

        # Choose a move based on the probabilities
        chosen_move = np.random.choice(allowed_moves, p=probabilities)
        return chosen_move



class Mcones:
    def __init__(self, hex_grid, birth_rate):
        self.green_indices = set()
        self.hex_grid = hex_grid
        self.birth_rate = birth_rate

    def get_green_indices(self):
        return self.green_indices

    def add_green_index(self, index):
        self.green_indices.add(index)

    def add_green_closest_to_center(self):
        if random.random() <= self.birth_rate:
            for index in self.hex_grid.sorted_indexes:
                if index not in self.green_indices and index not in self.hex_grid.blue_indices:
                    self.green_indices.add(index)
                    break


# Example usage
grid = HexGrid(size=45)
s_cones = Scones(grid, 160)

grid.draw(s_cones.blue_indices)

while len(s_cones.m_cones.get_green_indices()) < 1840:
    grid.draw(s_cones.move_sCones(), s_cones.m_cones.get_green_indices())

grid.create_gif(f"scones_v13_{s_cones.m_cones.birth_rate}br_{grid.size}Grid_decay_rate{s_cones.decay_rate}")

