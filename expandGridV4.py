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

    def draw(self, cell_indexes, plot=False):
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
        if plot:
            plt.show()
        else:
            self.save_plot(fig, total_cells)

    def save_plot(self, fig, filename):
        plt.savefig(f'output_plots/v4/{filename}.png')
        plt.close(fig)

    def create_gif(self, gif_name):
        filenames = [f for f in os.listdir('output_plots/v4') if f.endswith('.png')]
        # Sorting files numerically based on the number in the filename
        filenames.sort(key=lambda x: int(x.split('.')[0]))

        images = []
        for filename in filenames:
            images.append(imageio.v3.imread(f'output_plots/v4/{filename}'))
        imageio.mimsave(f'output_plots/v4/{gif_name}.gif', images, duration=0.5)

        for filename in filenames:
            os.remove(f'output_plots/v4/{filename}')


class Scones:
    def __init__(self, hex_grid, s_cone_params):
        self.hex_grid = hex_grid
        self.s_cone_params = s_cone_params
        self.cell_indexes = self.init(s_cone_params["init_mode"])  # Map of cell indexes to cell color {index: color}
        print(self.cell_indexes)
        self.debug = {"birth_colors": [], "birth_probabilities": []}

        self.stopSignal = False

    def init(self, init_mode):
        if init_mode == "bfs":
            return self.bfs_init()
        elif init_mode == "random":
            print("Random init not implemented yet!")
            # return self.random_init()

    def bfs_init(self):
        total_cells_to_place = self.s_cone_params["s_cones_init_count"] + self.s_cone_params["m_cones_init_count"]

        # Step 1: Fetch Hexagon Indexes
        indexes_to_populate = []
        start_hex = self.hex_grid.find_hexagon_index(0, 0)
        visited_hex = {start_hex}
        queue = deque([start_hex])
        while total_cells_to_place > 0 and queue:
            cur_hex = queue.popleft()
            indexes_to_populate.append(cur_hex)
            total_cells_to_place -= 1
            neighbours, neighbour_indexes = self.hex_grid.find_neighbours(cur_hex)
            for index in neighbour_indexes:
                if index not in visited_hex:
                    queue.append(index)
                    visited_hex.add(index)

        random.shuffle(indexes_to_populate)

        # Step 2: Populate with Colors
        cell_indexes = {}
        blue_cells_to_place = self.s_cone_params["s_cones_init_count"]
        green_cells_to_place = self.s_cone_params["m_cones_init_count"]
        for index in indexes_to_populate:
            if blue_cells_to_place > 0:
                cell_indexes[index] = "b"
                blue_cells_to_place -= 1
            elif green_cells_to_place > 0:
                cell_indexes[index] = "aquamarine"
                green_cells_to_place -= 1

        return cell_indexes


    def random_init(self):
        # Not implemented in this current version!! ###
        blue_indices = set()
        blue_cells_to_place = self.s_cone_count

        # Calculate the radius of the circle for initialization
        radius = self.hex_grid.size // 4  # You can adjust this as needed

        while blue_cells_to_place > 0:
            # Generate a random point
            random_x = random.uniform(-self.hex_grid.size / 2, self.hex_grid.size / 2)
            random_y = random.uniform(-self.hex_grid.size / 2, self.hex_grid.size / 2)

            # Check if the point is within the circle
            if (random_x ** 2 + random_y ** 2) <= radius ** 2:
                index = self.hex_grid.find_closest_hexagon(random_x, random_y)
                if index not in blue_indices:
                    blue_indices.add(index)
                    blue_cells_to_place -= 1

        green_indices = self.fill_circle_with_green(radius, blue_indices)
        return blue_indices


    def move_cell_bfs(self):
        cell_to_move_index = random.choice(list(self.cell_indexes.keys()))
        # cell_to_move_index = self.hex_grid.find_closest_hexagon(0, 0)

        path = self.find_shortest_path_to_empty(cell_to_move_index)
        if path:
            self.move_along_path(path)

        self.cell_indexes[cell_to_move_index] = self.cell_birth()

        if len(self.cell_indexes) > self.s_cone_params["s_cones_final_count"] + self.s_cone_params["m_cones_final_count"]:
            self.stopSignal = True
            return



    def find_shortest_path_to_empty(self, start_index):
        queue = deque([(start_index, [start_index])])
        visited = set()

        while queue:
            current_index, path = queue.popleft()
            visited.add(current_index)
            _, neighbours = self.hex_grid.find_neighbours(current_index)

            for neighbour in neighbours:
                if neighbour not in visited:
                    if neighbour not in self.cell_indexes:
                        return path + [neighbour]
                    else:
                        queue.append((neighbour, path + [neighbour]))
                        visited.add(neighbour)
        return None

    def move_along_path(self, path):
        if len(path) < 2:
            return  # No movement needed

        moving_color = self.cell_indexes[path[0]]
        for i in range(len(path) - 1, 0, -1):
            self.cell_indexes[path[i]] = self.cell_indexes[path[i - 1]]

        self.cell_indexes[path[0]] = moving_color
        del self.cell_indexes[path[0]]

    def move_cell_dfs(self):
        # Randomly select a cell to start the cascade
        # cell_to_move_index = random.choice(list(self.cell_indexes.keys()))
        cell_to_move_index = self.hex_grid.find_closest_hexagon(0, 0)
        print("Moving cell at index", cell_to_move_index)

        moving_color = self.cell_indexes[cell_to_move_index]
        move = self.choose_move(cell_to_move_index)

        if move not in self.cell_indexes:
            # Direct move if the spot is empty
            self.cell_indexes[move] = moving_color
        else:
            # If the spot is occupied, start a cascade move
            self.cascade_move_dfs(move, moving_color, visited={cell_to_move_index})

        # A new cell is born that takes the place of the cell that moved
        del self.cell_indexes[cell_to_move_index]
        # self.cell_indexes[cell_to_move_index] = self.cell_birth()
        #
        # if len(self.cell_indexes) > self.s_cone_params["s_cones_final_count"] + self.s_cone_params["m_cones_final_count"]:
        #     self.stopSignal = True
        #     return

    def cascade_move_dfs(self, current_index, moving_color, visited=None):
        if visited is None:
            visited = set()

        # Mark the current cell as visited
        visited.add(current_index)
        neighbours, neighbour_indexes = self.hex_grid.find_neighbours(current_index)

        # Sort neighbours by increasing distance from the center
        sorted_neighbours = sorted(neighbour_indexes,
                                   key=lambda ix: (ix in self.cell_indexes, -self.hex_grid.calculate_distance(ix)))

        for next_move in sorted_neighbours:
            if next_move in visited:
                continue  # Skip already visited cells to prevent cycles

            if next_move not in self.cell_indexes:
                self.cell_indexes[next_move] = self.cell_indexes[current_index]
                self.cell_indexes[current_index] = moving_color
                return
            else:
                # Continue the cascade with the color of the cell that is being displaced
                displaced_color = self.cell_indexes[current_index]
                self.cell_indexes[current_index] = moving_color
                self.cascade_move_dfs(next_move, displaced_color, visited)
                return
        return

    def cell_birth(self):
        # A blue cell or green cell is born with a probability given by the parameters in the config and the current cell count

        # First get current blue and green cell counts
        blue_cell_count = sum([1 for color in self.cell_indexes.values() if color == "b"])
        green_cell_count = sum([1 for color in self.cell_indexes.values() if color == "aquamarine"])

        if blue_cell_count == self.s_cone_params["s_cones_final_count"]:
            return "aquamarine"
        elif green_cell_count == self.s_cone_params["m_cones_final_count"]:
            return "b"
        else:
            prob_green = self.quadratic_probability(current_count=green_cell_count,
                                                    final_count=self.s_cone_params["m_cones_final_count"], max_probability=1)

            prob_blue = self.quadratic_probability(current_count=blue_cell_count,
                                                   final_count=self.s_cone_params["s_cones_final_count"], max_probability=self.s_cone_params["max_probability"])

            # print(f"Green probability: {prob_green}, Blue probability: {prob_blue}")

            norm_prob_green, norm_prob_blue = self.normalize_probabilities(prob_green, prob_blue)

            # print(f"Normalized green probability: {norm_prob_green}, Normalized blue probability: {norm_prob_blue}")
            choice = np.random.choice(['aquamarine', 'b'], p=[norm_prob_green, norm_prob_blue])
            self.debug["birth_colors"].append(choice)
            self.debug["birth_probabilities"].append((norm_prob_green, norm_prob_blue))
            return choice

    def quadratic_probability(self, current_count, final_count, max_probability):
        """
        Calculate the birth probability based on a quadratic function.

        :param current_count: Current count of cells
        :param final_count: Final count of cells
        :param max_probability: Maximum probability of birth
        :param a: Determines the width of the parabola
        :param dh: Shifts the parabola horizontally
        :return: Probability of birth at the current count
        """

        half_final_count = final_count / 2

        a = max_probability / ((final_count / 2)**2)
        # Quadratic function
        probability = -a * (current_count - half_final_count) ** 2 + max_probability
        return max(probability, 0.01)  # Ensure probability is not negative

    def normalize_probabilities(self, prob_green, prob_blue):
        total_prob = prob_green + prob_blue
        norm_prob_green = prob_green / total_prob
        norm_prob_blue = prob_blue / total_prob
        return norm_prob_green, norm_prob_blue

    def choose_move(self, cell_to_move):
        _, moves_indexes = self.hex_grid.find_neighbours(cell_to_move)

        # If there is an empty neighbour, move to it, it's the most likely thing to happen
        for move in moves_indexes:
            if move not in self.cell_indexes:
                return move

        # Calculate the normalized count of green cells
        cell_ratio = len(self.cell_indexes) / len(self.hex_grid.hex_centers)

        # Update the Gaussian wave's mean based on the green ratio and velocity m
        mu = self.s_cone_params['m'] * cell_ratio

        # Calculate the 80% percentile of the Gaussian distribution
        p80 = mu + 0.842 * self.s_cone_params['c']

        # Check the distance of the cell from the center
        current_distance = self.hex_grid.calculate_distance(cell_to_move)

        if abs(current_distance - mu) <= p80 - mu:
            # Rank the moves based on how much they increase the distance from the center
            move_distances = [(move, self.hex_grid.calculate_distance(move)) for move in moves_indexes]
            move_distances.sort(key=lambda x: x[1] - current_distance, reverse=True)

            # Assign probabilities proportional to the rank
            total_ranks = sum(range(1, len(moves_indexes) + 1))
            probabilities = [(len(moves_indexes) - rank) / total_ranks for rank, _ in enumerate(move_distances)]

            # Extract moves from sorted list
            sorted_moves = [move for move, _ in move_distances]
        else:
            # If the cell is outside the 80% percentile, use Gaussian probabilities
            probabilities = []
            for move in moves_indexes:
                distance = self.hex_grid.calculate_distance(move)
                probability = np.exp(-((distance - mu) ** 2) / (2 * self.s_cone_params['c'] ** 2))
                probabilities.append(probability)

            # Normalize probabilities to sum up to 1
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]

            sorted_moves = moves_indexes

        # Choose a move based on the probabilities
        chosen_move = np.random.choice(sorted_moves, p=probabilities)
        return chosen_move


# Example usage
grid = HexGrid(size=40)
s_cones_parameters = {
    "s_cones_init_count": 1,
    "m_cones_init_count": 900,

    "s_cones_final_count": 80,
    "m_cones_final_count": 920,

    "init_mode": "bfs",
    "max_probability": 0.5
}

s_cones = Scones(grid, s_cones_parameters)
# grid.draw(s_cones.cell_indexes, plot=True)

# s_cones.move_cell_bfs()
# grid.draw(s_cones.cell_indexes, plot=True)
#
# for i in range(20):
#     s_cones.move_cell_bfs()
#     grid.draw(s_cones.cell_indexes, plot=True)

i = 0
while s_cones.stopSignal is False:
    s_cones.move_cell_bfs()
    if i % 50 == 0:
        grid.draw(s_cones.cell_indexes, plot=False)

    i += 1

grid.draw(s_cones.cell_indexes, plot=False)

grid.create_gif(f"scones_v2_sConesInit={s_cones_parameters['s_cones_init_count']}_"
                f"mConesInit={s_cones_parameters['m_cones_init_count']}_"
                f"sConesFinal={s_cones_parameters['s_cones_final_count']}_"
                f"mConesFinal={s_cones_parameters['m_cones_final_count']}_"
                f"maxProb={s_cones_parameters['max_probability']}_"
                f"gridSize={grid.size}")



