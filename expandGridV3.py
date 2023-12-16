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

    def draw(self, blue_indices, green_indices=None):
        if green_indices is None:
            green_indices = set()
        hex_centers = self.hex_centers
        self.blue_indices = blue_indices

        # Color the selected hexagons blue and the rest white
        colors = ['b' if i in blue_indices else ('aquamarine' if i in green_indices else 'w') for i in
                  range(len(hex_centers))]

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
    def __init__(self, hex_grid, s_cone_params):
        self.m_cones = None
        self.s_cone_count = None
        # Gaussian parameters
        self.a = None
        self.c = None
        self.dfs = False
        self.time_step = 0

        self.hex_grid = hex_grid
        self.init_params(s_cone_params)
        self.blue_indices = self.init_blue_indices(s_cone_params["init_mode"])


    def init_params(self, s_cone_params):
        self.s_cone_count = s_cone_params["s_cones_final_count"]
        self.m_cones = Mcones(self.hex_grid, s_cone_params["m_cones_birth_rate"])
        self.a = s_cone_params["a"]
        self.m = s_cone_params["m"]
        self.c = s_cone_params["c"]
        self.dfs = s_cone_params["dfs"]

    def init_blue_indices(self, init_mode):
        if self.s_cone_count == 0:
            return Exception("s_cone_count must be greater than 0")

        if init_mode == "bfs":
            return self.bfs_init()
        elif init_mode == "random":
            return self.random_init()


    def bfs_init(self):
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

    def random_init(self):
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

    def fill_circle_with_green(self, radius, blue_indices):
        center_x, center_y = self.hex_grid.x_center, self.hex_grid.y_center

        for index, (x, y) in enumerate(self.hex_grid.hex_centers):
            if index not in blue_indices:
                distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if distance_from_center <= radius:
                    self.m_cones.add_green_index(index)
        return

    def move_sCones(self):
        new_blue_indices = set()
        self.time_step += 1

        def dfs_move(cell_to_move, visited = None):
            if visited is None:
                visited = set()

            neighbours, neighbour_indexes = self.hex_grid.find_neighbours(cell_to_move)
            # Sort neighbours by increasing distance from the center
            sorted_neighbours = sorted(neighbour_indexes, key=lambda ix: self.hex_grid.calculate_distance(ix), reverse=True)

            for index in sorted_neighbours:
                if index in visited:
                    continue  # Skip already visited cells to prevent cycles

                visited.add(index)
                if index not in self.blue_indices and index not in new_blue_indices:
                    return index  # Found a cell that can be moved
                elif index in self.blue_indices:
                    result = dfs_move(index, visited)
                    if result is not None:
                        return result  # Propagate the move up the call stack
            return None

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
                if self.dfs is False:
                    new_blue_indices.add(cell_to_move)

                elif self.dfs is True:
                    # Use DFS to find a new position
                    new_position = dfs_move(cell_to_move)
                    if new_position is not None:
                        new_blue_indices.add(new_position)


        self.blue_indices = new_blue_indices
        return self.blue_indices

    def choose_move(self, cell_to_move, allowed_moves):
        # Calculate the normalized count of green cells
        green_ratio = len(self.m_cones.get_green_indices()) / len(self.hex_grid.hex_centers)

        # Update the Gaussian wave's mean based on the green ratio and velocity m
        mu = self.m * green_ratio

        # Calculate the 80% percentile of the Gaussian distribution
        p80 = mu + 0.842 * self.c

        # Check the distance of the cell from the center
        current_distance = self.hex_grid.calculate_distance(cell_to_move)

        if abs(current_distance - mu) <= p80 - mu:
            # Rank the moves based on how much they increase the distance from the center
            move_distances = [(move, self.hex_grid.calculate_distance(move)) for move in allowed_moves]
            move_distances.sort(key=lambda x: x[1] - current_distance, reverse=True)

            # Assign probabilities proportional to the rank
            total_ranks = sum(range(1, len(allowed_moves) + 1))
            probabilities = [(len(allowed_moves) - rank) / total_ranks for rank, _ in enumerate(move_distances)]

            # Extract moves from sorted list
            sorted_moves = [move for move, _ in move_distances]
        else:
            # If the cell is outside the 80% percentile, use Gaussian probabilities
            probabilities = []
            for move in allowed_moves:
                distance = self.hex_grid.calculate_distance(move)
                probability = self.a * np.exp(-((distance - mu) ** 2) / (2 * self.c ** 2))
                probabilities.append(probability)

            # Normalize probabilities to sum up to 1
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]

            sorted_moves = allowed_moves

        # Choose a move based on the probabilities
        chosen_move = np.random.choice(sorted_moves, p=probabilities)
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
s_cones_parameters = {
    "s_cones_final_count": 160,
    "m_cones_final_count": 1840,
    "m_cones_birth_rate": 0.6,

    # Gaussian parameters
    "c": 30,  # Width of the bell, adjust as needed
    "m": 45,  # Coefficient of speed of the offset of the bell, adjust as needed

    "init_mode": "random",
    "dfs": True
}

s_cones = Scones(grid, s_cones_parameters)

grid.draw(s_cones.blue_indices)

while len(s_cones.m_cones.get_green_indices()) < 1840:
    grid.draw(s_cones.move_sCones(), s_cones.m_cones.get_green_indices())

grid.create_gif(f"scones_v14_gauss_"
                f"{s_cones.m_cones.birth_rate}"
                f"br_{grid.size}_"
                f"m={s_cones_parameters['m']}_"
                f"a={s_cones_parameters['a']}_"
                f"c={s_cones_parameters['c']}_"
                f"init={s_cones_parameters['init_mode']}")
