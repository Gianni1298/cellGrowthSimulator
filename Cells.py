import random
from collections import deque

from hexalattice.hexalattice import *


class Cells:
    def __init__(self, hex_grid, params):
        self.hex_grid = hex_grid
        self.params = params
        self.final_cell_count = int(params["grid_size"] * params["grid_size"] * 0.3)
        self.s_cones_final_count = int(self.final_cell_count * params["sCones_to_mCones_ratio"])
        self.m_cones_final_count = int(self.final_cell_count * (1 - params["sCones_to_mCones_ratio"]))
        self.cell_indexes = self.init(params["init_mode"])  # Map of cell indexes to cell color {index: color}
        self.move_mode = params["move_mode"]
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
        total_cells_to_place = self.params["s_cones_init_count"] + self.params["m_cones_init_count"]

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
        blue_cells_to_place = self.params["s_cones_init_count"]
        green_cells_to_place = self.params["m_cones_init_count"]
        for index in indexes_to_populate:
            if blue_cells_to_place > 0:
                cell_indexes[index] = "b"
                blue_cells_to_place -= 1
            elif green_cells_to_place > 0:
                cell_indexes[index] = "aquamarine"
                green_cells_to_place -= 1

        return cell_indexes

    def move_cell_bfs(self, savePlot=False):
        cell_to_move_index = random.choice(list(self.cell_indexes.keys()))

        if self.move_mode == "line":
            movement_direction = random.choice(self.hex_grid.directions)
            path = self.find_path_following_direction(cell_to_move_index, movement_direction)
        else:
            path = self.find_shortest_path_to_empty(cell_to_move_index)

        if path:
            self.move_along_path(path)
            self.cell_indexes[cell_to_move_index] = self.cell_birth()

        if len(self.cell_indexes) > self.final_cell_count:
            self.stopSignal = True
            return

        if savePlot:
            if len(self.cell_indexes) % 10 == 0 or len(self.cell_indexes) >= self.final_cell_count:
                self.hex_grid.draw(self.cell_indexes)


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

    def find_path_following_direction(self, start_index, direction):
        path = [start_index]

        # Find the path to an empty cell in the given direction
        x, y = self.hex_grid.hex_centers[start_index]
        dx, dy = direction

        next_hex =  self.hex_grid.find_hexagon_index(x + dx, y + dy)
        while next_hex in self.cell_indexes:
            path = path + [next_hex]
            x, y = x + dx, y + dy
            next_hex = self.hex_grid.find_hexagon_index(x + dx, y + dy)

        return path + [next_hex]

    def move_along_path(self, path):
        if len(path) < 2:
            return  # No movement needed

        moving_color = self.cell_indexes[path[0]]
        for i in range(len(path) - 1, 0, -1):
            self.cell_indexes[path[i]] = self.cell_indexes[path[i - 1]]

        self.cell_indexes[path[0]] = moving_color
        del self.cell_indexes[path[0]]


    def cell_birth(self):
        # A blue cell or green cell is born with a probability given by the parameters in the config and the current cell count

        # First get current blue and green cell counts
        blue_cell_count = sum([1 for color in self.cell_indexes.values() if color == "b"])
        green_cell_count = sum([1 for color in self.cell_indexes.values() if color == "aquamarine"])

        if blue_cell_count == self.s_cones_final_count:
            return "aquamarine"
        elif green_cell_count == self.m_cones_final_count:
            return "b"
        else:
            prob_green = self.quadratic_probability(current_count=green_cell_count,
                                                    final_count=self.m_cones_final_count, max_probability=1)

            prob_blue = self.quadratic_probability(current_count=blue_cell_count,
                                                   final_count=self.s_cones_final_count, max_probability=self.params["max_probability"])

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
        mu = self.params['m'] * cell_ratio

        # Calculate the 80% percentile of the Gaussian distribution
        p80 = mu + 0.842 * self.params['c']

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
                probability = np.exp(-((distance - mu) ** 2) / (2 * self.params['c'] ** 2))
                probabilities.append(probability)

            # Normalize probabilities to sum up to 1
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]

            sorted_moves = moves_indexes

        # Choose a move based on the probabilities
        chosen_move = np.random.choice(sorted_moves, p=probabilities)
        return chosen_move

    def find_closed_loop(self):
        # Step 1: Find the starting point
        starting_point = None
        for cell_index in self.cell_indexes:
            _, neighbour_indexes = self.hex_grid.find_neighbours(cell_index)
            for neighbour in neighbour_indexes:
                if neighbour not in self.cell_indexes:
                    starting_point = neighbour
                    break
            if starting_point is not None:
                break

        if starting_point is None:
            print("The perimeter of the population couldn't be found")
            return []  # No loop found

        # Step 2: Initialize loop tracking
        loop = [starting_point]
        current_cell = starting_point

        # Step 3: Iterate over neighbors to find the loop
        while True:
            _, neighbour_indexes = self.hex_grid.find_neighbours(current_cell)
            potential_next_cells = []
            for neighbour in neighbour_indexes:
                if neighbour not in self.cell_indexes and neighbour not in loop:
                    adjacent_to_population = any(n in self.cell_indexes for n in self.hex_grid.find_neighbours(neighbour)[1])
                    if adjacent_to_population:
                    # Count the number of neighbours
                        neighbour_count = len([n for n in self.hex_grid.find_neighbours(neighbour)[1] if n not in self.cell_indexes])
                        potential_next_cells.append((neighbour, neighbour_count))

            if not potential_next_cells:
                print("A perimeter has been found")
                break  # Loop is closed or no next cell found

            # Choose the next cell as the one with the most neighbours
            next_cell = max(potential_next_cells, key=lambda x: x[1])[0]
            loop.append(next_cell)
            current_cell = next_cell

        # Step 4: Find the x, y points of the loop
        points = [self.hex_grid.hex_centers[index] for index in loop]
        return points
