import random
from collections import deque

from hexalattice.hexalattice import *


class Cells:
    def __init__(self, hex_grid, s_cone_params):
        self.hex_grid = hex_grid
        self.s_cone_params = s_cone_params
        self.final_cell_count = s_cone_params["s_cones_final_count"] + s_cone_params["m_cones_final_count"]
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

    def move_cell_bfs(self, savePlot=False):
        cell_to_move_index = random.choice(list(self.cell_indexes.keys()))
        # cell_to_move_index = self.hex_grid.find_closest_hexagon(0, 0)

        path = self.find_shortest_path_to_empty(cell_to_move_index)
        if path:
            self.move_along_path(path)

        self.cell_indexes[cell_to_move_index] = self.cell_birth()

        if len(self.cell_indexes) > self.s_cone_params["s_cones_final_count"] + self.s_cone_params["m_cones_final_count"]:
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

