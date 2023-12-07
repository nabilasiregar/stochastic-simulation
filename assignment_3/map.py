import numpy as np
import matplotlib.pyplot as plt

SMALL_MAP = "TSP-Configurations/eil51.tsp.txt"
MEDIUM_MAP = "TSP-Configurations/a280.tsp.txt"
LARGE_MAP = "TSP-Configurations/pcb442.tsp.txt"

SMALL_OPT = "TSP-Configurations/eil51.opt.tour.txt"
MEDIUM_OPT = "TSP-Configurations/a280.opt.tour.txt"
LARGE_OPT = "TSP-Configurations/pcb442.opt.tour.txt"


class Map:
    def __init__(self, csv):
        self.nodes = {}
        self.paths = []
        self.csv = csv
        self.read_csv()

    def read_csv(self):
        with open(self.csv) as f:
            file = f.readlines()
        for line in file:
            if not line.strip()[0].isnumeric():
                continue
            line = line.split()
            self.nodes[int(line[0])] = [float(line[1]), float(line[2])]

    def plot(self):
        plt.figure(figsize=(10, 10))
        for node in self.nodes:
            plt.plot(self.nodes[node][0], self.nodes[node][1], 'ro')
            plt.text(self.nodes[node][0], self.nodes[node][1], str(node))
        if any(self.paths):
            path = [self.nodes[i] for i in self.paths]
            plt.plot([i[0] for i in path], [i[1] for i in path], 'b-')

    def add_paths(self, paths):
        if type(paths) is not str:
            self.paths = paths.copy()
            self.paths.append(paths[0])
        else:
            self.paths = []
            with open(paths) as f:
                file = f.readlines()
            for line in file:
                if not line[0].isnumeric():
                    continue
                self.paths.append(abs(int(line)))
            self.paths.append(self.paths[0])

    def distance_between_nodes(self, node1, node2):
        x_diff = node1[0] - node2[0]
        y_diff = node1[1] - node2[1]
        return np.sqrt(x_diff**2 + y_diff**2)

    def calculate_path_length(self, paths=None):
        '''Calculates the path length of the current path'''
        if paths is not None:
            self.add_paths(paths)
        length = 0
        for i in range(len(self.paths)-1):
            length += self.distance_between_nodes(
                self.nodes[self.paths[i]], self.nodes[self.paths[i+1]])
        return length


# Setting the operators for the map
def inverse(path):
    '''Creates an inverse of the path between two nodes
    '''
    node1 = np.random.randint(0, len(path))
    node2 = np.random.randint(0, len(path))
    new_path = path.copy()
    new_path[min(node1, node2):max(node1, node2)] = new_path[min(
        node1, node2):max(node1, node2)][::-1]
    assert len(new_path) == len(path), "Inverse caused an error"
    return new_path


def insert(path):
    '''Picks a random node and inserts it into a random position in the path'''
    node1 = np.random.randint(0, len(path))
    node2 = np.random.randint(0, len(path))
    new_path = path.copy()
    new_path.insert(node1, new_path.pop(node2))
    assert len(new_path) == len(path), "Inserting caused an error"
    return new_path


def swap(path):
    '''Swaps two random nodes in the path'''
    node1 = np.random.randint(0, len(path))
    node2 = np.random.randint(0, len(path))
    new_path = path.copy()
    new_path[node1], new_path[node2] = new_path[node2], new_path[node1]
    assert len(new_path) == len(path), "swapping caused an error"
    return new_path


def swap_routes(path):
    ''' Picks a random subroute, removes it from one part of the 
        path and inserts it elsewhere'''
    node1 = np.random.randint(0, len(path))
    node2 = np.random.randint(0, len(path))
    subroute = path[min(node1, node2):max(node1, node2)]
    new_path = path.copy()
    del new_path[min(node1, node2):max(node1, node2)]
    insertion_point = np.random.randint(0, len(new_path))
    for position in subroute[::-1]:
        new_path.insert(insertion_point, position)
    assert len(new_path) == len(path), "swapping routes caused an error"
    return new_path


def get_neighbor(path):
    '''Returns a random neighbour of the path'''
    operators = [inverse, insert, swap, swap_routes]
    selection = np.random.randint(0, 4)
    return operators[selection](path)


def sim_annealing(map, T, alpha, stopping_T, stopping_iter, starting_path):
    '''
    This function will take in a map and return the best path found using simulated annealing
    params:
    map: the map object
    T: the starting temperature
    alpha: the cooling factor
    stopping_T: the temperature to stop at
    stopping_iter: the number of iterations to stop at
    starting_path: the starting path, this can be changed to test convergence for different initial paths


    returns:
    best: the best path found
    length: the length of the best path
    iterations: the number of iterations it took to find the best path
    '''
    solution = starting_path.copy()
    max_length = len(map.nodes)
    iter = 0
    t_list = []
    length_list = []

    while iter < stopping_iter and T > stopping_T:
        neighbor = get_neighbor(solution)

        assert len(
            neighbor) == max_length, f'Added an edge in iteration {iter}, current path length: {len(neighbor)}, max length: {max_length}'
        new_length = map.calculate_path_length(neighbor)
        length_diff = new_length - map.calculate_path_length(solution)
        if length_diff <= 0:
            solution = neighbor

        else:
            e = np.exp(-(1/length_diff*T))
            if np.random.random() >= e:
                solution = neighbor

        if iter % 1000 == 0:
            print(
                f'Iteration {iter}, current path length: {map.calculate_path_length(solution)}, temperature: {T}')

        iter += 1
        T *= alpha
        t_list.append(T)
        length_list.append(map.calculate_path_length(solution))
    return solution, map.calculate_path_length(solution), iter, t_list, length_list
