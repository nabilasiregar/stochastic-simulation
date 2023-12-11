import numpy as np
import matplotlib.pyplot as plt

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


def get_temperature_list(map, list_length, p0, starting_path):
    solution = starting_path.copy()
    temperature_list = []
    i = 0

    while i < list_length:
        neighbor = get_neighbor(solution)
        current_length = map.calculate_path_length(solution)
        new_length = map.calculate_path_length(neighbor)
        if new_length < current_length:
            solution = neighbor
        
        temp = (-1 * abs(new_length - current_length)) / (np.log(p0))
        temperature_list.append(temp)
        i += 1
    
    temperature_list = sorted(temperature_list, reverse=True)

    return temperature_list


def sim_annealing_list(map, k, stopping_iter, starting_path):
    temperature_list = get_temperature_list(map, 120, 0.9, starting_path)
    k = 0
    solution = starting_path.copy()
    max_length = len(map.nodes) + 1
    length_list = []

    while k < stopping_iter:
        if not temperature_list:
            break
        max_temp = temperature_list.pop(0)
        k += 1
        t = 0
        c = 0
        iter = 0
        while iter < stopping_iter:
            neighbor = get_neighbor(solution)
            iter += 1
            assert len(
                neighbor) == max_length - 1, f'Added an edge in iteration {iter}, current path length: {len(neighbor)}, max length: {max_length}'
            new_length = map.calculate_path_length(neighbor)
            length_diff = new_length - map.calculate_path_length(solution)
            if length_diff <= 0:
                solution = neighbor

            else:
                e = np.exp(-(1/length_diff*max_temp))
                r = np.random.random()
                if r >= e:
                    t = (t - length_diff)/(np.log(r))
                    c += 1
                    solution = neighbor
        if c != 0:
            temperature_list.append(t/c)
            temperature_list = sorted(temperature_list, reverse=True)

        length_list.append(map.calculate_path_length(solution))

    return solution, map.calculate_path_length(solution), iter, length_list