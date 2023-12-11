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
