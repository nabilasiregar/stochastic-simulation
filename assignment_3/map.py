import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.colors import to_rgba

def read_csv(csv):
    nodes = []
    with open(csv) as f:
        file = f.readlines()
    for line in file:
        if not line.strip()[0].isnumeric():
            continue
        line = line.split()
        nodes.append([float(line[1]), float(line[2])])
    return np.array(nodes)

def add_paths(csv):
    paths = []
    with open(csv) as f:
        file = f.readlines()
    for line in file:
        if not line[0].isnumeric():
            continue
        paths.append(abs(int(line)))
    return np.array(paths)

def plotmap(nodes, ax=None, path = None, title=None):
    node_color = to_rgba('#83BCFF')
    path_color = to_rgba('#EE204D')

    if ax is None:
        ax = plt.plot()
    for node in nodes:
        ax.plot(node[0], node[1], 'o', color=node_color)
    if any(path):
        for i in range(len(path)-1):
            ax.plot([nodes[path[i]-1][0], nodes[path[i+1]-1][0]], [nodes[path[i]-1][1], nodes[path[i+1]-1][1]], '-', color=path_color)
        ax.plot([nodes[path[-1]-1][0], nodes[path[0]-1][0]], [nodes[path[-1]-1][1], nodes[path[0]-1][1]], '-', color=path_color)
    if title:
        ax.set_title(title)
@njit
def distance_between_nodes(node1, node2):
    x_diff = node1[0] - node2[0]
    y_diff = node1[1] - node2[1]
    return np.sqrt(x_diff**2 + y_diff**2)

@njit
def calculate_path_length(paths, nodes):
    '''Calculates the path length of the current path'''
    length = 0
    for i in range(len(paths)-1):
        dist = distance_between_nodes(nodes[paths[i]-1], nodes[paths[i+1]-1])
        length += dist
    dist = distance_between_nodes(nodes[paths[-1]-1], nodes[paths[0]-1])
    length += dist
    return length
