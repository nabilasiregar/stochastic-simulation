import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def read_csv(csv):
    """
    Reads a CSV file and extracts node coordinates.

    Parameters:
    csv (str): The path to the CSV file.

    Returns:
    numpy.ndarray: An array of node coordinates.
    """
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
    """
    Reads a CSV file and extracts paths.

    Parameters:
    csv (str): The path to the CSV file.

    Returns:
    numpy.ndarray: An array of paths.
    """
    paths = []
    with open(csv) as f:
        file = f.readlines()
    for line in file:
        if not line[0].isnumeric():
            continue
        paths.append(abs(int(line)))
    return np.array(paths)

def plotmap(nodes, ax=None, path = None):
    """
    Plots the nodes and paths on a map.

    This function takes an array of nodes and an optional path, then plots the nodes
    and the path on a matplotlib axis. If no axis is provided, it creates a new plot.

    Parameters:
    nodes: Array of node coordinates.
    ax: A matplotlib axis. Defaults to None.
    path (list, optional): A list of node indices forming a path. Defaults to None.
    """
    if ax is None:
        ax = plt.plot()
    for node in nodes:
        ax.plot(node[0], node[1], 'ro')
    if any(path):
        for i in range(len(path)-1):
            ax.plot([nodes[path[i]-1][0], nodes[path[i+1]-1][0]], [nodes[path[i]-1][1], nodes[path[i+1]-1][1]], 'b-')
        ax.plot([nodes[path[-1]-1][0], nodes[path[0]-1][0]], [nodes[path[-1]-1][1], nodes[path[0]-1][1]], 'b-')

@njit
def distance_between_nodes(node1, node2):
    """
    Calculates the Euclidean distance between two nodes.

    Parameters:
    node1 (list): The [x, y] coordinates of the first node.
    node2 (list): The [x, y] coordinates of the second node.

    Returns:
    float: The Euclidean distance between the two nodes.
    """
    x_diff = node1[0] - node2[0]
    y_diff = node1[1] - node2[1]
    return np.sqrt(x_diff**2 + y_diff**2)

@njit
def calculate_path_length(paths, nodes):
    """
    Calculates the total path length for a given sequence of nodes.

    This function iterates over a sequence of nodes, calculating the total
    distance traveled if one were to visit the nodes in the order specified.

    Parameters:
    paths (list): A list of node indices representing the path.
    nodes (numpy.ndarray): An array of node coordinates.

    Returns:
    float: The total path length.
    """
    length = 0
    for i in range(len(paths)-1):
        dist = distance_between_nodes(nodes[paths[i]-1], nodes[paths[i+1]-1])
        length += dist
    dist = distance_between_nodes(nodes[paths[-1]-1], nodes[paths[0]-1])
    length += dist
    return length
