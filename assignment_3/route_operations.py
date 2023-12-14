import numpy as np
from numba import njit, int32

@njit
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
    new_path[node1+1:] = new_path[node1:-1]
    new_path[node1] = path[node2]
    assert len(new_path) == len(path), "Inserting caused an error"
    return new_path

@njit
def swap(path):
    '''Swaps two random nodes in the path'''
    node1 = np.random.randint(0, len(path))
    node2 = np.random.randint(0, len(path))
    new_path = path.copy()
    new_path[node1], new_path[node2] = new_path[node2], new_path[node1]
    assert len(new_path) == len(path), "swapping caused an error"
    return new_path

@njit
def swap_routes(path):
    '''Picks a random subroute, removes it from one part of the path and inserts it elsewhere'''
    node1 = np.random.randint(0, len(path))
    node2 = np.random.randint(0, len(path))
    start, end = min(node1, node2), max(node1, node2)
    subroute = path[start:end]
    new_path = np.concatenate((path[:start], path[end:]))
    insertion_point = np.random.randint(0, len(new_path))
    new_path = np.concatenate((new_path[:insertion_point], subroute, new_path[insertion_point:]))
    assert len(new_path) == len(path), "Swapping routes caused an error"
    return new_path

@njit
def get_neighbor(path):
    '''Returns a random neighbor of the path'''
    # operators = [inverse, insert, swap, swap_routes]  
    # selection = np.random.randint(0, len(operators))
    # new_path = operators[selection](path)
    return inverse(path)

@njit
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
