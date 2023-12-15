import numpy as np
from map import *
from numba import njit, int32

@njit
def get_neighbor(path, selection):
    '''Returns a random neighbor of the path'''
    node1 = np.random.randint(0, len(path))
    node2 = np.random.randint(0, len(path))
    while node1 == node2:
        node2 = np.random.randint(0, len(path))

    new_path = path.copy()

    if selection == 0:
        # Inverse
        new_path[min(node1, node2):max(node1, node2)] = new_path[min(node1, node2):max(node1, node2)][::-1]
        
    elif selection == 1:
        # Swap
        new_path[node1], new_path[node2] = new_path[node2], new_path[node1]
        
    elif selection == 2:
        # Swap routes
        start, end = min(node1, node2), max(node1, node2)
        subroute = path[start:end]
        new_path = np.concatenate((path[:start], path[end:]))
        insertion_point = np.random.randint(0, len(new_path))
        new_path = np.concatenate((new_path[:insertion_point], subroute, new_path[insertion_point:]))
        
    else:
        # Handle invalid selection
        raise ValueError("Invalid selection value")

    assert len(new_path) == len(path), "Operation caused an error"
    assert len(np.unique(new_path)) == len(new_path), f"Operation caused an error using slection {selection}\n {new_path}"
    return new_path

def get_temperature_list(nodes, list_length, p0, starting_path):
    solution = starting_path 
    temperature_list = [] 
    i = 0

    while i < list_length:
        selection = np.random.randint(0, 3)
        neighbor = get_neighbor(solution, selection)

        current_length = calculate_path_length(solution, nodes)
        new_length = calculate_path_length(neighbor, nodes)
        if new_length < current_length:
            solution = neighbor
        
        temp = (-1 * abs(new_length - current_length)) / (np.log(p0))
        temperature_list.append(temp)
        i += 1
        
        temperature_list = sorted(temperature_list, reverse=True)

    return temperature_list