import numpy as np

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
    '''Picks a random subroute, removes it from one part of the path and inserts it elsewhere'''
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