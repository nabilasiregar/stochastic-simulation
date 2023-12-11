import numpy as np
from route_operations import get_neighbor

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
