import numpy as np
from numba import njit
from route_operations import get_neighbor, get_temperature_list

@njit
def sim_annealing(nodes, T, alpha, stopping_T, chain_length , starting_path):
    '''
    This function will take in a map and return the best path found using simulated annealing
    params:
    map: the map object
    T: the starting temperature
    alpha: the cooling factor
    stopping_T: the temperature to stop at
    stopping_iter: the number of iterations to stop at
    starting_path: the starting path, this can be changed to test convergence for different initial paths


    length: the length of the best path
    iterations: the number of iterations it took to find the best path
    '''
    solution = starting_path
    iter = 0
    t_list = []
    length_list = []

    
    best_path = np.copy(starting_path)
    best_length = calculate_path_length(starting_path, nodes)
    best_length = np.inf
    while T > stopping_T:
        for i in range(chain_length):
            neighbor = get_neighbor(solution)
            

            new_length = calculate_path_length(neighbor, nodes)
            length_diff = new_length - calculate_path_length(solution, nodes)

            if length_diff < 0:
                solution = neighbor
            else:
                e = np.exp(-length_diff / T)
                if np.random.random() <= e:
                    solution = neighbor

            if new_length < best_length:
                best_path[:] = neighbor
                best_length = new_length

            iter += 1
            t_list.append(T)
            length_list.append(new_length)

        T *= alpha
    return best_path, best_length, iter, t_list, length_list

@njit
def fast_annealing(nodes, T, alpha, stopping_T, chain_length, starting_path):
    '''
    This function will take in a map and return the best path found using simulated annealing
    params:
    map: the map object
    T: the starting temperature
    alpha: the cooling factor
    stopping_T: the temperature to stop at
    stopping_iter: the number of iterations to stop at
    starting_path: the starting path, this can be changed to test convergence for different initial paths


    length: the length of the best path
    iterations: the number of iterations it took to find the best path
    '''
    solution = starting_path
    iter = 0
    t_list = []
    length_list = []

    
    best_path = np.copy(starting_path)
    best_length = calculate_path_length(starting_path, nodes)
    best_length = np.inf
    while T > stopping_T:
        for i in range(chain_length):
            neighbor = get_neighbor(solution)
            

            new_length = calculate_path_length(neighbor, nodes)
            length_diff = new_length - calculate_path_length(solution, nodes)

            if length_diff < 0:
                solution = neighbor
            else:
                e = np.exp(-length_diff / T)
                if np.random.random() <= e:
                    solution = neighbor

            if new_length < best_length:
                best_path[:] = neighbor
                best_length = new_length

            iter += 1
            t_list.append(T)
            length_list.append(new_length)

        T *= alpha
    return best_path, best_length, iter, t_list, length_list