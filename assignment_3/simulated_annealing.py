from map import *
from route_operations import *
import numpy as np
from numba import njit

@njit
def sim_annealing(nodes, T, alpha, stopping_T, chain_length , starting_path):
    '''
    This function will take in a map and return the best path found using simulated annealing
    params:
    map: the map object
    T: the starting temperature
    alpha: the cooling factor
    stopping_T: the temperature to stop at
    chain_length: the length of Markov Chain to complete the tour
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
            selection = np.random.randint(0, 3)
            neighbor = get_neighbor(solution, selection)
            

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
            if iter % 10000 == 0:
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
        for _ in range(chain_length):
            selection = np.random.randint(0, 3)
            neighbor = get_neighbor(solution, selection)
            

            new_length = calculate_path_length(neighbor, nodes)
            length_diff = new_length - calculate_path_length(solution, nodes)

            if length_diff < 0:
                solution = neighbor
            else:
                e = length_diff / T
                rho = 1/ (1+e)
                if np.random.random() <= rho:
                    solution = neighbor

            if new_length < best_length:
                best_path[:] = neighbor
                best_length = new_length

            iter += 1
            
            if iter % 10000 == 0:
                t_list.append(T)
                length_list.append(new_length)

        T *= alpha
    return best_path, best_length, iter, t_list, length_list

@njit
def sim_annealing_list(nodes, temp_list_length, chain_length, starting_path, temperature_list):
    '''
    Optimizes a path using simulated annealing with a list of temperature values.

    This function applies simulated annealing to optimize a path based on a list of
    temperature values. It iteratively selects temperatures from the list and explores
    neighboring paths to find an optimal or near-optimal solution.

    Parameters:
    nodes: An array of node coordinates.
    temp_list_length (int): The number of temperature values to use from the list.
    chain_length (int): The number of iterations to perform at each temperature.
    starting_path (numpy.ndarray): The initial path for the optimization process.
    temperature_list (list): A list of temperature values to use during the optimization.

    Returns:
    Tuple[numpy.ndarray, float, int, List[float], List[float]]:
        - The optimized path as a numpy array of node indices.
        - The length of the optimized path.
        - The total number of iterations performed.
        - A list of temperature values used during the optimization.
        - A list of corresponding path lengths at each temperature.

    '''
    k = 0
    solution = starting_path
    max_length = len(nodes) + 1
    length_list = []
    t_list = []
    
    best_path = np.copy(starting_path)
    best_length = calculate_path_length(starting_path, nodes)
    best_length = np.inf
    
    TEMP_STOP = False
    while k < temp_list_length and not TEMP_STOP:
        max_temp = temperature_list[0]
        if len(temperature_list) == 1:
            TEMP_STOP = True
        else:
            temperature_list = temperature_list[1:]
        k += 1
        t = 0
        iter = 0
        while iter < chain_length:
            iter += 1
            selection = np.random.randint(0, 3)
            neighbor = get_neighbor(solution, selection)

            new_length = calculate_path_length(neighbor, nodes)
            length_diff = new_length - calculate_path_length(solution, nodes)
            if length_diff <= 0:
                solution = neighbor

            else:
                e = np.exp(-(1/length_diff)*max_temp)
                r = np.random.random()
                if r >= e:
                    t = (t - length_diff)/(np.log(r))
                    solution = neighbor

            if new_length < best_length:
                best_path[:] = neighbor
                best_length = new_length
                
        t_list.append(max_temp)
        length_list.append(new_length)

    return solution, calculate_path_length(solution, nodes), iter, t_list, length_list
