import numpy as np
from route_operations import get_neighbor, get_temperature_list

def sim_annealing(map, T, alpha, stopping_T, chain_length , starting_path):
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
    iter = 0
    t_list = []
    length_list = []

    best_path = []
    best_length = np.inf
    while T > stopping_T:


        for i in range(chain_length):
            neighbor = get_neighbor(solution)

            new_length = map.calculate_path_length(neighbor)
            length_diff = new_length - map.calculate_path_length(solution)

            if length_diff <= 0:
                solution = neighbor

            else:
                # e = (1 + np.exp(length_diff/T))**-1
                e = np.exp(-length_diff/T)
                
                if np.random.random() <= e:
                    solution = neighbor
                # if iter % 1000 == 0:
                #     print(
                #         f'Iteration {iter}, current path length: {map.calculate_path_length(solution)}, temperature: {T}, e = {e}')
            if map.calculate_path_length(solution) < best_length:
                best_path = solution.copy()
                best_length = map.calculate_path_length(solution)

            iter += 1
            #adaptive cooling
            t_list.append(T)
            length_list.append(map.calculate_path_length(solution))
        T *= alpha
    return best_path, best_length, iter, t_list, length_list

def fast_annealing(map, T, alpha, stopping_T, chain_length, starting_path):
    '''
    This function will take in a map and return the best path found using a faster version of  annealing,
    this is done by using a different acceptance function which allows for faster cooling
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
    iter = 0
    t_list = []
    length_list = []

    best_path = []
    best_length = np.inf
    while T > stopping_T:


        for i in range(chain_length):
            neighbor = get_neighbor(solution)

            new_length = map.calculate_path_length(neighbor)
            length_diff = new_length - map.calculate_path_length(solution)

            if length_diff <= 0:
                solution = neighbor

            else:
                rho = length_diff/T
                e = 1/(1+rho)
                
                if np.random.random() <= e:
                    solution = neighbor
                # if iter % 1000 == 0:
                #     print(
                #         f'Iteration {iter}, current path length: {map.calculate_path_length(solution)}, temperature: {T}, e = {e}')
            if map.calculate_path_length(solution) < best_length:
                best_path = solution.copy()
                best_length = map.calculate_path_length(solution)

            iter += 1
            #adaptive cooling
            t_list.append(T)
            length_list.append(map.calculate_path_length(solution))
        T *= alpha
    return best_path, best_length, iter, t_list, length_list


@njit
def sim_annealing_list(nodes, k, stopping_iter, starting_path, temperature_list):
    k = 0
    solution = starting_path
    max_length = len(nodes) + 1
    length_list = []

    best_path = np.copy(starting_path)
    best_length = calculate_path_length(starting_path, nodes)
    best_length = np.inf

    TEMP_STOP = False
    while k < stopping_iter or not TEMP_STOP:

        max_temp = temperature_list[0]

        if len(temperature_list) == 1:
            TEMP_STOP = True
        else:
            temperature_list = temperature_list[1:]
        k += 1
        t = 0
        c = 0
        iter = 0
        while iter < stopping_iter:
            neighbor = get_neighbor(np.copy(solution))
            iter += 1
            assert len(
                neighbor) == max_length - 1, f'Added an edge in iteration {iter}, current path length: {len(neighbor)}, max length: {max_length}'
            new_length = calculate_path_length(neighbor, nodes)
            length_diff = new_length - calculate_path_length(solution, nodes)
            if length_diff <= 0:
                solution = neighbor

            else:
                e = np.exp(-(1/length_diff*max_temp))
                r = np.random.random()
                if r >= e:
                    t = (t - length_diff)/(np.log(r))
                    c += 1
                    solution = neighbor

            if new_length < best_length:
                best_path[:] = neighbor
                best_length = new_length
        if c != 0:
            temperature_list.append(t/c)
            temperature_list = sorted(temperature_list, reverse=True)

        length_list.append(calculate_path_length(solution, nodes))

    return solution, calculate_path_length(solution, nodes), iter, length_list