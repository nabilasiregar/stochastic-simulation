from map import *
from simulated_annealing import *
from map_config import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    nodes = read_csv(SMALL_MAP)
    paths = add_paths(SMALL_OPT)
    known_best_length = calculate_path_length(paths, nodes)

    # Define parameters for simulated annealing
    initial_temperature = 1000.0
    cooling_factor = 0.995
    stopping_temperature = 0.1
    chain_length = 1000
    starting_path = np.random.permutation(range(1,len(nodes)+1))

    #Run Simulated Annealing
    best_path, best_length, iterations, t_list, length_list = sim_annealing(
        nodes, initial_temperature, cooling_factor, stopping_temperature, chain_length, starting_path)
    
    # Display results
    print(f'Best path length found: {best_length}, provided path length: {known_best_length}, found in {iterations} iterations')

    # Optionally, plot the final path
    fig, axs = plt.subplots(1,3)
    fig.set_size_inches(18,6)
    plotmap(nodes, axs[0], paths)
    plotmap(nodes, axs[1], best_path)
    axs[0].set_title(f"Optimal Path, length:{known_best_length:.2f} ")
    axs[1].set_title(f"Generated path, length:{best_length:.2f}")
    axs[2].loglog(t_list, length_list, color='pink')
    axs[2].plot(t_list, length_list, color='pink')
    axs[2].invert_xaxis()
    axs[2].set_xlabel("Temperature")
    axs[2].set_ylabel("length")
    axs[2].set_title("Loss")
    plt.show()

if __name__ == "__main__":
    main()
