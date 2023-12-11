from map import Map
from simulated_annealing import *
from config import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    map = Map(SMALL_MAP)

    # Define parameters for simulated annealing
    initial_temperature = 1000.0
    cooling_factor = 0.995
    stopping_temperature = 0.1
    stopping_iterations = 1000

    initial_path = list(map.nodes.keys())
    np.random.shuffle(initial_path)

    #Run Simulated Annealing
    best_path, best_length, iterations, temp_list, length_list = sim_annealing(
        map, initial_temperature, cooling_factor, stopping_temperature, stopping_iterations, initial_path)
    # Display results
    print(f"Best path: {best_path}")
    print(f"Length of the best path: {best_length}")
    print(f"Total iterations: {iterations}")

    # Optionally, plot the final path
    map.add_paths(best_path)
    map.plot()
    plt.show()

if __name__ == "__main__":
    main()
