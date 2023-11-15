import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def data_stats(filename, alpha):
    methods = ["Uniform Square", "Uniform Circle", "Latin Hypercube", "Orthogonal"]
    data = pd.read_csv(filename)
    
    uniform_square_data = data.groupby("method").get_group("uniform_square")
    uniform_circle_data = data.groupby("method").get_group("uniform_circle")
    latin_hypercube_data = data.groupby("method").get_group("latin_hypercube")
    orthogonal_data = data.groupby("method").get_group("orthogonal")
    
    mean_uniform_square = uniform_square_data["mean_area"].iloc[0]
    mean_uniform_circle = uniform_circle_data["mean_area"].iloc[0]
    mean_latin_hypercube = latin_hypercube_data["mean_area"].iloc[0]
    mean_orthogonal = orthogonal_data["mean_area"].iloc[0]
    
    means = [mean_uniform_square, mean_uniform_circle, mean_latin_hypercube, mean_orthogonal]
    
    std_uniform_square = np.std(uniform_square_data.loc[:,"area"])
    std_uniform_circle = np.std(uniform_circle_data.loc[:,"area"])
    std_latin_hypercube = np.std(latin_hypercube_data.loc[:,"area"])
    std_orthogonal = np.std(orthogonal_data.loc[:,"area"])
    
    stds = [std_uniform_square, std_uniform_circle, std_latin_hypercube, std_orthogonal]
    
    for i in range(len(stds)):
        standard_error = stds[i]/ 10
        df = 99
        conf_interval = stats.t.interval(1-alpha, df, means[i], scale=standard_error)
        print(f"{methods[i]} Confidence Interval: {conf_interval}")
        plt.errorbar(x=i, y=means[i], yerr=(conf_interval[1] - means[i]), fmt='o', label=methods[i])

    plt.xticks(range(len(means)), [methods[i] for i in range(len(means))])
    plt.ylabel('Area')
    plt.title('Confidence Intervals for Mandelbrot Set Area')
    plt.show()

    var_uniform_square = np.var(uniform_square_data.loc[:,"area"])
    var_uniform_circle = np.var(uniform_circle_data.loc[:,"area"])
    var_latin_hypercube = np.var(latin_hypercube_data.loc[:,"area"])
    var_orthogonal = np.var(orthogonal_data.loc[:,"area"])
    
    print()
    print("Variance Uniform Sampling over Square: " + str(var_uniform_square))
    print("Variance Uniform Sampling over Circle: " + str(var_uniform_circle))
    print("Variance Latin Hypercube over Square: " + str(var_latin_hypercube))
    print("Variance Orthogonal Sampling over Square: " + str(var_orthogonal))


data_stats("mandelbrot_estimations.csv", 0.01)
