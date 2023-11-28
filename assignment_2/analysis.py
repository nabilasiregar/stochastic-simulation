import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

palette = {
    "green": (139, 191, 159),
    "blue": (131, 188, 255),
    "midnight": (18, 69, 89),
    "violet": (89, 52, 79),
    "crayola": (238, 32, 77)
}

normalized_palette = {key: tuple(val / 255.0 for val in value)
                      for key, value in palette.items()}

colors = list(normalized_palette.values())
color_stops = [i / (len(colors) - 1) for i in range(len(colors))]
cmap = LinearSegmentedColormap.from_list(
    "custom_gradient", list(zip(color_stops, colors)))


def statistics(filename):
    #Collecting the data from the csv
    f = pd.read_csv(filename, header=0)
    df = pd.DataFrame(f)
    
    mm1_data = df.groupby("n_server").get_group(1)
    mm2_data = df.groupby("n_server").get_group(2)
    mm4_data = df.groupby("n_server").get_group(4)
    
    mm1_wait_mean = np.mean(mm1_data["system_time"].values)
    mm2_wait_mean = np.mean(mm2_data["system_time"].values)
    mm4_wait_mean = np.mean(mm4_data["system_time"].values)
    
    mm1_wait_std = np.std(mm1_data["system_time"].values)
    mm2_wait_std = np.std(mm2_data["system_time"].values)
    mm4_wait_std = np.std(mm4_data["system_time"].values)
    
    #Creating the general stats
    alpha = 0.01
    general_stats = df.groupby(["n_server"]).aggregate(["mean", "std", "min", "max"])["system_time"]
    print(general_stats.to_latex(float_format="%.3f"))
    
    #ANOVA
    statistic, p_value = stats.f_oneway(mm1_data["system_time"].values, mm2_data["system_time"].values, mm4_data["system_time"].values)
    print(f"P_value for ANOVA: {p_value}")
    print()
    
    #Creating the confidence intervals
    methods = ["M/M/1", "M/M/2", "M/M/4"]
    means = [mm1_wait_mean, mm2_wait_mean, mm4_wait_mean]
    stds = [mm1_wait_std, mm2_wait_std, mm4_wait_std]
    
    plt.figure(figsize=(10, 6))
    
    for i in range(len(stds)):
            standard_error = stds[i]/ 10
            df = 99
            conf_interval = stats.t.interval(1-alpha, df, means[i], scale=standard_error)
            rounded_conf_interval = tuple(round(value, 3) for value in conf_interval)
            print(f"{methods[i]} Confidence Interval: {rounded_conf_interval}")
            plt.errorbar(x=i, y=means[i], yerr=(conf_interval[1] - means[i]), fmt='o', label=methods[i], color = colors[i])

    plt.xticks(range(len(means)), [methods[i] for i in range(len(means))], fontsize = 12)
    plt.ylabel('Expected Waiting Time', fontsize = 16)
    plt.title('Confidence Intervals for Expected Waiting Times', fontsize = 16)
    plt.show()

statistics("./assignment_2/simulation_results/results.csv")

def expected_wait_time_n(n, lam, mu):
    delay_probability = ((n* rho) ** n / math.factorial(n)) * (((1 - rho) *  np.sum([(n * rho) ** m / math.factorial(m) for m in range(n)]) + (n * rho) ** n / math.factorial(n)) ** -1)
    expected_wait_time_n = delay_probability / (n * mu * (1 - rho))
    
    return expected_wait_time_n

def power_analysis(n, lam, mu):
    mm1_wait = (lam/(n * mu)) / (mu - lam)
    mm2_wait = expected_wait_time_n(2, lam, mu)
    mm4_wait = expected_wait_time_n(4, lam, mu)
    
    var_mm1_wait = (1/mm1_wait)^2
    var_mm2_wait = (1/mm2_wait)^2
    var_mm4_wait = (1/mm4_wait)^2
    
    
    
    