import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.stats.power as smp

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
    
    mm1_data = df.groupby("n_server").get_group(1)["waiting_time"].values
    mm2_data = df.groupby("n_server").get_group(2)["waiting_time"].values
    mm4_data = df.groupby("n_server").get_group(4)["waiting_time"].values
    
    mm1_wait_mean = np.mean(mm1_data)
    mm2_wait_mean = np.mean(mm2_data)
    mm4_wait_mean = np.mean(mm4_data)
    
    mm1_wait_std = np.std(mm1_data)
    mm2_wait_std = np.std(mm2_data)
    mm4_wait_std = np.std(mm4_data)
    
    methods = ["M/M/1", "M/M/2", "M/M/4"]
    means = [mm1_wait_mean, mm2_wait_mean, mm4_wait_mean]
    stds = [mm1_wait_std, mm2_wait_std, mm4_wait_std]
    
    #Creating the general stats
    alpha = 0.01
    general_stats = df.groupby(["n_server"]).aggregate(["mean", "std", "min", "max"])["waiting_time"]
    print(general_stats.to_latex(float_format="%.3f"))
    
    #ANOVA
    statistic, p_value = stats.f_oneway(mm1_data, mm2_data, mm4_data)
    print(f"P_value for ANOVA: {p_value}")
    print()
    
    #T-tests
    statistic_1, p_value_1 = stats.ttest_ind(a= mm1_data, b=mm2_data, equal_var=True)
    statistic_2, p_value_2 = stats.ttest_ind(a= mm1_data, b=mm4_data, equal_var=True)
    statistic_4, p_value_4 = stats.ttest_ind(a= mm2_data, b=mm4_data, equal_var=True)
    print(f"T-test p_value between n = 1, n = 2:  {p_value_1}")
    print(f"T-test p_value between n = 1, n = 4:  {p_value_2}")
    print(f"T-test p_value between n = 2, n = 4:  {p_value_4}")
    print()
    
    #Creating the confidence intervals
    
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

statistics("./simulation_results/results.csv")

def expected_wait_time_n(n, lam, mu):
    rho = lam/(n* mu)
    delay_probability = ((n* rho) ** n / math.factorial(n)) * (((1 - rho) *  np.sum([(n * rho) ** m / math.factorial(m) for m in range(n)]) + (n * rho) ** n / math.factorial(n)) ** -1)
    expected_wait_time_n = delay_probability / (n * mu * (1 - rho))
    
    return expected_wait_time_n

def power_analysis(lam, mu):
    mm1_wait = (lam/(mu)) / (mu - lam)
    mm2_wait = expected_wait_time_n(2, 2*lam, mu)
    mm4_wait = expected_wait_time_n(4, 4*lam, mu)
    
    var_mm1_wait = (1/mm1_wait)**2
    var_mm2_wait = (1/mm2_wait)**2
    var_mm4_wait = (1/mm4_wait)**2
    
    waiting_time_data = np.array([mm1_wait, mm2_wait, mm4_wait])
    overall_mean = np.mean(waiting_time_data)
    ss_total = np.sum((waiting_time_data - overall_mean) ** 2)
    ss_between = np.sum([(np.mean(queue_type) - overall_mean) ** 2 for queue_type in waiting_time_data])
    eta_squared = ss_between /ss_total
    
    sample_size = smp.FTestAnovaPower().solve_power(effect_size=1, alpha=0.01, k_groups=3, power=0.8, nobs=None)
    print(f"Number of samples needed for lam = {lam} and mu = {mu}: {int(np.ceil(sample_size))}")
    
power_analysis(0.3, 0.8)
    
    