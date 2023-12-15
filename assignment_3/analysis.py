import numpy as np
import pandas as pd
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

data = pd.read_csv("./results.csv", header=0)

def general_stats(csv_data):
    df = pd.DataFrame(data)
    alpha = 0.01

    point_estimated = df.groupby(["method"]).aggregate(
        ["mean", "std", "min", "max"])["best_length"]
    point_estimated["Confidence Interval"] = [stats.t.interval(
        1-alpha, 99, mean, std/33) for mean, std in zip(point_estimated["mean"], point_estimated["std"])]
    print(point_estimated.to_latex(float_format="%.3f"))

    array_like = df.groupby("method")["best_length"].apply(
        list).reset_index()["best_length"].values
    methods = df["method"].unique()

    bartlett_stat, bartlett_pvalue = stats.bartlett(*array_like)
    print(
        f"Bartlett's Test - Statistic: {bartlett_stat}, p-value: {bartlett_pvalue}")
    print()

    welch_result = pg.welch_anova(data=data, dv="best_length", between="method")
    print(f"Welch's ANOVA statistic: {welch_result['F'][0]}    p-value: {welch_result['p-unc'][0]}")

    posthoc_result = pg.pairwise_gameshowell(data=data, dv="best_length", between="method")
    print(posthoc_result)


def bar_plot(csv_data):
    df = pd.DataFrame(data)
    alpha = 0.01
    methods = ["Classic Simulated Annealing", "Fast Simulated Annealing", "List-Based Simulated Annealing"]
    
    sa_data = data.groupby("method").get_group("sim_annealing")["best_length"].iloc[0]
    fast_sa_data = data.groupby("method").get_group("fast_annealing")["best_length"].iloc[0]
    list_sa_data = data.groupby("method").get_group("list_sim_annealing")["best_length"].iloc[0]
    
    mean_sa = np.mean(sa_data)
    mean_fast_sa = np.mean(fast_sa_data)
    mean_list_sa = np.mean(list_sa_data)
    
    std_sa = np.std(sa_data)
    std_fast_sa = np.std(fast_sa_data)
    std_list_sa = np.std(list_sa_data)
    
    means = [mean_sa, mean_fast_sa, mean_list_sa]
    stds = [std_sa, std_fast_sa, std_list_sa]
    
    method_colors = {
    "Classic Simulated Annealing": "#8BBF9F",
    "Fast Simulated Annealing": "#83BCFF",
    "List-Based Simulated Annealing": "#124559",
    }
    
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        alpha = 0.05
        standard_error = stds[i]/ np.sqrt(20)
        df = 19
        color = to_rgba(method_colors[method])
        plt.bar(i, means[i], yerr=standard_error, capsize=5, color=color, label=method)      
    #plt.xticks([1.5, 5.5, 9.5], ["n = 1", "n = 2", "n = 4"], fontsize = 14)
    plt.ylabel('Expected Waiting Time', fontsize = 16)
    plt.yticks(fontsize = 14)
    plt.legend(["Classic", "Fast", "List-based"], loc='upper right', fontsize = 14)
    plt.title('Mean Best Path Lengths', fontsize = 16)
    plt.tight_layout()
    plt.show()


def error_plot(csv_data, method):
    df = pd.DataFrame(data)
    sa_data = data.groupby("method").get_group("sim_annealing")["best_length"].iloc[0]
    fast_sa_data = data.groupby("method").get_group("fast_annealing")["best_length"].iloc[0]
    list_sa_data = data.groupby("method").get_group("list_sim_annealing")["best_length"].iloc[0]
    
    if method == "sim_annealing":
        sa_data = data.groupby("method").get_group("sim_annealing")["best_length"].iloc[0]
    


general_stats(data)
bar_plot(data)


