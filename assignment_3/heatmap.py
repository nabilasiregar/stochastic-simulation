import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

csv_file_path = './multirun/train_results.csv'
data = pd.read_csv(csv_file_path)

def plot_heatmap():
    cooling_factor_bins = np.arange(0.8, 1.00, 0.02)
    chain_length_bins = np.arange(2000, 22000, 2000)

    data['cooling_factor_group'] = pd.cut(data['cooling_factor'], bins=cooling_factor_bins, include_lowest=True, labels=np.round(cooling_factor_bins[:-1], 2))
    data['chain_length_group'] = pd.cut(data['chain_length'], bins=chain_length_bins, include_lowest=True, labels=chain_length_bins[:-1])
    pivoted_df = data.pivot_table(index='cooling_factor_group', columns='chain_length_group', values='best_length', aggfunc='mean')

    # Index and columns represent the parameters and the cell values represent the averaged 'best_length'
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivoted_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average Path Length'}, annot_kws={"fontsize":11})
    ax.set_title('Tuning Cooling Factor and Markov-Chain Length Based on Path Length', fontsize=18, pad=20)
    plt.xlabel('Markov Chain Length', fontsize=16)
    plt.ylabel('Cooling Factor', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel('Average Path Length', fontsize=16)
    plt.show()

def plot_contour():
    data.columns = ['trial', 'initial_temperature', 'cooling_factor', 'chain_length', 'best_length', 'iterations']

    plt.figure(figsize=(12, 8))
    sns.kdeplot(
        data=data,
        x='chain_length',
        y='cooling_factor',
        weights='best_length',
        cmap="coolwarm",
        fill=True
    )

    plt.title("Best Path Length with Different Cooling Factors and Markov Chain Lengths")
    plt.xlabel("Markov Chain Length")
    plt.ylabel("Cooling Factor")
    plt.show()

def plot_raw_heatmap():
    pivoted_df = data.pivot_table(index='cooling_factor', columns='chain_length', values='best_length', aggfunc='mean')
    
    # Index and columns represent the parameters and the cell values represent the averaged 'best_length'
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivoted_df, annot=True, fmt=".2f", cmap="YlGnBu", 
            cbar_kws={'label': 'Average Path Length'}) 
    plt.title('Tuning Cooling Factor and Markov-Chain Length Based on Path Length', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Markov Chain Length', fontsize=16)
    plt.ylabel('Cooling Factor', fontsize=16)
    plt.show()

if __name__ == "__main__":
    plot_heatmap()