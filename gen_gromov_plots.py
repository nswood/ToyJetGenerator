# %%
# from datagenerator import jet_data_generator 
import numpy as np
import os
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams.update({'font.size': 18})
import pandas as pd

output_dir = 'Gromov_figures'
# %%
df = pd.read_csv('processed_gromov_toy_data.csv')

# %%
import matplotlib.pyplot as plt

# Get unique values for n_parts and n_prong
unique_n_parts = df['n_parts'].unique()
unique_n_prongs = df['n_prong'].unique()


# Create subplots
fig, axes = plt.subplots(len(unique_n_parts), len(unique_n_prongs), figsize=(5 * len(unique_n_prongs), 5 * len(unique_n_parts)), sharex=True, sharey=True)

# Plot histograms

for i, n_parts in enumerate(unique_n_parts):
    for j, n_prong in enumerate(unique_n_prongs):
        ax = axes[i, j]
        subset = df[(df['n_parts'] == n_parts) & (df['n_prong'] == n_prong)]
        if not subset.empty:
            ax.hist(subset['delta'], bins=50, density=True, alpha=0.6)
            ax.set_title(f'n_parts={n_parts}, n_prong={n_prong}')
            ax.set_xlabel('delta')
            ax.legend()

plt.suptitle(f'Histograms of delta for different n_parts and n_prongs')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, f'histograms.png'))
plt.close()

fig, axes = plt.subplots(len(unique_n_parts), len(unique_n_prongs), figsize=(5 * len(unique_n_prongs), 5 * len(unique_n_parts)), sharex=True, sharey=True)

# Plot histograms
for i, n_parts in enumerate(unique_n_parts):
    for j, n_prong in enumerate(unique_n_prongs):
        ax = axes[i, j]
        subset = df[(df['n_parts'] == n_parts) & (df['n_prong'] == n_prong)]
        ax.hist(subset['rel_delta'], bins=50, density=True, alpha=0.6, label=f'n_parts={n_parts}, n_prong={n_prong}')
        ax.set_title(f'n_parts={n_parts}, n_prong={n_prong}')
        ax.set_xlabel('rel_delta')
        ax.set_ylabel('Density')
        # ax.legend()

        

plt.tight_layout()
plt.savefig(os.path.join(output_dir,'per_jet_relative_distribution'))
plt.close()
# %%
df = pd.read_csv('per_particle_gromov_toy_data.csv')

df_shower = df[df['data_type'] == 'shower']
df_fs = df[df['data_type'] == 'final_state']

# Define the bins
rel_bins = np.linspace(0, 2, 50)
bins = np.linspace(0, 0.1, 50)

def plot_histograms(df, data_type, output_dir):
    
    if data_type == 'shower':
        part_col = 'final_state_parts'
    else:
        part_col = 'num_parts'
    grouped = df.groupby([part_col, 'num_prong'])
    for combo in product(df[part_col].unique(), df['num_prong'].unique()):
        plt.figure()
        plotted = False

        for k in df['k'].unique():
            try:
                subset = grouped.get_group(combo)
                k_grouped = subset.groupby(['k']).get_group(k)
                plt.hist(k_grouped['rel_delta'], bins=rel_bins, label=f'K = {k}', histtype='step', density=True)
                plt.title(f'n_parts={combo[0]}, n_prong={combo[1]}, data_type={data_type}')
                plt.legend()
                plt.xlabel('rel_delta')
                plt.ylabel('Density')
                plotted = True
            except KeyError:
                # This combination doesn't exist in the data
                pass

        if plotted:
            plt.savefig(os.path.join(output_dir, f'per_part_rel_delta_{combo[0]}_{combo[1]}_{data_type}.png'))
        plt.close()

        plt.figure()
        plotted = False

        for k in df['k'].unique():
            try:
                subset = grouped.get_group(combo)
                k_grouped = subset.groupby(['k']).get_group(k)
                plt.hist(k_grouped['delta'], bins=bins, label=f'K = {k}', histtype='step', density=True)
                plt.title(f'n_parts={combo[0]}, n_prong={combo[1]}, data_type={data_type}')
                plt.legend()
                plt.xlabel('delta')
                plt.xscale('log')
                plt.ylabel('Density')
                plotted = True
            except KeyError:
                # This combination doesn't exist in the data
                pass

        if plotted:
            plt.savefig(os.path.join(output_dir, f'per_part_delta_{combo[0]}_{combo[1]}_{data_type}.png'))
        plt.close()

    # Create an empty dictionary to store aggregated histogram data
    heatmap_data_rel = {}
    heatmap_data_delta = {}

    for combo in product(df[part_col].unique(), df['num_prong'].unique()):
        plt.figure()
        rel_histograms = []
        delta_histograms = []
        plotted = False

        for k in df['k'].unique():
            try:
                subset = grouped.get_group(combo)
                k_grouped = subset.groupby(['k']).get_group(k)
                plt.hist(k_grouped['delta'], bins=bins, label=f'K = {k}', histtype='step', density=True)
                plt.title(f'n_parts={combo[0]}, n_prong={combo[1]}, data_type={data_type}')
                plt.legend()
                plt.xlabel('delta')
                plt.xscale('log')
                plt.ylabel('Density')
                plotted = True
            except KeyError:
                # This combination doesn't exist in the data
                pass

        if plotted:
            plt.savefig(os.path.join(output_dir, f'per_part_delta_{combo[0]}_{combo[1]}_{data_type}.png'))
        plt.close()

    for combo in product(df[part_col].unique(), df['num_prong'].unique()):
        plt.figure()
        rel_histograms = []
        delta_histograms = []

        for k in df['k'].unique():
            try:
                subset = grouped.get_group(combo)
                k_grouped = subset.groupby(['k']).get_group(k)

                # Histogram for `rel_delta`
                hist_rel, _ = np.histogram(k_grouped['rel_delta'], bins=rel_bins, density=True)
                rel_histograms.append(hist_rel)
                
                # Plot histogram for `rel_delta`
                plt.hist(k_grouped['rel_delta'], bins=rel_bins, label=f'K = {k}', histtype='step', density=True)
                
            except KeyError:
                # Skip missing combinations
                pass
        plt.title(f'n_parts={combo[0]}, n_prong={combo[1]}, data_type={data_type}')
        plt.legend()
        plt.xlabel('rel_delta')
        plt.ylabel('Density')
        plt.savefig(os.path.join(output_dir, f'per_part_rel_delta_{combo[0]}_{combo[1]}_{data_type}.png'))
        plt.close()

        # Save histograms for heatmap aggregation
        heatmap_data_rel[combo] = np.array(rel_histograms)

        plt.figure()
        for k in df['k'].unique():
            try:
                # Histogram for `delta`
                hist_delta, _ = np.histogram(k_grouped['delta'], bins=bins, density=True)
                delta_histograms.append(hist_delta)
                
                # Plot histogram for `delta`
                plt.hist(k_grouped['delta'], bins=bins, label=f'K = {k}', histtype='step', density=True)
                
            except KeyError:
                pass
        plt.title(f'n_parts={combo[0]}, n_prong={combo[1]}, data_type={data_type}')
        plt.legend()
        plt.xlabel('delta')
        plt.xscale('log')
        plt.ylabel('Density')
        plt.savefig(os.path.join(output_dir, f'per_part_delta_{combo[0]}_{combo[1]}_{data_type}.png'))
        plt.close()

        # Save histograms for heatmap aggregation
        heatmap_data_delta[combo] = np.array(delta_histograms)

    

# Plot histograms for final state data
plot_histograms(df_fs, 'final_state', output_dir)

# Plot histograms for shower data
plot_histograms(df_shower, 'shower', output_dir)
