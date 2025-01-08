# %%
# from datagenerator import jet_data_generator 
from datagenerator_realistic import jet_data_generator as realistic_generator

from plotutils import plot_event 
import matplotlib.pyplot as plt
import time
import torch
import pandas as pd

# %%
import os
import numpy as np
# if not os.path.exists(os.path.join(os.getcwd(),'figures/paper')):
#     os.mkdir(os.path.join(os.getcwd(),'figures/paper'))

# %%
def delta_hyp(dismat):
    p = 0
    row = dismat[p, :][None, :]
    col = dismat[:, p][:, None]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = torch.minimum(XY_p[:, :, None], XY_p[None, :, :]).max(1).values
    return (maxmin - XY_p).max()

# %%
# quark_mass = [2,5,10]
# n_parts = [16,32,64]
quark_mass = [10]
n_parts = [64]
num_samples = 1000
# num_samples = 5
output_folder = "gromov_testing_datasets"


# %%

# %%
def process_datasets():
    gromov_outputs = []

    for qmass in quark_mass:
        for npts in n_parts:
            filename = f"{output_folder}/dataset_qmass{qmass}_nparts{npts}.pt"

            if os.path.exists(filename):
                dataset = torch.load(filename)
                data_samples = len(dataset[3])

                for i in range(data_samples):
                    all_particles = dataset[3][i]
                    e_eta_phi = []

                    for part in all_particles:
                        mom = part.mom
                        e_eta_phi.append([mom.e, mom.eta.item(), mom.phi.item()])

                    e_eta_phi = np.array(e_eta_phi)
                    four_momentum_tensor = torch.tensor(e_eta_phi)

                    # Extract energy, eta, phi
                    energies = four_momentum_tensor[:, 0]
                    normalized_energies = energies / energies.sum()
                    energies = normalized_energies
                    etas = four_momentum_tensor[:, 1]
                    phis = four_momentum_tensor[:, 2]

                    # Compute pairwise energy differences
                    energy_diffs = torch.abs(energies[:, None] - energies[None, :])
                    delta_eta = etas[:, None] - etas[None, :]
                    delta_phi = phis[:, None] - phis[None, :]
                    delta_phi = torch.remainder(delta_phi + np.pi, 2 * np.pi) - np.pi
                    delta_R = torch.sqrt(delta_eta ** 2 + delta_phi ** 2)

                    R = 1
                    dists = (energy_diffs * delta_R) / R
                    delta = delta_hyp(dists)

                    gromov_outputs.append({
                        'delta': delta.item(),
                        'n_parts': npts,
                        'quark_mass': qmass
                    })

            else:
                print(f"Dataset not found: {filename}")

    return gromov_outputs


# %%
output = process_datasets()

# %%
df = pd.DataFrame(output)

figure_path = "Gromov_figures"
os.makedirs(figure_path, exist_ok = True)

# %%
grouped = df.groupby(['n_parts', 'quark_mass'])
grouped.mean()

# %%
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True, sharey=True)
axes = axes.flatten()

bins = np.linspace(df['delta'].min(), df['delta'].max(), 50)

for ax, ((n_parts, quark_mass), group) in zip(axes, grouped):
    group['delta'].plot.hist(ax=ax, bins=bins, density=True, alpha=0.5)
    ax.set_title(f'n_parts={n_parts}, quark_mass={quark_mass}')
    
plt.tight_layout()
plt.xlabel('Gromov Delta')
plt.ylabel('Density')
plt.savefig(f'./{figure_path}/gromov_d_histograms.png')
plt.close()

