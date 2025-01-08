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


# %%
# quark_mass = [2,5,10]
# n_parts = [16,32,64]
quark_mass = [10]
n_parts = [64]
num_samples = 1000
output_folder = "gromov_testing_datasets"
os.makedirs(output_folder, exist_ok = True)

# %%

def generate_and_save_datasets():
    for qmass in quark_mass:
        for npts in n_parts:
            realistic_sig_2p_16part = realistic_generator("signal", qmass, npts, True)
            start = time.time()
            dataset = realistic_sig_2p_16part.generate_dataset(num_samples)
            end = time.time()

            print(f"Generated dataset for quark mass {qmass}, n_parts {npts} in {end - start:.2f}s")

            # Save dataset to a folder with naming convention
            filename = f"{output_folder}/dataset_qmass{qmass}_nparts{npts}.pt"
            torch.save(dataset, filename)
            print(f"Saved to {filename}")

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
generate_and_save_datasets()
