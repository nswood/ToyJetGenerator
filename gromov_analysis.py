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

def delta_hyp(dismat):
    p = 0
    row = dismat[p, :][None, :]
    col = dismat[:, p][:, None]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = torch.minimum(XY_p[:, :, None], XY_p[None, :, :]).max(1).values
    return (maxmin - XY_p).max()



n_parts = [32,64]
n_prong = [1,2,3,4]
output_folder = "Toy_datasets"


# %%
def process_datasets():
    gromov_outputs = []

    for prong in n_prong:
        for npts in n_parts:
            for file_id in range(25):
                filename = f"{output_folder}/dataset_nprong{prong}_nparts{npts}_{file_id}.pt"
                if os.path.exists(filename):
                    print(filename)
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
                            'n_prong': prong
                        })
                        

                else:
                    print(f"Dataset not found: {filename}")

    return gromov_outputs


def find_neighbors_knn(four_momentum_tensor, dists, index, k=5):
    # Randomly select a point
    num_points = four_momentum_tensor.shape[0]
    selected_point = four_momentum_tensor[index]
    
    # Find the indices of the k-nearest neighbors
    neighbors_indices = torch.topk(dists[index], k, largest=False).indices

    neighbors = four_momentum_tensor[neighbors_indices]

    return selected_point, neighbors, neighbors_indices

def process_datasets_per_particle(all_k, n_parts,n_prong):
    gromov_outputs = []

    for prong in n_prong:
        for npts in n_parts:
            for file_id in range(25):
                filename = f"{output_folder}/dataset_nprong{prong}_nparts{npts}_{file_id}.pt"
                if os.path.exists(filename):
                    print(filename)
                    dataset = torch.load(filename)
                    data_samples = len(dataset[3])

                    for i in range(data_samples):

                        ###############################
                        ###############################
                        # Only Final State Particles
                        ###############################
                        ###############################
                        final_state_particles = dataset[3][i]
                        e_eta_phi = []
                        results = []
                        for part in final_state_particles:
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
                        for j in range(len(four_momentum_tensor)):
                            n_parts = len(four_momentum_tensor)

                            for k in all_k:
                                try: 
                                    # print(k)
                                    selected_point, neighbors, neighbors_indices = find_neighbors_knn(four_momentum_tensor, dists, index=j, k=k)

                                    # Extract the submatrix for the neighbors
                                    neighbors_dists = dists[neighbors_indices][:, neighbors_indices]

                                    delta = delta_hyp(neighbors_dists)
                                    diam = neighbors_dists.max()
                                    rel_delta = (2 * delta) / diam

                                    # Calculate c based on relative delta mean
                                    c = (0.144 / rel_delta) ** 2

                                    # Save the results
                                    gromov_outputs.append({
                                        'file_id': file_id,
                                        'jet_id': i,
                                        'data_type':'final_state',
                                        'selected_point_energy': selected_point[0].item(),
                                        'selected_point_eta': selected_point[1].item(),
                                        'selected_point_phi': selected_point[2].item(),
                                        'num_parts': n_parts,
                                        'num_prong': prong,
                                        'k': k,
                                        'delta': delta.item(),
                                        'rel_delta': rel_delta.item(),
                                        'c': c.item(),
                                    })
                                except:
                                    print(f'Error in {filename},jet {i}, particle {j}, k {k}')
                        
                        ###############################
                        ###############################
                        # Full Shower
                        ###############################
                        ###############################

                        shower_particles = dataset[4][i]
                        e_eta_phi = []
                        results = []
                        for part in shower_particles:
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
                        for j in range(len(four_momentum_tensor)):
                            n_parts = len(four_momentum_tensor)

                            for k in all_k:
                                try: 
                                    # print(k)
                                    selected_point, neighbors, neighbors_indices = find_neighbors_knn(four_momentum_tensor, dists, index=j, k=k)

                                    # Extract the submatrix for the neighbors
                                    neighbors_dists = dists[neighbors_indices][:, neighbors_indices]

                                    delta = delta_hyp(neighbors_dists)
                                    diam = neighbors_dists.max()
                                    rel_delta = (2 * delta) / diam

                                    # Calculate c based on relative delta mean
                                    c = (0.144 / rel_delta) ** 2

                                    # Save the results
                                    gromov_outputs.append({
                                        'file_id': file_id,
                                        'jet_id': i,
                                        'data_type':'shower',
                                        'selected_point_energy': selected_point[0].item(),
                                        'selected_point_eta': selected_point[1].item(),
                                        'selected_point_phi': selected_point[2].item(),
                                        'num_parts': n_parts,
                                        'num_prong': prong,
                                        'k': k,
                                        'delta': delta.item(),
                                        'rel_delta': rel_delta.item(),
                                        'c': c.item(),
                                    })
                                except:
                                    print(f'Error in {filename},jet {i}, particle {j}, k {k}')


                        

                else:
                    print(f"Dataset not found: {filename}")
                df = pd.DataFrame(gromov_outputs)
                # if os.path.exists("per_particle_gromov_toy_data.csv"):
                #     df_existing = pd.read_csv("per_particle_gromov_toy_data.csv")
                #     df = pd.concat([df_existing, df], ignore_index=True)
                df.to_csv("per_particle_gromov_toy_data.csv", index=False)

    
# %%
# output = process_datasets()
all_k = [5,8,12]
n_parts = [32,64]
n_prong = [1,2,3,4]

process_datasets_per_particle(all_k, n_parts,n_prong)
# %%
df = pd.DataFrame(output)

# df.to_csv("per_particle_gromov_toy_data.csv", index=False)


