from datagenerator_realistic import jet_data_generator as realistic_generator
import os
import numpy as np
import copy
from pylorentz import Momentum4
import argparse
import h5py
import copy
from pylorentz import Momentum4
 
def soft_splitting_augmentation(gen_function, particles_list,n_splits_mid, n_splits_std):
    particles = copy.deepcopy(particles_list)
    n_splittings = int(np.random.normal(loc=n_splits_mid, scale=n_splits_std))
#     print(f'Splitting {n_splittings} particles with soft splitting')
    for _ in range(n_splittings):
        dau1, dau2, z, theta = gen_function.softsplit(particles[0])
        particles.pop(0)
        gen_function.reverse_insort(particles, dau1)
        gen_function.reverse_insort(particles, dau2)
        
    return particles

# new function
# randomly selects root
# merges it with lowest particle energy weighted distance 
def soft_merging_augmentation(gen_function, particles_list,n_merges_mid, n_merges_std):
    particles = copy.deepcopy(particles_list)
    all_e = []
    all_phi = []
    all_eta = []
    for p in particles:
        mom = p.mom
        all_e.append(mom[0])
        all_phi.append(mom.phi)
        all_eta.append(mom.eta)
        
    all_e = np.array(all_e)
    all_phi = np.array(all_phi)
    all_eta = np.array(all_eta)

    phi_diff = (all_phi[:, np.newaxis] - all_phi[np.newaxis, :]).squeeze()
    eta_diff = (all_eta[:, np.newaxis] - all_eta[np.newaxis, :]).squeeze()

    pairwise_matrix = ((phi_diff**2 + eta_diff**2) * (all_e[:, np.newaxis] + all_e[np.newaxis, :]) / 2) + 1000*np.eye(len(particles))
    print(pairwise_matrix.shape)
    n_merges = int(np.random.normal(loc=n_merges_mid, scale=n_merges_std))
    for _ in range(n_merges):
        root = int(np.abs(np.random.normal(loc=0, scale=2)))
        dau2_id = np.argmin(pairwise_matrix[-root])
        
        mother = gen_function.softcombine(particles[-root],particles[dau2_id])
        if root > dau2_id:
            particles.pop(root)
            particles.pop(dau2_id)
        else:
            particles.pop(dau2_id)
            particles.pop(root)
        # need to update pairwise matrix to remove particles dropped and add info for new particles 
#         gen_function.reverse_insort(particles, mother)
        mother_index = gen_function.reverse_insort(particles, mother)
        # Update the pairwise matrix:
        # 1. Remove the rows/columns of the merged particles
        all_e = []
        all_phi = []
        all_eta = []
        for p in particles:
            mom = p.mom
            all_e.append(mom[0])
            all_phi.append(mom.phi)
            all_eta.append(mom.eta)

        all_e = np.array(all_e)
        all_phi = np.array(all_phi)
        all_eta = np.array(all_eta)

        phi_diff = (all_phi[:, np.newaxis] - all_phi[np.newaxis, :]).squeeze()
        eta_diff = (all_eta[:, np.newaxis] - all_eta[np.newaxis, :]).squeeze()

        pairwise_matrix = ((phi_diff**2 + eta_diff**2) * (all_e[:, np.newaxis] + all_e[np.newaxis, :]) / 2) + 1000*np.eye(len(particles))

        
    return particles


def rotation_augmentation(gen_function, particles_list):
    particles = copy.deepcopy(particles_list)
    n_particles = len(particles_list)
    theta = np.random.uniform(low=0,high = np.pi)
#     print(f'Rotating {n_particles} particles')
    for i in range(n_particles):
        rotated_particle_mom = gen_function.rotatePhi(particles[i].mom,theta)
        particles[i].mom = rotated_particle_mom
        
    return particles

def rotation_matrix_x(theta):
    # Create an identity matrix
    rotation_matrix = np.eye(4)
    
    # Set the rotation part for the y-z plane
    rotation_matrix[1, 1] = np.cos(theta)
    rotation_matrix[1, 2] = -np.sin(theta)
    rotation_matrix[2, 1] = np.sin(theta)
    rotation_matrix[2, 2] = np.cos(theta)
    
    return rotation_matrix



def Lorentz_xy_rotation_augmentation(gen_function, particles_list):
    particles = copy.deepcopy(particles_list)
    n_particles = len(particles_list)
    theta = np.random.uniform(low=0,high = np.pi)
    M = rotation_matrix_x(theta)
#     print(f'Rotating {n_particles} particles')
    for i in range(n_particles):
        mom = particles[i].mom
        mom_vec = np.array([mom.p_t,mom.p_x,mom.p_y,mom.p_z])
        rotated_mom = M@mom_vec
        new_mom = Momentum4(rotated_mom[0],rotated_mom[1],rotated_mom[2],rotated_mom[3])
        particles[i].mom = new_mom
        
    return particles

def boost_matrix_z(eta):
    # Create an identity matrix
    rotation_matrix = np.eye(4)
    
    # Set the rotation part for the y-z plane
    rotation_matrix[0, 0] = np.cosh(eta)
    rotation_matrix[1, 3] = np.sinh(eta)
    rotation_matrix[0, 3] = np.sinh(eta)
    rotation_matrix[3, 3] = np.cosh(eta)
    
    return rotation_matrix
    
def Lorentz_z_boost_augmentation(gen_function, particles_list):
    particles = copy.deepcopy(particles_list)
    n_particles = len(particles_list)
    eta = np.random.normal(loc=0, scale=0.25)
    M = boost_matrix_z(eta)
#     print(f'Rotating {n_particles} particles')
    for i in range(n_particles):
        mom = particles[i].mom
        mom_vec = np.array([mom.p_t,mom.p_x,mom.p_y,mom.p_z])
        boosted_mom = M@mom_vec
        new_mom = Momentum4(boosted_mom[0],boosted_mom[1],boosted_mom[2],boosted_mom[3])
        particles[i].mom = new_mom
        
    return particles  


def apply_augmentation(gen_function,data,aug_fn,Lorentz = False):
    if Lorentz:
        augmented_jet = aug_fn(gen_function,list(data))
    else:
        augmented_jet = aug_fn(gen_function,list(data),4,2)
    return augmented_jet

def particle_list_to_arr(particle_list):
    
    arr = []
    for j in range(len(particle_list)):
        arr.append(particle_list[j].mom.p_t)
        arr.append(particle_list[j].mom.eta)
        arr.append(particle_list[j].mom.phi)
    arr = np.array(arr)
    return arr


import random
def gen_data_pairs(out_dir, n_prongs, n_samples,n_parts, functions):
    gen_function = realistic_generator("signal",n_prongs, n_parts, True)
    data_arr,_,_,data_particles = gen_function.generate_dataset(n_samples)
    aug_arr = []
    for d in data_particles:
        # Choose one primary augmentation
        cur_fun = random.choice(functions[0])
        aug_jet = apply_augmentation(gen_function, d,cur_fun)
        # Apply all Lorentz augmentations
        for f in functions[1]:
            aug_jet = apply_augmentation(gen_function, aug_jet,f,Lorentz = True)
        final_aug_arr = particle_list_to_arr(aug_jet)
        aug_arr.append(final_aug_arr)
    return data_arr, aug_arr
        
    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and save model results.")
    parser.add_argument('--opath', type=str, required=True, help="Output file path.")
    parser.add_argument('--samples', type=int, required=True, help="Number of data-augmentation pairs to generate.")
    parser.add_argument('--prong', type=int, required=True, help="Number prongs for dataset")
    parser.add_argument('--index', type=int, required=True, help="Number prongs for dataset")
    parser.add_argument('--n_parts', type=int, required=True, help="Mean number of particles per jet. Std is 10% of n_parts")
    
    functions = [[soft_splitting_augmentation,soft_merging_augmentation],[Lorentz_z_boost_augmentation,Lorentz_xy_rotation_augmentation]]
    
    # Parse the arguments
    args = parser.parse_args()
    os.makedirs(args.opath, exist_ok=True)
    batch_size = args.samples
    filename = os.path.join(args.opath, f'jet_{args.prong}_{args.index}.h5')
    data, aug_data = gen_data_pairs(args.opath, args.prong,batch_size, args.n_parts, functions)
    # Save the arrays
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=data, dtype=h5py.special_dtype(vlen=np.dtype('float64')))
        f.create_dataset('aug_data', data=aug_data, dtype=h5py.special_dtype(vlen=np.dtype('float64')))


if __name__ == '__main__':
    main()
    
   