# %%
# from datagenerator import jet_data_generator 
from datagenerator_resonances_realistic import jet_data_generator as realistic_generator

import time
import torch # type: ignore
import os
import argparse

def generate_and_save_datasets(nprong, npts, num_samples,file_id):
    realistic_sig_2p_16part = realistic_generator("signal", nprong, npts, True,
                resonance_data=[
                    {'mass': 180.0, 'relative_ratio': 1.0, 'decay_products': 3},
                    {'mass': 30.0, 'relative_ratio': 1.0, 'decay_products': 3},
                    {'mass': 80.0, 'relative_ratio': 1.0, 'decay_products': 2},
                    {'mass': 10.0, 'relative_ratio': 1.0, 'decay_products': 2},
                ], 
                total_resonance_prob=0.15, # Adjust total resonance probability as desired
                max_resonance_per_jet = 1)
    start = time.time()
    dataset = realistic_sig_2p_16part.generate_dataset(num_samples)
    end = time.time()

    print(f"Generated dataset for n_prong {nprong}, n_parts {npts} in {end - start:.2f}s")

    # Save dataset to a folder with naming convention
    filename = f"{output_folder}/dataset_nprong{nprong}_nparts{npts}_{file_id}.pt"
    torch.save(dataset, filename)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate and save datasets")
    parser.add_argument("--nprong", type=int, required=True, help="Number of prongs")
    parser.add_argument("--npts", type=int, required=True, help="Number of particles")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--output_folder", type=str, default="gromov_testing_datasets", help="Output folder")
    parser.add_argument("--file_id", type=str, default="", help="Optional file identifier")
    args = parser.parse_args()
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)   

    generate_and_save_datasets(args.nprong, args.npts, args.num_samples,args.file_id)
