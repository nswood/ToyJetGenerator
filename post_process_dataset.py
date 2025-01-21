import os
import torch
import numpy as np
import argparse
import time

def process_dataset(cur_dataset, max_particles=100):
    feature_list = []
    label_list = []

    for particles in cur_dataset:
        features = []
        labels = []
        for particle in particles:
            kin_features = [particle.mom.e, particle.mom.p_t, particle.mom.m, particle.mom.eta, particle.mom.phi]
            part_labels = [particle.part_label, particle.part_parent_label, particle.prong_label, particle.resonance_origin]
            features.append(kin_features)
            labels.append(part_labels)

        features = np.array(features)
        labels = np.array(labels)
        sorted_indices = np.argsort(-features[:, 0])
        features = features[sorted_indices]
        labels = labels[sorted_indices]

        if len(features) < max_particles:
            padding = max_particles - len(features)
            features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
            labels = np.pad(labels, ((0, padding), (0, 0)), mode='constant')
        elif len(features) > max_particles:
            features = features[:max_particles]
            labels = labels[:max_particles]

        feature_list.append(features)
        label_list.append(labels)

    return np.array(feature_list), np.array(label_list)

def process_and_save_datasets(input_dir, output_dir):
    final_state_datasets = []
    full_shower_datasets = []
    
    start_time = time.time()
    for i, filename in enumerate(os.listdir(input_dir)):
        file_start_time = time.time()
        
        print(f"Processing file {i + 1}/{len(os.listdir(input_dir))}")
        if filename.endswith('.pt'):
            file_path = os.path.join(input_dir, filename)
            data = torch.load(file_path)
            
            final_state_dataset = data[0]
            full_shower_dataset = data[1]
            
            processed_final_state, _ = process_dataset(final_state_dataset, max_particles=32)
            processed_full_shower, _ = process_dataset(full_shower_dataset, max_particles=100)
            
            final_state_datasets.append(processed_final_state)
            full_shower_datasets.append(processed_full_shower)
        
        file_end_time = time.time()
        print(f"Time taken for file {i + 1}: {file_end_time - file_start_time:.2f} seconds")
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    final_state_datasets = np.concatenate(final_state_datasets, axis=0)
    full_shower_datasets = np.concatenate(full_shower_datasets, axis=0)

    np.random.shuffle(final_state_datasets)
    np.random.shuffle(full_shower_datasets)

    final_state_output_path = os.path.join(output_dir, 'final_state_dataset.pt')
    full_shower_output_path = os.path.join(output_dir, 'full_shower_dataset.pt')

    torch.save(final_state_datasets, final_state_output_path)
    torch.save(full_shower_datasets, full_shower_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and save datasets.')
    parser.add_argument('input_dir', type=str, help='Path to the input directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    
    args = parser.parse_args()
    
    process_and_save_datasets(args.input_dir, args.output_dir)
