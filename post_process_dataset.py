import os
import torch
import numpy as np
import argparse
import time
import h5py

def process_dataset(cur_dataset, max_particles=100):
    feature_list = []
    label_list = []
    four_vector_list = []

    for particles in cur_dataset:
        features = []
        four_vectors = []
        labels = []
        for particle in particles:
            kin_features = [particle.mom.e, particle.mom.p_t, particle.mom.m, particle.mom.eta, particle.mom.phi]
            cur_four_vector = [particle.mom.p_x, particle.mom.p_y, particle.mom.p_z, particle.mom.e]
            part_labels = [particle.part_label, particle.part_parent_label, particle.prong_label, particle.resonance_origin]
            features.append(kin_features)
            four_vectors.append(cur_four_vector)
            labels.append(part_labels)

        features = np.array(features)
        four_vectors = np.array(four_vectors)
        labels = np.array(labels)
        sorted_indices = np.argsort(-features[:, 0])
        features = features[sorted_indices]
        four_vectors = four_vectors[sorted_indices]
        labels = labels[sorted_indices]

        if len(features) < max_particles:
            padding = max_particles - len(features)
            features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
            four_vectors = np.pad(four_vectors, ((0, padding), (0, 0)), mode='constant')
            labels = np.pad(labels, ((0, padding), (0, 0)), mode='constant')
        elif len(features) > max_particles:
            features = features[:max_particles]
            four_vectors = four_vectors[:max_particles]
            labels = labels[:max_particles]

        feature_list.append(features)
        four_vector_list.append(four_vectors)
        label_list.append(labels)

    return np.array(feature_list), np.array(four_vector_list), np.array(label_list)

def process_and_save_datasets(input_dir, output_dir):
    final_state_datasets = []
    final_state_four_vectors = []
    final_state_labels = []
    full_shower_datasets = []
    full_shower_four_vectors = []
    full_shower_labels = []
    
    start_time = time.time()
    for i, filename in enumerate(os.listdir(input_dir)):
        file_start_time = time.time()
        
        print(f"Processing file {i + 1}/{len(os.listdir(input_dir))}")
        if filename.endswith('.pt'):
            file_path = os.path.join(input_dir, filename)
            data = torch.load(file_path)
            
            final_state_dataset = data[0]
            full_shower_dataset = data[1]
            
            processed_final_state, final_state_four_vector, final_state_label = process_dataset(final_state_dataset, max_particles=32)
            processed_full_shower, full_shower_four_vector, full_shower_label = process_dataset(full_shower_dataset, max_particles=100)
            
            final_state_datasets.append(processed_final_state)
            final_state_four_vectors.append(final_state_four_vector)
            final_state_labels.append(final_state_label)
            full_shower_datasets.append(processed_full_shower)
            full_shower_four_vectors.append(full_shower_four_vector)
            full_shower_labels.append(full_shower_label)
        
        file_end_time = time.time()
        print(f"Time taken for file {i + 1}: {file_end_time - file_start_time:.2f} seconds")
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    final_state_datasets = np.concatenate(final_state_datasets, axis=0)
    final_state_four_vectors = np.concatenate(final_state_four_vectors, axis=0)
    final_state_labels = np.concatenate(final_state_labels, axis=0)
    full_shower_datasets = np.concatenate(full_shower_datasets, axis=0)
    full_shower_four_vectors = np.concatenate(full_shower_four_vectors, axis=0)
    full_shower_labels = np.concatenate(full_shower_labels, axis=0)
    
    # Shuffle the final state datasets, four vectors, and labels together
    final_state_indices = np.arange(final_state_datasets.shape[0])
    np.random.shuffle(final_state_indices)
    final_state_datasets = final_state_datasets[final_state_indices]
    final_state_four_vectors = final_state_four_vectors[final_state_indices]
    final_state_labels = final_state_labels[final_state_indices]

    # Shuffle the full shower datasets, four vectors, and labels together
    full_shower_indices = np.arange(full_shower_datasets.shape[0])
    np.random.shuffle(full_shower_indices)
    full_shower_datasets = full_shower_datasets[full_shower_indices]
    full_shower_four_vectors = full_shower_four_vectors[full_shower_indices]
    full_shower_labels = full_shower_labels[full_shower_indices]

    final_state_output_path = os.path.join(output_dir, 'final_state_dataset.h5')
    full_shower_output_path = os.path.join(output_dir, 'full_shower_dataset.h5')

    with h5py.File(final_state_output_path, 'w') as f:
        f.create_dataset('data', data=final_state_datasets)
        f.create_dataset('four_vectors', data=final_state_four_vectors)
        encoded_labels = np.array(final_state_labels, dtype='S')
        f.create_dataset('labels', data=encoded_labels)

    with h5py.File(full_shower_output_path, 'w') as f:
        f.create_dataset('data', data=full_shower_datasets)
        f.create_dataset('four_vectors', data=full_shower_four_vectors)
        encoded_labels = np.array(full_shower_labels, dtype='S')
        f.create_dataset('labels', data=encoded_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and save datasets.')
    parser.add_argument('input_dir', type=str, help='Path to the input directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    process_and_save_datasets(args.input_dir, args.output_dir)
