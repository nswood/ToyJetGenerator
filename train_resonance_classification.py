import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Import tqdm

# -- If needed, adjust sys.path or PYTHONPATH so Python can find weaver-core --
# sys.path.append(os.path.expanduser('~/weaver-core'))

from weaver.nn.model.MoG_part_lvl_MLP import MoG_part_lvl_MLP

# ----------------------------
# DATA LOADING UTILS
# ----------------------------
import h5py as h5
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def load_binary_resonance_dataset(path, n_samples=-1):
    with h5.File(path, 'r') as f:
        data = f['data'][:n_samples]
        labels = f['labels'][:n_samples].astype(str)
        four_vectors = f['four_vectors'][:n_samples]
    resonance_labels = labels[:,:,-1]
    padding_mask = resonance_labels != '0'
    binary_resonance_label = (labels != 'None') & (labels != '0')
    binary_resonance_label = binary_resonance_label[:,:,-1]
    
    return data,four_vectors,binary_resonance_label, padding_mask

def train_test_val_split(data, binary_resonance_label, four_vectors, padding_mask, train_percent, val_percent, batch_size):
    n_samples = data.shape[0]
    n_train = int(n_samples * train_percent)
    n_val = int(n_samples * val_percent)
    n_test = n_samples - n_train - n_val

    train_data = torch.tensor(data[:n_train])
    val_data = torch.tensor(data[n_train:n_train + n_val])
    test_data = torch.tensor(data[n_train + n_val:])

    train_labels = torch.tensor(binary_resonance_label[:n_train])
    val_labels = torch.tensor(binary_resonance_label[n_train:n_train + n_val])
    test_labels = torch.tensor(binary_resonance_label[n_train + n_val:])

    train_four_vectors = torch.tensor(four_vectors[:n_train])
    val_four_vectors  = torch.tensor(four_vectors[n_train:n_train + n_val])
    test_four_vectors  = torch.tensor(four_vectors[n_train + n_val:])

    train_padding = torch.tensor(padding_mask[:n_train])
    val_padding = torch.tensor(padding_mask[n_train:n_train + n_val])
    test_padding = torch.tensor(padding_mask[n_train + n_val:])

    train_dataset = TensorDataset(train_data, train_four_vectors, train_labels, train_padding)
    val_dataset = TensorDataset(val_data, val_four_vectors,  val_labels, val_padding)
    test_dataset = TensorDataset(test_data, test_four_vectors, test_labels, test_padding)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def prep_dataloaders(path,n_samples = -1,train_percent=0.8, val_percent=0.1, batch_size=32):
    data, four_vectors, binary_resonance_label, padding_mask = load_binary_resonance_dataset(path, n_samples)
    print('Data shape:', data.shape)
    print('Four vectors shape:', four_vectors.shape)
    print('Binary resonance label shape:', binary_resonance_label.shape)
    print('Padding mask shape:', padding_mask.shape)
    train_loader, val_loader, test_loader = train_test_val_split(data, binary_resonance_label, four_vectors, padding_mask, train_percent, val_percent, batch_size)
    return train_loader, val_loader, test_loader


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.base_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets, mask):
        loss_per_element = self.base_loss(outputs, targets)
        mask_flat = mask.view(-1).float()
        masked_loss = loss_per_element * mask_flat  # Exclude masked elements
        mean_loss = masked_loss.sum() / mask_flat.sum()  # Compute mean loss over unmasked elements
        return mean_loss

class ImportanceRegulatedLoss(nn.Module):
    def __init__(self, base_loss, w_1=0.1):
        super(ImportanceRegulatedLoss, self).__init__()
        self.base_loss = base_loss
        self.w_1 = w_1

    def forward(self, outputs, targets, router_1, router_2, mask):
        # Base loss with masking
        b_loss = self.base_loss(outputs, targets, mask)
        
        # Sum of gate values for each router (importance calculation)
        r_1_summed = torch.sum(router_1, dim=0)
        
        # Coefficient of Variation (CV) calculation
        cv_1 = torch.std(r_1_summed) / (torch.mean(r_1_summed) + 1e-8)
        
        # Importance loss component (from equation in image)
        L_importance_1 = self.w_1 * cv_1 ** 2
        
        # Total loss (base loss + importance regularization)
        total_loss = b_loss + L_importance_1
        
        return total_loss

        # Update the training loop to use the new loss function
def train_model(
    data_path='processed_toy_resonaces_dataset_1_21_25/full_shower_dataset.h5',
    n_samples=1000,
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=10,
    device='cuda'
):

    # 1) Prepare data loaders
    train_loader, val_loader, test_loader = prep_dataloaders(
        path=data_path,
        n_samples=n_samples,
        train_percent=0.8,
        val_percent=0.1,
        batch_size=batch_size
    )

    # 2) Instantiate model
    model = MoG_part_lvl_MLP(input_dim=5, num_classes=2,
                                particle_classification=True,
                                jet_classification=False)

    # Move to GPU if available
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 3) Define optimizer & loss
    base_criterion = MaskedCrossEntropyLoss()
    criterion = ImportanceRegulatedLoss(base_loss=base_criterion)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 4) Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        running_expert_prob = np.zeros(6)  # Initialize array to store running expert probabilities

        # Wrap the training loop with tqdm
        for batch_idx, (data, four_vec, labels, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move everything to device
            data   = data.float().to(device).permute(0,2,1)
            four_vec = four_vec.float().to(device).permute(0,2,1)
            labels = labels.long().to(device)
            mask   = mask.to(device)

            # Forward pass (the model expects x=data, v=four_vec)
            outputs, part_expert_probs = model(x=data, v=four_vec)

            # Compute the loss
            predictions_flat = outputs.view(-1, 2)
            labels_flat = labels.view(-1)
            total_loss = criterion(predictions_flat, labels_flat, part_expert_probs, part_expert_probs, mask)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_expert_prob += part_expert_probs.view(-1).mean(dim=0).cpu().detach().numpy()  # Update running expert probabilities

            # Compute predictions & accuracy
            predicted = torch.argmax(predictions_flat, dim=1)
            correct += (predicted == labels_flat.squeeze()).sum().item()
            total += labels_flat.size(0)

            # Update tqdm description with current average loss and expert probabilities
            tqdm.write(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {running_loss/(batch_idx+1):.4f} - Acc: {100*correct/total:.2f}% - Expert Prob: {running_expert_prob/(batch_idx+1)}")

        # Compute average training loss over the epoch
        train_loss = running_loss / len(train_loader)

        # 5) Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, four_vec, labels, mask in val_loader:
                data   = data.float().to(device).permute(0,2,1)
                four_vec = four_vec.float().to(device).permute(0,2,1)
                labels = labels.long().to(device)
                mask = mask.to(device)

                outputs, part_expert_probs = model(x=data, v=four_vec)
                
                predictions_flat = outputs.view(-1, 2)
                labels_flat = labels.view(-1)
                total_loss = criterion(predictions_flat, labels_flat, part_expert_probs, part_expert_probs, mask)

                val_loss += total_loss.item()

                # Compute predictions & accuracy
                predicted = torch.argmax(predictions_flat, dim=1)
                correct += (predicted == labels_flat.squeeze()).sum().item()
                total += labels_flat.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {100*val_acc:.2f}%")

    # 6) Testing (optional, can be done similarly)
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data, four_vec, labels, mask in test_loader:
            data   = data.float().to(device).permute(0,2,1)
            four_vec = four_vec.float().to(device).permute(0,2,1)
            labels = labels.long().to(device)
            mask = mask.to(device)

            outputs, part_expert_probs = model(x=data, v=four_vec)

            predictions_flat = outputs.view(-1, 2)
            labels_flat = labels.view(-1)
            total_loss = criterion(predictions_flat, labels_flat, part_expert_probs, part_expert_probs, mask)

            test_loss += total_loss.item()

            predicted = torch.argmax(predictions_flat, dim=1)
            correct += (predicted == labels_flat.squeeze()).sum().item()
            total += labels_flat.size(0)

    test_loss /= len(test_loader)
    test_acc = correct_test / total_test
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {100*test_acc:.2f}%")

    return model  # return the trained model


def main():
    trained_model = train_model(
        data_path='processed_toy_resonaces_dataset_1_21_25/full_shower_dataset.h5',
        n_samples=1000,
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=10,
        device='cuda'
    )

if __name__ == "__main__":
    main()
