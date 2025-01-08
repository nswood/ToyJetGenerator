import os
import h5py
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd


# Parameters
data_dir = 'CLR_data'  # Directory where the data is stored
input_shape = 135       # Target shape after padding
batch_size = 32
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Data from the 'test_data' Directory with Zero Padding
def load_data(data_dir, max_length):
    data = []
    aug_data = []
    labels = []

    # Loop through the files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".h5"):
            # Extract class label (jet prong type) from filename
            class_label = int(filename.split('_')[1])

            # Load HDF5 file
            with h5py.File(os.path.join(data_dir, filename), 'r') as f:
                # Extract 'data' and 'aug_data'
                data_batch = f['data'][:]
                aug_data_batch = f['aug_data'][:]

                # Zero-pad the data to the fixed length
                data_batch_padded = [np.pad(sample, (0, max_length - len(sample)), 'constant') if len(sample) < max_length else sample[:max_length] for sample in data_batch]
                aug_data_batch_padded = [np.pad(sample, (0, max_length - len(sample)), 'constant') if len(sample) < max_length else sample[:max_length] for sample in aug_data_batch]

                # Store the batches and labels
                data.append(data_batch_padded)
                aug_data.append(aug_data_batch_padded)
                labels.extend([class_label] * len(data_batch_padded))  # Label each sample with the corresponding class

    # Convert lists to numpy arrays
    data = np.concatenate(data, axis=0)
    aug_data = np.concatenate(aug_data, axis=0)
    labels = np.array(labels)

    # Convert to torch tensors
    return torch.tensor(data, dtype=torch.float32), torch.tensor(aug_data, dtype=torch.float32), labels

# 3. Custom Dataset Class for PyTorch
class JetDataset(Dataset):
    def __init__(self, data, aug_data, labels):
        self.data = data
        self.aug_data = aug_data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.aug_data[idx], self.labels[idx]

# 4. SimCLR Loss Function (NT-Xent Loss)
class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # Concatenate augmented pairs
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)  # Positive pairs: (z_i, z_j)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_pairs = torch.cat([sim_i_j, sim_j_i], dim=0)

        loss = -torch.log(torch.exp(positive_pairs) / torch.exp(sim).sum(dim=1)).mean()

        return loss

# 5. Define the DNN Model for Embeddings
class SimCLRModel(nn.Module):
    def __init__(self, input_dim, projection_dim=3):
        super(SimCLRModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=-1)

# 6. Data Loading and Preprocessing
data, aug_data, labels = load_data(data_dir, input_shape)

# Convert to PyTorch Dataset
dataset = JetDataset(data, aug_data, labels)

# Split into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 7. Training Loop with SimCLR Loss
def train_model(model, train_loader, optimizer, loss_fn, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data_batch, aug_data_batch, _ in train_loader:
            data_batch = data_batch.to(device)
            aug_data_batch = aug_data_batch.to(device)

            # Forward pass through the model
            z_i = model(data_batch)
            z_j = model(aug_data_batch)

            # Compute SimCLR loss
            loss = loss_fn(z_i, z_j)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader)}")

# 8. Visualize Embedding Space
def visualize_embeddings(model, test_loader, labels, output_dir='visualizations'):
    model.eval()
    embeddings = []
    targets = []

    with torch.no_grad():
        for data_batch, _, label_batch in test_loader:
            data_batch = data_batch.to(device)
            z_i = model(data_batch)
            embeddings.append(z_i.cpu().numpy())
            targets.append(label_batch.numpy())

        embeddings = np.concatenate(embeddings, axis=0)[0:5000]
        targets = np.concatenate(targets, axis=0)[0:5000]

        pca = PCA(n_components=2)

        embeddings_2d_pca = pca.fit_transform(embeddings)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        unique_targets = np.unique(targets)
        plt.figure(figsize=(8, 8))
        # Plot each target group with its own color
        for i, target in enumerate(unique_targets):
            mask = targets == target  # Filter for each unique target
            plt.scatter(embeddings_2d_pca[mask, 0], embeddings_2d_pca[mask, 1], 
                        alpha=0.5, label=f'{target} Prong Jet')
        plt.legend()
        plt.title('PCA Embedding Space (Scatter Plot)')
        plt.savefig(os.path.join(output_dir, 'embedding_pca_scatter.png'))
        plt.close()
        
        # Colors for each unique target, ensuring consistency
        colors = plt.get_cmap('tab10')(np.arange(len(unique_targets)))

        plt.figure(figsize=(8, 8))

        # Plot the contour for each target group with consistent colors
        for i, target in enumerate(unique_targets):
            mask = targets == target  # Filter for each unique target
            data = embeddings_2d_pca[mask]  # PCA points for this target

            # Perform kernel density estimation
            kde = gaussian_kde(data.T)

            # Create grid to evaluate kde
            x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
            y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
            X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

            # Evaluate kde on the grid points
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

            # Plot contour with a consistent color for each prong type
            plt.contour(X, Y, Z, levels=3, colors=[colors[i]], alpha=0.7)

        # Add a legend with consistent colors for each prong type
        for i, target in enumerate(unique_targets):
            plt.scatter([], [], color=colors[i], label=f'{target} Prong Jet')  # Invisible points for legend only

        plt.legend(title="Jet Types", loc='upper left')
        plt.title('PCA Embedding Space (Contour Plot)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'embedding_pca_contour.png'))
        plt.close()



        # Corner plot (pair plot) for raw embeddings
#         df = pd.DataFrame(embeddings)
#         df['Prongs'] = targets
#         sns.pairplot(df, hue='Prongs', palette='tab10', diag_kind='kde')
# #         plt.title('Corner Plot of Raw Embedding Space')
#         plt.savefig(os.path.join(output_dir, 'embedding_raw_scatter_corner.png'))
#         plt.close()
        
        df = pd.DataFrame(embeddings)
        df.columns = [f'Latent Dim {i+1}' for i in range(df.shape[1])]
        df['Prongs'] = targets
        ax = sns.pairplot(df, hue='Prongs', kind="kde", palette='tab10', diag_kind='kde')
#         ax = sns.pairplot(df, hue='Prongs', kind="scatter", palette='tab10', diag_kind='kde')
        ax._legend.remove()
        ax.add_legend(title="Jet Prongs", bbox_to_anchor=(0.5, -0.05), loc='center', ncol=4, frameon=False,title_fontsize=20)
        plt.setp(ax._legend.get_texts(), fontsize='25')
        plt.setp(ax._legend.get_title(), fontsize='20')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'embedding_raw_contour_corner.png'),bbox_inches='tight')
        plt.close()


# 9. Initialize and Train the Model
model = SimCLRModel(input_dim=input_shape).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = SimCLRLoss()

train_model(model, train_loader, optimizer, loss_fn, epochs)

# 10. Visualize embeddings for test dataset
visualize_embeddings(model, test_loader, labels, output_dir='visualizations')
