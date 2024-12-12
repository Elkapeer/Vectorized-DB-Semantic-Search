import torch
import numpy as np
import os
import pickle
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm


class GPUKMeans:
    def __init__(self, n_clusters: int, max_iters: int = 100, tolerance: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, batch_size: int = 6000):
        assert X.shape[1] == 70, f"Expected input dimension of 70, got {X.shape[1]}"
        num_samples = X.shape[0]

        print(f"Starting GPU KMeans clustering with cosine similarity")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Input shape: {X.shape}")
        print(f"Using device: {self.device}")
        print(f"Batch size: {batch_size}")

        # Convert input to PyTorch tensor and move to GPU
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Initialize centroids randomly
        print("Initializing centroids...")
        idx = torch.randperm(num_samples, device=self.device)[:self.n_clusters]
        self.cluster_centers_ = X_tensor[idx].clone()

        # Normalize centroids
        self.cluster_centers_ = torch.nn.functional.normalize(self.cluster_centers_, p=2, dim=1)

        prev_centroids = torch.zeros_like(self.cluster_centers_)
        n_batches = (num_samples + batch_size - 1) // batch_size

        for iteration in tqdm(range(self.max_iters), desc="KMeans iterations"):
            all_labels = []

            # Process data in batches
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                batch = X_tensor[i:batch_end]

                # Compute cosine similarity
                # Since vectors are normalized, dot product equals cosine similarity
                similarities = torch.mm(batch, self.cluster_centers_.t())

                # Convert similarities to labels (maximum similarity = minimum distance)
                batch_labels = torch.argmax(similarities, dim=1)
                all_labels.append(batch_labels)

            # Combine all batch labels
            self.labels_ = torch.cat(all_labels)

            # Update centroids
            prev_centroids.copy_(self.cluster_centers_)

            # Create a new tensor for updated centroids
            new_centroids = torch.zeros_like(self.cluster_centers_)

            for k in range(self.n_clusters):
                cluster_points = X_tensor[self.labels_ == k]
                if len(cluster_points) > 0:
                    # Average the points
                    new_centroids[k] = cluster_points.mean(dim=0)

            # Normalize all centroids in one go (more efficient)
            self.cluster_centers_ = torch.nn.functional.normalize(new_centroids, p=2, dim=1)

            # Check for convergence using cosine similarity
            similarity = torch.sum(self.cluster_centers_ * prev_centroids, dim=1).mean()
            if (1 - similarity) < self.tolerance:
                print(f"\nConverged after {iteration + 1} iterations")
                break

        # Convert back to numpy
        self.cluster_centers_ = self.cluster_centers_.cpu().numpy()
        self.labels_ = self.labels_.cpu().numpy()

        # Calculate and print cluster statistics
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        print("\nCluster size statistics:")
        print(f"Min cluster size: {counts.min()}")
        print(f"Max cluster size: {counts.max()}")
        print(f"Average cluster size: {counts.mean():.2f}")
        print(f"Empty clusters: {self.n_clusters - len(unique_labels)}")

        return self