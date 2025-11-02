# -*- coding: utf-8 -*-

#step03_train_test_split.py




import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from collections import defaultdict
import umap
from math import floor
import matplotlib.pyplot as plt
import seaborn as sns
import os



def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test



"""
Cluster-based Train-Test Split using UMAP + KMeans + Gap Statistic
"""

# ---------------- Utility Functions ---------------- #

def select_test_indices(values, indices, n):
    """Select n indices closest to the local median."""
    local_median = np.median(values[indices])
    sorted_idx = sorted(indices, key=lambda i: abs(values[i] - local_median))
    return sorted_idx[:n]


def recursive_split_large_clusters(data, indices, max_size=7, method='ward'):
    """Recursively split clusters that are too large."""
    if len(indices) <= max_size:
        return [indices]

    subset = data.loc[indices]
    Z = linkage(subset, method=method)
    n_clusters = max(2, len(indices) // max_size)
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    clustered = defaultdict(list)
    for idx, label in zip(indices, labels):
        clustered[label].append(idx)

    result = []
    for group in clustered.values():
        if len(group) > max_size:
            result.extend(recursive_split_large_clusters(data, group, max_size, method))
        else:
            result.append(group)
    return result


def gap_statistic(data, max_k=10, n_references=5, random_state=42):
    """Compute Gap Statistic for KMeans clustering."""
    np.random.seed(random_state)
    gaps, s_k = [], []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(data)
        disp_k = np.log(kmeans.inertia_)

        ref_disps = []
        for _ in range(n_references):
            random_data = np.random.uniform(np.min(data, axis=0), np.max(data, axis=0), size=data.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans_ref.fit(random_data)
            ref_disps.append(np.log(kmeans_ref.inertia_))

        gap = np.mean(ref_disps) - disp_k
        sk = np.std(ref_disps) * np.sqrt(1 + 1 / n_references)
        gaps.append(gap)
        s_k.append(sk)

    optimal_k = next((k + 2 for k in range(len(gaps) - 1)
                      if gaps[k] >= gaps[k + 1] - s_k[k + 1]), max_k)
    print ('optimal_k', optimal_k)
    return optimal_k


# ---------------- Main Function ---------------- #

def cluster_based_train_test_split(data, descriptor_cols, target_cols, test_size=0.2, random_state=42):
    """
    Performs cluster-based train-test split using UMAP + KMeans + Gap Statistic.
    
    Returns X_train, X_test, y_train, y_test similar to sklearn's train_test_split.
    """
    # Ensure output folder exists
    os.makedirs("output/plots", exist_ok=True)

    # --- Preprocess ---
    descriptor_data = data[descriptor_cols].fillna(data[descriptor_cols].mean())
    y = data[target_cols]
    sample_values = y.iloc[:, 0].values  # use first target to guide sampling

    # --- Standardize descriptors ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(descriptor_data)

    # --- UMAP dimensionality reduction ---
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(scaled_data)

    # # Convert embedding to DataFrame
    umap_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])

    # --- Save UMAP embedding plot--- 
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=umap_df, x='UMAP_1', y='UMAP_2', alpha=0.7, s=50)
    plt.title("UMAP Projection")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig("output/plots/umap_embedding.png", dpi=300)
    plt.close()

    # --- Optimal number of clusters via Gap Statistic ---
    best_k = gap_statistic(embedding, max_k=10, n_references=10, random_state=random_state)

    # --- KMeans clustering ---
    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embedding)

    data['Cluster'] = cluster_labels + 1
    umap_df['Cluster'] = cluster_labels + 1

    # --- Save KMeans cluster plot--- 
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=best_k)
    sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='Cluster', palette=palette, data=umap_df, alpha=0.7, s=50)
    plt.title(f"UMAP + KMeans Clusters (k={best_k})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig("output/plots/umap_kmeans_clusters.png", dpi=300, bbox_inches='tight')
    plt.close()


    # --- Compute cluster sizes and test quotas ---
    total_samples = len(data)
    total_test_quota = floor(total_samples * test_size)
    cluster_sizes = umap_df['Cluster'].value_counts().sort_index()
    cluster_test_quotas = (cluster_sizes / total_samples * total_test_quota).round().astype(int)
    test_quota_dic = cluster_test_quotas.to_dict()

    # --- Train/Test selection ---
    train_indices = []
    test_indices = []
    selected_test_set = set()

    for cluster_id in umap_df['Cluster'].unique():
        cluster_indices = umap_df[umap_df['Cluster'] == cluster_id].index.tolist()
        cluster_data = descriptor_data.loc[cluster_indices]
        cluster_test_quota = test_quota_dic[cluster_id]

        test_cluster_indices = []

        if len(cluster_indices) == 1:
            train_indices.extend(cluster_indices)
            remaining_test_needed = cluster_test_quota

        else:
            final_clusters = recursive_split_large_clusters(cluster_data, cluster_indices, max_size=7)
            remaining_test_needed = cluster_test_quota

            for cluster in final_clusters:
                if remaining_test_needed <= 0:
                    train_indices.extend(cluster)
                    continue
                test_sample = select_test_indices(sample_values, cluster, 1)
                test_cluster_indices.append(test_sample[0])
                selected_test_set.add(test_sample[0])
                remaining_test_needed -= 1
                train_indices.extend([i for i in cluster if i not in test_sample])

        test_indices.extend(test_cluster_indices)

    train_indices = [i for i in range(len(sample_values)) if i not in selected_test_set]

    # --- Return split data ---
    X_train = descriptor_data.iloc[train_indices]
    X_test = descriptor_data.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test
