# %% [markdown]
# # Import Libraries

# %%
%matplotlib inline
import imageio
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy import linalg
from random import sample
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# %% [markdown]
# ---
# # Load Data

# %%
dataset = pd.read_csv('tsne_scores.csv')
dataset.shape

# %%
dataset.head()

# %%
dataset.info()

# %%
dataset.isnull().sum()

# %% [markdown]
# ---
# # DBSCAN Parameter Selection
# In this section, the minimum number of points (`minPts`) is set to twice the number of data dimensions. The epsilon (`eps`) value is determined using the kNN method and the knee point identified by `KneeLocator`.

# %%
# Determine minPts and eps for DBSCAN
# Use only feature columns, excluding an existing cluster column if present
features = dataset.drop(columns=['cluster'], errors='ignore')
n_features = features.shape[1]
minPts = 2 * n_features

# Compute k-nearest-neighbor distances (k = minPts)
neigh = NearestNeighbors(n_neighbors=minPts)
nbrs = neigh.fit(features)
distances, indices = nbrs.kneighbors(features)
distances = np.sort(distances[:, -1])  # Distance to the minPts-th nearest neighbor for each point

# Find the elbow point to determine eps
kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve='convex', direction='increasing')
eps = distances[kneedle.knee] if kneedle.knee is not None else np.percentile(distances, 90)  # If no elbow is found, use the 90th percentile
print(f"minPts: {minPts}, eps: {eps}")

# %%
# Plot the k-nearest neighbor distance graph to inspect the knee point
plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{minPts}-th Nearest Neighbor Distance')
plt.title('k-NN Distance Plot for DBSCAN Epsilon Selection')
if kneedle.knee is not None:
    plt.axvline(kneedle.knee, color='r', linestyle='--', label='Knee (Epsilon)')
    plt.legend()
plt.show()

# %% [markdown]
# ---
# # Running DBSCAN and Displaying the Results
# In this section, the clustering is performed using the obtained parameters, and the results are visualized in the feature space.

# %%
# Run DBSCAN with the selected parameters
dbscan = DBSCAN(eps=eps, min_samples=minPts)
labels = dbscan.fit_predict(dataset)

# Add cluster labels to the dataset
dataset['cluster'] = labels

# Show the number of points in each cluster
print(dataset['cluster'].value_counts())

# Plot clustering results in 2D (if the data has two feature dimensions)
if dataset.shape[1] - 1 == 2:
    plt.figure(figsize=(8, 6))
    for cluster in np.unique(labels):
        mask = dataset['cluster'] == cluster
        plt.scatter(
            dataset.loc[mask, dataset.columns[0]],
            dataset.loc[mask, dataset.columns[1]],
            label=f'Cluster {cluster}'
        )
    plt.xlabel(dataset.columns[0])
    plt.ylabel(dataset.columns[1])
    plt.title('DBSCAN Clustering Results')
    plt.legend()
    plt.show()
else:
    print('The data has more than two dimensions. Dimensionality reduction is required for visualization.')


