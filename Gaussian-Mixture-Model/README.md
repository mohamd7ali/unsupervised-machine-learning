# Gaussian Mixture Model (GMM) Clustering

This module demonstrates unsupervised clustering using the Gaussian Mixture Model (GMM) and compares its performance with KMeans on a synthetic dataset. The project is implemented in Python and visualized using Jupyter Notebook.


## Overview

Clustering is a fundamental task in unsupervised machine learning, aiming to group similar data points together. This project:
- Generates a synthetic dataset with four clusters.
- Applies KMeans clustering as a baseline.
- Applies Gaussian Mixture Model (GMM) clustering for probabilistic cluster assignment.
- Visualizes and compares the results of both algorithms.


## Features
- **Synthetic Data Generation:** Uses `sklearn.datasets.make_blobs` to create a dataset with clear cluster structure.
- **KMeans Clustering:** Provides a baseline clustering using the KMeans algorithm.
- **Gaussian Mixture Model:** Fits a GMM to the data, allowing for soft (probabilistic) cluster assignments.
- **Visualization:**
  - Scatter plots of clustering results.
  - Ellipse overlays to represent GMM component covariances.
  - Point sizes proportional to cluster membership probabilities.
- **Reusable Functions:**
  - `draw_ellipse`: Visualizes covariance ellipses for GMM components.
  - `plot_gmm`: Plots GMM clustering results with ellipses.


## File Structure

- `gaussian-mixture-model.ipynb` — Main Jupyter Notebook with code, explanations, and visualizations.
- `README.md` — Project documentation (this file).


## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Python packages:
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd unsupervised-machine-learning/Gaussian-Mixture-Model
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib seaborn scikit-learn
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook gaussian-mixture-model.ipynb
   ```


## Usage

1. Open `gaussian-mixture-model.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the notebook cells sequentially to:
   - Import libraries
   - Generate and visualize the dataset
   - Apply KMeans and GMM clustering
   - Visualize and compare results


## Results
- KMeans provides hard cluster assignments and works well for spherical clusters.
- GMM provides soft assignments (probabilities) and models clusters as Gaussian distributions, capturing more complex cluster shapes.
- Visualizations include both cluster assignments and GMM component ellipses.

## **Author:** 
Mohammad Ali Etemadi Naeen
