# Unsupervised Machine Learning Project

This repository provides hands-on implementations and visualizations of popular unsupervised machine learning algorithms, including Hierarchical Clustering, DBSCAN, and K-Means for data compression. The project is structured as a collection of Jupyter Notebooks, each focusing on a specific algorithm, with clear explanations, code, and visual outputs.

## Project Structure

- **DBSCAN/**
	- `DBSCAN.ipynb`: Implementation and visualization of the DBSCAN clustering algorithm.
	- `tsne_scores.csv`: t-SNE scores for dimensionality reduction and visualization.
	- `README.md`: Details specific to DBSCAN usage and results.

- **KMeans-Compression/**
	- `KMeans-Compression.ipynb`: Demonstrates K-Means clustering for data compression tasks.
	- `data.mat`: Example dataset used in the notebook.
	- `README.md`: Details specific to K-Means compression.

- **Hierarchical-Clustering**
	- `HierarchicalClustering.ipynb`: Comprehensive notebook on hierarchical clustering, including dendrograms, silhouette analysis, and application to the Iris dataset.
	- `README.md`: Details specific to hierarchical clustering.

## Features

- **Step-by-step explanations** for each algorithm, including mathematical background and practical considerations.
- **Visualization** of clustering results, dendrograms, and silhouette scores to aid understanding.
- **Application to real datasets** such as the Iris dataset.
- **Well-documented code** for easy learning and experimentation.

## Getting Started

1. **Clone the repository:**
	 ```bash
	 git clone https://github.com/yourusername/unsupervised-machine-learning.git
	 cd unsupervised-machine-learning
	 ```
2. **Install dependencies:**
	 - Recommended: Use a Python virtual environment.
	 - Required packages include `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `scipy`.
	 - Install with pip:
		 ```bash
		 pip install -r requirements.txt
		 ```
	 - Or manually:
		 ```bash
		 pip install numpy pandas scikit-learn matplotlib scipy
		 ```
3. **Open the notebooks:**
	 - Use Jupyter Notebook or JupyterLab:
		 ```bash
		 jupyter notebook
		 ```
	 - Navigate to the desired notebook in the browser interface.

## Notebooks Overview

- **DBSCAN.ipynb**: Explores density-based clustering, parameter selection, and visualizes clusters in 2D.
- **KMeans-Compression.ipynb**: Applies K-Means for vector quantization and data compression, with practical examples.
- **HierarchicalClustering.ipynb**: Covers agglomerative and divisive clustering, dendrogram interpretation, and silhouette analysis.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## **Author:** 
Mohammad Ali Etemadi Naeen