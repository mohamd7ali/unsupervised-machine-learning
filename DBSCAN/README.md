## DBSCAN Clustering on t-SNE Data


This project demonstrates the use of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm to cluster data that has been reduced to two dimensions using t-SNE. The workflow includes parameter selection, clustering, and visualization of results.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Analysis Workflow](#analysis-workflow)
- [How to Run](#how-to-run)
- [Results](#results)
- [Author](#author)

---

### Project Structure

- `DBSCAN.ipynb` — Jupyter notebook containing the full analysis, parameter selection, clustering, and visualization code.
- `tsne_scores.csv` — Input dataset containing two-dimensional t-SNE features.
- `README.md` — Project documentation (this file).

---

### Dataset

The dataset (`tsne_scores.csv`) consists of two columns:

- `t-SNE-1`: First t-SNE feature
- `t-SNE-2`: Second t-SNE feature

Each row represents a data point in the reduced feature space.

---

### Analysis Workflow

1. **Data Loading & Inspection**
	- The dataset is loaded and checked for missing values and data types.

2. **Parameter Selection for DBSCAN**
	- The minimum number of points (`minPts`) is set to twice the number of features (here, 4).
	- The epsilon (`eps`) parameter is determined using the k-nearest neighbor (kNN) distance plot and the `KneeLocator` algorithm to find the optimal knee point.

3. **Clustering**
	- DBSCAN is applied using the selected parameters.
	- Cluster labels are assigned to each data point.

4. **Visualization**
	- The results are visualized in 2D, with each cluster shown in a different color.

---

### How to Run

1. Open `DBSCAN.ipynb` in Jupyter Notebook or VS Code.
2. Run all cells sequentially to reproduce the analysis and plots.
3. Ensure the following Python packages are installed:
	- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `kneed`, `scipy`, `imageio`

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib scikit-learn kneed scipy imageio
```

---

### Results

- The notebook outputs the number of points in each cluster and visualizes the clusters in the t-SNE feature space.
- The kNN distance plot helps justify the choice of epsilon for DBSCAN.

---

### **Author:** 
Mohammad Ali Etemadi Naeen
