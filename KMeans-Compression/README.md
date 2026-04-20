# Image Compression with K-means Clustering

This project demonstrates how to use the K-means clustering algorithm for image compression. By reducing the number of unique colors in an image, K-means enables efficient compression while preserving visual quality. The project is implemented in Python and is organized as a Jupyter Notebook for step-by-step exploration and visualization.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [References](#references)

## Overview
K-means clustering is an unsupervised machine learning algorithm that partitions data into $K$ distinct clusters based on feature similarity. In this project, K-means is applied to the RGB values of an image's pixels, grouping similar colors together and reconstructing the image using only the cluster centroids. This results in a compressed image with a reduced color palette.

## Project Structure

```
unsupervised-machine-learning/
│
├── KMeans-Compression.ipynb   # Main Jupyter Notebook
├── data.mat                   # Dataset for Clustering
├── cristiano-ronaldo.jpg      # Test Image
├── README.md                  # Project Documentation
```

## Requirements

To run this project, you need the following Python packages:

- numpy
- matplotlib
- scipy
- imageio

You can install the required packages using pip:

```bash
pip install numpy matplotlib scipy imageio
```

## Usage

1. **Clone the repository** and navigate to the project directory.
2. **Open** the `KMeans-Compression.ipynb` notebook in Jupyter or VS Code.
3. **Run the notebook cells** sequentially to:
	- Load and visualize the dataset
	- Implement and test K-means clustering
	- Apply K-means to image data for compression
	- Visualize the original and compressed images

> **Note:** Replace the image filename in the notebook if you wish to use your own image for compression.

## Methodology

The notebook is organized into the following sections:

1. **Implementing K-means:**
	- Functions for finding closest centroids and updating centroid positions.
2. **K-means on Example Dataset:**
	- Demonstrates clustering on a simple 2D dataset.
3. **Random Initialization:**
	- Shows the effect of different centroid initializations.
4. **Image Compression with K-means:**
	- Loads an image, reshapes pixel data, applies K-means, and reconstructs the compressed image.

## Results

- The notebook visualizes both the original and compressed images.
- Compression is achieved by reducing the number of colors (e.g., to 16) while maintaining perceptual quality.
- The effect of different initializations and cluster counts can be explored interactively.

---
**Author:** 
Mohammad Ali Etemadi Naeen
