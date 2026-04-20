# %% [markdown]
# # Image Compression with K-means Clustering

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from random import sample
import scipy.misc
import matplotlib.cm as cm
from scipy import linalg
import imageio

# %% [markdown]
# ## 1: Implementing K-means

# %% [markdown]
# ### 1.1: Finding closest centroids

# %%
datafile = 'data.mat' 
mat = scipy.io.loadmat( datafile )
X = mat['X']

# %%
#Choose the number of centroids... K = 3
K = 3
initial_centroids = np.array([[3,3],[6,2],[8,5]]) # example initial value of centroids

# %%
# Visualizing the data, please don't change the code below!
def plotData(myX,mycentroids,myidxs = None):
    """
    Fucntion to plot the data and color it accordingly.
    myidxs should be the latest iteraction index vector
    mycentroids should be a vector of centroids, one per iteration
    """
    
    colors = ['b','g','gold','darkorange','salmon','olivedrab']
    
    assert myX[0].shape == mycentroids[0][0].shape
    assert mycentroids[-1].shape[0] <= len(colors)

    #If idxs is supplied, divide up X into colors
    if myidxs is not None:
        assert myidxs.shape[0] == myX.shape[0]
        subX = []
        for x in range(mycentroids[0].shape[0]):
            subX.append(np.array([myX[i] for i in range(myX.shape[0]) if myidxs[i] == x]))
    else:
        subX = [myX]
        
    fig = plt.figure(figsize=(7,5))
    for x in range(len(subX)):
        newX = subX[x]
        plt.plot(newX[:,0],newX[:,1],'o',color=colors[x],
                 alpha=0.75, label='Data Points: Cluster %d'%x)
    plt.xlabel('x1',fontsize=14)
    plt.ylabel('x2',fontsize=14)
    plt.title('Plot of X Points',fontsize=16)
    plt.grid(True)

    #Drawing a history of centroid movement
    tempx, tempy = [], []
    for mycentroid in mycentroids:
        tempx.append(mycentroid[:,0])
        tempy.append(mycentroid[:,1])
    
    for x in range(len(tempx[0])):
        plt.plot(tempx, tempy, 'rx--', markersize=8)

    leg = plt.legend(loc=4, framealpha=0.5)

# %%
plotData(X,[initial_centroids])

# %%
def distSquared(point1, point2):
    # this function should return the squared distance between two points
    return np.sum((point1 - point2) ** 2)

# %%
def findClosestCentroids(myX, mycentroids):
    """
    Function takes in the (m,n) X matrix
    (where m is the # of points, n is # of features per point)
    and the (K,n) centroid seed matrix
    (where K is the # of centroids (clusters)
    and returns a (m,1) vector of cluster indices 
    per point in X (0 through K-1)
    """
    idxs = np.zeros((myX.shape[0],1))
    
    #Loop through each data point in X
    for x in range(idxs.shape[0]):
        # Get the current point
        point = myX[x]
        
        #Then compare this point to each centroid,
        #Keep track of shortest distance and index of shortest distance
        mindist, idx = 9999999, 0
        for i in range(mycentroids.shape[0]):
            # Calculate distance to this centroid
            dist = distSquared(point, mycentroids[i])
            if dist < mindist:
                mindist = dist
                idx = i
            
        #With the best index found, modify the result idx vector
        idxs[x] = idx
        
    return idxs

# %%
idxs = findClosestCentroids(X, initial_centroids)

print(idxs[:3].astype(int).ravel())


# %%
plotData(X,[initial_centroids],idxs)

# %% [markdown]
# ### 1.2: Computing centroid means

# %%
def computeCentroids(myX, myidxs):
    """
    Function takes in the X matrix and the index vector
    and computes a new centroid matrix.
    """
    myidxs = myidxs.astype(int).ravel()
    centroids = np.zeros((K, myX.shape[1]))
    
    for k in range(K):
        points = myX[myidxs == k]
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
    
    return centroids

# %% [markdown]
# ## 2: K-means on example dataset

# %%
def runKMeans(myX, initial_centroids, K, n_iter):
    centroid_history = []
    current_centroids = initial_centroids
    for myiter in range(n_iter):
        centroid_history.append(current_centroids.copy())
        idxs = findClosestCentroids(myX, current_centroids)
        current_centroids = computeCentroids(myX, idxs)
    return idxs, centroid_history

# %%
idxs, centroid_history = runKMeans(X, initial_centroids, K, 10)

# %%
plotData(X,centroid_history,idxs)

# %% [markdown]
# ## 3: Random initialization

# %%
def chooseKRandomCentroids(myX, K):
    rand_indices = sample(range(0,myX.shape[0]),K)
    return np.array([myX[i] for i in rand_indices])

# %%
# Let's choose random initial centroids and see the resulting 
for x in range(3):
    rand_centroids = chooseKRandomCentroids(X, K)
    idxs, centroid_history = runKMeans(X, rand_centroids, K, 10)
    plotData(X, centroid_history, idxs)

# %% [markdown]
# ## 4: Image compression with K-means

# %% [markdown]
# ### 4.1: K-means on pixels

# %%
datafile = '' 

# This creates a three-dimensional matrix A whose first two indices 
# identify a pixel position and whose last index represents red, green, or blue.
A = imageio.imread('cristiano-ronaldo.jpg')  # replace with your image filename if different

print("A shape is ", A.shape)
dummy = plt.imshow(A)

# %%
# Divide every entry in A by 255 so all values are in the range of 0 to 1
A = A / 255.

# Unroll the image to shape (16384,3) (16384 is 128*128)
A_unrolled = A.reshape(-1, 3)

# Run k-means on this data, forming 16 clusters, with random initialization
myK = 16
rand_centroids = chooseKRandomCentroids(A_unrolled, myK)
idxs, centroid_history = runKMeans(A_unrolled, rand_centroids, myK, 10)

# %%
# Now I have 16 centroids, each representing a color.
# Let's assign an index to each pixel in the original image dictating
# which of the 16 colors it should be
final_centroids = centroid_history[-1]
idxs = findClosestCentroids(A_unrolled, final_centroids)

# %%
final_centroids = centroid_history[-1]
# Now loop through the original image and form a new image
# that only has 16 colors in it
final_image = np.zeros((idxs.shape[0],3))
for x in range(final_image.shape[0]):
    final_image[x] = final_centroids[int(idxs[x])]

# %%
# Display the Original image and the Compressed image with 16 colors 
plt.figure()
dummy = plt.imshow(A)
plt.title('Original Image')
plt.figure()
final_image = final_centroids[idxs.astype(int).ravel()]
dummy = plt.imshow(final_image.reshape(A.shape))
plt.title('Compressed Image with 16 colors')


