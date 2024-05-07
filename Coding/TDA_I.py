---
title: "Topological Data Analysis Part I"
subtitle: "Clustering - Coding"
author: "Neeraj Namani"
output: pdf_document
---

### Exercise 5.18

1. Write a Python function GaussianClusters(k,n) which takes as input a number k, and integer n>0, and does the following:

- Compute an i.i.d. sample $c_1,\ldots,c_k $ from the unit square $[0,1]^2$.

- For each $i \in \{1, \ldots,k\}$, compute an i.i.d. sample $X_i$ of n points from a Gaussian distribution with mean $c_i$ and covariance matrix

\[
 = \begin{bmatrix}
0.1 & 0 \\
0 & 0.1
\end{bmatrix}
\]

- Output $X = \bigcup_{i=1}^{k}X_i$ as a list. $\\$


```{python}
import numpy as np

def GaussianClusters(k, n):
    """
    Generate k Gaussian clusters with n points in each cluster.
    
    Parameters:
        k (int): The number of clusters.
        n (int): The number of points per cluster.
        
    Returns:
        list: A list of points representing all the clusters.
    """
    # Initialize an empty list to store all the points from all clusters
    all_points = []
    
    # Generate k random centers within the unit square [0, 1]²
    centers = np.random.rand(k, 2)
    
    # Covariance matrix for each Gaussian distribution
    cov = np.array([[0.1, 0], [0, 0.1]])
    
    # Generate n points around each center with Gaussian distribution
    for center in centers:
        # Sample n points from a Gaussian distribution centered at `center`
        cluster_points = np.random.multivariate_normal(center, cov, n)
        # Append the points of this cluster to the overall list
        all_points.extend(cluster_points)
    
    return all_points

# Example usage:
clusters = GaussianClusters(3, 50)  # Generate 3 clusters with 50 points each

# Convert to numpy array for easier slicing
clusters_array = np.array(clusters)

# Print the first 10 points
print("Points from the generated clusters:")
for point in clusters_array:
    print(point)

```


2. Run this function with n = 100 and k = 3, and plot the output. $\\$

```{python}
import numpy as np
import matplotlib.pyplot as plt

def GaussianClusters(k, n):
    """
    Generate k Gaussian clusters with n points in each cluster.
    
    Parameters:
        k (int): The number of clusters.
        n (int): The number of points per cluster.
        
    Returns:
        list: A list of points representing all the clusters.
    """
    # Initialize an empty list to store all the points from all clusters
    all_points = []
    
    # Generate k random centers within the unit square [0, 1]²
    centers = np.random.rand(k, 2)
    
    # Covariance matrix for each Gaussian distribution
    cov = np.array([[0.1, 0], [0, 0.1]])
    
    # Generate n points around each center with Gaussian distribution
    for center in centers:
        # Sample n points from a Gaussian distribution centered at `center`
        cluster_points = np.random.multivariate_normal(center, cov, n)
        # Append the points of this cluster to the overall list
        all_points.extend(cluster_points)
    
    return np.array(all_points), centers

# Run the function with n = 100 and k = 3
k, n = 3, 100
points, centers = GaussianClusters(k, n)

# Plot the generated clusters
plt.figure(figsize=(8, 8))
plt.scatter(points[:, 0], points[:, 1], c='blue', marker='o', s=10, alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Cluster Centers')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title(f'Gaussian Clusters with k={k} and n={n} points per cluster')
plt.legend()
plt.grid(True)
plt.show()

```
This was the plot for the Gaussian clusters generated with n = 100 points and k = 3 clusters. The blue dots represent the data points of the clusters, while the red 'x' markers show the centers of these clusters. The distribution is based on the Gaussian distribution you specified with a mean at the cluster centers and a covariance matrix (Given in the Exercise 5.18). $\\$

### Exercise 5.19

1. Implement Lloyd's algorithm from scratch in Python, choosing the initial cluster centers randomly from the set X. (Make sure the cluster centers are distinct, i.e., chosen randomly $without$ replacement.) $\\$

```{python}
import numpy as np

def initialize_centers(X, k):
    """ Randomly initializes k distinct cluster centers from the dataset X. """
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centers):
    """ Assigns each point in X to the nearest cluster center. """
    distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centers(X, labels, k):
    """ Updates the cluster centers based on the current cluster assignment. """
    new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centers

def lloyds_algorithm(X, k, max_iter=100, tol=1e-4):
    """ Implements Lloyd's Algorithm for k-means clustering. """
    centers = initialize_centers(X, k)
    for _ in range(max_iter):
        labels = assign_clusters(X, centers)
        new_centers = update_centers(X, labels, k)
        # Check for convergence
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    return centers, labels

# Example usage
np.random.seed(42)  # For reproducibility
X = np.random.rand(300, 2)  # Generate some random data
k = 3  # Number of clusters
centers, labels = lloyds_algorithm(X, k)

# Plotting the results
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')
plt.title('K-Means Clustering with Lloyd\'s Algorithm')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()
```

2. Compute a data set $X$ as in Exercise 5.18 (ii). (Make sure that the $c_i$ are chosen far enough apart so that a plot reveals clear three-cluster structure in $X$.) $\\$

To create a dataset $X$ as described in Exercise 5.18 (ii), where we need a clear three-cluster structure, it is important to ensure that the centers $c_i$ of these clusters are chosen to be far enough apart. $\\$

Given the constraints and goals, we will use Gaussian clusters for this dataset with k = 3 and n = 100 for each cluster.$\\$

```{python}
import numpy as np
import matplotlib.pyplot as plt

def GaussianClusters(k, n, centers, cov):
    """
    Generate k Gaussian clusters each with n points around specified centers.
    
    Parameters:
        k (int): Number of clusters.
        n (int): Number of points per cluster.
        centers (np.array): An array of centers for the clusters.
        cov (np.array): Covariance matrix for the Gaussian distribution of each cluster.
    
    Returns:
        np.array: An array of points representing all the clusters.
    """
    all_points = []
    
    # Generate n points around each center with Gaussian distribution
    for center in centers:
        cluster_points = np.random.multivariate_normal(center, cov, n)
        all_points.extend(cluster_points)
    
    return np.array(all_points)

# Define the parameters
k, n = 3, 100
# Define centers far apart within the unit square
centers = np.array([[0.2, 0.2], [0.5, 0.8], [0.8, 0.3]])
covariance_matrix = np.array([[0.01, 0], [0, 0.01]])  # Small covariance 

# Generate the clusters
X = GaussianClusters(k, n, centers, covariance_matrix)

# Plotting the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
plt.title('Gaussian Clusters with Clear Three-Cluster Structure')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()
```

3. Taking k = 3, run your implementation of Lloyd's algorithm on $X$ 20 times (i.e., for 20 different random choices of initial cluster centers). For each iteration, compute the cost of the clustering. What are the lowest, highest, and mean value of the cost among the 20 runs? $\\$

```{python}
import numpy as np

def lloyds_algorithm(X, k, max_iter=100, tol=1e-4):
    """ Implements Lloyd's Algorithm for k-means clustering. """
    def initialize_centers(X, k):
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]

    def assign_clusters(X, centers):
        distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update_centers(X, labels, k):
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        return new_centers

    def compute_cost(X, labels, centers):
        cost = 0
        for i in range(k):
            cluster_points = X[labels == i]
            cost += ((cluster_points - centers[i])**2).sum()
        return cost

    # Randomly initialize the centers
    centers = initialize_centers(X, k)
    labels = assign_clusters(X, centers)
    for _ in range(max_iter):
        new_centers = update_centers(X, labels, k)
        new_labels = assign_clusters(X, new_centers)
        # Check for convergence
        if np.array_equal(labels, new_labels):
            break
        centers = new_centers
        labels = new_labels

    # Compute final cost
    final_cost = compute_cost(X, labels, centers)
    return centers, labels, final_cost

# Define the parameters and create the dataset
k, n = 3, 100
centers = np.array([[0.2, 0.2], [0.5, 0.8], [0.8, 0.3]])
covariance_matrix = np.array([[0.01, 0], [0, 0.01]])
X = GaussianClusters(k, n, centers, covariance_matrix)

# Run Lloyd's algorithm 20 times
costs = []
for _ in range(20):
    _, _, cost = lloyds_algorithm(X, k)
    costs.append(cost)

# Analyzing the costs
lowest_cost = min(costs)
highest_cost = max(costs)
mean_cost = np.mean(costs)

lowest_cost, highest_cost, mean_cost

```

The results indicate that the initial selection of cluster centers can significantly affect the performance and outcome of the k-means clustering algorithm, as shown by the variation in costs across different runs. The lowest cost suggests a very efficient clustering that closely matches the inherent structure of the dataset, while the highest cost likely represents a less optimal initial positioning of centers. $\\$


4. Plot the clusterings yielding the lowest and highest values of cost, using colors to distinguish between the different clusters. Use two separate plots for the two clusterings. $\\$


```{python}
import matplotlib.pyplot as plt

# Running Lloyd's algorithm 20 times to store both the costs and clustering results 
results = []
for _ in range(20):
    centers, labels, cost = lloyds_algorithm(X, k)
    results.append((centers, labels, cost))

# Sorting results by cost to easily access the lowest and highest cost clusterings
results_sorted = sorted(results, key=lambda x: x[2])
lowest_cost_result = results_sorted[0]
highest_cost_result = results_sorted[-1]

# Function to plot the clusters
def plot_clusters(X, centers, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o',
    edgecolor='k', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', 
    s=100, label='Centers')
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting the lowest and highest cost clusterings
plot_clusters(X, lowest_cost_result[0], lowest_cost_result[1],
f'Lowest Cost Clustering (Cost: {lowest_cost_result[2]:.2f})')
plot_clusters(X, highest_cost_result[0], highest_cost_result[1],
f'Highest Cost Clustering (Cost: {highest_cost_result[2]:.2f})')

```

5. Compute a 3-means clustering of $K$ using scikit-learn's implementation of k-means in Python. Again, compute the cost, and plot the resulting clustering. How do the results compare? $\\$

```{python}
from sklearn.cluster import KMeans
import warnings

# Suppress specific warnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module='threadpoolctl')

# Using scikit-learn's KMeans to perform the clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
sklearn_labels = kmeans.labels_
sklearn_centers = kmeans.cluster_centers_

# Compute the cost for the sklearn k-means clustering
sklearn_cost = np.sum((X - sklearn_centers[sklearn_labels])**2)

# Plot the results from sklearn's KMeans
plot_clusters(X, sklearn_centers, sklearn_labels,
f'Scikit-learn K-Means Clustering (Cost: {sklearn_cost:.2f})')

# Return the sklearn cost for comparison
sklearn_cost

```

- Cost: The Cost from scikit-learn's k-means is very close to the lowest cost we obtained in the custom implementation (also around 5.55). This indicates that scikit-learn's implementation is efficient and capable of finding near-optimal solutions for the clustering problem. $\\$

- Clustering Quality: The visual inspection of the clustering suggests that scikit-learn's implementation has successfully grouped the data into distinct and well-separated clusters, similar to the best runs of the custom implementation. $\\$

6. Run scikit-learn's implementation of k-means for $k = 1,\ldots20$, and plot the costs of the resulting clusterings as a function of k. At which values of k do you see an "elbow"? $\\$

```{python}
from sklearn.cluster import KMeans

# Running k-means for k from 1 to 20
ks = range(1, 21)
costs = []

for k in ks:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    # Compute the cost (sum of squared distances of samples to their closest cluster center)
    costs.append(kmeans.inertia_)

# Plotting the costs
plt.figure(figsize=(10, 6))
plt.plot(ks, costs, marker='o')
plt.title('K-Means Clustering Costs as a Function of k')
plt.xlabel('Number of Clusters k')
plt.ylabel('Cost (Sum of Squared Distances)')
plt.grid(True)
plt.xticks(ks)
plt.show()

# Output costs for inspection
costs

```
The "elbow" in the plot is typically identified where the cost curve starts to flatten out after a steep decline, indicating diminishing returns on the cost reduction as more clusters are added. This point is often considered as a good choice for the number of clusters because it represents a balance between minimizing the cost and the complexity of the model - not using many clusters. $\\$

The elbow method suggests k = 3 as a suitable choice for the number of clusters, aligning with the initial setup of the dataset where 3 distinct clusters were used to generate the data. This choice seems optimal in terms of achieving a significant reduction in clustering cost without unnecessarily increasing the number of clusters.$\\$

7. Use scikit-learn to compute the single linkage dendrogram of $X$. Does the dendrogram clearly reveal the presence of three clusters in $X$? $\\$

```{python}
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Compute the linkage matrix using single linkage
Z = linkage(X, method='single')

# Plot the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(Z)
plt.title('Dendrogram for Single Linkage Clustering')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

```

- Presence of Three Clusters: The dendrogram structure suggests potential cluster formations at various levels of granularity. Notably, there are a few long vertical lines (large jumps in height) which typically indicate the formation of distinct clusters. $\\$

- Identifying Clusters: The dendrogram does show regions where points are grouped closely at lower heights, and these groupings merge at higher distances. This characteristic can be used to infer the presence of clusters. $\\$

- Optimal Cluster Count: For a clear identification of three clusters, we would look for three vertical lines joining at a significantly higher height than others, which is not very distinct here. Single linkage often results in a "chaining effect", where clusters are elongated and might merge gradually, which can make the separation into exactly three clusters less obvious at a glance. $\\$

8. Do the same for the average linkage dendrogram of $X$. Does the dendrogram clearly reveal the presence of three clusters in $X$? $\\$

```{python}
# Compute the linkage matrix using average linkage
Z_average = linkage(X, method='average')

# Plot the dendrogram for average linkage
plt.figure(figsize=(10, 8))
dendrogram(Z_average)
plt.title('Dendrogram for Average Linkage Clustering')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

```

- Clearer Cluster Formation: The dendrigram with average linkage shows clearer and more distinct cluster formations than the single linkage dendrogram. This is because average linkage tends to mitigate the chaining effect seen in single linkage, providing a more balanced view of cluster distance. $\\$

- Presence of Three Clusters: The dendrogram reveals what appears to be three main clusters merging at higher levels. These clusters are indicated by the three major groupings of lines that merge at distinctly higher distances than the rest of the lower merges. $\\$

- Interpretation of Clusters: The vertical distances between the merges (height of the joins) are more uniform before combining into these three groups, suggesting that the data naturally forms into three clusters before any further merging occurs.






