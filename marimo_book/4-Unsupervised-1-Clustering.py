import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 4- Unsupervised Learning 

    Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels and with a minimum of human supervision.

    ## 4.1 Clustering 

    ### Objectives:
    1. Understand the concept of clustering in unsupervised learning.
    2. Learn about commonly used clustering techniques: K-Means and Hierarchical Clustering.
    3. Explore practical applications of clustering.
    4. Hands-on: Implement K-Means clustering using Scikit-learn.

    By the end of this session, you will be able to group data points into clusters based on similarity and interpret clustering results.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🤔 What is Clustering?

    Clustering is a type of unsupervised learning where the goal is to group data points into clusters based on their similarity.

    ---

    ### Key Characteristics:
    - No labeled data (i.e., the target variable is unknown).
    - Finds hidden patterns or structures in data.

    ---

    ### Examples of Clustering:
    1. **Customer Segmentation**:
       - Group customers based on purchasing behavior.
    2. **Document Clustering**:
       - Group news articles based on topics.
    3. **Image Segmentation**:
       - Identify regions of interest in an image (e.g., foreground and background).

    ---

    ### How Clustering Works:
    1. Compute a similarity or distance measure (e.g., Euclidean distance, cosine similarity) between data points.
    2. Group similar data points into clusters.
    3. Evaluate the compactness and separation of clusters.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔑 Types of Clustering

    ### 1️⃣ **Partition-Based Clustering**:
    - Divides data into non-overlapping subsets.
    - Example Algorithm: **K-Means Clustering**.
    - Application: Customer segmentation.

    ---

    ### 2️⃣ **Hierarchical Clustering**:
    - Builds a tree-like structure of clusters (dendrogram).
    - Can be agglomerative (bottom-up) or divisive (top-down).
    - Application: Organizing documents into categories.

    ---

    ### 3️⃣ **Density-Based Clustering**:
    - Groups data points based on dense regions.
    - Example Algorithm: **DBSCAN**.
    - Application: Detecting anomalies or outliers.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## What is K-Means?

    K-Means is one of the most widely used clustering algorithms. It groups data into \( K \) clusters by minimizing the variance within each cluster.

    ---

    ### Key Steps:
    1. Choose the number of clusters \( K \).
    2. Initialize \( K \) cluster centroids randomly.
    3. Assign each data point to the nearest centroid.
    4. Update centroids as the mean of all points assigned to that cluster.
    5. Repeat steps 3 and 4 until convergence.

    ---

    ### Example:
    Suppose you have sales data for customers, and you want to group them into three segments:
    1. High spenders.
    2. Moderate spenders.
    3. Low spenders.

    ---

    ### Visual Representation:
    - Data points are grouped based on proximity to centroids.
    - Final clusters are compact and well-separated.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📏 How Do We Evaluate Clustering Results?

    ### 1️⃣ Silhouette Score:
    - Measures how similar a point is to its cluster compared to other clusters.
    - Ranges from -1 to 1:
      - 1: Perfect clustering.
      - 0: Overlapping clusters.
      - -1: Points assigned to the wrong cluster.

    ---

    ### 2️⃣ Inertia (Sum of Squared Distances):
    - Measures the compactness of clusters.
    - Lower inertia indicates tighter clusters.

    ---

    ### 3️⃣ Visualization:
    - Use scatterplots or dendrograms to visualize clusters and their separations.

    """
    )
    return


@app.cell
def _():
    # Import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    return KMeans, make_blobs, plt, silhouette_score


@app.cell
def _(make_blobs):
    # 1. Generate Synthetic Data for Clustering
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)
    return (X,)


@app.cell
def _(X, plt):
    # Visualize the data
    plt.scatter(X[:, 0], X[:, 1], s=50, color='blue')
    plt.title('Synthetic Data for Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    return


@app.cell
def _(KMeans, X):
    # 2. Apply K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    return (kmeans,)


@app.cell
def _(kmeans):
    # Cluster centers and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return centroids, labels


@app.cell
def _(X, centroids, labels, plt):
    # Visualize the clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return


@app.cell
def _(X, labels, silhouette_score):
    # 3. Evaluate Clustering Performance
    sil_score = silhouette_score(X, labels)
    print(f"Silhouette Score: {sil_score:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📝 Key Takeaways:
    1. Clustering is an unsupervised learning technique for grouping similar data points.
    2. Common clustering methods include partition-based (K-Means), hierarchical, and density-based clustering.
    3. Evaluation metrics like silhouette score help measure the quality of clusters.
    4. Visualizing clusters is essential for interpreting results.

    Next, we will explore other unsupervised learning techniques!

    For more information on clustering, refer to the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/clustering.html).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## In-class Activity: Implementing another Clustering with the dataset.
    - Choose an unsupervised learning technique (e.g., hierarchical clustering) and apply it to the dataset. 
    - See scikit-learn's documentation for more information on [clustering algorithms](https://scikit-learn.org/stable/modules/clustering.html).
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

