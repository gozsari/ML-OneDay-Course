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
    ## 🧪 Interactive Playground: Choosing the Number of Clusters

    In real problems we don't know the "right" number of clusters. Here you control both the **true** structure of the data and the **k** used by K-Means.

    - The **elbow plot** shows inertia for k = 2 … 10. Inertia always decreases as k grows; look for the "elbow" where it stops dropping quickly.
    - The **silhouette plot** usually peaks near the true number of clusters.
    - Increase the **cluster spread** to make the clusters overlap and see how both metrics become less clear.
    """
    )
    return


@app.cell
def _(mo):
    # Controls for the clustering playground
    cl_true_centers = mo.ui.slider(2, 8, value=4, step=1, label="True number of clusters", show_value=True)
    cl_std = mo.ui.slider(0.3, 3.0, value=1.0, step=0.1, label="Cluster spread (std)", show_value=True)
    cl_k = mo.ui.slider(2, 10, value=3, step=1, label="K-Means k", show_value=True)
    mo.hstack([cl_true_centers, cl_std, cl_k])
    return cl_k, cl_std, cl_true_centers


@app.cell
def _(KMeans, cl_std, cl_true_centers, make_blobs, silhouette_score):
    # Generate data and evaluate K-Means for every k from 2 to 10
    X_cl, _ = make_blobs(n_samples=300, centers=cl_true_centers.value, cluster_std=cl_std.value, random_state=7)

    cl_k_values = list(range(2, 11))
    cl_inertias, cl_silhouettes = [], []
    for _k in cl_k_values:
        _km = KMeans(n_clusters=_k, random_state=42, n_init=10).fit(X_cl)
        cl_inertias.append(_km.inertia_)
        cl_silhouettes.append(silhouette_score(X_cl, _km.labels_))
    return X_cl, cl_inertias, cl_k_values, cl_silhouettes


@app.cell
def _(KMeans, X_cl, cl_inertias, cl_k, cl_k_values, cl_silhouettes, mo, plt):
    # Fit K-Means with the selected k and plot clusters, elbow, and silhouette curves
    _km = KMeans(n_clusters=cl_k.value, random_state=42, n_init=10).fit(X_cl)
    _i = cl_k_values.index(cl_k.value)

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    _ax1.scatter(X_cl[:, 0], X_cl[:, 1], c=_km.labels_, cmap='viridis', s=30)
    _ax1.scatter(_km.cluster_centers_[:, 0], _km.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
    _ax1.set_title(f'K-Means with k = {cl_k.value}')
    _ax1.set_xlabel('Feature 1')
    _ax1.set_ylabel('Feature 2')
    _ax1.legend()

    _ax2.plot(cl_k_values, cl_inertias, 'o-')
    _ax2.plot(cl_k.value, cl_inertias[_i], 'ro', markersize=12)
    _ax2.set_title('Elbow Plot (Inertia)')
    _ax2.set_xlabel('k')
    _ax2.set_ylabel('Inertia')

    _ax3.plot(cl_k_values, cl_silhouettes, 'o-', color='green')
    _ax3.plot(cl_k.value, cl_silhouettes[_i], 'ro', markersize=12)
    _ax3.set_title('Silhouette Score')
    _ax3.set_xlabel('k')
    _ax3.set_ylabel('Silhouette score')
    _fig.tight_layout()

    _best_k = cl_k_values[cl_silhouettes.index(max(cl_silhouettes))]
    mo.vstack([
        mo.md(
            f"**Selected k = {cl_k.value}:** silhouette `{cl_silhouettes[_i]:.2f}`, inertia `{cl_inertias[_i]:.0f}` "
            f"&nbsp;|&nbsp; **Best k by silhouette:** `{_best_k}`"
        ),
        _fig,
    ])
    return


@app.cell
def _(mo):
    # Quick check question
    cl_quiz = mo.ui.radio(
        [
            "Points are deep inside their own cluster",
            "Clusters overlap and points sit near the boundary between clusters",
            "Every point is assigned to the wrong cluster",
        ],
        value=None,
        label="What does a silhouette score close to 0 mean?",
    )
    return (cl_quiz,)


@app.cell(hide_code=True)
def _(cl_quiz, mo):
    if cl_quiz.value is None:
        _feedback = mo.md("Select an option above.")
    elif cl_quiz.value == "Clusters overlap and points sit near the boundary between clusters":
        _feedback = mo.callout(
            mo.md("✅ Correct! A score near 0 means a point is about as close to a neighbouring cluster as to its own. Try increasing the cluster spread above."),
            kind="success",
        )
    else:
        _feedback = mo.callout(
            mo.md("❌ Not quite. Scores near 1 mean well-separated clusters and scores near −1 mean wrong assignments. What's in between?"),
            kind="danger",
        )
    mo.vstack([cl_quiz, _feedback])
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

