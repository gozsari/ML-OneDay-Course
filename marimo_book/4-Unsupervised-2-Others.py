import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 4.2 Other Unsupervised Learning Techniques

    ## Objectives:
    1. Understand unsupervised learning techniques beyond clustering.
    2. Learn about dimensionality reduction methods such as PCA.
    3. Explore anomaly detection techniques.
    4. Hands-on: Apply Principal Component Analysis (PCA) to a dataset.

    By the end of this session, you will have a deeper understanding of unsupervised learning methods and their applications.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🌟 Dimensionality Reduction

    ### What is Dimensionality Reduction?
    Dimensionality reduction reduces the number of input features while retaining the essential patterns in the data. It is useful for:
    - Simplifying datasets.
    - Visualizing high-dimensional data.
    - Speeding up computations.

    ---

    ### Principal Component Analysis (PCA)
    PCA is one of the most popular dimensionality reduction techniques.

    ---

    ### Key Steps:
    1. Standardize the dataset (mean = 0, variance = 1).
    2. Compute the covariance matrix.
    3. Identify the principal components (eigenvectors of the covariance matrix).
    4. Project the data onto the new subspace defined by the top principal components.

    ---

    ### Applications:
    1. Image Compression.
    2. Reducing features for faster model training.
    3. Visualizing high-dimensional data (e.g., reducing 100 features to 2).

    ---

    ### Hands-On Example: PCA
    We will implement PCA on a synthetic dataset later in this section.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Hands-On: Dimensionality Reduction with PCA
    Let's apply PCA to a dataset to reduce its dimensionality on a synthetic dataset.
    """
    )
    return


@app.cell
def _():
    # Import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    return PCA, StandardScaler, make_classification, np, plt


@app.cell
def _(StandardScaler, make_classification):
    # 1. Generate Synthetic Data
    _X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    scaler = StandardScaler()
    # Standardize the data
    X_scaled = scaler.fit_transform(_X)
    return X_scaled, y


@app.cell
def _(PCA, X_scaled):
    # 2. Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


@app.cell
def _(X_pca, plt, y):
    # 3. Visualize the Results
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', label='Data Points')
    plt.title('PCA Projection to 2D')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Class')
    plt.legend()
    plt.show()
    return


@app.cell
def _(pca):
    # 4. Explained Variance
    _explained_variance = pca.explained_variance_ratio_
    print(f'Explained Variance by Each Component: {_explained_variance}')
    print(f'Total Explained Variance: {sum(_explained_variance):.2f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**NOTE**: The explained variance ratio tells us how much information is preserved by each principal component. We can use this information to decide how many principal components to keep. """)
    return


@app.cell
def _(PCA, X_scaled):
    pca_1 = PCA(n_components=5)
    X_pca_1 = pca_1.fit_transform(X_scaled)
    _explained_variance = pca_1.explained_variance_ratio_
    print(f'Explained Variance by Each Component: {_explained_variance}')
    print(f'Total Explained Variance: {sum(_explained_variance):.2f}')
    return


@app.cell
def _(PCA, X_scaled):
    pca_2 = PCA(n_components=7)
    X_pca_2 = pca_2.fit_transform(X_scaled)
    _explained_variance = pca_2.explained_variance_ratio_
    print(f'Explained Variance by Each Component: {_explained_variance}')
    print(f'Total Explained Variance: {sum(_explained_variance):.2f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧪 Interactive Playground: How Many Components Should We Keep?

    Instead of trying 2, 5 and 7 components by hand, use the slider to choose any number of components for the same 10-feature dataset.

    - The **bars** show how much variance each component explains, and the **line** shows the cumulative total.
    - The **reconstruction error** measures how much information is lost when we compress the data to *n* components and then map it back to the original 10 features. More components means lower error, but less compression.
    """
    )
    return


@app.cell
def _(mo):
    # Control for the PCA playground
    pca_n_components = mo.ui.slider(1, 10, value=2, step=1, label="Number of components", show_value=True)
    pca_n_components
    return (pca_n_components,)


@app.cell
def _(PCA, X_scaled, mo, np, pca_n_components, plt):
    # Explained variance for all components, plus reconstruction error for the selected number
    _pca_full = PCA().fit(X_scaled)
    _ratios = _pca_full.explained_variance_ratio_
    _cumulative = np.cumsum(_ratios)
    _n = pca_n_components.value

    _pca_n = PCA(n_components=_n).fit(X_scaled)
    _reconstructed = _pca_n.inverse_transform(_pca_n.transform(X_scaled))
    _reconstruction_error = np.mean((X_scaled - _reconstructed) ** 2)

    _components = np.arange(1, len(_ratios) + 1)
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _ax.bar(_components, _ratios, color=['tab:blue' if _c <= _n else 'lightgray' for _c in _components], label='Per component')
    _ax.plot(_components, _cumulative, 'o-', color='tab:orange', label='Cumulative')
    _ax.axvline(_n + 0.5, color='red', linestyle='--', label=f'Keep {_n} component(s)')
    _ax.set_xticks(_components)
    _ax.set_ylim(0, 1.05)
    _ax.set_title('Explained Variance Ratio')
    _ax.set_xlabel('Principal component')
    _ax.set_ylabel('Explained variance ratio')
    _ax.legend()
    _fig.tight_layout()

    mo.vstack([
        mo.md(
            f"**Variance kept with {_n} component(s):** `{_cumulative[_n - 1]:.1%}` &nbsp;|&nbsp; "
            f"**Reconstruction error (MSE):** `{_reconstruction_error:.3f}`"
        ),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🌟 Anomaly Detection

    ### What is Anomaly Detection?
    Anomaly detection identifies data points that deviate significantly from the normal patterns. It is often used for:
    - Fraud detection.
    - Monitoring network traffic for intrusions.
    - Detecting manufacturing defects.

    ---

    ### Techniques:
    1. **Density-Based Methods**:
       - DBSCAN (clustering-based).
       - Isolation Forest (tree-based).
    2. **Statistical Methods**:
       - Z-scores: Data points with high Z-scores (e.g., > 3) are anomalies.

    ---

    ### Applications:
    1. Detecting fraudulent transactions in banking.
    2. Identifying faulty sensors in IoT systems.
    3. Finding outliers in customer behavior data.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Hands-On: Anomaly Detection

    We will implement anomaly detection using the Isolation Forest algorithm on a synthetic dataset. 
    """
    )
    return


@app.cell
def _():
    # Import required libraries
    from sklearn.ensemble import IsolationForest
    from sklearn.datasets import make_blobs
    return IsolationForest, make_blobs


@app.cell
def _(make_blobs, np):
    # 1. Generate Synthetic Data
    _X, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
    outliers = np.random.uniform(low=-6, high=6, size=(20, 2))  # Add some outliers
    X_with_outliers = np.vstack([_X, outliers])
    return (X_with_outliers,)


@app.cell
def _(X_with_outliers, plt):
    # Visualize the data with outliers
    plt.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], color='blue', label='Data Points')
    plt.title('Data with Outliers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return


@app.cell
def _(IsolationForest, X_with_outliers):
    # 2. Apply Isolation Forest for Anomaly Detection
    iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Assume 5% contamination
    predictions = iso_forest.fit_predict(X_with_outliers)  # -1: Anomaly, 1: Normal
    return (predictions,)


@app.cell
def _(X_with_outliers, predictions):
    # Separate anomalies and normal points
    normal = X_with_outliers[predictions == 1]
    anomalies = X_with_outliers[predictions == -1]
    return anomalies, normal


@app.cell
def _(anomalies, normal, plt):
    # Visualize Results
    plt.scatter(normal[:, 0], normal[:, 1], color='blue', label='Normal Points')
    plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', label='Anomalies', edgecolor='k')
    plt.title('Anomaly Detection with Isolation Forest')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧪 Interactive Playground: Tuning Isolation Forest

    The `contamination` parameter tells Isolation Forest **what fraction of the data you expect to be anomalies**. Here we know exactly which points are injected outliers, so we can check how well it does.

    - If contamination is **too low**, real outliers are missed.
    - If contamination is **too high**, normal points get flagged as anomalies.
    - Try matching contamination to the actual outlier fraction shown below.
    """
    )
    return


@app.cell
def _(mo):
    # Controls for the anomaly detection playground
    iso_contamination = mo.ui.slider(0.01, 0.30, value=0.05, step=0.01, label="Contamination", show_value=True)
    iso_n_outliers = mo.ui.slider(0, 60, value=20, step=5, label="Injected outliers", show_value=True)
    mo.hstack([iso_contamination, iso_n_outliers])
    return iso_contamination, iso_n_outliers


@app.cell
def _(IsolationForest, iso_contamination, iso_n_outliers, make_blobs, mo, np, plt):
    # Build data with a known number of outliers, fit Isolation Forest, and compare
    _rng = np.random.default_rng(0)
    _X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
    _X_outliers = _rng.uniform(low=-6, high=6, size=(iso_n_outliers.value, 2))
    _X_all = np.vstack([_X_normal, _X_outliers])
    _is_injected = np.r_[np.zeros(len(_X_normal), dtype=bool), np.ones(len(_X_outliers), dtype=bool)]

    _flagged = IsolationForest(contamination=iso_contamination.value, random_state=42).fit_predict(_X_all) == -1

    _fig, _ax = plt.subplots(figsize=(7, 5))
    _ax.scatter(_X_all[~_flagged, 0], _X_all[~_flagged, 1], color='blue', alpha=0.6, label='Predicted normal')
    _ax.scatter(_X_all[_flagged, 0], _X_all[_flagged, 1], color='red', edgecolor='k', label='Predicted anomaly')
    _ax.scatter(_X_all[_is_injected, 0], _X_all[_is_injected, 1], facecolors='none', edgecolors='black', s=150, label='Injected outlier')
    _ax.set_title(f'Isolation Forest (contamination = {iso_contamination.value:.2f})')
    _ax.set_xlabel('Feature 1')
    _ax.set_ylabel('Feature 2')
    _ax.legend()
    _fig.tight_layout()

    _caught = int(np.sum(_flagged & _is_injected))
    _false_alarms = int(np.sum(_flagged & ~_is_injected))
    mo.vstack([
        mo.md(
            f"**Actual outlier fraction:** `{_is_injected.mean():.2f}` &nbsp;|&nbsp; "
            f"**Outliers caught:** `{_caught} / {int(_is_injected.sum())}` &nbsp;|&nbsp; "
            f"**Normal points flagged:** `{_false_alarms}`"
        ),
        _fig,
    ])
    return


@app.cell
def _(mo):
    # Quick check question
    pca_quiz = mo.ui.radio(
        [
            "You can keep 3 components and lose only about 5% of the variance",
            "The data contains exactly 3 classes",
            "The remaining 7 features of the original data are useless and can be deleted",
        ],
        value=None,
        label="On a 10-feature dataset, the first 3 principal components explain 95% of the variance. What does that tell you?",
    )
    return (pca_quiz,)


@app.cell(hide_code=True)
def _(mo, pca_quiz):
    if pca_quiz.value is None:
        _feedback = mo.md("Select an option above.")
    elif pca_quiz.value == "You can keep 3 components and lose only about 5% of the variance":
        _feedback = mo.callout(
            mo.md("✅ Correct! Principal components are *combinations* of all original features, so we compress the data rather than delete features."),
            kind="success",
        )
    else:
        _feedback = mo.callout(
            mo.md("❌ Not quite. Explained variance is about how much *information* is kept, not about classes or individual features."),
            kind="danger",
        )
    mo.vstack([pca_quiz, _feedback])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📝 Key Takeaways:
    1. Dimensionality reduction techniques like PCA help simplify data while preserving patterns.
    2. Anomaly detection is used to identify unusual data points in applications like fraud detection.
    3. Each of these techniques has distinct use cases in real-world scenarios.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

