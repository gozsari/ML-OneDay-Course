import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🔄 Understanding the Machine Learning Workflow

    In this section, we will explore the complete pipeline for a machine learning project, step by step. Each step plays a critical role in building effective machine learning models.

    ## What You’ll Learn:
    * The standard steps in the ML workflow.
    * Hands-on example demonstrating the entire workflow.

    By the end of this section, you will understand how to approach ML problems systematically and have a deeper appreciation for the engineering behind successful models.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🚀 Steps in the ML Workflow

    ### 1️⃣ **Define the Problem**
    - Clearly define the objective.
      - Example: Predicting whether a customer will churn based on their usage data.
    - Specify whether it's a regression, classification, or clustering task.

    ---

    ### 2️⃣ **Collect and Clean Data**
    - **Data Collection**: Gather data from sources such as databases, APIs, or CSV files.
    - **Data Cleaning**:
      - Handle missing values (e.g., replace with mean, median, or drop rows).
      - Remove duplicates.
      - Address outliers that could distort results.

    ---

    ### 3️⃣ **Explore and Visualize Data**
    - Generate summary statistics (e.g., mean, standard deviation, and correlation).
    - Visualize distributions and relationships using:
      - Histograms
      - Scatterplots
      - Heatmaps
    - Example: Plot the distribution of customer ages to check for skewness.

    ---

    ### 4️⃣ **Feature Engineering**
    Feature engineering transforms raw data into meaningful features that improve model performance.
    - **Feature Selection**: Choose only the most relevant features.
      - Example: Removing highly correlated features to avoid redundancy.
    - **Feature Transformation**:
      - Normalize numerical values (scaling data to 0-1 or z-score).
      - Convert categorical features into numerical ones using one-hot encoding.
    - **Feature Creation**: Create new features from existing ones.
      - Example: Extracting "month" from a "date" column.

    ---

    ### 5️⃣ **Split Data**
    - Divide the dataset into:
      - **Training Set**: For training the model.
      - **Validation Set**: For hyperparameter tuning.
      - **Test Set**: For final performance evaluation.
    - Typical split ratios: 70% training, 15% validation, 15% test.

    ---

    ### 6️⃣ **Choose and Train a Model**
    - Select an algorithm based on the task (e.g., regression for predicting numbers, classification for predicting categories).
    - Train the model using the training dataset.
    - Example models:
      - Regression: Linear Regression
      - Classification: Logistic Regression, Decision Trees
      - Clustering: K-Means

    ---

    ### 7️⃣ **Evaluate the Model**
    - Use appropriate metrics to assess model performance:
      - **Regression**: RMSE, MAE, R².
      - **Classification**: Accuracy, Precision, Recall, F1-score.
      - **Clustering**: Silhouette Score.
    - Visualize performance using confusion matrices or ROC curves.

    ---

    ### 8️⃣ **Hyperparameter Optimization**
    Hyperparameter tuning helps to improve model performance by finding the best parameter configuration.
    - **Grid Search**:
      - Example: Trying multiple combinations of `max_depth` and `min_samples_split` in Decision Trees.
    - **Random Search**:
      - Randomly samples hyperparameter combinations to find the best results.
    - **Automated Tools**:
      - Libraries like Scikit-learn's `GridSearchCV` or `RandomizedSearchCV`.
    - **Example Parameters to Optimize**:
      - Learning rate for Gradient Boosting.
      - Number of clusters in K-Means.

    ---

    ### 9️⃣ **Deploy the Model**
    - Integrate the trained model into an application or business workflow.
    - Monitor for model drift or performance degradation over time.

    Below is a hands-on example demonstrating the ML workflow.
    """
    )
    return


@app.cell
def _():
    # Hands-On Example: Simulating the ML Workflow
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report
    return (
        DecisionTreeClassifier,
        GridSearchCV,
        accuracy_score,
        classification_report,
        make_classification,
        plt,
        train_test_split,
    )


@app.cell
def _():
    # 1. Define the Problem
    print('Problem: Predict binary labels based on input features.')
    return


@app.cell
def _(make_classification):
    # 2. Collect and Preprocess Data
    X, y = make_classification(
        n_samples=200, 
        n_features=2, 
        n_classes=2, 
        n_informative=2,  # Use both features as informative
        n_redundant=0,     # No redundant features
        n_repeated=0,      # No repeated features
        n_clusters_per_class=1,  # Simplify for 2 features
        random_state=42
    )

    return X, y


@app.cell
def _(X, plt, y):
    # Visualize the raw data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Classes')
    plt.title('Raw Data Distribution')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return


@app.cell
def _(X, train_test_split, y):
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(DecisionTreeClassifier, X_train, y_train):
    # 4. Train a Model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(X_test, accuracy_score, classification_report, model, y_test):
    # 5. Evaluate the Model
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    return


@app.cell
def _(DecisionTreeClassifier, GridSearchCV, X_train, y_train):
    # 6. Hyperparameter Optimization using GridSearchCV
    param_grid = {'max_depth': [2, 4, 6, 8], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    return (grid_search,)


@app.cell
def _(grid_search):

    print('\nBest Parameters from GridSearchCV:', grid_search.best_params_)
    optimized_model = grid_search.best_estimator_

    return (optimized_model,)


@app.cell
def _(X_test, accuracy_score, optimized_model, y_test):
    # Evaluate the optimized model
    optimized_pred = optimized_model.predict(X_test)
    print(f'Optimized Model Accuracy: {accuracy_score(y_test, optimized_pred):.2f}')

    return (optimized_pred,)


@app.cell
def _(X_test, optimized_pred, plt):
    # Visualize Results
    plt.scatter(X_test[:, 0], X_test[:, 1], c=optimized_pred, cmap='coolwarm', label='Predictions')
    plt.title('Optimized Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧪 Interactive Playground: Overfitting and the Train/Test Split

    Use the controls below to change the depth of a decision tree, the amount of label noise in the data, and the size of the test set.

    - **Training accuracy** keeps going up as the tree gets deeper, because the tree memorises the training data.
    - **Test accuracy** stops improving (or even drops) after a certain depth.

    The gap between the two curves is **overfitting**. This is why step 5 (*Split Data*) and step 7 (*Evaluate the Model*) always use data the model has never seen.
    """
    )
    return


@app.cell
def _(mo):
    # Controls for the overfitting playground
    wf_depth = mo.ui.slider(1, 20, value=3, step=1, label="Tree max_depth", show_value=True)
    wf_noise = mo.ui.slider(0.0, 0.4, value=0.1, step=0.05, label="Label noise (flip_y)", show_value=True)
    wf_test_size = mo.ui.slider(0.1, 0.5, value=0.3, step=0.05, label="Test size", show_value=True)
    mo.hstack([wf_depth, wf_noise, wf_test_size])
    return wf_depth, wf_noise, wf_test_size


@app.cell
def _(
    DecisionTreeClassifier,
    accuracy_score,
    make_classification,
    train_test_split,
    wf_noise,
    wf_test_size,
):
    # Generate noisy data, split it, and measure train/test accuracy for every depth
    X_wf, y_wf = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=wf_noise.value,
        random_state=0,
    )
    X_wf_train, X_wf_test, y_wf_train, y_wf_test = train_test_split(
        X_wf, y_wf, test_size=wf_test_size.value, random_state=42
    )

    wf_depths = list(range(1, 21))
    wf_train_acc, wf_test_acc = [], []
    for _depth in wf_depths:
        _tree = DecisionTreeClassifier(max_depth=_depth, random_state=42).fit(X_wf_train, y_wf_train)
        wf_train_acc.append(accuracy_score(y_wf_train, _tree.predict(X_wf_train)))
        wf_test_acc.append(accuracy_score(y_wf_test, _tree.predict(X_wf_test)))
    return (
        X_wf,
        X_wf_test,
        X_wf_train,
        wf_depths,
        wf_test_acc,
        wf_train_acc,
        y_wf_test,
        y_wf_train,
    )


@app.cell
def _(
    DecisionTreeClassifier,
    X_wf,
    X_wf_test,
    X_wf_train,
    mo,
    plt,
    wf_depth,
    wf_depths,
    wf_test_acc,
    wf_train_acc,
    y_wf_test,
    y_wf_train,
):
    # Plot the decision boundary for the selected depth and the accuracy curves
    import numpy as _np

    _tree = DecisionTreeClassifier(max_depth=wf_depth.value, random_state=42).fit(X_wf_train, y_wf_train)
    _xx, _yy = _np.meshgrid(
        _np.linspace(X_wf[:, 0].min() - 1, X_wf[:, 0].max() + 1, 200),
        _np.linspace(X_wf[:, 1].min() - 1, X_wf[:, 1].max() + 1, 200),
    )
    _zz = _tree.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _ax1.contourf(_xx, _yy, _zz, alpha=0.25, cmap='coolwarm')
    _ax1.scatter(X_wf_train[:, 0], X_wf_train[:, 1], c=y_wf_train, cmap='coolwarm', edgecolor='k', s=25, label='Train')
    _ax1.scatter(X_wf_test[:, 0], X_wf_test[:, 1], c=y_wf_test, cmap='coolwarm', edgecolor='k', marker='^', s=45, label='Test')
    _ax1.set_title(f'Decision Boundary (max_depth={wf_depth.value})')
    _ax1.set_xlabel('Feature 1')
    _ax1.set_ylabel('Feature 2')
    _ax1.legend()

    _ax2.plot(wf_depths, wf_train_acc, 'o-', label='Train accuracy')
    _ax2.plot(wf_depths, wf_test_acc, 's-', label='Test accuracy')
    _ax2.axvline(wf_depth.value, color='gray', linestyle='--', label='Selected depth')
    _ax2.set_title('Accuracy vs. Tree Depth')
    _ax2.set_xlabel('max_depth')
    _ax2.set_ylabel('Accuracy')
    _ax2.legend()
    _fig.tight_layout()

    _i = wf_depth.value - 1
    mo.vstack([
        mo.md(
            f"**Train accuracy:** `{wf_train_acc[_i]:.2f}` &nbsp;|&nbsp; "
            f"**Test accuracy:** `{wf_test_acc[_i]:.2f}` &nbsp;|&nbsp; "
            f"**Gap:** `{wf_train_acc[_i] - wf_test_acc[_i]:.2f}`"
        ),
        _fig,
    ])
    return


@app.cell
def _(mo):
    # Quick check question
    wf_quiz = mo.ui.radio(
        [
            "Deploy the model",
            "Split the data into training and test sets",
            "Hyperparameter optimization",
        ],
        value=None,
        label="Which step should come right before training a model?",
    )
    return (wf_quiz,)


@app.cell(hide_code=True)
def _(mo, wf_quiz):
    if wf_quiz.value is None:
        _feedback = mo.md("Select an option above.")
    elif wf_quiz.value == "Split the data into training and test sets":
        _feedback = mo.callout(
            mo.md("✅ Correct! The test set must be held out *before* training so the model never sees it."),
            kind="success",
        )
    else:
        _feedback = mo.callout(
            mo.md("❌ Not quite. Look at the order of steps 5 and 6 in the workflow above."),
            kind="danger",
        )
    mo.vstack([wf_quiz, _feedback])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📝 Key Takeaways

    - A structured ML workflow ensures reproducibility and efficiency.
    - Data cleaning and feature engineering are crucial for good performance.
    - Hyperparameter optimization can significantly improve model results.
    - Always evaluate your model on unseen test data to gauge its real-world effectiveness.

    In the next section, we'll explore **Supervised Learning** techniques in more detail!
    """
    )
    return


if __name__ == "__main__":
    app.run()
