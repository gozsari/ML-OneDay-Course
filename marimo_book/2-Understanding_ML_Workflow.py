import marimo

__generated_with = "0.17.0"
app = marimo.App()


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
