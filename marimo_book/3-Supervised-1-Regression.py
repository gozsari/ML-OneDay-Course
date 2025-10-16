import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 3. Supervised Learning :rocket:

    :memo: **Definition:** Supervised learning is a type of machine learning where the model is trained on a labeled dataset. The model learns the relationship between the features and the labels and uses this relationship to make predictions on new data.

    - **Supervised learning** is the most common type of machine learning. 
    - It is the first type of learning that most people encounter. 
    - It is the type of learning that is used to train a model to predict an output based on an input. 
    - The input is called the **feature** X and the output is called the **label** Y. 
    - The model is trained on a dataset that contains both the features and the labels. 
    - The model learns the relationship between the features and the labels and uses this relationship to make predictions on new data.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3.1 Regression :chart_with_upwards_trend:

    ## Objectives:
    1. Understand what regression is and why it's important.
    2. Learn the key components of regression models.
    3. Explore the differences between linear and polynomial regression.
    4. Understand evaluation metrics used for regression models.
    5. Hands-on: Implement a simple linear regression model and evaluate its performance.

    By the end of this session, you will be able to apply regression to make predictions on continuous data and evaluate the quality of your models.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🤔 What is Regression?

    Regression is a supervised learning technique used to model the relationship between a dependent variable (target) and one or more independent variables (features). 

    ### Key Characteristics:
    - **Continuous Target Variable**: Regression predicts continuous outputs, such as prices, distances, or probabilities.
    - **Mathematical Relationship**: The model tries to find the best function that maps input features to the target variable.

    ### Regression vs. Classification:
    - **Regression**: Predicts continuous values (e.g., house prices, temperature).
    - **Classification**: Predicts discrete classes (e.g., spam vs. not spam).

    ---

    ### How Regression Works:
    1. Collect data with input features \( X \) and target \( y \).
    2. Use a regression algorithm to learn the mapping \( y = f(X) \).
    3. Apply the trained model to predict new target values for unseen data.

    For example:
    - Predicting house prices based on square footage (\( X \)) and predicting a new price (\( y \)).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔑 Types of Regression:

    ### 1️⃣ Linear Regression:
    - Models a straight-line relationship between features and the target.
    - Equation: \( y = mx + b \), where:
      - \( m \): slope (how much \( y \) changes for a unit change in \( x \)).
      - \( b \): intercept (value of \( y \) when \( x = 0 \)).

    ---

    ### 2️⃣ Polynomial Regression:
    - Extends linear regression to fit curves by adding polynomial terms.
    - Equation: \( y = ax^2 + bx + c \).
    - Example: Modeling relationships that aren't linear, like population growth.

    ---

    ### 3️⃣ Other Types of Regression:
    - **Ridge Regression**: Adds regularization (L2 penalty) to prevent overfitting.

    - **Lasso Regression**: Adds regularization (L1 penalty) and performs feature selection by setting some coefficients to zero.

    - **Logistic Regression**: Used for classification, not continuous targets.

    **Scikit-learn** provides a simple and efficient way to implement regression models in Python. See the [documentation](https://scikit-learn.org/stable/modules/linear_model.html) for more details.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📏 How Do We Evaluate Regression Models?

    ### Importance of Metrics:
    Metrics help determine how well the regression model predicts the target values. A good model minimizes errors and explains the variance in the target variable.

    ---

    ### Common Metrics:
    1. **Mean Absolute Error (MAE)**:
       - Measures the average absolute difference between predicted and actual values.
       - Lower MAE indicates better model performance.
       - Formula: MAE = 1/n * Σ |y - ŷ| where \( n \) is the number of samples, \( y \) is the actual value, and \( ŷ \) is the predicted value.
    ---

    1. **Mean Squared Error (MSE)**:
       - Measures the average squared difference between predicted and actual values.
       - Penalizes larger errors more heavily than MAE.
       - Formula: MSE = 1/n * Σ (y - ŷ)² where \( n \) is the number of samples, \( y \) is the actual value, and \( ŷ \) is the predicted value.

    ---

    1. **Root Mean Squared Error (RMSE)**:
       - The square root of MSE. Intuitive as it's in the same units as the target variable.
       - Formula: RMSE = √MSE

    ---

    1. **R² Score (Coefficient of Determination)**:
       - Measures the proportion of variance in the target variable explained by the features.
       - Values range from 0 (no explanation) to 1 (perfect fit).
       - Formula: R² = 1 - (Σ (y - ŷ)² / Σ (y - ȳ)²) where \( y \) is the actual value, \( ŷ \) is the predicted value, and \( ȳ \) is the mean of the actual values.

    ---

    ### When to Use Each Metric:
    - **MAE**: Robust to outliers, suitable for general comparisons.
    - **MSE**: Highlights larger errors, useful when large deviations are unacceptable.
    - **R²**: Explains the overall performance of the model relative to a baseline (mean).

    **Scikit-learn** provides functions to calculate these metrics for regression models. See the [documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) for more details.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🚀 Hands-on: Implementing Linear Regression""")
    return


@app.cell
def _():
    # Import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return (
        LinearRegression,
        mean_absolute_error,
        mean_squared_error,
        np,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell
def _(np):
    # 1. Generate Synthetic Data for Regression
    np.random.seed(42)
    X = 2.5 * np.random.randn(100, 1) + 1.5   # Feature (independent variable)
    y = 2 + 1.8 * X + np.random.randn(100, 1) # Target (dependent variable)
    return X, y


@app.cell
def _(X, plt, y):
    # Visualize the data
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.title('Synthetic Regression Data')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()
    return


@app.cell
def _(X, train_test_split, y):
    # 2. Split Data into Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(LinearRegression, X_train, y_train):
    # 3. Train a Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(model):
    # Model Parameters
    print(f"Intercept: {model.intercept_[0]:.2f}")
    print(f"Coefficient: {model.coef_[0][0]:.2f}")
    return


@app.cell
def _(X_test, model):
    # 4. Make Predictions
    y_pred = model.predict(X_test)
    return (y_pred,)


@app.cell
def _(mean_absolute_error, mean_squared_error, np, r2_score, y_pred, y_test):
    # 5. Evaluate the Model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    return


@app.cell
def _(X_test, plt, y_pred, y_test):
    # 6. Visualize Predictions
    # Sort by X for a clean line plot
    import numpy as _np
    _order = _np.argsort(X_test[:, 0])
    _Xs, _ys, _yps = X_test[_order], y_test[_order], y_pred[_order]
    plt.scatter(_Xs, _ys, color='blue', label='Actual')
    plt.plot(_Xs, _yps, color='red', label='Predicted')
    plt.title('Linear Regression Predictions')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📝 Key Takeaways:
    1. Regression is a powerful technique for predicting continuous outcomes.
    2. Linear regression models a straight-line relationship, but other types like polynomial regression can handle non-linear data.
    3. Evaluating models with metrics like MAE, MSE, RMSE, and R² is crucial for assessing their performance.
    4. Visualizing data and predictions helps understand model behavior.

    Up next, we’ll explore **Classification** techniques!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## In-class Activity:
    - Implement another regression algorithm using a different dataset.
    - Use scikit-learn to choose dataset and new regression algorithm.
    - Datasets: [Toy datasets](https://scikit-learn.org/1.5/datasets/toy_dataset.html)
    - Algorithms: [Scikit-learn algorithms](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
