import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 3. Supervised Learning :rocket:

    **Definition:** Supervised learning is a type of machine learning where the model is trained on a labeled dataset. The model learns the relationship between the features and the labels and uses this relationship to make predictions on new data.

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
    ## 🧪 Interactive Playground: Linear vs. Polynomial Regression

    Pick the *true* relationship that generates the data, then change the **polynomial degree** of the model.

    - With **degree 1** (a straight line) on curved data, the model **underfits**: both train and test errors are high.
    - With a **very high degree** and few samples, the model **overfits**: the train error is tiny but the test error grows.
    - The **residual plot** (actual − predicted) should look like random noise around 0 for a good model. A visible pattern means the model is missing structure in the data.
    """
    )
    return


@app.cell
def _(mo):
    # Controls for the regression playground
    reg_shape = mo.ui.dropdown(["Linear", "Quadratic", "Sine wave"], value="Sine wave", label="True relationship")
    reg_degree = mo.ui.slider(1, 15, value=1, step=1, label="Polynomial degree", show_value=True)
    reg_noise = mo.ui.slider(0.0, 3.0, value=0.5, step=0.1, label="Noise σ", show_value=True)
    reg_n = mo.ui.slider(20, 300, value=60, step=10, label="Samples", show_value=True)
    mo.hstack([reg_shape, reg_degree, reg_noise, reg_n])
    return reg_degree, reg_n, reg_noise, reg_shape


@app.cell
def _(
    LinearRegression,
    mean_absolute_error,
    mean_squared_error,
    np,
    r2_score,
    reg_degree,
    reg_n,
    reg_noise,
    reg_shape,
    train_test_split,
):
    # Generate data from the chosen relationship and fit a polynomial model
    from sklearn.pipeline import make_pipeline as _make_pipeline
    from sklearn.preprocessing import PolynomialFeatures as _PolynomialFeatures

    _true_functions = {
        "Linear": lambda x: 2 + 1.8 * x,
        "Quadratic": lambda x: 1 + 0.5 * x - 0.8 * x**2,
        "Sine wave": lambda x: 3 * np.sin(1.5 * x),
    }
    reg_true_fn = _true_functions[reg_shape.value]

    _rng = np.random.default_rng(0)
    X_reg = _rng.uniform(-3, 3, size=(reg_n.value, 1))
    y_reg = reg_true_fn(X_reg[:, 0]) + reg_noise.value * _rng.standard_normal(reg_n.value)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    reg_model = _make_pipeline(
        _PolynomialFeatures(degree=reg_degree.value, include_bias=False),
        LinearRegression(),
    )
    reg_model.fit(X_reg_train, y_reg_train)
    reg_train_pred = reg_model.predict(X_reg_train)
    reg_test_pred = reg_model.predict(X_reg_test)

    def _metrics(y_true, y_hat):
        _mse = mean_squared_error(y_true, y_hat)
        return mean_absolute_error(y_true, y_hat), _mse, np.sqrt(_mse), r2_score(y_true, y_hat)

    reg_metrics = {
        "Train": _metrics(y_reg_train, reg_train_pred),
        "Test": _metrics(y_reg_test, reg_test_pred),
    }
    return (
        X_reg,
        X_reg_test,
        X_reg_train,
        reg_metrics,
        reg_model,
        reg_test_pred,
        reg_train_pred,
        reg_true_fn,
        y_reg,
        y_reg_test,
        y_reg_train,
    )


@app.cell
def _(
    X_reg_test,
    X_reg_train,
    mo,
    np,
    plt,
    reg_degree,
    reg_metrics,
    reg_model,
    reg_test_pred,
    reg_train_pred,
    reg_true_fn,
    y_reg,
    y_reg_test,
    y_reg_train,
):
    # Plot the fitted curve and the residuals, and show train/test metrics
    _grid = np.linspace(-3, 3, 300).reshape(-1, 1)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _ax1.scatter(X_reg_train, y_reg_train, color='blue', alpha=0.7, label='Train')
    _ax1.scatter(X_reg_test, y_reg_test, color='orange', edgecolor='k', label='Test')
    _ax1.plot(_grid, reg_true_fn(_grid[:, 0]), 'g--', label='True relationship')
    _ax1.plot(_grid, reg_model.predict(_grid), 'r-', linewidth=2, label=f'Model (degree {reg_degree.value})')
    _ax1.set_ylim(y_reg.min() - 2, y_reg.max() + 2)
    _ax1.set_title('Model Fit')
    _ax1.set_xlabel('Feature')
    _ax1.set_ylabel('Target')
    _ax1.legend()

    _ax2.scatter(reg_train_pred, y_reg_train - reg_train_pred, color='blue', alpha=0.7, label='Train')
    _ax2.scatter(reg_test_pred, y_reg_test - reg_test_pred, color='orange', edgecolor='k', label='Test')
    _ax2.axhline(0, color='gray', linestyle='--')
    _ax2.set_title('Residuals (Actual − Predicted)')
    _ax2.set_xlabel('Predicted value')
    _ax2.set_ylabel('Residual')
    _ax2.legend()
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {_name} | {_mae:.2f} | {_mse:.2f} | {_rmse:.2f} | {_r2:.2f} |"
        for _name, (_mae, _mse, _rmse, _r2) in reg_metrics.items()
    )
    mo.vstack([
        mo.md("| Set | MAE | MSE | RMSE | R² |\n|---|---|---|---|---|\n" + _rows),
        _fig,
    ])
    return


@app.cell
def _(mo):
    # Quick check question
    reg_quiz = mo.ui.radio(
        ["MAE", "MSE", "R²"],
        value=None,
        label="Which metric punishes large errors the most, because it squares them?",
    )
    return (reg_quiz,)


@app.cell(hide_code=True)
def _(mo, reg_quiz):
    if reg_quiz.value is None:
        _feedback = mo.md("Select an option above.")
    elif reg_quiz.value == "MSE":
        _feedback = mo.callout(
            mo.md("✅ Correct! Squaring makes one large error count much more than several small ones, so MSE (and RMSE) is sensitive to outliers."),
            kind="success",
        )
    else:
        _feedback = mo.callout(
            mo.md("❌ Not quite. Look again at the *Common Metrics* section: which one squares the errors?"),
            kind="danger",
        )
    mo.vstack([reg_quiz, _feedback])
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
