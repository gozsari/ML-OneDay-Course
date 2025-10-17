import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 3. Supervised Learning: Regression (Interactive)

    Welcome! This enhanced Marimo notebook turns the original lesson into a **hands-on lab**.
    Students can manipulate data, try different models, and **see in real time** how
    choices affect results. Use the controls and follow the prompts ✅

    **Learning goals**
    1. Understand what regression is and when to use it.
    2. Compare Linear, Ridge, Lasso, and Polynomial regression.
    3. Practice with train/test splits and cross‑validation.
    4. Interpret metrics (MAE, RMSE, R²) and plots (fit & residuals).
    """
    )
    return


@app.cell
def _():
    import marimo as _mo
    import numpy as _np

    ds_choice = _mo.ui.radio(
        options={
            "synthetic": "Synthetic (1D)",
            "diabetes": "Diabetes (sklearn)",
            "california": "California Housing (sklearn)",
            "upload": "Upload CSV",
        },
        value="synthetic",
        label="Dataset"
    )

    syn_n = _mo.ui.slider(50, 1000, value=200, step=10, label="Synthetic: #samples")
    syn_noise = _mo.ui.slider(0.0, 3.0, value=1.0, step=0.1, label="Synthetic: noise σ")
    syn_slope = _mo.ui.number(1.8, label="Synthetic: true slope m")
    syn_intercept = _mo.ui.number(2.0, label="Synthetic: true intercept b")
    syn_seed = _mo.ui.number(42, label="Random seed")

    uploaded = _mo.ui.file(label="Upload CSV (optional)")

    _mo.hstack([ds_choice]).layout(justify="start", gap="1rem")
    _mo.hstack([syn_n, syn_noise, syn_slope, syn_intercept, syn_seed]).layout(gap="0.75rem")
    return (
        ds_choice,
        syn_intercept,
        syn_n,
        syn_noise,
        syn_seed,
        syn_slope,
        uploaded,
    )


@app.cell
def _(
    ds_choice,
    mo,
    syn_intercept,
    syn_n,
    syn_noise,
    syn_seed,
    syn_slope,
    uploaded,
):
    import pandas as pd
    from sklearn.datasets import load_diabetes, fetch_california_housing

    def make_synthetic(n, slope, intercept, noise, seed):
        rng = _np.random.default_rng(seed)
        X = rng.normal(loc=1.5, scale=2.5, size=(n, 1))
        y = intercept + slope * X[:, 0] + rng.normal(0, noise, size=n)
        df = pd.DataFrame({"x": X[:, 0], "y": y})
        return df

    if ds_choice.value == "synthetic":
        df = make_synthetic(
            syn_n.value, syn_slope.value, syn_intercept.value, syn_noise.value, syn_seed.value
        )
        note = "Synthetic 1D dataset (one feature: x)"
    elif ds_choice.value == "diabetes":
        d = load_diabetes()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df["target"] = d.target
        note = "Diabetes regression dataset"
    elif ds_choice.value == "california":
        d = fetch_california_housing()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df["target"] = d.target
        note = "California Housing dataset"
    else:
        if uploaded.value is not None:
            df = pd.read_csv(uploaded.value)
            note = f"Uploaded CSV with shape {df.shape}"
        else:
            df = make_synthetic(200, 1.8, 2.0, 1.0, 42)
            note = "No CSV uploaded → fallback to synthetic"

    mo.md(f"**Dataset:** {note} — shape **{df.shape}**")
    df.head()
    return (df,)


@app.cell
def _(df, mo):
    import numpy as np

    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    default_target = "y" if "y" in df.columns else ("target" if "target" in df.columns else numeric_cols[-1])
    default_features = [c for c in numeric_cols if c != default_target]
    if len(default_features) == 0:
        default_features = [numeric_cols[0]]

    target_sel = mo.ui.dropdown(numeric_cols, value=default_target, label="Target column")
    feat_sel = mo.ui.multiselect(numeric_cols, value=default_features[:2], label="Feature column(s)")

    mo.hstack([target_sel, feat_sel]).layout(gap="1rem")
    return feat_sel, target_sel


@app.cell
def _(mo):
    model_choice = mo.ui.radio(
        options=[
            ("LinearRegression", "lin"),
            ("Ridge", "ridge"),
            ("Lasso", "lasso"),
        ],
        value="lin",
        label="Model"
    )

    degree = mo.ui.slider(1, 10, value=1, step=1, label="Polynomial degree")
    alpha = mo.ui.slider(0.0001, 10.0, value=1.0, step=0.0001, label="Regularization α (Ridge/Lasso)")
    standardize = mo.ui.switch(value=True, label="Standardize features")

    test_size = mo.ui.slider(0.1, 0.5, value=0.2, step=0.05, label="Test size (fraction)")
    random_state = mo.ui.number(42, label="Random state")

    mo.hstack([model_choice, degree, alpha, standardize]).layout(gap="0.75rem")
    mo.hstack([test_size, random_state]).layout(gap="0.75rem")
    return alpha, degree, model_choice, random_state, standardize, test_size


@app.cell
def _(alpha, degree, df, feat_sel, model_choice, target_sel):
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    feats = list(feat_sel.value) if isinstance(feat_sel.value, (list, tuple)) else [feat_sel.value]
    target = target_sel.value

    X = df[feats].copy()
    y = df[target].values

    # Build pipeline
    steps = []
    if degree.value > 1:
        steps.append(("poly", PolynomialFeatures(include_bias=False, degree=degree.value)))

    # Standardization controlled outside (added in next cell)
    model_key = model_choice.value
    if model_key == "lin":
        model = LinearRegression()
    elif model_key == "ridge":
        model = Ridge(alpha=alpha.value)
    else:
        model = Lasso(alpha=alpha.value, max_iter=10000)

    # We'll finalize steps (incl. scaler) in the next cell for clarity
    return X, feats, model, y


@app.cell
def _(X, degree, mo, model, random_state, standardize, test_size, y):
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    steps = []
    if degree.value > 1:
        steps.append(("poly", PolynomialFeatures(include_bias=False, degree=degree.value)))

    if standardize.value:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    pipe = Pipeline(steps)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size.value, random_state=int(random_state.value)
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mo.md(f"**Metrics (test):** MAE = `{mae:.3f}`, RMSE = `{rmse:.3f}`, R² = `{r2:.3f}`")
    return X_test, pipe, y_pred, y_test


@app.cell
def _(X_test, feats, y_pred, y_test):
    import numpy as np
    import matplotlib.pyplot as plt

    # If 1 feature: show fit line on sorted feature
    if X_test.shape[1] == 1:
        xs = X_test.iloc[:, 0].to_numpy()
        order = np.argsort(xs)
        xs_sorted = xs[order]
        y_sorted = y_test[order]
        yp_sorted = y_pred[order]

        fig1, ax1 = plt.subplots()
        ax1.scatter(xs_sorted, y_sorted, label="Actual", alpha=0.7)
        ax1.plot(xs_sorted, yp_sorted, label="Predicted", linewidth=2)
        ax1.set_xlabel(feats[0])
        ax1.set_ylabel("Target")
        ax1.set_title("Prediction line vs actual (test set)")
        ax1.legend()
        plt.show()
    else:
        # If multi-feature: scatter y_true vs y_pred
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred, alpha=0.7)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax1.plot(lims, lims, linestyle="--")
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_title("Actual vs Predicted (test set)")
        plt.show()

    # Residuals plot
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_pred, residuals, alpha=0.7)
    ax2.axhline(0, linestyle="--")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual (y - ŷ)")
    ax2.set_title("Residuals vs Predicted (test set)")
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(df, feat_sel, mo, pipe, target_sel):
    from sklearn.model_selection import cross_val_score
    import numpy as np

    feats = list(feat_sel.value) if isinstance(feat_sel.value, (list, tuple)) else [feat_sel.value]
    X_full = df[feats]
    y_full = df[target_sel.value]

    k = mo.ui.slider(3, 10, value=5, step=1, label="Cross‑validation folds (k)")
    mo.hstack([k])

    r2_scores = cross_val_score(pipe, X_full, y_full, cv=k.value, scoring="r2")
    neg_mse = cross_val_score(pipe, X_full, y_full, cv=k.value, scoring="neg_mean_squared_error")

    mo.md(
        f"**CV R²:** mean=`{r2_scores.mean():.3f}` ± `{r2_scores.std():.3f}`  "+
        f"**CV RMSE:** mean=`{np.sqrt((-neg_mse).mean()):.3f}`"
    )
    return (feats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Try it yourself 🧪
    - Increase the **polynomial degree** until you start to overfit (watch R² on test vs CV).
    - Switch between **Linear, Ridge, and Lasso**; tune **α**. What changes?
    - Upload a small CSV, pick the **target**, and compare feature subsets.

    ### Quick quiz ✅
    """
    )
    return


@app.cell
def _(mo):
    q1 = mo.ui.radio(
        [
            ("Regression predicts continuous targets.", True),
            ("Regression predicts discrete class labels.", False),
        ],
        value=None,
        label="Q1: Which statement is true?",
    )

    q2 = mo.ui.radio(
        [
            ("MAE is more sensitive to large errors than MSE.", False),
            ("MSE penalizes large errors more than MAE.", True),
        ],
        value=None,
        label="Q2: MAE vs MSE",
    )

    check = mo.ui.button(label="Check answers")

    out = mo.ui.text(label="Feedback", disabled=True)

    if check.value:
        correct = (q1.value is True) and (q2.value is True)
        msg = "✅ Correct!" if correct else "❌ Not quite — review the metrics section above."
        out.value = msg

    mo.vstack([q1, q2, check, out]).layout(gap="0.75rem")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Key takeaways
    1. **Model choice matters**: Ridge/Lasso help when features are noisy or numerous.
    2. **Capacity vs. data**: Higher polynomial degree can overfit — monitor CV.
    3. **Metrics tell the story**: Use MAE/RMSE/R² together, not in isolation.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
