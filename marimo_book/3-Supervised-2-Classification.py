import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 3.2 Classification 

    ## Objectives:
    1. Understand the concept of classification and its role in supervised learning.
    2. Explore the types of classification: binary, multi-class, and multi-label.
    3. Learn about common classification algorithms: Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN).
    4. Explore evaluation metrics for classification models.
    5. Hands-on: Implement a binary classification model using Scikit-learn.

    By the end of this session, you will be able to classify data into discrete categories, handle different classification scenarios, and evaluate model performance.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🤔 What is Classification?

    Classification is a supervised learning technique where the goal is to predict a discrete category (class) for an input.

    ---

    ### Examples of Classification:
    1. **Spam Detection**:
       - Input: Email text
       - Output: Spam or Not Spam (binary classification).

    2. **Disease Diagnosis**:
       - Input: Patient symptoms.
       - Output: Disease type (multi-class classification).

    3. **Image Recognition**:
       - Input: Image pixels.
       - Output: Multiple objects in the image (multi-label classification).

    ---

    ### How Classification Works:
    1. Train the model on labeled data (inputs with corresponding class labels).
    2. Learn decision boundaries or probabilities for each class.
    3. Use the trained model to classify new, unseen inputs.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔑 Types of Classification

    ### 1️⃣ **Binary Classification**:
    - **Definition**: Two possible class labels.
    - **Examples**:
      - Spam vs. Not Spam.
      - Disease diagnosis: Positive vs. Negative.
    - **Decision Boundary**: A line, curve, or hyperplane separates the two classes.

    ---

    ### 2️⃣ **Multi-Class Classification**:
    - **Definition**: More than two possible class labels, but each input is assigned to only one class.
    - **Examples**:
      - Image classification: Dog, Cat, or Bird.
      - Flower classification: Iris Setosa, Iris Versicolor, Iris Virginica.
    - **Special Requirement**: Algorithms like Logistic Regression and Decision Trees need to handle more than two classes.

    ---

    ### 3️⃣ **Multi-Label Classification**:
    - **Definition**: One input can belong to multiple classes simultaneously.
    - **Examples**:
      - Image with multiple objects: Dog, Ball, Tree.
      - Movie genres: A film can be both Comedy and Drama.
    - **Approach**: Typically solved by transforming the problem into multiple binary classification tasks.

    ---

    ### Key Differences:
    | Classification Type | Input Example            | Output Example            |
    |---------------------|--------------------------|---------------------------|
    | **Binary**          | Patient symptoms         | Disease: Yes or No        |
    | **Multi-Class**     | Image of a fruit         | Apple, Banana, or Orange  |
    | **Multi-Label**     | Movie description        | Genres: Comedy, Action    |

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🔑 Common Algorithms:
    1. **Logistic Regression**:
       - Despite its name, Logistic Regression is used for classification tasks.
       - Uses the sigmoid function to map predictions to probabilities:
         \[
         \sigma(x) = \frac{1}{1 + e^{-x}}
         \]
       - Example: Classify customers as "churn" or "not churn".

    2. **Decision Trees**:
       - Splits data into branches based on feature values, leading to leaf nodes that represent classes.
       - Easy to visualize and interpret.
       - Example: Classify animals based on characteristics (e.g., feathers, number of legs).

    3. **K-Nearest Neighbors (KNN)**:
       - A non-parametric method that assigns a class based on the majority class of the \( k \) nearest neighbors.
       - Works well for small datasets but can be computationally expensive for large datasets.

    ---

    ### Other Algorithms (Briefly Mentioned):
    - **Support Vector Machines (SVM)**: Separates classes with a hyperplane that maximizes the margin.
    - **Random Forest**: An ensemble method that combines multiple decision trees.
    - **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.

    **Scikit-learn** provides a wide range of classification algorithms and tools to build and evaluate models. See the [documentation](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) for more details.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Evaluation Metrics for Classification

    We evaluate classification models using various metrics to understand their performance. The common evaluation metrics are "Accuracy," "Precision," "Recall," "F1 Score," which are calculated based on the confusion matrix.

    ### 1️⃣ **Confusion Matrix**:
    - A table that summarizes the performance of a classification model.
    - Contains four metrics: True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN).
    - Used to calculate other evaluation metrics.

    | Actual/Predicted | Positive | Negative |
    |------------------|----------|----------|
    | Positive         | TP       | FP       |
    | Negative         | FN       | TN       |

    True Positive (TP): Correctly predicted positive instances.
    True Negative (TN): Correctly predicted negative instances.
    False Positive (FP): Incorrectly predicted as positive.
    False Negative (FN): Incorrectly predicted as negative.

    ### 2️⃣ **Accuracy**:
    - **Definition**: The proportion of correctly classified instances.
    - **Formula**: Accuracy = correct predictions / total predictions = (TP + TN) / (TP + TN + FP + FN).
    - Can be misleading for imbalanced datasets. :warning:

    ### 3️⃣ **Precision**:
    - **Definition**: The proportion of correctly predicted positive instances among all predicted positives.
    - **Formula**: Precision = TP / (TP + FP).
    - High precision indicates low false positive rate 
  
    ### 4️⃣ **Recall (Sensitivity)**:
    - **Definition**: The proportion of correctly predicted positive instances among all actual positives.
    - **Formula**: Recall = TP / (TP + FN).
    - High recall indicates low false negative rate.

    ### 5️⃣ **F1 Score**:
    - **Definition**: The harmonic mean of precision and recall.
    - **Formula**: F1 Score = 2 * (Precision * Recall) / (Precision + Recall).
    - Useful when you want to balance precision and recall.
    - F1 Score ranges from 0 to 1, where 1 is the best score.
    - F1 Score is a better metric for imbalanced datasets.

    For multi-class classification, these metrics can be calculated for each class separately or averaged across all classes. Also, you can use the "macro" or "weighted" average to handle class imbalance. 

    For more details on evaluation metrics, refer to the Scikit-learn documentation on [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📏 Evaluation Metrics for Classification

    We evaluate classification models using various metrics to understand their performance. Common evaluation metrics include accuracy, precision, recall, F1-score, and Matthew's correlation coefficient (MCC) based on the confusion matrix.

    ### Confusion Matrix:
    - A table summarizing true positives, false positives, true negatives, and false negatives.

    |                | Predicted Positive | Predicted Negative |
    |----------------|---------------------|---------------------|
    | **Actual Positive** | True Positive (TP)     | False Negative (FN)     |
    | **Actual Negative** | False Positive (FP)    | True Negative (TN)      |

    **True Positive (TP)**: Correctly predicted positive samples.

    **False Positive (FP)**: Incorrectly predicted positive samples.

    **True Negative (TN)**: Correctly predicted negative samples.

    **False Negative (FN)**: Incorrectly predicted negative samples.

    ### 1️⃣ Accuracy:
    - Percentage of correctly predicted samples.
    - Formula: Accuracy = \(\frac{TP + TN}{TP + TN + FP + FN}\)
    - Limitation: Can be misleading for imbalanced datasets.

    ### 2️⃣ Precision:
    - Of all the predicted positives, how many are truly positive?
    - Formula:
        \[
        \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
        \]

    ### 3️⃣ Recall (Sensitivity):
    - Of all the actual positives, how many were correctly predicted?
    - Formula:
        \[
        \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
        \]

    ### 4️⃣ F1-Score:
    - Harmonic mean of precision and recall. Useful when you want to balance both.
    - Formula:
        \[
        \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
        \]

    ### 5️⃣ Matthew's Correlation Coefficient (MCC):
    - Takes into account true and false positives and negatives. Useful for imbalanced datasets.
    - Formula:
        \[
        \text{MCC} = \frac{(TP \times TN) - (FP \times FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
        \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🛠️ Hands-on: Binary Classification with Scikit-learn

    Let's implement a binary classification model using Scikit-learn. We will use the Breast Cancer Wisconsin dataset to classify tumors as benign or malignant.
    """
    )
    return


@app.cell
def _():
    # Import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    return (
        ConfusionMatrixDisplay,
        LogisticRegression,
        classification_report,
        confusion_matrix,
        load_breast_cancer,
        np,
        plt,
        train_test_split,
    )


@app.cell
def _(load_breast_cancer):
    X, y = load_breast_cancer(return_X_y=True)

    # print the shape of X and y
    print(X.shape)
    print(y.shape)
    return X, y


@app.cell
def _(X, y):
    # see dataset as a panda dataframe
    import pandas as pd
    df = pd.DataFrame(data=X)
    df['target'] = y
    print(df.head())
    return (df,)


@app.cell
def _(df):
    print(df['target'].unique())
    return


@app.cell
def _(X, plt, y):
    # Visualize the data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Classes')
    plt.title('Breast Cancer Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return


@app.cell
def _(X, train_test_split, y):
    # 2. Split Data into Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(LogisticRegression, X_train, y_train):
    # 3. Train a Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(X_test, model):
    # 4. Make Predictions
    y_pred = model.predict(X_test)
    return (y_pred,)


@app.cell
def _(classification_report, y_pred, y_test):
    # 5. Evaluate the Model
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return


@app.cell
def _(ConfusionMatrixDisplay, confusion_matrix, model, plt, y_pred, y_test):
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    return


@app.cell
def _(np, y_pred, y_test):
    # Evaluate the model using the metrics
    accuracy = np.mean(y_test == y_pred)
    print(f"Accuracy: {accuracy}")

    # Precision, Recall, F1-Score
    precision = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test == 1)
    f1_score = 2 * precision * recall / (precision + recall)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧪 Interactive Playground: Models and the Decision Threshold

    A classifier usually outputs a **probability**, and we turn it into a class by comparing it with a **threshold** (0.5 by default).

    In this dataset label `0` means **malignant** and `1` means **benign**. In this playground we treat **malignant as the positive class**, because that's the case we care most about catching.

    - **Lower the threshold:** more tumours are flagged as malignant, so **recall goes up** (fewer missed cancers) and **precision goes down** (more false alarms).
    - **Raise the threshold:** the opposite happens.
    - **Switch the model** to see how different algorithms compare on the same split.
    """
    )
    return


@app.cell
def _(mo):
    # Controls for the classification playground
    clf_model_choice = mo.ui.dropdown(
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"],
        value="Logistic Regression",
        label="Model",
    )
    clf_threshold = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05, label="Decision threshold", show_value=True)
    mo.hstack([clf_model_choice, clf_threshold], justify="start", gap=2)
    return clf_model_choice, clf_threshold


@app.cell
def _(LogisticRegression, X_test, X_train, clf_model_choice, y_test, y_train):
    # Train the selected model and get the probability of "malignant" for each test sample
    from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
    from sklearn.pipeline import make_pipeline as _make_pipeline
    from sklearn.preprocessing import StandardScaler as _StandardScaler
    from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier

    _models = {
        "Logistic Regression": _make_pipeline(_StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": _DecisionTreeClassifier(max_depth=4, random_state=42),
        "K-Nearest Neighbors": _make_pipeline(_StandardScaler(), _KNeighborsClassifier(n_neighbors=15)),
    }
    # Positive class = malignant (label 0 in the original dataset)
    clf_y_train_pos = (y_train == 0).astype(int)
    clf_y_test_pos = (y_test == 0).astype(int)

    clf_playground_model = _models[clf_model_choice.value].fit(X_train, clf_y_train_pos)
    clf_test_proba = clf_playground_model.predict_proba(X_test)[:, 1]
    return clf_test_proba, clf_y_test_pos


@app.cell
def _(
    ConfusionMatrixDisplay,
    clf_model_choice,
    clf_test_proba,
    clf_threshold,
    clf_y_test_pos,
    confusion_matrix,
    mo,
    plt,
):
    # Apply the threshold, then show the confusion matrix, metrics, and probability distribution
    from sklearn.metrics import (
        accuracy_score as _accuracy_score,
        f1_score as _f1_score,
        matthews_corrcoef as _matthews_corrcoef,
        precision_score as _precision_score,
        recall_score as _recall_score,
    )

    _pred = (clf_test_proba >= clf_threshold.value).astype(int)
    _cm = confusion_matrix(clf_y_test_pos, _pred, labels=[0, 1])

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ConfusionMatrixDisplay(_cm, display_labels=["Benign", "Malignant"]).plot(ax=_ax1, cmap='Blues', colorbar=False)
    _ax1.set_title(f'{clf_model_choice.value} (threshold = {clf_threshold.value:.2f})')

    _ax2.hist(clf_test_proba[clf_y_test_pos == 0], bins=20, range=(0, 1), alpha=0.6, label='Actually benign')
    _ax2.hist(clf_test_proba[clf_y_test_pos == 1], bins=20, range=(0, 1), alpha=0.6, label='Actually malignant')
    _ax2.axvline(clf_threshold.value, color='red', linestyle='--', label='Threshold')
    _ax2.set_title('Predicted Probability of Malignant')
    _ax2.set_xlabel('Probability')
    _ax2.set_ylabel('Number of test samples')
    _ax2.legend()
    _fig.tight_layout()

    _tn, _fp, _fn, _tp = _cm.ravel()
    mo.vstack([
        mo.md(
            f"**Accuracy:** `{_accuracy_score(clf_y_test_pos, _pred):.3f}` &nbsp;|&nbsp; "
            f"**Precision:** `{_precision_score(clf_y_test_pos, _pred, zero_division=0):.3f}` &nbsp;|&nbsp; "
            f"**Recall:** `{_recall_score(clf_y_test_pos, _pred, zero_division=0):.3f}` &nbsp;|&nbsp; "
            f"**F1:** `{_f1_score(clf_y_test_pos, _pred, zero_division=0):.3f}` &nbsp;|&nbsp; "
            f"**MCC:** `{_matthews_corrcoef(clf_y_test_pos, _pred):.3f}`\n\n"
            f"Missed cancers (false negatives): **{_fn}** &nbsp;|&nbsp; False alarms (false positives): **{_fp}**"
        ),
        _fig,
    ])
    return


@app.cell
def _(mo):
    # Quick check question
    clf_quiz = mo.ui.radio(
        ["Accuracy", "Precision", "Recall"],
        value=None,
        label="In cancer screening, missing a malignant tumour is very costly. Which metric should you prioritise?",
    )
    return (clf_quiz,)


@app.cell(hide_code=True)
def _(clf_quiz, mo):
    if clf_quiz.value is None:
        _feedback = mo.md("Select an option above.")
    elif clf_quiz.value == "Recall":
        _feedback = mo.callout(
            mo.md("✅ Correct! Recall measures how many of the actual positives we catch. Try lowering the threshold above to see recall go up."),
            kind="success",
        )
    else:
        _feedback = mo.callout(
            mo.md("❌ Not quite. Which metric counts the *false negatives* (missed positive cases)?"),
            kind="danger",
        )
    mo.vstack([clf_quiz, _feedback])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📝 Key Takeaways:
    1. Classification is used to predict discrete categories (e.g., spam vs. not spam).
    2. Types include binary, multi-class, and multi-label classification.
    3. Common algorithms include Logistic Regression, Decision Trees, K-Nearest Neighbors (KNN) and many more.
    4. Evaluation metrics like accuracy, precision, recall, and F1-score help assess model performance.

    Next, we will explore **Unsupervised Learning** techniques!

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## In-class Activity:
    - Implement another classification algorithm using a different dataset.
    - Use scikit-learn to choose dataset and new classification algorithm.
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

