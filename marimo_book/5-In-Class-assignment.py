import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # In-class assignment: Build a classifier model

    ## Objective:

    Students will:

    - Develop and train a classification model using a dataset of your choice.
    - Save their trained models as pickle files.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 1: Load the Dataset
    Instructions for Students:
    - Load the dataset from scikit-learn.
    - Explore the dataset to understand its structure and key characteristics.
    """
    )
    return


@app.cell
def _():
    # Load dataset from scikit-learn
    from sklearn import datasets

    # load iris dataset
    iris = datasets.load_iris()
    return (iris,)


@app.cell
def _(iris):
    # visualize the data
    import matplotlib.pyplot as plt

    # get the first two features
    X = iris.data[:, :2]
    y = iris.target

    # plot the data
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Setosa')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Versicolor')
    plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='green', label='Virginica')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend()
    plt.show()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 2: Normalize the Features (if necessary)
    Instructions for Students:

    - If the dataset contains numerical features, normalize the features using a normalization technique.
    - If the dataset contains categorical features, encode the categorical features using an encoding technique.
    - Split the dataset into training and validation sets.
    """
    )
    return


@app.cell
def _(X, y):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split


    # Normalize numerical features
    scaler = StandardScaler() # You can also use MinMaxScaler, RobustScaler, etc. check https://scikit-learn.org/stable/modules/preprocessing.html
    X_norm = scaler.fit_transform(X)


    # Split the dataset

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42) # 80% training, 20% validation data (you can change the split ratio)

    print("Training set shape:", X_train.shape)
    print("Validation set shape:", X_test.shape)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 3: Hyperparameter Tuning and Model Training
    Instructions for Students:

    - Choose a classification model (e.g., Logistic Regression, Random Forest, Gradient Boosting, etc.) and train it on the training dataset.
    - Decide on the hyperparameters for the model and perform hyperparameter tuning using the training dataset.
    - Train the model using the training dataset and evaluate its performance on the validation dataset.
    - Save the trained model as a pickle file.
    """
    )
    return


@app.cell
def _(X_train, y_train):
    # Import the classifier from the sklearn library
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    # or
    # from sklearn.linear_model import LogisticRegression
    # or
    # from sklearn.svm import SVC
    # or
    # from sklearn.tree import DecisionTreeClassifier

    # Import the GridSearchCV or RandomizedSearchCV for hyperparameter tuning
    from sklearn.model_selection import GridSearchCV



    # Define the hyperparameters to tune and their possible values as a dictionary (param_grid)
    param_grid = {
        'n_estimators': [100, 200, 300],
        # this is to show format, you can add more hyperparameters

    }

    # Create the model
    clf = RandomForestClassifier() # Your classifier here, e.g. RandomForestClassifier(), DecisionTreeClassifier(), LogisticRegression(), etc.

    # Randomized search, you can also use GridSearchCV instead however RandomizedSearchCV is faster, GridSearchCV is exhaustive search over a specified parameter values
    # You may adjust n_iter, cv, and verbose parameters as needed
    # n_iter: Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    # cv: Determines the cross-validation splitting strategy. None, to use the default 3-fold cross validation.
    # verbose: Controls the verbosity: the higher, the more messages.
    clf_opt = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)

    # Fit the model

    clf_opt.fit(X_train, y_train)

    # print the best parameters
    print("Best parameters:", clf_opt.best_params_)
    return (clf_opt,)


@app.cell
def _(X_test, clf_opt, y_test):
    # Make predictions
    y_pred = clf_opt.predict(X_test)

    # Evaluate the model
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


    # if you are not satisfied with the model, you can retrain the model with different hyperparameters or try a different model

    # If you are satisfied with the model, you can save it for later use 
    import joblib
    joblib.dump(clf_opt, 'model.pkl')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Useful Links:
    - [Scikit-learn datasets](https://scikit-learn.org/1.5/api/sklearn.datasets.html)
    - [Scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
    - [Scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)
    - [Scikit-learn classifiers](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)




    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

