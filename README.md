

# Introduction to Machine Learning: One-Day Course 
This is a one-day machine learning introductory course for beginners. The course covers the basics of supervised and unsupervised learning, including regression, classification, clustering, dimensinality reduction and anomaly detection. It also includes hands-on exercises and examples using popular Machine Learning (ML) libraries like Scikit-learn.

The [presentation](presentation/ML_intro.pdf) is used to guide the instructor through the course, providing a structured outline of the topics to be covered. 

## Table of Contents
1. [Introduction to Machine Learning](#1-introduction-to-machine-learning)
2. [Understanding the Machine Learning Workflow](#2-understanding-the-machine-learning-workflow)
3. [Supervised Learning](#3-supervised-learning)
   - [3.1 Regression](#31-regression)
   - [3.2 Classification](#32-classification)
4. [Unsupervised Learning](#4-unsupervised-learning)
   - [4.1 Clustering](#41-clustering)
   - [4.2 Other Unsupervised Learning Techniques](#42-other-unsupervised-learning-techniques)
5. [In-Class Assignment](#5-in-class-assignment)

---

## 1. Introduction to Machine Learning
- **What is AI and ML?**
  - AI: The ability of machines to simulate intelligent behavior.
  - ML: A branch of AI where models are trained to learn from data and improve over time.
- **Applications**:
  - ChatGPT, Netflix recommendations, fraud detection, self-driving cars, etc.
- **Types of ML**:
  - Supervised Learning, Unsupervised Learning, Reinforcement Learning.
- **Key Terminology**:
  - Dataset, Features, Labels, Model, Training, Testing, Hyperparameters, Overfitting, Underfitting.

### Highlights
- Comparison between supervised and unsupervised learning using Linear Regression and K-Means examples.
- Basic visualizations of regression and clustering tasks.

**Related notebook:** [Introduction to Machine Learning](notebooks/1-Introduction_to_Machine_Learning.ipynb)

---

## 2. Understanding the Machine Learning Workflow

### Steps in the Workflow
1. **Define the Problem**: Set objectives (e.g., regression, classification).
2. **Collect and Clean Data**: Handle missing values, duplicates, outliers.
3. **Explore and Visualize Data**: Use summary statistics and visual tools like histograms and scatterplots.
4. **Feature Engineering**:
   - **Selection**: Remove irrelevant features.
   - **Transformation**: Normalize and encode data.
   - **Creation**: Generate new features.
5. **Split Data**: Divide into training, validation, and test sets.
6. **Choose and Train a Model**: Select an algorithm based on the task.
7. **Evaluate the Model**: Use metrics like RMSE, Accuracy, and Silhouette Score.
8. **Hyperparameter Optimization**: Use `GridSearchCV` or `RandomizedSearchCV` for fine-tuning.

### Highlights
- End-to-end example of an ML pipeline using Scikit-learn.
- Visualization of preprocessing and evaluation results.

**Related notebook:** [Machine Learning Workflow](notebooks/2-Understanding_ML_Workflow.ipynb)

---

## 3. Supervised Learning

### 3.1 Regression

- **Goal**: Predict continuous outputs (e.g., house prices, temperature).
- **Common Algorithms**:
  - Linear Regression, Polynomial Regression, Ridge, and Lasso.
- **Evaluation Metrics**:
  - MAE, MSE, RMSE, \( R^2 \).

#### Highlights
- Hands-on example of Linear Regression with visualization of results.
- Analysis of regression coefficients.

**Related notebook:** [Supervised Learning - Regression](notebooks/3-Supervised-1-Regression.ipynb)

---

### 3.2 Classification

- **Goal**: Predict discrete categories (e.g., spam detection, disease diagnosis).
- **Types**:
  - Binary, Multi-Class, Multi-Label Classification.
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

#### Highlights
- Logistic Regression example for binary classification.
- Hands-on exercise with Random Forest Classifier.
- Visualization of confusion matrix results.

**Related notebook:** [Supervised Learning - Classification](notebooks/3-Supervised-2-Classification.ipynb)

---

## 4. Unsupervised Learning

### 4.1 Clustering

- **Goal**: Group data points into clusters based on similarity without labels.
- **Types**:
  - Partition-Based: K-Means.
  - Hierarchical: Agglomerative Clustering.
  - Density-Based: DBSCAN.
- **Evaluation Metrics**:
  - Silhouette Score, Inertia, Visualization.

#### Highlights
- K-Means Clustering example with synthetic data.
- Visualizing clusters and centroids.

**Related notebook:** [Unsupervised Learning - Clustering](notebooks/4-Unsupervised-1-Clustering.ipynb)

---

### 4.2 Other Unsupervised Learning Techniques

- **Dimensionality Reduction**:
  - Reduces input features while preserving patterns.
  - Example: PCA (Principal Component Analysis).
- **Anomaly Detection**:
  - Identifies outliers or unusual patterns.
  - Examples: Isolation Forest, Z-scores.
- **Association Rule Mining**:
  - Finds relationships between items (e.g., market basket analysis).

#### Highlights
- PCA visualization of high-dimensional data projected into 2D.
- Hands-on example of Isolation Forest for anomaly detection.
- Apriori algorithm for discovering association rules.

**Related notebook:** [Unsupervised Learning - Other Techniques](notebooks/4-Unsupervised-2-Others.ipynb)

---

## 5. In-Class Assignment

- **Objective**:
  - Develop a classification model using a dataset of your choice.
  - Save the model as a pickle file.
- **Steps**:
  - Preprocess the data (handle missing values, encode categorical variables).
  - Train, evaluate and optimize the model.
  - Submit the pickle file of the trained model.

  **Related notebook:** [In-Class Assignment](notebooks/5-In-Class-assignment.ipynb)

---
## Using with GitHub Codespaces

You can run this repository directly in [GitHub Codespaces](https://github.com/features/codespaces) without needing to set up anything locally.

### Steps:
1. Click the **Code** button in the repository.
2. Select **Open with Codespaces**.
3. If you don’t see the option, click **Create codespace on main**.

Once the Codespace environment loads:
- All dependencies (e.g., Python packages) specified in `requirements.txt` will be automatically installed.
- You can open and run the Jupyter Notebooks directly in the Codespace.

---

### Acknowledgements
Thanks to [**Leon Boschman**](https://github.com/lboschman) for contributing his ideas, slides and feedback to this course material.