# 📚 Course Content: Introduction to Machine Learning - One Day Workshop  

🔹 **Duration**: ~6-8 hours  
🔹 **Prerequisites**: Basic Python knowledge recommended  
🔹 **Slides**: [Course Presentation](presentation/ML_intro.pdf)  
🔹 **Notebooks**: [Course Materials](notebooks/)  

---

## 🗂️ Table of Contents  
- [📚 Course Content: Introduction to Machine Learning - One Day Workshop](#-course-content-introduction-to-machine-learning---one-day-workshop)
  - [🗂️ Table of Contents](#️-table-of-contents)
  - [📖 1. Introduction to Machine Learning](#-1-introduction-to-machine-learning)
    - [🔹 What is AI and ML?](#-what-is-ai-and-ml)
    - [🔹 Applications of ML](#-applications-of-ml)
    - [🔹 Types of ML](#-types-of-ml)
  - [📖 2. Understanding the Machine Learning Workflow](#-2-understanding-the-machine-learning-workflow)
  - [📖 3. Supervised Learning](#-3-supervised-learning)
    - [🔹 3.1 Regression](#-31-regression)
    - [🔹 3.2 Classification](#-32-classification)
  - [📖 4. Unsupervised Learning](#-4-unsupervised-learning)
    - [🔹 4.1 Clustering](#-41-clustering)
    - [🔹 4.2 Other Unsupervised Learning Techniques](#-42-other-unsupervised-learning-techniques)
  - [📖 5. In-Class Assignment](#-5-in-class-assignment)

---

## 📖 1. Introduction to Machine Learning  
⏳ **Estimated Time**: ~1 hour  

### 🔹 What is AI and ML?  
✔ AI: The ability of machines to simulate intelligent behavior.  
✔ ML: A subset of AI where models learn from data and improve over time.  

### 🔹 Applications of ML  
✅ ChatGPT, Netflix recommendations, fraud detection, self-driving cars.  

### 🔹 Types of ML  
- **Supervised Learning** (Regression & Classification)  
- **Unsupervised Learning** (Clustering & Dimensionality Reduction)  
- **Reinforcement Learning** (Learning from rewards & penalties)  

📂 **Notebook**: [Intro to ML](notebooks/1-Introduction_to_Machine_Learning.ipynb)  

---

## 📖 2. Understanding the Machine Learning Workflow  
⏳ **Estimated Time**: ~1 hour  

1️⃣ **Define the Problem** – What are you trying to solve?  
2️⃣ **Collect & Clean Data** – Handle missing values, outliers.  
3️⃣ **Explore & Visualize Data** – Understand relationships using histograms, scatterplots.  
4️⃣ **Feature Engineering** – Selecting, transforming, and creating features.  
5️⃣ **Train-Test Split** – Splitting data into training and testing sets.  
6️⃣ **Choose & Train a Model** – Select a machine learning algorithm.  
7️⃣ **Evaluate Model Performance** – RMSE, Accuracy, Precision, Recall.  
8️⃣ **Optimize Hyperparameters** – GridSearchCV, RandomizedSearchCV.  

📂 **Notebook**: [ML Workflow](notebooks/2-Understanding_ML_Workflow.ipynb)  

---

## 📖 3. Supervised Learning  
⏳ **Estimated Time**: ~2 hours  

### 🔹 3.1 Regression  
✔ Predicts **continuous values** (e.g., house prices).  
✔ Common algorithms:  
   - **Linear Regression**, **Polynomial Regression**, **Ridge**, **Lasso**.  
✔ Evaluation Metrics:  
   - MAE, MSE, RMSE, \( R^2 \) score.  

📂 **Notebook**: [Regression](notebooks/3-Supervised-1-Regression.ipynb)  

---

### 🔹 3.2 Classification  
✔ Predicts **discrete categories** (e.g., spam vs. not spam).  
✔ Types:  
   - **Binary Classification**, **Multi-Class Classification**, **Multi-Label Classification**.  
✔ Algorithms:  
   - **Logistic Regression**, **Random Forest**, **Support Vector Machines**.  
✔ Evaluation Metrics:  
   - Accuracy, Precision, Recall, F1-Score, Confusion Matrix.  

📂 **Notebook**: [Classification](notebooks/3-Supervised-2-Classification.ipynb)  

---

## 📖 4. Unsupervised Learning  
⏳ **Estimated Time**: ~2 hours  

### 🔹 4.1 Clustering  
✔ Groups similar data points together **without labels**.  
✔ Types:  
   - **Partition-Based**: K-Means  
   - **Hierarchical**: Agglomerative Clustering  
   - **Density-Based**: DBSCAN  
✔ Evaluation: Silhouette Score, Inertia.  

📂 **Notebook**: [Clustering](notebooks/4-Unsupervised-1-Clustering.ipynb)  

---

### 🔹 4.2 Other Unsupervised Learning Techniques  
✔ **Dimensionality Reduction**: PCA (Principal Component Analysis)  
✔ **Anomaly Detection**: Isolation Forest, Z-score  
✔ **Use Cases**: Fraud detection, noise reduction, pattern discovery.  

📂 **Notebook**: [Other Unsupervised Techniques](notebooks/4-Unsupervised-2-Others.ipynb)  

---

## 📖 5. In-Class Assignment  
⏳ **Estimated Time**: ~1-2 hours  

🔹 **Objective**: Build a classification model and submit predictions.  
🔹 **Steps**:  
1️⃣ Preprocess the dataset (handle missing values, categorical variables).  
2️⃣ Train and evaluate the model.  
3️⃣ Save the model as a `.pkl` file.  
4️⃣ Submit the final predictions.  

📂 **Notebook**: [Assignment](notebooks/5-In-Class-assignment.ipynb)  

---
