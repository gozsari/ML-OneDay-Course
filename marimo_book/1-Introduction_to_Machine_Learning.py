import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🌟 Introduction to Machine Learning

    Welcome to the first session of our Machine Learning course! 🎉

    ## What You’ll Learn:
    1. Understand what Machine Learning (ML) is.
    2. Explore the three main types of ML: Supervised, Unsupervised, and Reinforcement Learning.
    3. See examples of ML applications in various fields.
    4. Learn key ML terminology and the standard ML workflow.

    At the end of this session, you'll have a strong foundation to start building your own ML models!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🤖 What is Machine Learning?

    Machine Learning is a branch of Artificial Intelligence (AI) where computers are trained to learn from data and make decisions or predictions without being explicitly programmed.

    The goal of ML is to develop algorithms that can learn and improve over time. These algorithms are used to build models that can:
    - Make predictions (e.g., weather forecasts)
    - Classify data (e.g., spam detection)
    - Identify patterns (e.g., customer segmentation)
    - Optimize decisions (e.g., route planning)
    - And much more!


    ### 🔑 Key Characteristics:
    1. **Data-Driven**: ML relies on historical data to make predictions or decisions.
    2. **Self-Improving**: ML systems get better with experience as more variety of data becomes available.

    ### 🤔 Why is ML Important?

    ML powers some of the most impactful technologies in the world today:
    - Netflix and YouTube recommendations 🎥
    - Google Translate 🌐
    - Fraud detection in banking 🏦
    - Self-driving cars 🚗
    - Medical diagnosis 
    - And many more!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧠 Types of Machine Learning

    ML is broadly categorized into three types:

    1️⃣ **Supervised Learning**:
    - Trains on labeled data (inputs with known outputs).
    - **Example**: Predicting house prices based on features like square footage and number of bedrooms.
    - Example dataset format:

    | Area (sq. ft.) | Bedrooms | Price ($) |
    |----------------|----------|-----------|
    | 1500           | 3        | 200,000   |
    | 2000           | 4        | 300,000   |
    | 1200           | 2        | 150,000   |

    - The goal is to learn a mapping from inputs (features) to outputs (labels).

    2️⃣ **Unsupervised Learning**:
    - Trains on unlabeled data (no known outputs).
    - **Example**: Grouping customers into segments based on their shopping behavior.
    - Example dataset format:

    | Age | Income ($) | Married | Children | Homeowner |
    |-----|------------|---------|----------|----------|
    | 25  | 50,000     | No      | 0        | Yes      |
    | 45  | 80,000     | Yes     | 2        | No       |
    | 30  | 70,000     | Yes     | 3        | Yes      |

    - The goal is to find patterns or intrinsic structures in the data. There are no explicit labels (output) to predict. We have features and we need to find patterns in them.

    3️⃣ **Reinforcement Learning**:
    - Learns by interacting with an environment and receiving rewards or penalties.
    - **Example**: Training a robot to walk or teaching an AI to play chess.
    - No fixed dataset format.
    - The goal is to learn a sequence of actions that maximize a reward signal. Examples include game playing, robotics, and autonomous driving.
    - The agent learns to achieve a goal in an uncertain, potentially complex environment.

    ### 📚 Other Types of ML:
    There are several other types of ML that combine elements of supervised, unsupervised, and reinforcement learning:

     **Semi-Supervised Learning**:
    - Trains on a mix of labeled and unlabeled data.
    - Useful when labeled data is scarce or expensive to obtain.
    - Example: Image recognition with a few labeled images and many unlabeled images.

     **Self-Supervised Learning**:
    - Trains on data without human-annotated labels.
    - Labels are generated from the input data itself.
    - Example: Training a model to predict missing words in a sentence.

    **Transfer Learning**:
    - Transfers knowledge from one task to another.
    - Useful when you have a small dataset for the target task.
    - Example: Fine-tuning a pre-trained model on a new dataset.
    - Transfer learning is widely used in computer vision and natural language processing tasks.
    """
    )
    return


@app.cell
def _():
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    return KMeans, LinearRegression, np, plt


@app.cell
def _(LinearRegression, np):
    # Supervised Learning Example: Regression
    X_supervised = np.array([[1], [2], [3], [4], [5]])
    y_supervised = np.array([2, 4, 6, 8, 10])

    model = LinearRegression()
    model.fit(X_supervised, y_supervised)
    y_pred = model.predict(X_supervised)
    return X_supervised, y_pred, y_supervised


@app.cell
def _(KMeans, np):
    # Unsupervised Learning Example: Clustering
    X_unsupervised = np.random.rand(50, 2) * 100
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_unsupervised)
    return X_unsupervised, labels


@app.cell
def _(X_supervised, X_unsupervised, labels, plt, y_pred, y_supervised):
    # Plot both examples
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_supervised, y_supervised, color='blue', label='Data')
    plt.plot(X_supervised, y_pred, color='red', label='Model')
    plt.title('Supervised Learning: Linear Regression')
    plt.xlabel('Input Feature')
    plt.ylabel('Target Label')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X_unsupervised[:, 0], X_unsupervised[:, 1], c=labels, cmap='viridis')
    plt.title('Unsupervised Learning: K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🌍 Real-World Applications of Machine Learning

    ### 🔬 Healthcare:
    - **Disease Prediction**: Predicting diabetes risk based on patient data.
    - **Medical Imaging**: Detecting tumors in MRI scans.

    ### 💳 Finance:
    - **Fraud Detection**: Identifying suspicious transactions.
    - **Algorithmic Trading**: Predicting stock prices.

    ### 🛒 E-Commerce:
    - **Product Recommendations**: Suggesting products on Amazon.
    - **Customer Sentiment Analysis**: Analyzing reviews.

    ### 🚗 Autonomous Vehicles:
    - **Self-Driving Cars**: Detecting road signs and pedestrians.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📖 Machine Learning Terminology

    1️⃣ **Dataset**: A collection of data used for training and testing.

    2️⃣ **Feature**: An individual property of the data.

    3️⃣ **Label**: The target output in supervised learning.

    4️⃣ **Model**: A mathematical representation of the data.

    5️⃣ **Training**: The process of fitting a model to the data.

    6️⃣ **Testing**: Evaluating a model on unseen data.

    7️⃣ **Prediction**: An output generated by the model.

    8️⃣ **Hyperparameters**: Settings that control the learning process. They are set before training the model.

    9️⃣ **Evaluation Metric**: A measure used to assess model performance. 

    1️⃣0️⃣ **Overfitting**: When a model performs well on training data but poorly on unseen data.

    1️⃣1️⃣ **Underfitting**: When a model is too simple to capture the patterns in the data.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔄 ML Workflow

    1. **Define the Problem**: What are you trying to predict or solve?
    2. **Collect and Clean Data**: Ensure your data is accurate and complete.
    3. **Feature Engineering**: Select relevant features and transform the data into a suitable format(like one-hot encoding).
    4. **Split Data**: Divide into training, validation, and test sets.
    5. **Choose a Model**: Decide on a suitable algorithm (e.g., regression, clustering).
    6. **Hyperparameter Tuning**: Optimize the model settings for best performance.
    7. **Train the Model**: Teach the model to learn patterns in the training data.
    8. **Evaluate the Model**: Use metrics like accuracy or RMSE to measure performance.
    9. **Deploy the Model**: Use it in real-world applications.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Useful Libraries:
    - **Scikit-learn**: For building ML models. link: https://scikit-learn.org/stable/
    - **TensorFlow**: For deep learning. link: https://www.tensorflow.org/
    - **PyTorch**: For deep learning. link: https://pytorch.org/
    """
    )
    return


if __name__ == "__main__":
    app.run()
