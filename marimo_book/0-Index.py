import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # 🎓 Machine Learning One-Day Course Book

    Welcome to the interactive Machine Learning course! This book is built using marimo, a reactive Python notebook environment.

    ## 📚 Table of Contents

    ### Getting Started
    Click on any chapter below to start learning. Each chapter is an interactive marimo notebook that you can run and experiment with.

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _():
    chapters = [
        {
            "number": "1",
            "title": "Introduction to Machine Learning",
            "file": "1-Introduction_to_Machine_Learning.py",
            "playground": "Regression and k-means playgrounds, quiz",
            "description": "Learn the fundamentals of ML, types of learning, and key terminology.",
            "duration": "45 min"
        },
        {
            "number": "2",
            "title": "Understanding ML Workflow",
            "file": "2-Understanding_ML_Workflow.py",
            "playground": "Overfitting / train-test split playground, quiz",
            "description": "Explore the complete pipeline for machine learning projects.",
            "duration": "45 min"
        },
        {
            "number": "3.1",
            "title": "Supervised Learning - Regression",
            "file": "3-Supervised-1-Regression.py",
            "playground": "Polynomial degree and residuals playground, quiz",
            "description": "Master regression techniques for predicting continuous values.",
            "duration": "60 min"
        },
        {
            "number": "3.2",
            "title": "Supervised Learning - Classification",
            "file": "3-Supervised-2-Classification.py",
            "playground": "Model and decision-threshold playground, quiz",
            "description": "Learn classification methods for discrete categories.",
            "duration": "60 min"
        },
        {
            "number": "4.1",
            "title": "Unsupervised Learning - Clustering",
            "file": "4-Unsupervised-1-Clustering.py",
            "playground": "Choosing k: elbow and silhouette playground, quiz",
            "description": "Discover how to group similar data points using clustering.",
            "duration": "45 min"
        },
        {
            "number": "4.2",
            "title": "Unsupervised Learning - Other Techniques",
            "file": "4-Unsupervised-2-Others.py",
            "playground": "PCA components and Isolation Forest playgrounds, quiz",
            "description": "Explore dimensionality reduction and anomaly detection.",
            "duration": "45 min"
        },
        {
            "number": "5",
            "title": "In-Class Assignment",
            "file": "5-In-Class-assignment.py",
            "playground": "Optional model and hyperparameter explorer",
            "description": "Apply your knowledge by building a classifier model.",
            "duration": "90 min"
        },
    ]
    return (chapters,)


@app.cell(hide_code=True)
def _(chapters, mo):
    # Summary table of chapters with durations and quick commands
    header = "| # | Title | Duration | Interactive | Open Command |\n|---|-------|----------|-------------|--------------|\n"
    rows = []
    for ch in chapters:
        cmd = f"marimo edit {ch['file']}"
        rows.append(f"| {ch['number']} | {ch['title']} | {ch.get('duration','')} | {ch.get('playground','')} | `{cmd}` |")
    mo.md(header + "\n".join(rows))
    return


@app.cell(hide_code=True)
def _(chapters, mo):
    # Create chapter cards
    chapter_cards = []

    for chapter in chapters:
        card_content = f"""
        ### Chapter {chapter['number']}: {chapter['title']}

        {chapter['description']}

        **Duration:** {chapter.get('duration','')}

        **🧪 Interactive:** {chapter.get('playground','')}

        **File:** `{chapter['file']}`

        To open this chapter, run:
        ```bash
        marimo edit {chapter['file']}
        ```

        ---
        """
        chapter_cards.append(mo.md(card_content))

    mo.vstack(chapter_cards)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🚀 How to Use This Book

    ### Option 1: Navigate via Command Line
    Open any chapter by running:
    ```bash
    marimo edit <chapter-file>.py
    ```

    ### Option 2: Use Marimo's File Browser
    From the `marimo_book` folder, open the whole folder instead of a single file:
    ```bash
    marimo edit .
    ```
    Marimo shows a file browser in your browser where every chapter is one click away.

    ### Option 3: Run a Chapter as an App
    Open a chapter in read-only app mode (code hidden, widgets still interactive):
    ```bash
    marimo run <chapter-file>.py
    ```

    ## 🧪 Interactive Playgrounds

    Every chapter includes a **playground**: sliders and dropdowns that re-run the model as you change them. Chapters 1 to 4 also have a short quiz question. Look for the 🧪 headings.

    ## 💡 Key Features of Marimo

    - **Reactive Execution**: Cells automatically update when dependencies change
    - **No Hidden State**: No need to restart kernels or worry about execution order
    - **Pure Python**: Notebooks are stored as `.py` files, making them git-friendly
    - **Interactive**: Built-in UI elements for interactive data exploration

    ## 📦 Requirements

    Make sure you have all dependencies installed (run from the `marimo_book` folder):
    ```bash
    pip install -r ../requirements.txt
    ```

    ## 🎯 Learning Path

    1. **Start with Chapter 1** to understand ML fundamentals
    2. **Follow through Chapters 2-4** to learn different ML techniques
    3. **Complete Chapter 5** to apply your knowledge

    ### Quick Tips
    - You can open this index at any time with `marimo edit 0-Index.py`.
    - Prefer running chapters in order the first time.
    - Each chapter is self-contained and reactive.

    ---

    Happy Learning! 🌟
    """
    )
    return


if __name__ == "__main__":
    app.run()
