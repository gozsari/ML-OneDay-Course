# 🎉 Marimo Book


## 📚 Contents

### Interactive Notebooks (index + 7 chapters)
✅ **0-Index.py** - Interactive table of contents with navigation  
✅ **1-Introduction_to_Machine_Learning.py** - ML fundamentals (regression and k-means playgrounds, quiz)  
✅ **2-Understanding_ML_Workflow.py** - Complete ML pipeline (overfitting / train-test split playground, quiz)  
✅ **3-Supervised-1-Regression.py** - Regression techniques (polynomial degree and residuals playground, quiz)  
✅ **3-Supervised-2-Classification.py** - Classification methods (model and decision-threshold playground, quiz)  
✅ **4-Unsupervised-1-Clustering.py** - Clustering algorithms (elbow and silhouette playground, quiz)  
✅ **4-Unsupervised-2-Others.py** - PCA and anomaly detection (PCA and Isolation Forest playgrounds, quiz)  
✅ **5-In-Class-assignment.py** - Hands-on assignment (optional model explorer)  



### Launch Scripts
✅ **start_book.sh** - Linux/Mac launcher (executable)  
✅ **start_book.bat** - Windows launcher  

## 🚀 How to Start

Install the dependencies first (from the repository root), following [HOW_TO_CREATE_ENV.md](../HOW_TO_CREATE_ENV.md):

```bash
pip install -r requirements.txt
```

### Quick Start (Recommended)

```bash
cd marimo_book
./start_book.sh
```

For Windows:

```bash
cd marimo_book
start_book.bat
```

Then select **Option 1** to open the interactive index, or **Option 9** to browse all chapters.

### Manual Start

Open the index:

```bash
cd marimo_book
marimo edit 0-Index.py
```

Or open the whole folder to get a file browser with every chapter:

```bash
cd marimo_book
marimo edit .
```

This will open in your browser at: http://localhost:2718

### GitHub Codespaces

Marimo also works inside a Codespace. From the repository root, run:

```bash
marimo edit marimo_book --host 0.0.0.0 --headless --no-token
```

Then open the forwarded port **2718** when Codespaces prompts you. Forwarded ports in Codespaces are private to your account by default, so you can leave marimo's own access token turned off. Don't make the port public.

