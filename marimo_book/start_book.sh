#!/bin/bash

# Machine Learning One-Day Course - Marimo Book Launcher
# This script helps you start the marimo book

# Run from the folder that contains this script, so it works from anywhere
cd "$(dirname "$0")" || exit 1

echo "🎓 Machine Learning One-Day Course - Marimo Book"
echo "================================================"
echo ""

# Check if marimo is installed
if ! command -v marimo &> /dev/null; then
    echo "❌ Marimo is not installed!"
    echo "📦 Installing marimo and dependencies..."
    pip install -r ../requirements.txt
fi

echo "✅ Marimo is installed!"
echo ""
echo "📚 Available options:"
echo ""
echo "  1. Open the Index (Table of Contents)"
echo "  2. Chapter 1: Introduction to Machine Learning"
echo "  3. Chapter 2: Understanding ML Workflow"
echo "  4. Chapter 3.1: Supervised Learning - Regression"
echo "  5. Chapter 3.2: Supervised Learning - Classification"
echo "  6. Chapter 4.1: Unsupervised Learning - Clustering"
echo "  7. Chapter 4.2: Unsupervised Learning - Other Techniques"
echo "  8. Chapter 5: In-Class Assignment"
echo "  9. Browse all chapters (marimo file browser)"
echo " 10. Exit"
echo ""

# Default to index if no argument provided
if [ $# -eq 0 ]; then
    read -p "Enter your choice (1-10): " choice
else
    choice=$1
fi

case $choice in
    1)
        echo "🚀 Opening Index..."
        marimo edit 0-Index.py
        ;;
    2)
        echo "🚀 Opening Chapter 1..."
        marimo edit 1-Introduction_to_Machine_Learning.py
        ;;
    3)
        echo "🚀 Opening Chapter 2..."
        marimo edit 2-Understanding_ML_Workflow.py
        ;;
    4)
        echo "🚀 Opening Chapter 3.1..."
        marimo edit 3-Supervised-1-Regression.py
        ;;
    5)
        echo "🚀 Opening Chapter 3.2..."
        marimo edit 3-Supervised-2-Classification.py
        ;;
    6)
        echo "🚀 Opening Chapter 4.1..."
        marimo edit 4-Unsupervised-1-Clustering.py
        ;;
    7)
        echo "🚀 Opening Chapter 4.2..."
        marimo edit 4-Unsupervised-2-Others.py
        ;;
    8)
        echo "🚀 Opening Chapter 5..."
        marimo edit 5-In-Class-assignment.py
        ;;
    9)
        echo "🚀 Opening the file browser with all chapters..."
        marimo edit .
        ;;
    10)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run again and select 1-10."
        exit 1
        ;;
esac
