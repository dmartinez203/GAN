#!/bin/bash

# Quick Start Script for GAN Image Generator
# This script helps you get started quickly

echo "🎨 GAN-based Fake Image Generator - Quick Start"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Found: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "What would you like to do?"
echo "1) Install dependencies"
echo "2) Run Streamlit app"
echo "3) Run Jupyter notebook"
echo "4) Open HTML demo"
echo "5) Run complete setup (install + streamlit)"
echo "6) Exit"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "📦 Installing dependencies..."
        pip3 install -r requirements.txt
        echo "✅ Installation complete!"
        echo ""
        echo "Next steps:"
        echo "  - Run Streamlit: streamlit run app.py"
        echo "  - Run Jupyter: jupyter notebook gan_mnist_notebook.ipynb"
        ;;
    2)
        echo ""
        echo "🚀 Starting Streamlit app..."
        echo "The app will open in your browser at http://localhost:8501"
        echo ""
        streamlit run app.py
        ;;
    3)
        echo ""
        echo "📓 Starting Jupyter notebook..."
        if command -v jupyter &> /dev/null; then
            jupyter notebook gan_mnist_notebook.ipynb
        else
            echo "❌ Jupyter not found. Installing..."
            pip3 install jupyter
            jupyter notebook gan_mnist_notebook.ipynb
        fi
        ;;
    4)
        echo ""
        echo "🌐 Opening HTML demo..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open index.html
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            xdg-open index.html
        else
            echo "Please open index.html in your browser manually"
        fi
        ;;
    5)
        echo ""
        echo "🔧 Running complete setup..."
        echo ""
        echo "Step 1/2: Installing dependencies..."
        pip3 install -r requirements.txt
        echo ""
        echo "Step 2/2: Starting Streamlit app..."
        streamlit run app.py
        ;;
    6)
        echo "Goodbye! 👋"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "For more information, see README.md"
echo "For cloud deployment, see DEPLOYMENT.md"
