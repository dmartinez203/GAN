# 🎨 GAN-based Fake Image Generator

A comprehensive implementation of Generative Adversarial Networks (GANs) for generating synthetic MNIST digit images. This project includes multiple interfaces: Streamlit web app, Jupyter notebook, standalone HTML demo, and Python scripts.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Option 1: Streamlit Web App (Recommended)](#option-1-streamlit-web-app-recommended)
  - [Option 2: Jupyter Notebook](#option-2-jupyter-notebook)
  - [Option 3: Python Script](#option-3-python-script)
  - [Option 4: HTML Demo](#option-4-html-demo)
- [Google Colab Deployment](#google-colab-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Algorithm Overview](#algorithm-overview)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results & Analysis](#results--analysis)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## 🎯 Problem Statement

Fake images have become ubiquitous on social media, affecting public discourse on these platforms. It is essential for students to understand how these images are created. This project demonstrates how Generative Adversarial Networks (GANs) can generate synthetic images that look like real ones by training on the MNIST dataset of handwritten digits.

**Educational Goals:**
- Understand the architecture and training dynamics of GANs
- Learn how neural networks can generate realistic fake images
- Explore the balance between Generator and Discriminator in adversarial training
- Analyze the quality progression of generated images across training epochs

---

## ✨ Features

### 🖥️ Multiple Interfaces
- **Streamlit Web App**: Interactive training with real-time metrics and visualization
- **Jupyter Notebook**: Detailed analysis with code explanations and milestone plots
- **Python Script**: Command-line training for automation
- **HTML Demo**: Standalone demonstration page

### 🧠 Model Capabilities
- Fully-connected Generator network (3 layers + output)
- Discriminator network for binary classification
- LeakyReLU activation and Batch Normalization
- Support for 400+ epoch training with milestone checkpoints

### 📊 Visualization & Monitoring
- Real-time loss curves (Generator & Discriminator)
- Discriminator accuracy tracking
- Generated image grids at training milestones
- Interactive parameter controls

### 💾 Model Management
- Save/load trained models
- Export generated images
- Training history tracking

---

## 📁 Project Structure

```
GAN Based Application/
├── app.py                          # Streamlit web application
├── gan_mnist_notebook.ipynb        # Jupyter notebook with full analysis
├── gan_mnist_notebook.py           # Python script version
├── index.html                      # Standalone HTML demo
├── requirements.txt                # Python dependencies
├── colab_requirements.txt          # Google Colab requirements
├── colab_install.txt               # Colab installation commands
├── README.md                       # This file
├── gan_outputs/                    # Generated images (created during training)
│   ├── epoch_1.png
│   ├── epoch_30.png
│   ├── epoch_100.png
│   └── epoch_400.png
└── Front End example/              # Reference implementation (for learning)
```

---

## 🚀 Installation

### Prerequisites
- **Python**: 3.8 or higher
- **pip**: Latest version
- **Operating System**: Windows, macOS, or Linux
- **Optional**: GPU (NVIDIA CUDA or Apple Silicon) for faster training

### Step 1: Clone or Download Repository

```bash
# If using Git
git clone <repository-url>
cd "GAN Based Application"

# Or download and extract ZIP file, then navigate to folder
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install tensorflow numpy matplotlib tqdm pillow streamlit pandas
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import streamlit as st; print('Streamlit version:', st.__version__)"
```

---

## 🎮 Usage

### Option 1: Streamlit Web App (Recommended)

The Streamlit app provides the most interactive experience with real-time training visualization.

#### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

#### Using the Streamlit App

1. **Configure Parameters** (Sidebar):
   - Noise Dimension: 100 (default)
   - Learning Rate: 0.0002
   - Batch Size: 128
   - Epochs per training session: 1-100

2. **Initialize Models**:
   - Click "🚀 Initialize Models" in sidebar
   - Wait for confirmation message

3. **Train the GAN**:
   - Set desired number of epochs
   - Click "🏋️ Train GAN"
   - Watch real-time progress and metrics

4. **Generate Images**:
   - Navigate to "🖼️ Generate Images" tab
   - Select grid size
   - Click "🎲 Generate New Batch"

5. **View Metrics**:
   - Check "📈 Training Metrics" tab for loss curves
   - Monitor Discriminator accuracy over time

6. **Save Model**:
   - Click "💾 Save Generator" to persist trained model
   - Load later with "📂 Load Generator"

#### Streamlit Tips
- Train in small batches (10-20 epochs) to see incremental progress
- For best results, train 100-400 epochs total
- Use GPU if available (automatically detected)
- Generated images improve significantly after 100+ epochs

---

### Option 2: Jupyter Notebook

The notebook provides detailed explanations, code comments, and analysis.

#### Running the Notebook

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter notebook gan_mnist_notebook.ipynb
```

#### Using the Notebook

1. **Run cells sequentially** from top to bottom
2. **Key sections**:
   - Problem Statement & Algorithm
   - Model Building (Generator & Discriminator)
   - Data Loading & Preprocessing
   - Training Function
   - Training Execution (modify epochs as needed)
   - Analysis & Findings

3. **Training**:
   - Locate the "Run training" cell
   - Uncomment the training line:
     ```python
     history = train_gan(generator, discriminator, gan, x_train, noise_dim=noise_dim, epochs=400, batch_size=128)
     ```
   - For quick tests, use `epochs=10`
   - For assignment submission, use `epochs=400`

4. **Generated Images**:
   - Images saved in `gan_outputs/` folder
   - View milestones: epoch_1.png, epoch_30.png, epoch_100.png, epoch_400.png

---

### Option 3: Python Script

Run training directly from command line.

```bash
# Convert notebook to Python script (if needed)
jupyter nbconvert --to python gan_mnist_notebook.ipynb

# Run the script
python gan_mnist_notebook.py
```

**Note**: Modify the script to uncomment training lines before running.

---

### Option 4: HTML Demo

Open `index.html` in a web browser for a demonstration interface. This is a static demo showing the project structure and documentation.

**Note**: The HTML demo does not include actual model training. Use Streamlit or Jupyter for full functionality.

---

## ☁️ Google Colab Deployment

### Step 1: Upload Files to Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `gan_mnist_notebook.ipynb`
3. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator → GPU → Save

### Step 2: Install Dependencies

Add this cell at the beginning of the notebook:

```python
# Install required packages
!pip install tensorflow numpy matplotlib tqdm Pillow

# Verify GPU
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### Step 3: Run Training

Execute all cells sequentially. Training on Colab GPU is much faster than CPU.

**Colab Tips:**
- Free tier has session time limits (~12 hours)
- Save models to Google Drive to persist between sessions:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  generator.save('/content/drive/MyDrive/gan_generator.h5')
  ```

---

## 🌐 Cloud Deployment

### Deploying Streamlit App to Streamlit Cloud

1. **Push code to GitHub**:
   ```bash
   git init
   git add app.py requirements.txt
   git commit -m "Add GAN Streamlit app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Click "Deploy"

3. **Note**: Cloud deployment requires `requirements.txt` with all dependencies

### Alternative: Deploy to Heroku, AWS, or Azure

See deployment guides:
- [Streamlit deployment docs](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- [Heroku Python deployment](https://devcenter.heroku.com/articles/getting-started-with-python)

---

## 🧮 Algorithm Overview

### GAN Architecture

A GAN consists of two neural networks competing against each other:

1. **Generator (G)**: Creates fake images from random noise
2. **Discriminator (D)**: Distinguishes real images from fake ones

### Training Process

```
For each training iteration:
  1. Train Discriminator:
     a. Feed real images → D predicts "real" (label = 1)
     b. Generate fake images from noise → D predicts "fake" (label = 0)
     c. Update D weights to improve classification
  
  2. Train Generator:
     a. Generate fake images from noise
     b. Feed to D (with frozen weights)
     c. Update G weights to fool D (trick it into predicting "real")
  
  3. Repeat for multiple epochs until equilibrium
```

### Loss Functions

- **Discriminator Loss**: Binary cross-entropy on real/fake classification
- **Generator Loss**: Binary cross-entropy (trying to maximize D's error)

---

## 🏗️ Model Architecture

### Generator Network

```
Input: Random noise vector (100 dimensions)
│
├─ Dense(256) → LeakyReLU(0.2) → BatchNormalization
├─ Dense(512) → LeakyReLU(0.2) → BatchNormalization
├─ Dense(1024) → LeakyReLU(0.2) → BatchNormalization
├─ Dense(784) → Tanh
└─ Reshape → (28, 28, 1)

Output: 28×28 grayscale image, values in [-1, 1]
```

**Total Parameters**: ~1.2M

### Discriminator Network

```
Input: 28×28 grayscale image
│
├─ Flatten → (784,)
├─ Dense(1024) → LeakyReLU(0.2) → BatchNormalization
├─ Dense(512) → LeakyReLU(0.2) → BatchNormalization
├─ Dense(256) → LeakyReLU(0.2) → BatchNormalization
└─ Dense(1) → Sigmoid

Output: Probability [0, 1] (0=fake, 1=real)
```

**Total Parameters**: ~1.4M

---

## 🏋️ Training Process

### Hyperparameters (Default)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Noise Dimension | 100 | Size of input noise vector |
| Learning Rate | 0.0002 | Adam optimizer learning rate |
| Beta 1 | 0.5 | Adam optimizer momentum |
| Batch Size | 128 | Number of images per batch |
| Epochs | 400 | Total training iterations |

### Training Schedule

- **Epochs 1-30**: Generator learns basic structure, outputs are noisy
- **Epochs 30-100**: Recognizable digit shapes emerge
- **Epochs 100-200**: Details improve, fewer artifacts
- **Epochs 200-400**: High-quality digits, minimal noise

### Milestone Checkpoints

Images are saved at:
- Epoch 1 (baseline)
- Epoch 30 (early progress)
- Epoch 100 (mid-training)
- Epoch 200 (late-training)
- Epoch 400 (final result)

---

## 📊 Results & Analysis

### Expected Performance

After 400 epochs of training:

**Quantitative Metrics:**
- Generator Loss: ~0.5-1.5 (lower is better at fooling discriminator)
- Discriminator Loss: ~0.3-0.7 (balanced with generator)
- Discriminator Accuracy: ~50-70% (equilibrium indicator)

**Qualitative Assessment:**
- ✅ Clear digit shapes (0-9 recognizable)
- ✅ Smooth edges (minimal pixelation)
- ⚠️ Occasional artifacts (fully-connected architecture limitation)
- ⚠️ Some mode collapse (certain digits more common)

### Comparison: Fully-Connected vs Convolutional

This implementation uses **fully-connected (Dense) layers** for simplicity. For production-quality images, consider:

- **DCGAN (Deep Convolutional GAN)**: Uses Conv2D/ConvTranspose2D layers
- **Progressive GAN**: Gradually increases image resolution
- **StyleGAN**: State-of-the-art image generation

### Sample Output Progression

```
Epoch 1:    [noise/static]
Epoch 30:   [vague shapes]
Epoch 100:  [recognizable digits, blurry]
Epoch 400:  [clear digits with minor artifacts]
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Errors

**Problem**: `ImportError: No module named 'tensorflow'`

**Solution**:
```bash
pip install --upgrade pip
pip install tensorflow>=2.10.0
```

For Apple Silicon Mac (M1/M2/M3):
```bash
pip install tensorflow-macos tensorflow-metal
```

#### 2. Streamlit Not Found

**Problem**: `streamlit: command not found`

**Solution**:
```bash
pip install streamlit
# Verify installation
streamlit --version
```

#### 3. Training is Very Slow

**Solutions**:
- Reduce batch size: `batch_size=64` or `32`
- Reduce epochs for testing: `epochs=10`
- Use GPU if available (Colab, cloud instance)
- Train overnight for full 400 epochs

#### 4. Out of Memory Errors

**Solutions**:
- Reduce batch size: `batch_size=32`
- Close other applications
- Use Colab with GPU runtime
- Restart kernel/runtime

#### 5. Generated Images Look Like Noise

**Possible Causes**:
- Model not trained enough (train more epochs)
- Model not initialized (check "Initialize Models" was clicked)
- Training diverged (restart with fresh model)

#### 6. Streamlit Port Already in Use

**Problem**: `Port 8501 is already in use`

**Solution**:
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## 📚 References

### Academic Papers

1. **Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y.** (2014).  
   *Generative adversarial nets.*  
   Advances in neural information processing systems, 27.

2. **Radford, A., Metz, L., & Chintala, S.** (2015).  
   *Unsupervised representation learning with deep convolutional generative adversarial networks.*  
   arXiv preprint arXiv:1511.06434.

3. **Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X.** (2016).  
   *Improved techniques for training gans.*  
   Advances in neural information processing systems, 29.

### Online Resources

- **TensorFlow Documentation**: https://www.tensorflow.org/guide/keras
- **Keras Examples - GANs**: https://keras.io/examples/generative/dcgan_overriding_train_step/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/

### Tutorials

- **DeepLearning.AI GAN Specialization**: https://www.deeplearning.ai/courses/generative-adversarial-networks-gans-specialization/
- **PyImageSearch GAN Tutorial**: https://pyimagesearch.com/2021/09/13/intro-to-generative-adversarial-networks-gans/

---

## 📝 Assignment Submission Checklist

Before submitting, ensure:

- ✅ **Code runs without errors** (test all interfaces)
- ✅ **Training completed** (at least 400 epochs)
- ✅ **Milestone images generated** (epoch_1, epoch_30, epoch_100, epoch_400)
- ✅ **Streamlit app functional** (demonstrates understanding of deployment)
- ✅ **Jupyter notebook complete** with:
  - Problem statement
  - Algorithm explanation
  - Code with comments
  - Generated images at milestones
  - Analysis of findings
  - References
- ✅ **README.md present** (this file, with instructions)
- ✅ **Cloud deployment** (Streamlit Cloud or Colab link)
- ✅ **Code explanations included** (comments, docstrings, markdown cells)

---

## 🤝 Contributing

This is an educational project. For improvements or bug fixes:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is created for educational purposes. Feel free to use and modify for learning.

---

## 👨‍💻 Author

Created as part of a machine learning course assignment demonstrating GANs for synthetic image generation.

---

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- TensorFlow/Keras teams
- Streamlit for the web framework
- Ian Goodfellow for inventing GANs

---

## 📞 Support

For questions or issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review error messages carefully
3. Consult TensorFlow/Streamlit documentation
4. Ask your instructor or TA

---

**Last Updated**: November 2025  
**Version**: 1.0.0

---

## Quick Command Reference

```bash
# Installation
pip install -r requirements.txt

# Run Streamlit App
streamlit run app.py

# Run Jupyter Notebook
jupyter notebook gan_mnist_notebook.ipynb

# Convert notebook to Python
jupyter nbconvert --to python gan_mnist_notebook.ipynb

# Run Python script
python gan_mnist_notebook.py

# View HTML demo
open index.html  # macOS
start index.html  # Windows
```

**Happy Training! 🎨🚀**
