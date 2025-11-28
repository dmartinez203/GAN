# 🌐 Cloud Deployment Guide

This guide provides step-by-step instructions for deploying your GAN application to the cloud, satisfying the "cloud deployment" requirement mentioned in the assignment rubric.

---

## Table of Contents

1. [Streamlit Cloud Deployment (Recommended)](#streamlit-cloud-deployment-recommended)
2. [Google Colab Deployment](#google-colab-deployment)
3. [Heroku Deployment](#heroku-deployment-alternative)
4. [AWS/Azure Deployment](#awsazure-deployment-advanced)

---

## 🎯 Streamlit Cloud Deployment (Recommended)

**Easiest and free for public repos!**

### Prerequisites
- GitHub account
- Your GAN project code in a GitHub repository
- `requirements.txt` file (already included)

### Step 1: Prepare Your Repository

Ensure your repo has these files:
```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

### Step 2: Push to GitHub

```bash
# Initialize git repository (if not already done)
git init

# Add files
git add app.py requirements.txt README.md

# Commit
git commit -m "Add GAN Streamlit app for cloud deployment"

# Create a GitHub repository at github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/gan-image-generator.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit: https://share.streamlit.io
   - Click "Sign up" and use your GitHub account

2. **Create New App**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/gan-image-generator`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: Choose a custom name (e.g., `gan-demo-yourname`)

3. **Deploy**:
   - Click "Deploy!"
   - Wait 2-5 minutes for deployment
   - Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

4. **Share the Link**:
   - Copy the URL
   - Include it in your assignment submission
   - Test it in an incognito/private window to verify public access

### Troubleshooting Streamlit Cloud

**Issue**: Deployment fails with dependency errors

**Solution**: Check `requirements.txt` has exact versions:
```txt
tensorflow==2.13.0
streamlit==1.28.0
numpy==1.24.3
matplotlib==3.7.1
pillow==10.0.0
```

**Issue**: App crashes on startup

**Solution**: Add resource limits in `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200

[runner]
magicEnabled = true
```

---

## 📓 Google Colab Deployment

**Best for Jupyter notebooks with GPU access**

### Step 1: Upload Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook → Select `gan_mnist_notebook.ipynb`

### Step 2: Configure Runtime

1. Runtime → Change runtime type
2. Hardware accelerator → **GPU**
3. Click "Save"

### Step 3: Install Dependencies

Add this cell at the top of your notebook:

```python
# Install required packages
!pip install tensorflow numpy matplotlib tqdm Pillow

# Verify GPU availability
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)
```

### Step 4: Save to Google Drive (Optional)

To persist your trained model between sessions:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save model after training
generator.save('/content/drive/MyDrive/gan_generator.h5')
discriminator.save('/content/drive/MyDrive/gan_discriminator.h5')

# Load model in future sessions
# generator = tf.keras.models.load_model('/content/drive/MyDrive/gan_generator.h5')
```

### Step 5: Share Your Notebook

1. File → Share
2. Change to "Anyone with the link"
3. Copy the sharing link
4. Include in your assignment submission

**Colab Link Format**: 
```
https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID
```

---

## 🚀 Heroku Deployment (Alternative)

**For custom backend + frontend setup**

### Prerequisites
- Heroku account (free tier available)
- Heroku CLI installed
- Git repository

### Step 1: Prepare Files

Create `Procfile` in project root:
```
web: sh setup.sh && streamlit run app.py
```

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

Make executable:
```bash
chmod +x setup.sh
```

### Step 2: Deploy to Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create gan-image-generator-yourname

# Add buildpack for Python
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

### Step 3: Configure Resources

```bash
# Set environment variables (if needed)
heroku config:set TENSORFLOW_INTRA_OP_PARALLELISM=2
heroku config:set TENSORFLOW_INTER_OP_PARALLELISM=2

# Check logs
heroku logs --tail
```

**Note**: Heroku free tier has memory limits. Training large models may require paid dynos.

---

## ☁️ AWS/Azure Deployment (Advanced)

### AWS EC2 Deployment

1. **Launch EC2 Instance**:
   - Instance type: t2.medium or higher
   - AMI: Ubuntu 20.04 LTS
   - Security group: Open port 8501

2. **SSH into Instance**:
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Install Dependencies**:
```bash
sudo apt update
sudo apt install python3-pip -y
pip3 install -r requirements.txt
```

4. **Run Streamlit**:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

5. **Access**: `http://your-ec2-ip:8501`

### Azure App Service

1. Create App Service (Python 3.8+)
2. Deploy via Git or Azure CLI
3. Configure startup command: `streamlit run app.py --server.port 8000`

---

## 📊 Deployment Comparison

| Platform | Cost | Difficulty | GPU Support | Best For |
|----------|------|------------|-------------|----------|
| **Streamlit Cloud** | Free | ⭐ Easy | ❌ No | Quick demos, public repos |
| **Google Colab** | Free | ⭐ Easy | ✅ Yes | Notebooks, GPU training |
| **Heroku** | Free/$7+/mo | ⭐⭐ Medium | ❌ No | Custom apps, APIs |
| **AWS EC2** | $10+/mo | ⭐⭐⭐ Hard | ✅ Yes (GPU instances) | Production, scaling |
| **Azure** | $10+/mo | ⭐⭐⭐ Hard | ✅ Yes | Enterprise, integration |

---

## ✅ Assignment Submission Checklist

For the "cloud deployment" requirement, you must provide:

1. **Deployment Link**: Public URL to your deployed app
   - Streamlit Cloud: `https://your-app.streamlit.app`
   - Colab: `https://colab.research.google.com/drive/...`
   - Heroku: `https://your-app.herokuapp.com`

2. **Documentation**: Include in README or separate doc:
   - Which platform you used
   - Link to deployed application
   - Instructions for accessing (if auth required)
   - Screenshots of working app (optional but recommended)

3. **Explanations**: Add comments explaining:
   - Why you chose this platform
   - How the deployment works
   - Any limitations or constraints

### Example Submission Note

```
CLOUD DEPLOYMENT:
Platform: Streamlit Cloud
URL: https://gan-demo-alexmartinez.streamlit.app
Status: Live and publicly accessible

The application is deployed on Streamlit Cloud for easy access and 
demonstration. Users can train the GAN model, generate images, and 
view real-time metrics without any local setup.

Screenshots are included in the screenshots/ folder showing:
1. Initial interface
2. Training in progress
3. Generated images at epoch 100
4. Final results dashboard
```

---

## 🐛 Common Deployment Issues

### Issue: App won't start
**Solution**: Check logs for errors, verify `requirements.txt`

### Issue: Out of memory
**Solution**: Reduce batch size, use smaller model, upgrade instance

### Issue: Slow training
**Solution**: Use GPU instance (Colab), reduce epochs for demo

### Issue: Files not persisting
**Solution**: Use cloud storage (Google Drive, S3) for model checkpoints

---

## 📞 Support Resources

- **Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Colab FAQ**: https://research.google.com/colaboratory/faq.html
- **Heroku Python**: https://devcenter.heroku.com/categories/python-support
- **AWS Free Tier**: https://aws.amazon.com/free/

---

**Deployment Complete! 🎉**

Your GAN application is now accessible from anywhere in the world!
