"""
GAN-based Fake Image Generator - Streamlit Application
Interactive web app for training and generating MNIST digit images using GANs
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
import os
from io import BytesIO
import time

# Page configuration
st.set_page_config(
    page_title="GAN Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'discriminator' not in st.session_state:
    st.session_state.discriminator = None
if 'gan' not in st.session_state:
    st.session_state.gan = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
if 'trained_epochs' not in st.session_state:
    st.session_state.trained_epochs = 0
if 'x_train' not in st.session_state:
    st.session_state.x_train = None

# Model building functions
@st.cache_resource
def build_generator(noise_dim=100):
    """Build generator network"""
    model = keras.Sequential(name='Generator')
    model.add(layers.Input(shape=(noise_dim,)))
    # Layer 1
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    # Layer 2
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    # Layer 3
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    # Output layer
    model.add(layers.Dense(28 * 28, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

@st.cache_resource
def build_discriminator(input_shape=(28, 28, 1)):
    """Build discriminator network"""
    model = keras.Sequential(name='Discriminator')
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    # Layer 1
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    # Layer 2
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    # Layer 3
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    # Output probability
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

@st.cache_data
def load_mnist_data():
    """Load and preprocess MNIST data"""
    (x_train, _), (_, _) = datasets.mnist.load_data()
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)
    return x_train

def plot_generated_images(generator, noise_input, n_images=25):
    """Generate and plot images"""
    gen_images = generator.predict(noise_input, verbose=0)
    gen_images = (gen_images + 1.0) / 2.0
    
    side = int(np.sqrt(n_images))
    fig, axes = plt.subplots(side, side, figsize=(8, 8))
    fig.patch.set_facecolor('#f0f4ff')
    
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx < n_images:
                axes[i, j].imshow(gen_images[idx, :, :, 0], cmap='gray')
                axes[i, j].axis('off')
                idx += 1
    
    plt.tight_layout()
    return fig

def train_gan_epochs(generator, discriminator, gan, data, noise_dim, epochs, batch_size, progress_bar, status_text):
    """Train GAN for specified epochs"""
    half_batch = batch_size // 2
    batches_per_epoch = data.shape[0] // batch_size
    
    real_label = np.ones((half_batch, 1))
    fake_label = np.zeros((half_batch, 1))
    
    history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
    
    for epoch in range(1, epochs + 1):
        d_losses = []
        d_accs = []
        g_losses = []
        
        idx = np.random.permutation(data.shape[0])
        data_shuffled = data[idx]
        
        for batch in range(batches_per_epoch):
            start = batch * batch_size
            end = start + half_batch
            
            # Train Discriminator
            real_imgs = data_shuffled[start:end]
            d_loss_real = discriminator.train_on_batch(real_imgs, real_label)
            
            noise = np.random.normal(0, 1, (half_batch, noise_dim))
            gen_imgs = generator.predict(noise, verbose=0)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_label)
            
            if isinstance(d_loss_real, list):
                d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
                d_acc = 0.5 * (d_loss_real[1] + d_loss_fake[1])
            else:
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_acc = 0.0
            
            d_losses.append(d_loss)
            d_accs.append(d_acc)
            
            # Train Generator
            noise2 = np.random.normal(0, 1, (batch_size, noise_dim))
            g_loss = gan.train_on_batch(noise2, np.ones((batch_size, 1)))
            g_losses.append(g_loss)
        
        history['d_loss'].append(np.mean(d_losses))
        history['d_acc'].append(np.mean(d_accs))
        history['g_loss'].append(np.mean(g_losses))
        
        progress_bar.progress(epoch / epochs)
        status_text.text(f'Epoch {epoch}/{epochs} - D_loss: {history["d_loss"][-1]:.4f}, G_loss: {history["g_loss"][-1]:.4f}')
    
    return history

# Header
st.markdown('<h1 class="main-header">🎨 GAN-based Fake Image Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Train a Generative Adversarial Network to create synthetic MNIST digits</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    noise_dim = st.number_input("Noise Dimension", min_value=50, max_value=200, value=100, step=10)
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.0002, step=0.0001, format="%.4f")
    batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2)
    
    st.markdown("---")
    st.header("🎯 Training")
    
    epochs_to_train = st.slider("Epochs to Train", min_value=1, max_value=100, value=10, step=1)
    
    if st.button("🚀 Initialize Models", use_container_width=True):
        with st.spinner("Building models..."):
            st.session_state.generator = build_generator(noise_dim)
            st.session_state.discriminator = build_discriminator()
            
            # Compile discriminator
            opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
            st.session_state.discriminator.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            
            # Build GAN
            st.session_state.discriminator.trainable = False
            gan_input = layers.Input(shape=(noise_dim,))
            gan_output = st.session_state.discriminator(st.session_state.generator(gan_input))
            st.session_state.gan = models.Model(inputs=gan_input, outputs=gan_output, name='GAN')
            st.session_state.gan.compile(optimizer=opt, loss='binary_crossentropy')
            
            # Load data
            st.session_state.x_train = load_mnist_data()
            st.session_state.trained_epochs = 0
            st.session_state.training_history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
            
        st.success("✅ Models initialized!")
    
    if st.button("🏋️ Train GAN", use_container_width=True, disabled=st.session_state.generator is None):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner(f"Training for {epochs_to_train} epochs..."):
            history = train_gan_epochs(
                st.session_state.generator,
                st.session_state.discriminator,
                st.session_state.gan,
                st.session_state.x_train,
                noise_dim,
                epochs_to_train,
                batch_size,
                progress_bar,
                status_text
            )
            
            # Update session state
            st.session_state.training_history['d_loss'].extend(history['d_loss'])
            st.session_state.training_history['g_loss'].extend(history['g_loss'])
            st.session_state.training_history['d_acc'].extend(history['d_acc'])
            st.session_state.trained_epochs += epochs_to_train
        
        st.success(f"✅ Training complete! Total epochs: {st.session_state.trained_epochs}")
    
    st.markdown("---")
    st.header("💾 Model Management")
    
    if st.button("💾 Save Generator", use_container_width=True, disabled=st.session_state.generator is None):
        st.session_state.generator.save('saved_generator.h5')
        st.success("✅ Generator saved!")
    
    if st.button("📂 Load Generator", use_container_width=True):
        if os.path.exists('saved_generator.h5'):
            st.session_state.generator = keras.models.load_model('saved_generator.h5')
            st.success("✅ Generator loaded!")
        else:
            st.error("❌ No saved model found!")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🖼️ Generate Images", "📈 Training Metrics", "ℹ️ About"])

with tab1:
    st.header("Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Epochs</h3>
            <h1>{st.session_state.trained_epochs}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.training_history['g_loss']:
            latest_g_loss = st.session_state.training_history['g_loss'][-1]
        else:
            latest_g_loss = 0.0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Generator Loss</h3>
            <h1>{latest_g_loss:.4f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.training_history['d_loss']:
            latest_d_loss = st.session_state.training_history['d_loss'][-1]
        else:
            latest_d_loss = 0.0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Discriminator Loss</h3>
            <h1>{latest_d_loss:.4f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.generator is not None:
        st.subheader("🎨 Sample Generated Images")
        num_samples = st.slider("Number of images", 9, 64, 25, step=1)
        
        if st.button("Generate Samples"):
            noise = np.random.normal(0, 1, (num_samples, noise_dim))
            fig = plot_generated_images(st.session_state.generator, noise, num_samples)
            st.pyplot(fig)
            plt.close()
    else:
        st.info("👈 Initialize models from the sidebar to get started!")

with tab2:
    st.header("Generate Fake Images")
    
    if st.session_state.generator is not None:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Settings")
            num_images = st.selectbox("Grid Size", [9, 16, 25, 36, 49, 64], index=2)
            
            if st.button("🎲 Generate New Batch", use_container_width=True):
                noise = np.random.normal(0, 1, (num_images, noise_dim))
                fig = plot_generated_images(st.session_state.generator, noise, num_images)
                st.session_state.current_fig = fig
        
        with col1:
            if 'current_fig' in st.session_state:
                st.pyplot(st.session_state.current_fig)
            else:
                st.info("Click 'Generate New Batch' to create images")
    else:
        st.warning("⚠️ Please initialize the models first from the sidebar!")

with tab3:
    st.header("Training Metrics")
    
    if st.session_state.training_history['d_loss']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loss Curves")
            fig, ax = plt.subplots(figsize=(8, 5))
            epochs = list(range(1, len(st.session_state.training_history['d_loss']) + 1))
            ax.plot(epochs, st.session_state.training_history['d_loss'], label='Discriminator Loss', color='#667eea', linewidth=2)
            ax.plot(epochs, st.session_state.training_history['g_loss'], label='Generator Loss', color='#764ba2', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Discriminator Accuracy")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, st.session_state.training_history['d_acc'], color='#e74c3c', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Discriminator Accuracy Over Time')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Display metrics table
        st.subheader("Metrics Summary")
        import pandas as pd
        df = pd.DataFrame({
            'Epoch': epochs,
            'D Loss': st.session_state.training_history['d_loss'],
            'G Loss': st.session_state.training_history['g_loss'],
            'D Accuracy': st.session_state.training_history['d_acc']
        })
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("📊 Train the model to see metrics!")

with tab4:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Problem Statement
    Fake images have become ubiquitous on social media, affecting public discourse. This application demonstrates
    how Generative Adversarial Networks (GANs) can create synthetic images that resemble real ones, helping
    students understand the technology behind fake image generation.
    
    ### 🔬 Algorithm
    This GAN implementation uses:
    
    1. **Generator Network**: Transforms random noise (100-dim vector) into 28×28 grayscale images
       - 3 Dense layers with LeakyReLU activation and Batch Normalization
       - Tanh output activation to produce pixel values in [-1, 1]
    
    2. **Discriminator Network**: Binary classifier that distinguishes real from fake images
       - 3 Dense layers with LeakyReLU activation and Batch Normalization
       - Sigmoid output for probability classification
    
    3. **Training Process**: Alternating optimization
       - Train Discriminator on real MNIST images and generated fake images
       - Train Generator (via stacked GAN model) to fool the Discriminator
    
    ### 📊 Dataset
    - **MNIST**: 60,000 handwritten digit images (28×28 grayscale)
    - Normalized to [-1, 1] range to match Generator's tanh output
    
    ### 🛠️ Technology Stack
    - **TensorFlow/Keras**: Deep learning framework
    - **Streamlit**: Interactive web application framework
    - **NumPy**: Numerical computing
    - **Matplotlib**: Visualization
    
    ### 📚 References
    - Goodfellow et al., 2014. *Generative Adversarial Nets*
    - Radford et al., 2015. *Unsupervised Representation Learning with Deep Convolutional GANs*
    - TensorFlow Keras Documentation: https://www.tensorflow.org/guide/keras
    
    ### 👨‍💻 Usage Instructions
    1. Click "Initialize Models" in the sidebar
    2. Adjust training parameters as needed
    3. Click "Train GAN" to start training
    4. Generate and visualize synthetic images
    5. Save your trained model for later use
    
    ### ⚡ Performance Tips
    - Use GPU for faster training (automatically detected)
    - Start with 10-20 epochs for quick tests
    - Train 100-400 epochs for better quality images
    - Larger batch sizes train faster but use more memory
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>GAN-based Fake Image Generator | Built with Streamlit & TensorFlow</p>
    <p>📧 For questions or issues, please refer to the README.md</p>
</div>
""", unsafe_allow_html=True)
