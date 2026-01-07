"""
AI Image Generator
------------------
Created by: Michal Matějček
Made in Czech Republic 🇨🇿

A Streamlit application that generates images from text descriptions
using Stable Diffusion v1.5.
"""

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import time
from pathlib import Path

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Creative AI Studio",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# LOAD EXTERNAL CSS
# ==============================================================================

def load_css():
    """Load external CSS file for styling."""
    css_file = Path(__file__).parent / "assets" / "style.css"
    
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS
load_css()

# ==============================================================================
# MODEL LOADING
# ==============================================================================

@st.cache_resource
def load_model():
    """
    Load Stable Diffusion model from cache.
    
    This function runs only once and caches the model in memory for
    better performance. Uses @st.cache_resource decorator to prevent
    reloading on every interaction.
    
    Returns:
        StableDiffusionPipeline: Loaded model ready for inference
    """
    try:
        # Load Stable Diffusion v1.5 from LOCAL CACHE (offline mode)
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,  # Disabled for faster generation
            local_files_only=True  # FORCE OFFLINE - use only cached model
        )
        
        # Move model to CPU (Mac compatible)
        pipe = pipe.to("cpu")
        
        return pipe
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==============================================================================
# IMAGE GENERATION
# ==============================================================================

def generate_image(pipe, prompt, num_inference_steps=50, guidance_scale=7.5):
    """
    Generate an image from a text prompt using Stable Diffusion.
    
    Args:
        pipe (StableDiffusionPipeline): The loaded SD model
        prompt (str): Text description of the desired image
        num_inference_steps (int): Number of denoising steps (20-100)
                                   Higher = better quality but slower
        guidance_scale (float): How closely to follow the prompt (5-15)
                               Higher = more literal interpretation
    
    Returns:
        PIL.Image: Generated image, or None if error occurred
    """
    try:
        # Generate image without gradient calculation (inference only)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        
        # Return the first generated image
        return result.images[0]
        
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """Main application logic and UI."""
    
    # -------------------------------------------------------------------------
    # Load AI Model (only once, at the start)
    # -------------------------------------------------------------------------
    with st.spinner("Loading AI model..."):
        pipe = load_model()
    
    if pipe is None:
        st.error("Failed to load model. Please check your installation")
        st.stop()
    
    # -------------------------------------------------------------------------
    # Hero Section
    # -------------------------------------------------------------------------
    st.markdown(
        '<h1 class="hero-title">CREATIVE AI STUDIO</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-subtitle">Transform words into art with artificial intelligence</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-flag">Made in Czech Republic | Michal Matějček</p>',
        unsafe_allow_html=True
    )
    
    st.success("Model loaded and ready to create")
    
    # -------------------------------------------------------------------------
    # Input Section
    # -------------------------------------------------------------------------
    st.markdown("### Describe Your Image")
    
    prompt = st.text_input(
        "",
        placeholder="e.g., Mountain landscape at dusk with fog, realistic style",
        key="prompt_input",
        label_visibility="collapsed"
    )
    
    # -------------------------------------------------------------------------
    # Advanced Settings
    # -------------------------------------------------------------------------
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            num_steps = st.slider(
                "Quality (steps)",
                min_value=20,
                max_value=100,
                value=50,
                step=10,
                help="Higher = better quality but slower"
            )
        
        with col2:
            guidance = st.slider(
                "Prompt Strength",
                min_value=5.0,
                max_value=15.0,
                value=7.5,
                step=0.5,
                help="How closely to follow your prompt"
            )
    
    # -------------------------------------------------------------------------
    # Generate Button
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("GENERATE IMAGE")
    
    # -------------------------------------------------------------------------
    # Generation Logic
    # -------------------------------------------------------------------------
    if generate_btn:
        if not prompt:
            st.warning("Please enter an image description")
        else:
            st.markdown("---")
            
            # Show progress
            progress_text = st.empty()
            progress_text.markdown(
                '<p class="progress-title">Creating your image...</p>',
                unsafe_allow_html=True
            )
            
            # Generate the image
            with st.spinner("Generating... (this may take 30-60 seconds)"):
                start_time = time.time()
                
                image = generate_image(
                    pipe=pipe,
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance
                )
                
                generation_time = time.time() - start_time
            
            # Display results
            if image:
                progress_text.empty()
                
                # Show statistics
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(
                        f"""
                        <div style="text-align: center; margin: 2rem 0;">
                            <span class="stats-badge">{generation_time:.1f}s</span>
                            <span class="stats-badge">{num_steps} steps</span>
                            <span class="stats-badge">Strength {guidance}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Display the generated image
                st.image(image, use_column_width=True)
                st.markdown(
                    f'<p class="caption-text">"{prompt}"</p>',
                    unsafe_allow_html=True
                )
                
                # Download button
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    # Convert image to bytes for download
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="DOWNLOAD IMAGE",
                        data=byte_im,
                        file_name=f"ai_image_{prompt[:30].replace(' ', '_')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.error("Generation failed. Please try again")
    
    # -------------------------------------------------------------------------
    # Example Prompts
    # -------------------------------------------------------------------------
    with st.expander("Example Prompts"):
        examples = [
            "Mountain landscape at dusk with reflection in lake, photorealistic",
            "Futuristic city with flying cars, cyberpunk style, neon lights",
            "Cute corgi wearing sunglasses, cartoon style, colorful background",
            "Ancient temple in jungle, mystical atmosphere, ray of light",
            "Abstract colorful swirls, modern art, vibrant colors",
            "Cozy coffee shop with warm lighting, detailed, realistic"
        ]
        
        for example in examples:
            st.markdown(
                f'<div class="example-prompt">• {example}</div>',
                unsafe_allow_html=True
            )
    
    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------
    st.markdown(
        '<div class="footer-text">Powered by Stable Diffusion v1.5 | '
        'Running 100% offline on your Mac</div>',
        unsafe_allow_html=True
    )

# ==============================================================================
# RUN APPLICATION
# ==============================================================================

if __name__ == "__main__":
    main()