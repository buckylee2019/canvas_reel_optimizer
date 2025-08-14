import streamlit as st
import os
import json
from PIL import Image
import sys
import random
from datetime import datetime
import concurrent.futures
import base64
from io import BytesIO
import boto3
from typing import Tuple
from botocore.exceptions import ValidationError, ClientError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    SYSTEM_TEXT_ONLY,
    SYSTEM_IMAGE_TEXT,
    SYSTEM_CANVAS,
    MODEL_OPTIONS,
    VIDEO_MODEL_OPTIONS,
    MODEL_REGIONS,
    MODEL_BUCKETS,
    DEFAULT_BUCKET,
    DEFAULT_GUIDELINE,
    GENERATED_VIDEOS_DIR,
    PROMPT_SAMPLES,
    CANVAS_SIZE,
    LITE_MODEL_ID,
    PRO_MODEL_ID,
    REEL_MODEL_ID,
    RAY_MODEL_ID
)
from shot_video import (
    ReelGenerator,
    generate_shots,
    generate_shot_image,
    generate_reel_prompts,
    generate_shot_vidoes,
    sistch_vidoes,
    extract_last_frame
)
from utils import *
from generation import (
    optimize_prompt,
    optimize_canvas_prompt,
    generate_image_pair,
    generate_single_image,
    generate_video,
    generate_video_with_model,
    generate_comparison_videos,
    generate_comparison_videos_with_model
)

###########################
# Helper Classes for Outpainting #
###########################

class NovaCanvasResizer:
    """Production-ready AI image resizer using Amazon Nova Canvas."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize the resizer with AWS Bedrock client."""
        self.bedrock = boto3.client(service_name="bedrock-runtime", region_name=region_name)
        self.model_id = "amazon.nova-canvas-v1:0"
        
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string with proper format validation."""
        # Ensure RGB mode for consistency
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use PNG format for better quality
        buffer = BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _validate_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Validate and adjust dimensions for Nova Canvas requirements."""
        # Nova Canvas requirements - updated based on API validation
        min_width = 320  # Minimum width requirement
        min_height = 320  # Minimum height requirement  
        max_dim = 2048
        
        # Ensure within bounds
        adjusted_width = max(min_width, min(max_dim, width))
        adjusted_height = max(min_height, min(max_dim, height))
        
        # Ensure even dimensions (recommended for AI models)
        adjusted_width = adjusted_width if adjusted_width % 2 == 0 else adjusted_width + 1
        adjusted_height = adjusted_height if adjusted_height % 2 == 0 else adjusted_height + 1
        
        return adjusted_width, adjusted_height
    
    def _create_canvas_and_mask(self, original_image: Image.Image, target_size: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
        """Create canvas and mask for outpainting."""
        orig_width, orig_height = original_image.size
        target_width, target_height = target_size
        
        # Ensure RGB mode
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Create canvas with neutral background
        canvas = Image.new('RGB', target_size, color=(128, 128, 128))
        
        # Calculate scale to fit original within target (don't upscale)
        scale = min(target_width / orig_width, target_height / orig_height, 1.0)
        
        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize if needed
        if scale < 1.0:
            original_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            new_width, new_height = orig_width, orig_height
        
        # Center the image on canvas
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste original onto canvas
        canvas.paste(original_image, (x_offset, y_offset))
        
        # Create mask: BLACK for preserve, WHITE for fill (CORRECTED)
        mask = Image.new('RGB', target_size, color=(255, 255, 255))  # White background (areas to fill)
        black_area = Image.new('RGB', (new_width, new_height), color=(0, 0, 0))  # Black area (preserve original)
        mask.paste(black_area, (x_offset, y_offset))
        
        return canvas, mask
    
    def resize_image(self, 
                    original_image: Image.Image, 
                    target_width: int, 
                    target_height: int,
                    prompt: str = "seamless background extension",
                    quality: str = "standard",
                    cfg_scale: float = 7.0) -> Image.Image:
        """
        Resize image using Nova Canvas outpainting.
        
        Args:
            original_image: PIL Image object to resize
            target_width: Target width in pixels
            target_height: Target height in pixels
            prompt: Text prompt for background generation
            quality: Quality setting ("standard" or "premium")
            cfg_scale: CFG scale for generation control
            
        Returns:
            PIL Image object with the resized result
            
        Raises:
            Exception: If the resize operation fails
        """
        try:
            # Validate target dimensions
            target_width, target_height = self._validate_dimensions(target_width, target_height)
            
            # Check if resize is needed
            if original_image.size == (target_width, target_height):
                return original_image
            
            # Create canvas and mask
            canvas, mask = self._create_canvas_and_mask(original_image, (target_width, target_height))
            
            # Convert to base64
            canvas_b64 = self._image_to_base64(canvas)
            mask_b64 = self._image_to_base64(mask)
            
            # Prepare API request
            inference_params = {
                "taskType": "OUTPAINTING",
                "outPaintingParams": {
                    "image": canvas_b64,
                    "maskImage": mask_b64,
                    "text": prompt,
                    "outPaintingMode": "DEFAULT"
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "quality": quality,
                    "cfgScale": cfg_scale
                }
            }
            
            # Invoke Nova Canvas
            response = self.bedrock.invoke_model(
                body=json.dumps(inference_params),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response.get("body").read())
            
            # Check for errors
            if "error" in response_body:
                raise Exception(f"Nova Canvas API error: {response_body['error']}")
            
            # Extract result
            images = response_body.get("images", [])
            if not images:
                raise Exception("No images returned from Nova Canvas")
            
            # Decode result
            image_bytes = base64.b64decode(images[0])
            result_image = Image.open(BytesIO(image_bytes))
            
            return result_image
            
        except Exception as e:
            raise Exception(f"Resize failed: {str(e)}")
    
    def outpaint_image(self, 
                      original_image: Image.Image, 
                      prompt: str,
                      mask_prompt: str = "",
                      target_width: int = None, 
                      target_height: int = None,
                      quality: str = "standard",
                      cfg_scale: float = 7.0) -> Image.Image:
        """
        Outpaint image using Nova Canvas.
        
        Args:
            original_image: PIL Image object to outpaint
            prompt: Text prompt for outpainting
            mask_prompt: Mask prompt for specific area targeting
            target_width: Target width (optional, uses original if not specified)
            target_height: Target height (optional, uses original if not specified)
            quality: Quality setting ("standard" or "premium")
            cfg_scale: CFG scale for generation control
            
        Returns:
            PIL Image object with the outpainted result
        """
        try:
            # Use original dimensions if target not specified
            if target_width is None or target_height is None:
                target_width, target_height = original_image.size
            
            # Validate target dimensions
            target_width, target_height = self._validate_dimensions(target_width, target_height)
            
            # Convert to base64
            image_b64 = self._image_to_base64(original_image)
            
            # Prepare API request for outpainting
            inference_params = {
                "taskType": "OUTPAINTING",
                "outPaintingParams": {
                    "image": image_b64,
                    "text": prompt,
                    "outPaintingMode": "DEFAULT"
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "quality": quality,
                    "cfgScale": cfg_scale,
                    "width": target_width,
                    "height": target_height
                }
            }
            
            # Add mask prompt if provided
            if mask_prompt:
                inference_params["outPaintingParams"]["maskPrompt"] = mask_prompt
            
            # Invoke Nova Canvas
            response = self.bedrock.invoke_model(
                body=json.dumps(inference_params),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response.get("body").read())
            
            # Check for errors
            if "error" in response_body:
                raise Exception(f"Nova Canvas API error: {response_body['error']}")
            
            # Extract result
            images = response_body.get("images", [])
            if not images:
                raise Exception("No images returned from Nova Canvas")
            
            # Decode result
            image_bytes = base64.b64decode(images[0])
            result_image = Image.open(BytesIO(image_bytes))
            
            return result_image
            
        except Exception as e:
            raise Exception(f"Outpainting failed: {str(e)}")

# Initialize global resizer instance
@st.cache_resource
def get_nova_resizer():
    """Get cached Nova Canvas resizer instance."""
    return NovaCanvasResizer()

def analyze_image_with_claude(image_input, analysis_type="both"):
    """
    Analyze uploaded image using Claude Sonnet to suggest mask prompts and themes.
    
    Args:
        image_input: PIL Image or file path
        analysis_type: "mask", "theme", or "both"
    
    Returns:
        Dictionary with suggestions
    """
    try:
        # Convert image to base64
        if isinstance(image_input, str):
            # It's a file path
            with open(image_input, "rb") as img_file:
                image_bytes = img_file.read()
        elif hasattr(image_input, 'save'):
            # It's a PIL Image
            buffer = BytesIO()
            image_input.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
        else:
            # It's an uploaded file
            image_bytes = image_input.getvalue()
        
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create Claude client
        bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
        
        # Prepare prompts based on analysis type
        if analysis_type == "mask":
            prompt = """Analyze this image and suggest specific mask prompts for outpainting/editing. 
            
            Provide 3-5 mask prompt suggestions that would be useful for targeting specific areas of this image for AI editing. Focus on:
            - Main objects or subjects
            - Background elements
            - Clothing or accessories
            - Specific regions (sky, ground, walls, etc.)
            
            Return only the mask prompt words/phrases that the image model should understand, one per line:
            background
            clothing
            sky
            person
            etc."""
            
        elif analysis_type == "theme":
            prompt = """Analyze this image and suggest creative themes and prompts for outpainting/extending this image.
            
            Provide 4-6 creative outpainting suggestions that would enhance or extend this image. Consider:
            - Background extensions
            - Atmospheric changes
            - Style modifications
            - Environmental additions
            - Lighting enhancements
            
            Return creative outpainting prompts, one per line:
            extend with serene mountain landscape
            add warm golden hour lighting
            transform to professional studio setting
            etc."""
            
        else:  # both
            prompt = """Analyze this image and provide suggestions for both mask prompts and creative themes for outpainting.
            
            MASK PROMPTS (for targeting specific areas):
            Suggest 3-4 mask prompts for targeting specific parts of this image. Return simple words/phrases the image model can understand:
            
            CREATIVE THEMES (for outpainting/extending):
            Suggest 4-5 creative outpainting themes to enhance or extend this image. Return complete prompt phrases:
            
            Keep suggestions concise and practical. No bullet points or markdown formatting."""
        
        # Prepare the request
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        # Invoke Claude Sonnet
        response = bedrock.invoke_model(
            body=json.dumps(request_body),
            modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
            accept="application/json",
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response.get("body").read())
        suggestions = response_body.get("content", [{}])[0].get("text", "")
        
        return {
            "success": True,
            "suggestions": suggestions,
            "type": analysis_type
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "type": analysis_type
        }

def parse_suggestions(suggestions_text, suggestion_type):
    """Parse Claude's suggestions into structured format."""
    lines = suggestions_text.split('\n')
    
    if suggestion_type == "both":
        mask_prompts = []
        themes = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if "MASK PROMPTS" in line.upper():
                current_section = "mask"
            elif "CREATIVE THEMES" in line.upper() or "THEMES" in line.upper():
                current_section = "theme"
            elif line and not line.startswith(('Suggest', 'Return', 'Keep', 'No ')):
                # Skip instruction lines, process actual suggestions
                # Remove any bullet points or formatting
                suggestion = line.lstrip('•-*').strip()
                if current_section == "mask" and suggestion:
                    mask_prompts.append(suggestion)
                elif current_section == "theme" and suggestion:
                    themes.append(suggestion)
        
        return {"mask_prompts": mask_prompts, "themes": themes}
    
    else:
        # Single type suggestions
        suggestions = []
        for line in lines:
            line = line.strip()
            # Skip instruction lines and empty lines
            if line and not line.startswith(('Provide', 'Return', 'Focus', 'Consider', 'Format', 'Keep')):
                # Remove any bullet points or formatting
                suggestion = line.lstrip('•-*').strip()
                if suggestion:
                    suggestions.append(suggestion)
        
        return {f"{suggestion_type}s": suggestions}

def generate_outpainted_image(original_image, prompt, mask_prompt="", target_width=None, target_height=None, quality="standard", cfg_scale=7.0):
    """Generate outpainted image using the production-ready Nova Canvas resizer with optional resizing."""
    try:
        # Get the resizer instance
        resizer = get_nova_resizer()
        
        # Debug: Check if resize_image method exists
        if not hasattr(resizer, 'resize_image'):
            st.error("resize_image method not found. Clearing cache and retrying...")
            clear_resizer_cache()
            resizer = get_nova_resizer()
        
        # If target dimensions are provided, use resize_image method for proper resizing
        if target_width and target_height and (target_width != original_image.size[0] or target_height != original_image.size[1]):
            st.info(f"Resizing from {original_image.size} to {target_width}×{target_height}")
            # Use the resize method which handles canvas creation and masking
            result_image = resizer.resize_image(
                original_image=original_image,
                target_width=target_width,
                target_height=target_height,
                prompt=prompt,
                quality=quality,
                cfg_scale=cfg_scale
            )
        else:
            st.info("Using regular outpainting (same dimensions)")
            # Use regular outpainting for same dimensions
            result_image = resizer.outpaint_image(
                original_image=original_image,
                prompt=prompt,
                mask_prompt=mask_prompt,
                quality=quality,
                cfg_scale=cfg_scale
            )
        
        if result_image:
            # Save the result to a file for consistency with other functions
            os.makedirs("generated_images", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"generated_images/outpainted_{timestamp}.png"
            result_image.save(output_path)
            return output_path
        
        return None
        
    except Exception as e:
        st.error(f"Error generating outpainted image: {str(e)}")
        return None

def load_image_as_base64(image_file):
    """Helper function for preparing image data."""
    if isinstance(image_file, str):
        with open(image_file, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    else:
        # Handle PIL Image or uploaded file
        with BytesIO() as byte_io:
            if hasattr(image_file, 'save'):
                # Resize image if too large (Nova Canvas has 4.1M pixel limit)
                if hasattr(image_file, 'size'):
                    width, height = image_file.size
                    if width * height > 4100000:  # 4.1M pixels
                        # Calculate new dimensions maintaining aspect ratio
                        ratio = (4100000 / (width * height)) ** 0.5
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        image_file = image_file.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                image_file.save(byte_io, format="PNG")
            else:
                # It's already bytes
                byte_io.write(image_file.getvalue())
            image_bytes = byte_io.getvalue()
            return base64.b64encode(image_bytes).decode("utf-8")

# Page config
st.set_page_config(
    page_title="Nova Reel & Canvas Prompt Optimizer",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'bucket_name' not in st.session_state:
    st.session_state.bucket_name = DEFAULT_BUCKET
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = PRO_MODEL_ID
if 'canvas_model' not in st.session_state:
    st.session_state.canvas_model = "amazon.nova-canvas-v1:0"
if 'reel_model' not in st.session_state:
    st.session_state.reel_model = REEL_MODEL_ID

def aspect_ratio_to_dimensions(aspect_ratio, base_size=1024):
    """Convert aspect ratio string to width and height"""
    if ":" not in aspect_ratio:
        return base_size, base_size
    
    try:
        w_ratio, h_ratio = aspect_ratio.split(":")
        w_ratio, h_ratio = float(w_ratio), float(h_ratio)
        
        # Calculate dimensions maintaining the aspect ratio
        if w_ratio >= h_ratio:
            width = base_size
            height = int(base_size * h_ratio / w_ratio)
        else:
            height = base_size
            width = int(base_size * w_ratio / h_ratio)
        
        # Ensure dimensions are multiples of 8 (common requirement for image generation)
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        return width, height
    except:
        return base_size, base_size

def display_image_with_copy_button(image_input, label=""):
    """Display image with a copy button - handles both file paths and PIL Image objects"""
    try:
        # Handle different input types
        if isinstance(image_input, str):
            # It's a file path
            if image_input and os.path.exists(image_input):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(image_input, caption=label, use_column_width=True)
                with col2:
                    if st.button(f"Copy to Video Gen", key=f"copy_{label}"):
                        st.session_state.copied_image = image_input
                        st.success("Image copied!")
                return image_input
        elif isinstance(image_input, list):
            # It's a list of file paths - use the first one
            if image_input and len(image_input) > 0:
                first_image = image_input[0]
                if isinstance(first_image, str) and os.path.exists(first_image):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.image(first_image, caption=label, use_column_width=True)
                    with col2:
                        if st.button(f"Copy to Video Gen", key=f"copy_{label}"):
                            st.session_state.copied_image = first_image
                            st.success("Image copied!")
                    return first_image
        elif hasattr(image_input, 'save'):
            # It's a PIL Image object
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(image_input, caption=label, use_column_width=True)
            with col2:
                if st.button(f"Copy to Video Gen", key=f"copy_{label}"):
                    # Save PIL Image to temporary file for copying
                    temp_path = f"generated_images/temp_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    os.makedirs("generated_images", exist_ok=True)
                    image_input.save(temp_path)
                    st.session_state.copied_image = temp_path
                    st.success("Image copied!")
            return image_input
        else:
            st.warning(f"Invalid image input type: {type(image_input)}")
            return None
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        return None

def display_video(video_path, label=""):
    """Display video with download button"""
    if video_path and os.path.exists(video_path):
        st.video(video_path)
        
        # Create columns for caption and download button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(label)
        with col2:
            # Add download button
            try:
                with open(video_path, "rb") as file:
                    video_bytes = file.read()
                    # Get filename from path
                    filename = os.path.basename(video_path)
                    st.download_button(
                        label="📥 Download",
                        data=video_bytes,
                        file_name=filename,
                        mime="video/mp4",
                        help=f"Download {label.lower() if label else 'video'}"
                    )
            except Exception as e:
                st.error(f"Error preparing download: {str(e)}")
    return video_path

def main():
    st.title("🎬 Nova Reel & Canvas Prompt Optimizer")
    st.markdown("A Streamlit interface for optimizing prompts and generating videos using Amazon's Nova Canvas and Reel models.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model Selection
        model_display_names = list(MODEL_OPTIONS.keys())
        selected_display_name = st.selectbox(
            "Optimization Model",
            options=model_display_names,
            index=0  # Default to first option
        )
        st.session_state.selected_model = MODEL_OPTIONS[selected_display_name]
        
        st.session_state.canvas_model = st.selectbox(
            "Canvas Model",
            options=["amazon.nova-canvas-v1:0"],
            index=0
        )
        
        st.session_state.reel_model = st.selectbox(
            "Reel Model", 
            options=[REEL_MODEL_ID],
            index=0
        )
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Generation", "🎥 Video Generation", "🎬 Long Video Generation"])
    
    with tab1:
        st.header("Image Generation with Nova Canvas")
        
        # Mode selection: Text-to-Image or Image-to-Image (Outpainting)
        generation_mode = st.radio(
            "Generation Mode:",
            ["Text-to-Image", "Image-to-Image (Outpainting)"],
            horizontal=True,
            help="Choose between generating from text or editing an uploaded image"
        )
        
        # Image upload section (for outpainting mode)
        uploaded_image = None
        if generation_mode == "Image-to-Image (Outpainting)":
            st.subheader("Upload Image for Outpainting")
            uploaded_image = st.file_uploader(
                "Upload an image to edit/extend:",
                type=['png', 'jpg', 'jpeg'],
                key="canvas_upload",
                help="Upload the image you want to edit or extend"
            )
            
            if uploaded_image:
                # Display uploaded image
                original_img = Image.open(uploaded_image)
                col_img1, col_img2 = st.columns([1, 2])
                with col_img1:
                    st.image(original_img, caption=f"Original: {original_img.size[0]}×{original_img.size[1]}", use_column_width=True)
                with col_img2:
                    st.info(f"**Original Size:** {original_img.size[0]} × {original_img.size[1]} pixels")
                    st.info("You can now describe what you want to add or change in the prompt below.")
                    
                    # AI Analysis Section
                    st.markdown("### 🤖 AI Suggestions")
                    col_suggest1, col_suggest2 = st.columns(2)
                    
                    with col_suggest1:
                        if st.button("🎯 Suggest Mask Prompts", help="Get AI suggestions for targeting specific areas"):
                            with st.spinner("Analyzing image for mask suggestions..."):
                                result = analyze_image_with_claude(uploaded_image, "mask")
                                if result["success"]:
                                    st.session_state.mask_suggestions = parse_suggestions(result["suggestions"], "mask")
                                else:
                                    st.error(f"Error getting mask suggestions: {result['error']}")
                    
                    with col_suggest2:
                        if st.button("🎨 Suggest Themes", help="Get AI suggestions for creative outpainting themes"):
                            with st.spinner("Analyzing image for theme suggestions..."):
                                result = analyze_image_with_claude(uploaded_image, "theme")
                                if result["success"]:
                                    st.session_state.theme_suggestions = parse_suggestions(result["suggestions"], "theme")
                                else:
                                    st.error(f"Error getting theme suggestions: {result['error']}")
                    
                    # Get both suggestions at once
                    if st.button("🚀 Get All Suggestions", help="Get both mask prompts and theme suggestions"):
                        with st.spinner("Analyzing image for all suggestions..."):
                            result = analyze_image_with_claude(uploaded_image, "both")
                            if result["success"]:
                                parsed = parse_suggestions(result["suggestions"], "both")
                                st.session_state.mask_suggestions = {"masks": parsed.get("mask_prompts", [])}
                                st.session_state.theme_suggestions = {"themes": parsed.get("themes", [])}
                            else:
                                st.error(f"Error getting suggestions: {result['error']}")
                
                # Display suggestions if available
                if 'mask_suggestions' in st.session_state or 'theme_suggestions' in st.session_state:
                    st.markdown("---")
                    
                    # Clear suggestions button
                    col_clear1, col_clear2 = st.columns([3, 1])
                    with col_clear2:
                        if st.button("🗑️ Clear All Suggestions", help="Clear all AI suggestions"):
                            if 'mask_suggestions' in st.session_state:
                                del st.session_state.mask_suggestions
                            if 'theme_suggestions' in st.session_state:
                                del st.session_state.theme_suggestions
                            if 'selected_mask_prompt' in st.session_state:
                                del st.session_state.selected_mask_prompt
                            if 'selected_theme' in st.session_state:
                                del st.session_state.selected_theme
                            st.rerun()
                    
                    # Display suggestions in dropdown format
                    col_mask_dropdown, col_theme_dropdown = st.columns(2)
                    
                    with col_mask_dropdown:
                        if 'mask_suggestions' in st.session_state:
                            st.markdown("#### 🎯 Mask Prompt Suggestions")
                            mask_suggestions = st.session_state.mask_suggestions.get("masks", [])
                            if mask_suggestions:
                                # Add "None" option at the beginning
                                mask_options = ["None (don't use mask prompt)"] + mask_suggestions
                                
                                # Get current selection index
                                current_mask = st.session_state.get('selected_mask_prompt', '')
                                if current_mask and current_mask in mask_suggestions:
                                    default_index = mask_suggestions.index(current_mask) + 1
                                else:
                                    default_index = 0
                                
                                selected_mask_option = st.selectbox(
                                    "Select mask prompt:",
                                    options=mask_options,
                                    index=default_index,
                                    key="mask_dropdown",
                                    help="Choose a mask prompt to target specific areas"
                                )
                                
                                # Update session state based on selection
                                if selected_mask_option == "None (don't use mask prompt)":
                                    if 'selected_mask_prompt' in st.session_state:
                                        del st.session_state.selected_mask_prompt
                                else:
                                    st.session_state.selected_mask_prompt = selected_mask_option
                                    st.success(f"✓ Selected: {selected_mask_option}")
                            else:
                                st.info("No mask suggestions available")
                    
                    with col_theme_dropdown:
                        if 'theme_suggestions' in st.session_state:
                            st.markdown("#### 🎨 Theme Suggestions")
                            theme_suggestions = st.session_state.theme_suggestions.get("themes", [])
                            if theme_suggestions:
                                # Add "None" option at the beginning
                                theme_options = ["None (don't use theme)"] + theme_suggestions
                                
                                # Get current selection index
                                current_theme = st.session_state.get('selected_theme', '')
                                if current_theme and current_theme in theme_suggestions:
                                    default_index = theme_suggestions.index(current_theme) + 1
                                else:
                                    default_index = 0
                                
                                selected_theme_option = st.selectbox(
                                    "Select theme:",
                                    options=theme_options,
                                    index=default_index,
                                    key="theme_dropdown",
                                    help="Choose a theme for creative outpainting"
                                )
                                
                                # Update session state based on selection
                                if selected_theme_option == "None (don't use theme)":
                                    if 'selected_theme' in st.session_state:
                                        del st.session_state.selected_theme
                                else:
                                    st.session_state.selected_theme = selected_theme_option
                                    st.success(f"✓ Selected: {selected_theme_option}")
                            else:
                                st.info("No theme suggestions available")
        
        # Input prompt - different labels based on mode
        if generation_mode == "Text-to-Image":
            # Prompt templates (only for text-to-image)
            template_options = ["Custom"] + list(PROMPT_SAMPLES.keys())
            selected_template = st.selectbox("Choose a template:", template_options)
            
            if selected_template != "Custom":
                default_prompt = PROMPT_SAMPLES[selected_template]
            else:
                default_prompt = ""
            
            prompt_label = "Enter your prompt:"
            prompt_help = "Describe what you want to generate"
            default_prompt_value = default_prompt
        else:
            prompt_label = "Enter your outpainting prompt:"
            prompt_help = "Describe what you want to add, change, or extend in the image"
            # Use selected theme if available
            selected_theme = st.session_state.get('selected_theme', '')
            if selected_theme:
                default_prompt_value = selected_theme
                st.info(f"🎨 Using selected theme: {selected_theme}")
            else:
                default_prompt_value = ""
        
        canvas_prompt = st.text_area(
            prompt_label,
            value=default_prompt_value,
            height=100,
            help=prompt_help
        )
        
        # Additional outpainting settings
        mask_prompt = ""
        target_width = None
        target_height = None
        
        if generation_mode == "Image-to-Image (Outpainting)":
            
            # Use selected mask prompt if available
            selected_mask = st.session_state.get('selected_mask_prompt', "")
            if selected_mask:
                st.info(f"🎯 Using selected mask prompt: {selected_mask}")
            
            mask_prompt = st.text_input(
                "Mask Prompt (optional):",
                value=selected_mask,
                help="Specify which part of the image to focus on (e.g., 'background', 'clothing', 'sky')"
            )
            
            # Resize options for outpainting (in expandable section)
            with st.expander("🔧 Resize Options (Optional)", expanded=False):
                st.markdown("**Resize the image to different dimensions while outpainting**")
                resize_enabled = st.checkbox("Enable Resize", help="Resize the image to different dimensions while outpainting")
                
                if resize_enabled and uploaded_image:
                    original_img = Image.open(uploaded_image)
                    orig_width, orig_height = original_img.size
                    
                    col_resize1, col_resize2, col_resize3 = st.columns([1, 1, 1])
                    
                    with col_resize1:
                        target_width = st.number_input(
                            "Target Width",
                            min_value=320,
                            max_value=2048,
                            value=orig_width,
                            step=1,
                            help="New width in pixels (minimum 320)"
                        )
                    
                    with col_resize2:
                        target_height = st.number_input(
                            "Target Height", 
                            min_value=320,
                            max_value=2048,
                            value=orig_height,
                            step=1,
                            help="New height in pixels (minimum 320)"
                        )
                    
                    with col_resize3:
                        pass  # Quick presets removed
                    
                    # Show what will change
                    if target_width != orig_width or target_height != orig_height:
                        st.info(f"**Original:** {orig_width}×{orig_height} → **Target:** {target_width}×{target_height}")
                        width_diff = target_width - orig_width
                        height_diff = target_height - orig_height
                        if width_diff > 0 or height_diff > 0:
                            st.warning(f"**AI will fill:** {max(0, width_diff)} pixels width, {max(0, height_diff)} pixels height")
                    
                    # Preview what the canvas will look like
                    if st.checkbox("Show Canvas Preview", help="Preview how the image will be positioned"):
                        resizer = get_nova_resizer()
                        canvas_preview, _ = resizer._create_canvas_and_mask(original_img, (target_width, target_height))
                        st.image(canvas_preview, caption=f"Canvas Preview: {target_width}×{target_height}", width=400)
                else:
                    # Initialize variables when resize is not enabled
                    target_width = None
                    target_height = None
            with st.expander("🔧 Debug Options"):
                if st.button("Clear Resizer Cache", help="Clear cached resizer instance if having issues"):
                    clear_resizer_cache()
                    st.success("Cache cleared! Try generating again.")
        # Canvas settings
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            seed = st.number_input("Seed", value=-1, help="Random seed (-1 for random)")
        with col2:
            cfg_scale = st.slider("CFG Scale", 1.0, 10.0, 7.0, 0.1)
        with col3:
            # Extract just the aspect ratios for display
            aspect_ratios = []
            for size in CANVAS_SIZE:
                if "(" in size and ")" in size:
                    ratio = size.split("(")[1].split(")")[0]
                    if ratio not in aspect_ratios:
                        aspect_ratios.append(ratio)
            aspect_ratio = st.selectbox("Aspect Ratio", aspect_ratios[:10])  # Show first 10 unique ratios
        with col4:
            if generation_mode == "Text-to-Image":
                comparison_mode = st.checkbox("Enable Comparison Mode", value=True)
            else:
                quality = st.selectbox("Quality", ["standard", "premium"], help="Higher quality for better results")
        
        # Generate buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔧 Optimize Prompt", key="optimize_canvas"):
                if canvas_prompt:
                    with st.spinner("Optimizing prompt..."):
                        try:
                            optimized_prompt, negative_prompt = optimize_canvas_prompt(
                                canvas_prompt,
                                st.session_state.selected_model
                            )
                            st.session_state.canvas_optimized_prompt = optimized_prompt
                            st.session_state.canvas_negative_prompt = negative_prompt
                            st.success("Prompt optimized!")
                        except Exception as e:
                            st.error(f"Error optimizing prompt: {str(e)}")
                else:
                    st.warning("Please enter a prompt first")
        
        with col2:
            generate_button_text = "🎨 Generate Image"  # Same text for both modes
            if st.button(generate_button_text, key="generate_canvas"):
                if canvas_prompt:
                    if generation_mode == "Image-to-Image (Outpainting)" and not uploaded_image:
                        st.warning("Please upload an image for outpainting mode")
                    else:
                        with st.spinner("Generating image..."):
                            try:
                                if generation_mode == "Text-to-Image":
                                    # Original text-to-image generation
                                    width, height = aspect_ratio_to_dimensions(aspect_ratio)
                                    if comparison_mode and 'canvas_optimized_prompt' in st.session_state:
                                        # Generate comparison
                                        original_result, optimized_result = generate_image_pair(
                                            canvas_prompt,
                                            st.session_state.canvas_optimized_prompt,
                                            st.session_state.canvas_negative_prompt,
                                            "standard", 1, height, width, seed, cfg_scale
                                        )
                                        st.session_state.canvas_original_image = original_result
                                        st.session_state.canvas_optimized_image = optimized_result
                                    else:
                                        # Generate single image
                                        result = generate_single_image(
                                            canvas_prompt,
                                            st.session_state.canvas_negative_prompt if 'canvas_negative_prompt' in st.session_state else "",
                                            "standard", 1, height, width, seed, cfg_scale
                                        )
                                        st.session_state.canvas_single_image = result
                                else:
                                    # Outpainting mode
                                    original_img = Image.open(uploaded_image)
                                    prompt_to_use = st.session_state.canvas_optimized_prompt if 'canvas_optimized_prompt' in st.session_state else canvas_prompt
                                    
                                    result = generate_outpainted_image(
                                        original_img,
                                        prompt_to_use,
                                        mask_prompt,
                                        target_width,
                                        target_height,
                                        quality,
                                        cfg_scale
                                    )
                                    if result:
                                        st.session_state.canvas_outpainted_image = result
                                
                                st.success("Image generated!")
                            except Exception as e:
                                st.error(f"Error generating image: {str(e)}")
                else:
                    st.warning("Please enter a prompt first")
        
        # Display optimized prompts
        if 'canvas_optimized_prompt' in st.session_state:
            st.subheader("Optimized Prompts")
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Optimized Prompt", st.session_state.canvas_optimized_prompt, height=100)
            with col2:
                st.text_area("Negative Prompt", st.session_state.canvas_negative_prompt, height=100)
        
        # Display generated images
        if generation_mode == "Text-to-Image":
            if comparison_mode and 'canvas_original_image' in st.session_state:
                st.subheader("Generated Images - Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    display_image_with_copy_button(st.session_state.canvas_original_image, "Original")
                with col2:
                    display_image_with_copy_button(st.session_state.canvas_optimized_image, "Optimized")
            elif 'canvas_single_image' in st.session_state:
                st.subheader("Generated Image")
                display_image_with_copy_button(st.session_state.canvas_single_image, "Generated")
        else:
            # Outpainting mode results
            if 'canvas_outpainted_image' in st.session_state:
                st.subheader("Outpainting Result")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    display_image_with_copy_button(st.session_state.canvas_outpainted_image, "Outpainted")
                
                with col2:
                    st.markdown("### Download Options")
                    
                    # Convert image to bytes for download
                    img_buffer = BytesIO()
                    # Load the image from file path
                    img = Image.open(st.session_state.canvas_outpainted_image)
                    img.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    
                    st.download_button(
                        label="Download Outpainted Image",
                        data=img_bytes,
                        file_name="outpainted_image.png",
                        mime="image/png"
                    )
                    
                    # Show comparison with original if available
                    if uploaded_image:
                        st.markdown("### Before & After")
                        original_img = Image.open(uploaded_image)
                        st.write(f"**Original:** {original_img.size[0]}×{original_img.size[1]}")
                        st.write(f"**Result:** {img.size[0]}×{img.size[1]}")
                
                # Clear results button
                st.markdown("### Try Again")
                if st.button("Clear Results", key="clear_outpaint"):
                    if 'canvas_outpainted_image' in st.session_state:
                        del st.session_state.canvas_outpainted_image
                    st.rerun()
        
        # Tips for outpainting
        if generation_mode == "Image-to-Image (Outpainting)":
            with st.expander("**Outpainting Tips & AI Suggestions Guide**"):
                st.markdown("""
                ### **🤖 AI Suggestions Feature:**
                - **Suggest Mask Prompts**: AI analyzes your image and suggests specific areas to target (e.g., "background", "clothing", "sky")
                - **Suggest Themes**: AI provides creative ideas for extending or enhancing your image
                - **Get All Suggestions**: Get both mask prompts and themes in one analysis
                - **Use Button**: Click "Use" next to any suggestion to automatically fill the prompt fields
                - **Visual Feedback**: Selected suggestions are highlighted with a ✓ checkmark
                
                ### **Best Practices:**
                - Upload your image first, then use AI suggestions for better results
                - Try different mask prompts to target specific areas precisely
                - Combine AI theme suggestions with your own creative ideas
                - Use mask prompts for precision, themes for creativity
                
                ### **Manual Prompt Examples:**
                - **Background Extension**: "extend the blue conference background, maintain professional atmosphere"
                - **Adding Elements**: "add beautiful flowers in the foreground, natural lighting"
                - **Style Changes**: "convert to artistic painting style, vibrant colors"
                - **Object Replacement**: "replace the background with a modern office setting"
                
                ### **Mask Prompt Examples:**
                - "background" - focus on background areas
                - "clothing" - focus on clothing items
                - "sky" - focus on sky areas
                - "foreground" - focus on foreground elements
                
                ### **Settings Guide:**
                - **CFG Scale 3.0-5.0**: More natural, conservative changes
                - **CFG Scale 6.0-8.0**: More creative, dramatic changes
                - **Premium Quality**: Better results for final outputs
                - **Standard Quality**: Faster generation for testing
                
                ### **AI Analysis Powered by:**
                - **Claude Sonnet 3.5**: Advanced vision model for intelligent image analysis
                - **Context-Aware**: Understands image content, composition, and potential enhancements
                - **Creative Suggestions**: Provides both practical and artistic recommendations
                """)
    
    with tab2:
        st.header("Video Generation with Nova Reel")
        
        # Input options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            reel_prompt = st.text_area(
                "Enter your video prompt:",
                height=100,
                help="Describe the video you want to generate"
            )
        
        with col2:
            # Image upload or copy from canvas
            uploaded_image = st.file_uploader("Upload Image (optional)", type=['png', 'jpg', 'jpeg'])
            
            if 'copied_image' in st.session_state:
                st.success("Image copied from Canvas!")
                st.image(st.session_state.copied_image, caption="Copied Image", use_column_width=True)
        
        # Video settings
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            selected_video_model = st.selectbox(
                "Video Model",
                options=list(VIDEO_MODEL_OPTIONS.keys()),
                index=0,
                help="Choose the video generation model"
            )
        with col2:
            # Show duration selector for Luma Ray, seed for others
            if "Luma Ray" in selected_video_model:
                video_duration = st.selectbox(
                    "Duration",
                    options=["5s", "9s"],
                    index=0,
                    help="Video duration (Luma Ray only)"
                )
            else:
                video_seed = st.number_input("Video Seed", value=-1, help="Random seed (-1 for random)")
        with col3:
            if "Luma Ray" not in selected_video_model:
                video_cfg = st.slider("Video CFG Scale", 1.0, 10.0, 7.0, 0.1)
            else:
                st.empty()  # Placeholder for Luma Ray (no CFG scale)
        with col4:
            video_comparison = st.checkbox("Enable Video Comparison", value=True)
        
        # Generate buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔧 Optimize Video Prompt", key="optimize_reel"):
                if reel_prompt:
                    with st.spinner("Optimizing video prompt..."):
                        try:
                            optimized_prompt, length_info = optimize_prompt(
                                reel_prompt,
                                DEFAULT_GUIDELINE,  # Add guideline path
                                st.session_state.selected_model
                            )
                            st.session_state.reel_optimized_prompt = optimized_prompt
                            st.session_state.reel_negative_prompt = ""  # optimize_prompt doesn't return negative prompt
                            st.success(f"Video prompt optimized! {length_info}")
                        except Exception as e:
                            st.error(f"Error optimizing prompt: {str(e)}")
                else:
                    st.warning("Please enter a prompt first")
        
        with col2:
            if st.button("🎥 Generate Video", key="generate_reel"):
                if reel_prompt:
                    with st.spinner("Generating video... This may take several minutes."):
                        try:
                            # Determine input image
                            input_image = None
                            if uploaded_image:
                                # Convert uploaded image to PIL Image
                                # Streamlit UploadedFile needs to be handled specially
                                try:
                                    # Reset file pointer to beginning
                                    uploaded_image.seek(0)
                                    # Read the raw bytes and create a fresh PIL Image
                                    image_bytes = uploaded_image.read()
                                    input_image = Image.open(BytesIO(image_bytes))
                                    # Convert to RGB if needed
                                    if input_image.mode != 'RGB':
                                        input_image = input_image.convert('RGB')
                                    st.info(f"Using uploaded image: {input_image.size}, mode: {input_image.mode}")
                                except Exception as e:
                                    st.error(f"Error loading uploaded image: {str(e)}")
                                    input_image = None
                                    st.info("Generating text-only video instead")
                            elif 'copied_image' in st.session_state:
                                # Load copied image as PIL Image
                                try:
                                    if st.session_state.copied_image and os.path.exists(st.session_state.copied_image):
                                        input_image = Image.open(st.session_state.copied_image)
                                        # Convert to RGB if needed
                                        if input_image.mode != 'RGB':
                                            input_image = input_image.convert('RGB')
                                        st.info(f"Using copied image: {input_image.size}, mode: {input_image.mode}")
                                        st.info(f"Copied image path: {st.session_state.copied_image}")
                                    else:
                                        input_image = None
                                        st.warning(f"Copied image path not found: {st.session_state.copied_image}")
                                        st.info("Generating text-only video instead")
                                except Exception as e:
                                    input_image = None
                                    st.error(f"Error loading copied image: {str(e)}")
                                    st.info("Generating text-only video instead")
                            else:
                                st.info("No image provided, generating text-only video")
                            
                            # Get selected model ID and corresponding bucket
                            selected_model_id = VIDEO_MODEL_OPTIONS[selected_video_model]
                            model_bucket = MODEL_BUCKETS.get(selected_model_id, st.session_state.bucket_name)
                            st.info(f"Using video model: {selected_video_model} ({selected_model_id})")
                            st.info(f"Using bucket: {model_bucket}")
                            
                            # Prepare model-specific parameters
                            if "Luma Ray" in selected_video_model:
                                # Use duration for Luma Ray, default seed
                                model_seed = 0
                                model_duration = video_duration if 'video_duration' in locals() else "5s"
                            else:
                                # Use seed for Nova Reel, default duration
                                model_seed = video_seed if 'video_seed' in locals() else 0
                                model_duration = "5s"
                            
                            if video_comparison and 'reel_optimized_prompt' in st.session_state:
                                # Generate comparison videos
                                st.info(f"Generating comparison videos...")
                                original_video, optimized_video = generate_comparison_videos_with_model(
                                    reel_prompt,
                                    st.session_state.reel_optimized_prompt,
                                    model_bucket,
                                    selected_model_id,
                                    input_image,
                                    model_seed,
                                    model_duration
                                )
                                st.session_state.reel_original_video = original_video
                                st.session_state.reel_optimized_video = optimized_video
                            else:
                                # Generate single video
                                st.info(f"Generating single video...")
                                video_result = generate_video_with_model(
                                    reel_prompt,
                                    model_bucket,
                                    selected_model_id,
                                    input_image,
                                    model_seed,
                                    model_duration
                                )
                                st.session_state.reel_single_video = video_result
                            
                            st.success("Video generated!")
                        except Exception as e:
                            st.error(f"Error generating video: {str(e)}")
                else:
                    st.warning("Please enter a prompt first")
        
        # Display optimized prompts
        if 'reel_optimized_prompt' in st.session_state:
            st.subheader("Optimized Video Prompts")
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Optimized Video Prompt", st.session_state.reel_optimized_prompt, height=100)
            with col2:
                st.text_area("Negative Video Prompt", st.session_state.reel_negative_prompt, height=100)
        
        # Display generated videos
        if video_comparison and 'reel_original_video' in st.session_state:
            st.subheader("Generated Videos - Comparison")
            col1, col2 = st.columns(2)
            with col1:
                display_video(st.session_state.reel_original_video, "Original Video")
            with col2:
                display_video(st.session_state.reel_optimized_video, "Optimized Video")
        elif 'reel_single_video' in st.session_state:
            st.subheader("Generated Video")
            display_video(st.session_state.reel_single_video, "Generated Video")
    
    with tab3:
        st.header("Long Video Generation")
        st.markdown("Generate longer videos by creating storyboards and stitching multiple scenes together.")
        
        # Information about the process
        with st.expander("ℹ️ How Long Video Generation Works"):
            st.markdown("""
            ### 🎬 Long Video Generation Process:
            
            1. **Story Analysis**: Your story is analyzed and broken down into individual shots
            2. **Shot Generation**: Each shot gets a detailed description for visual consistency
            3. **Image Creation**: High-quality images are generated for each shot using Nova Canvas
            4. **Prompt Optimization**: Video prompts are optimized for each shot based on the generated images
            5. **Video Generation**: Individual videos are created from each image using Nova Reel
            6. **Video Stitching**: All videos are combined into a single long video
            7. **Caption Addition**: Captions are added to enhance the viewing experience
            
            ### ⏱️ Expected Timeline:
            - **Shot Generation**: 30-60 seconds
            - **Image Generation**: 2-3 minutes per shot
            - **Video Generation**: 5-10 minutes per shot
            - **Stitching & Captions**: 1-2 minutes
            
            **Total Time**: 15-30 minutes for a 4-shot video
            
            ### 💡 Tips for Best Results:
            - Use detailed, cinematic descriptions
            - Include character descriptions and setting details
            - Mention camera angles and lighting preferences
            - Keep narrative flow logical and coherent
            """)
        
        # Story input
        story_prompt = st.text_area(
            "Enter your story/concept:",
            height=150,
            help="Describe the overall story or concept for your long video. Be specific about characters, setting, and sequence of events.",
            placeholder="Example: A young astronaut discovers a mysterious alien artifact on Mars. The story begins with her exploring a vast red canyon, then finding a glowing crystal in a cave, followed by the crystal activating and showing visions of an ancient civilization, and finally her making contact with the alien intelligence."
        )
        
        # Image upload for long video generation
        st.subheader("📸 Optional: Upload First Frame Image")
        long_video_uploaded_image = st.file_uploader(
            "Upload an image to use as the first frame of your long video:",
            type=['png', 'jpg', 'jpeg'],
            key="long_video_image_upload",
            help="Upload an image that will be used directly as the first frame. Subsequent frames will be generated to maintain visual consistency with this starting image."
        )
        
        # Display uploaded image
        if long_video_uploaded_image:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(long_video_uploaded_image, caption="First Frame Image", use_column_width=True)
            with col2:
                st.info("✅ First frame image uploaded! This will be used directly as the first frame of your long video, and subsequent frames will be generated to maintain visual consistency.")
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col1, col2, col3 = st.columns(3)
            with col1:
                num_shots = st.slider("Number of shots", 2, 8, 4, help="Number of individual shots to generate")
                shot_duration = st.slider("Duration per Shot (seconds)", 3, 10, 6, help="Duration for each individual shot")
            with col2:
                image_seed = st.number_input("Image Generation Seed", value=0, help="Random seed for image generation (-1 for random)")
                cfg_scale = st.slider("CFG Scale", 1.0, 10.0, 7.0, 0.1, help="Controls how closely the image follows the prompt")
            with col3:
                similarity_strength = st.slider("Similarity Strength", 0.1, 1.0, 0.8, 0.1, help="Controls consistency between shots")
                is_continuous = st.checkbox("Continuous narrative", value=True, help="Maintain character and setting consistency across shots")
        
        # Generate buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Generate Shots", key="generate_shots"):
                if story_prompt:
                    with st.spinner("Generating storyboard..."):
                        try:
                            # Create ReelGenerator instance
                            reel_gen = ReelGenerator(
                                canvas_model_id=st.session_state.canvas_model,  # Use canvas model for images
                                reel_model_id=st.session_state.reel_model,      # Use reel model for videos
                                text_model_id=st.session_state.selected_model,  # Use selected model for text generation
                                region='us-east-1',
                                bucket_name=st.session_state.bucket_name
                            )
                            shots = generate_shots(reel_gen, story_prompt, num_shots, is_continuous)
                            st.session_state.story_shots = shots
                            st.session_state.reel_generator = reel_gen  # Store for later use
                            st.success(f"Generated {len(shots)} shots!")
                        except Exception as e:
                            st.error(f"Error generating shots: {str(e)}")
                            import traceback
                            st.error(f"Detailed error: {traceback.format_exc()}")
                else:
                    st.warning("Please enter a story first")
        
        with col2:
            if st.button("🎬 Generate Long Video", key="generate_long_video"):
                if 'story_shots' in st.session_state:
                    with st.spinner("Generating long video... This will take several minutes."):
                        try:
                            # Create ReelGenerator instance if not exists
                            if 'reel_generator' not in st.session_state:
                                st.session_state.reel_generator = ReelGenerator(
                                    canvas_model_id=st.session_state.canvas_model,  # Use canvas model for images
                                    reel_model_id=st.session_state.reel_model,      # Use reel model for videos
                                    text_model_id=st.session_state.selected_model,  # Use selected model for text generation
                                    region='us-east-1',
                                    bucket_name=st.session_state.bucket_name
                                )
                            
                            reel_gen = st.session_state.reel_generator
                            shots = st.session_state.story_shots
                            
                            # Generate timestamp for this session
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            
                            # Step 1: Generate images for each shot
                            st.info("Step 1/4: Generating images for each shot...")
                            progress_bar = st.progress(0)
                            
                            # Prepare reference image if uploaded
                            reference_image = None
                            if long_video_uploaded_image:
                                try:
                                    # Reset file pointer and convert to PIL Image
                                    long_video_uploaded_image.seek(0)
                                    image_bytes = long_video_uploaded_image.read()
                                    reference_image = Image.open(BytesIO(image_bytes))
                                    if reference_image.mode != 'RGB':
                                        reference_image = reference_image.convert('RGB')
                                    st.success(f"✅ First frame image processed: {reference_image.size}, mode: {reference_image.mode}")
                                    st.info("🎬 Your uploaded image will be used directly as the first frame of the long video")
                                except Exception as e:
                                    st.error(f"Could not process reference image: {str(e)}")
                                    reference_image = None
                            else:
                                st.info("📝 No first frame image provided - all frames will be generated from text prompts")
                            
                            image_files = generate_shot_image(
                                reel_gen, 
                                shots, 
                                timestamp, 
                                seed=image_seed, 
                                cfg_scale=cfg_scale, 
                                similarity_strength=similarity_strength, 
                                is_continues_shot=is_continuous,
                                reference_image=reference_image
                            )
                            progress_bar.progress(25)
                            st.success(f"Generated {len(image_files)} images!")
                            
                            # Step 2: Generate optimized prompts for video generation
                            st.info("Step 2/4: Optimizing prompts for video generation...")
                            reel_prompts = generate_reel_prompts(reel_gen, shots, image_files, skip=True)
                            progress_bar.progress(50)
                            st.success(f"Generated {len(reel_prompts)} optimized prompts!")
                            
                            # Step 3: Generate videos from images
                            st.info("Step 3/4: Generating videos from images... (This may take 10-15 minutes)")
                            video_files = generate_shot_vidoes(reel_gen, image_files, reel_prompts)
                            progress_bar.progress(75)
                            st.success(f"Generated {len(video_files)} videos!")
                            
                            # Step 4: Stitch videos together and add captions
                            st.info("Step 4/4: Stitching videos together and adding captions...")
                            final_video, caption_video_file = sistch_vidoes(reel_gen, video_files, shots, timestamp)
                            progress_bar.progress(100)
                            
                            if final_video:
                                st.success("Long video generated successfully!")
                                
                                # Display the final video
                                st.subheader("🎬 Generated Long Video")
                                
                                if caption_video_file and os.path.exists(caption_video_file):
                                    # Show both videos if captions were successfully added
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        display_video(final_video, "Final Stitched Video")
                                    with col2:
                                        display_video(caption_video_file, "Video with Captions")
                                else:
                                    # Show only the final video if captions failed
                                    display_video(final_video, "Final Stitched Video")
                                    if not caption_video_file:
                                        st.warning("Captions could not be added to the video")
                                
                                # Store results in session state
                                st.session_state.final_video = final_video
                                st.session_state.caption_video = caption_video_file
                                st.session_state.generated_images = image_files
                                st.session_state.generated_videos = video_files
                                
                                # Show generation summary
                                with st.expander("📊 Generation Summary"):
                                    st.write(f"**Total Shots:** {len(shots)}")
                                    st.write(f"**Images Generated:** {len(image_files)}")
                                    st.write(f"**Videos Generated:** {len(video_files)}")
                                    st.write(f"**Final Video:** {final_video}")
                                    st.write(f"**Captioned Video:** {caption_video_file}")
                                    st.write(f"**Generation Timestamp:** {timestamp}")
                                    
                                    # Show individual shots
                                    st.write("**Shot Details:**")
                                    for i, (shot_key, shot_desc) in enumerate(shots.items()):
                                        st.write(f"**Shot {i+1} ({shot_key}):** {shot_desc[:100]}...")
                                
                                # Add cleanup option
                                if st.button("🧹 Clean Up Temporary Files"):
                                    try:
                                        import shutil
                                        temp_dirs = [f'shot_images/{timestamp}', f'shot_videos/{timestamp}']
                                        for temp_dir in temp_dirs:
                                            if os.path.exists(temp_dir):
                                                shutil.rmtree(temp_dir)
                                        st.success("Temporary files cleaned up!")
                                    except Exception as e:
                                        st.warning(f"Could not clean up all temporary files: {str(e)}")
                            else:
                                st.error("Failed to generate final video. Please check the logs.")
                                
                        except Exception as e:
                            st.error(f"Error generating long video: {str(e)}")
                            import traceback
                            st.error(f"Detailed error: {traceback.format_exc()}")
                else:
                    st.warning("Please generate shots first")
        
        # Display generated shots
        if 'story_shots' in st.session_state:
            st.subheader("Generated Storyboard")
            for i, (shot_key, shot_desc) in enumerate(st.session_state.story_shots.items()):
                with st.expander(f"Shot {i+1}: {shot_key}"):
                    st.write(shot_desc)
        
        # Display generated content if available
        if 'final_video' in st.session_state:
            st.subheader("🎬 Long Video Results")
            
            # Video display
            if st.session_state.get('caption_video') and os.path.exists(st.session_state.caption_video):
                # Show both videos if captions are available
                col1, col2 = st.columns(2)
                with col1:
                    display_video(st.session_state.final_video, "Final Stitched Video")
                with col2:
                    display_video(st.session_state.caption_video, "Video with Captions")
            else:
                # Show only the final video if no captions
                display_video(st.session_state.final_video, "Final Stitched Video")
                if not st.session_state.get('caption_video'):
                    st.info("Caption video not available")
            
            # Download options
            st.subheader("📥 Download Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📱 Generate QR Code for Final Video"):
                    try:
                        qr_path = generate_qr_code(st.session_state.final_video, st.session_state.bucket_name)
                        if qr_path:
                            st.image(qr_path, caption="Scan to download video")
                        else:
                            st.error("Failed to generate QR code")
                    except Exception as e:
                        st.error(f"Error generating QR code: {str(e)}")
            
            with col2:
                if 'generated_images' in st.session_state:
                    if st.button("🖼️ Show Generated Images"):
                        st.subheader("Generated Images")
                        cols = st.columns(3)
                        for i, img_path in enumerate(st.session_state.generated_images):
                            with cols[i % 3]:
                                if os.path.exists(img_path):
                                    st.image(img_path, caption=f"Shot {i+1}")
            
            with col3:
                if 'generated_videos' in st.session_state:
                    if st.button("🎥 Show Individual Videos"):
                        st.subheader("Individual Shot Videos")
                        for i, video_path in enumerate(st.session_state.generated_videos):
                            if os.path.exists(video_path):
                                display_video(video_path, f"Shot {i+1} Video")

if __name__ == "__main__":
    main()
