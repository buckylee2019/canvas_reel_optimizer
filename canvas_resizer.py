#!/usr/bin/env python3
"""
Nova Canvas Resizer - Standalone module for image outpainting
Extracted from streamlit_app.py to avoid import conflicts
"""

import boto3
import time
from functools import wraps
import base64
import json
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, base_delay=1):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class NovaCanvasResizer:
    """Production-ready AI image resizer using Amazon Nova Canvas."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize the resizer with direct boto3 client."""
        self.region_name = region_name
        self.model_id = "amazon.nova-canvas-v1:0"
        
    def get_bedrock_client(self):
        """Get bedrock client directly with boto3."""
        return boto3.client('bedrock-runtime', region_name=self.region_name)
        
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string with proper format validation."""
        # Ensure RGB mode for consistency
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use PNG format for better quality
        buffer = BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        return base64_string
    
    def _base64_to_image(self, base64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        try:
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_bytes))
            return image
        except Exception as e:
            raise Exception(f"Failed to decode base64 image: {str(e)}")
    
    def _create_canvas_and_mask(self, original_image: Image.Image, target_size: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
        """Create canvas and mask for outpainting."""
        orig_width, orig_height = original_image.size
        target_width, target_height = target_size
        
        # Calculate scaling to fit original image in target canvas
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        scale = min(scale_x, scale_y, 1.0)  # Don't upscale beyond original
        
        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize original image
        resized_original = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create canvas
        canvas = Image.new('RGB', target_size, (255, 255, 255))
        
        # Calculate position to center the image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste resized image onto canvas
        canvas.paste(resized_original, (x_offset, y_offset))
        
        # Create mask (black = preserve, white = outpaint)
        mask = Image.new('RGB', target_size, (255, 255, 255))  # White background
        black_region = Image.new('RGB', (new_width, new_height), (0, 0, 0))  # Black for original image area
        mask.paste(black_region, (x_offset, y_offset))
        
        return canvas, mask
    
    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def resize_image(self, 
                    original_image: Image.Image, 
                    target_width: int, 
                    target_height: int,
                    prompt: str = "professional product photography, clean background, high quality",
                    quality: str = "premium",
                    cfg_scale: float = 7.0) -> Image.Image:
        """
        Resize image using Nova Canvas outpainting.
        
        Args:
            original_image: PIL Image to resize
            target_width: Target width in pixels
            target_height: Target height in pixels
            prompt: Text prompt for outpainting
            quality: Image quality ('standard' or 'premium')
            cfg_scale: CFG scale for generation control
            
        Returns:
            PIL Image with new dimensions
        """
        try:
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
                    "height": target_height,
                    "width": target_width,
                    "cfgScale": cfg_scale,
                    "quality": quality
                }
            }
            
            # Invoke Nova Canvas
            bedrock_client = self.get_bedrock_client()
            response = bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(inference_params),
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            if 'images' in response_body and len(response_body['images']) > 0:
                result_b64 = response_body['images'][0]
                result_image = self._base64_to_image(result_b64)
                
                logger.info(f"Successfully resized image to {target_width}x{target_height}")
                return result_image
            else:
                raise Exception("No images returned from Nova Canvas")
                
        except Exception as e:
            logger.error(f"Resize failed: {str(e)}")
            raise Exception(f"Image resize failed: {str(e)}")
    
    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def outpaint_image(self, 
                      original_image: Image.Image,
                      prompt: str,
                      mask_prompt: str = "",
                      target_width: Optional[int] = None,
                      target_height: Optional[int] = None,
                      quality: str = "premium",
                      cfg_scale: float = 7.0) -> Image.Image:
        """
        Outpaint image using Nova Canvas.
        
        Args:
            original_image: PIL Image object to outpaint
            prompt: Text prompt for outpainting
            mask_prompt: Mask prompt for specific area targeting
            target_width: Target width (optional, uses original if not specified)
            target_height: Target height (optional, uses original if not specified)
            quality: Image quality ('standard' or 'premium')
            cfg_scale: CFG scale for generation control
            
        Returns:
            PIL Image with outpainting applied
        """
        try:
            # Use original dimensions if target not specified
            if target_width is None:
                target_width = original_image.width
            if target_height is None:
                target_height = original_image.height
            
            # Convert image to base64
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
                    "height": target_height,
                    "width": target_width,
                    "cfgScale": cfg_scale,
                    "quality": quality
                }
            }
            
            # Add mask prompt if provided
            if mask_prompt:
                inference_params["outPaintingParams"]["maskPrompt"] = mask_prompt
            
            # Invoke Nova Canvas using connection manager
            bedrock_client = self.get_bedrock_client()
            response = bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(inference_params),
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            if 'images' in response_body and len(response_body['images']) > 0:
                result_b64 = response_body['images'][0]
                result_image = self._base64_to_image(result_b64)
                
                logger.info(f"Successfully outpainted image")
                return result_image
            else:
                raise Exception("No images returned from Nova Canvas")
            
        except Exception as e:
            raise Exception(f"Outpainting failed: {str(e)}")
