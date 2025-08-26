#!/usr/bin/env python3
"""
Virtual Try-On functionality for DPA using Nova Canvas
Integrates product replacement in generated images
"""

import io
import json
import base64
import boto3
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class VirtualTryOnGenerator:
    """Virtual Try-On generator using Nova Canvas for product replacement in DPA images"""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize the VTO generator with AWS Bedrock client."""
        self.bedrock = boto3.client(service_name="bedrock-runtime", region_name=region_name)
        self.model_id = "amazon.nova-canvas-v1:0"
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for Nova Canvas."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def generate_product_try_on(
        self,
        source_image: Image.Image,
        product_image: Image.Image,
        product_category: str = "UPPER_BODY",
        mask_prompt: Optional[str] = None,
        quality: str = "standard",
        num_images: int = 1,
        preserve_face: bool = True,
        preserve_hands: bool = True,
        preserve_body_pose: bool = True
    ) -> Tuple[Optional[List[Image.Image]], Optional[Image.Image]]:
        """
        Generate virtual try-on for product replacement in DPA images.
        
        Args:
            source_image: Base image (person or scene)
            product_image: Product to be placed/replaced
            product_category: Product category for garment-based masking
            mask_prompt: Custom prompt for prompt-based masking
            quality: Image quality ("standard" or "premium")
            num_images: Number of images to generate
            preserve_face: Whether to preserve face
            preserve_hands: Whether to preserve hands
            preserve_body_pose: Whether to preserve body pose
            
        Returns:
            Tuple of (result_images, mask_image)
        """
        try:
            # Determine mask type based on inputs
            if mask_prompt:
                mask_type = "PROMPT"
            else:
                mask_type = "GARMENT"
            
            # Prepare VTO parameters
            vto_params = {
                "sourceImage": self._image_to_base64(source_image),
                "referenceImage": self._image_to_base64(product_image),
                "maskType": mask_type,
                "maskExclusions": {
                    "preserveFace": preserve_face,
                    "preserveHands": preserve_hands,
                    "preserveBodyPose": preserve_body_pose
                },
                "returnMask": True  # Return mask for debugging
            }
            
            # Add mask-specific parameters
            if mask_type == "GARMENT":
                vto_params["garmentBasedMask"] = {
                    "garmentClass": product_category
                }
            elif mask_type == "PROMPT":
                vto_params["promptBasedMask"] = {
                    "maskPrompt": mask_prompt
                }
            
            # Prepare inference parameters
            inference_params = {
                "taskType": "VIRTUAL_TRY_ON",
                "virtualTryOnParams": vto_params,
                "imageGenerationConfig": {
                    "numberOfImages": num_images,
                    "quality": quality,
                    "cfgScale": 5.0
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
            
            if response_body.get("error"):
                logger.error(f"VTO Error: {response_body.get('error')}")
                return None, None
            
            # Extract images
            result_images = []
            images = response_body.get("images", [])
            
            for img_b64 in images:
                image_bytes = base64.b64decode(img_b64)
                image_buffer = io.BytesIO(image_bytes)
                image = Image.open(image_buffer)
                result_images.append(image)
            
            # Extract mask if available
            mask_img = None
            if response_body.get("maskImage"):
                mask_bytes = base64.b64decode(response_body["maskImage"])
                mask_buffer = io.BytesIO(mask_bytes)
                mask_img = Image.open(mask_buffer)
            
            return result_images, mask_img
            
        except ClientError as e:
            logger.error(f"AWS Client Error in VTO: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error in VTO: {e}")
            return None, None
    
    def generate_product_placement(
        self,
        scene_image: Image.Image,
        product_image: Image.Image,
        placement_prompt: str,
        quality: str = "standard"
    ) -> Optional[List[Image.Image]]:
        """
        Generate product placement in scene using prompt-based masking.
        
        Args:
            scene_image: Background scene image
            product_image: Product to place in scene
            placement_prompt: Prompt describing where to place product
            quality: Image quality
            
        Returns:
            List of result images or None if failed
        """
        result_images, _ = self.generate_product_try_on(
            source_image=scene_image,
            product_image=product_image,
            mask_prompt=placement_prompt,
            quality=quality,
            preserve_face=False,
            preserve_hands=False,
            preserve_body_pose=False
        )
        
        return result_images
    
    def generate_clothing_try_on(
        self,
        person_image: Image.Image,
        clothing_image: Image.Image,
        clothing_type: str = "UPPER_BODY",
        quality: str = "standard"
    ) -> Optional[List[Image.Image]]:
        """
        Generate clothing try-on using garment-based masking.
        
        Args:
            person_image: Image of person
            clothing_image: Clothing product image
            clothing_type: Type of clothing (UPPER_BODY, LOWER_BODY, FULL_BODY, etc.)
            quality: Image quality
            
        Returns:
            List of result images or None if failed
        """
        result_images, _ = self.generate_product_try_on(
            source_image=person_image,
            product_image=clothing_image,
            product_category=clothing_type,
            quality=quality
        )
        
        return result_images

# DPA Integration Functions

def enhance_dpa_with_try_on(
    base_image: Image.Image,
    product_image: Image.Image,
    product_category: str,
    enhancement_type: str = "auto"
) -> Optional[Image.Image]:
    """
    Enhance DPA generated image with virtual try-on product replacement.
    
    Args:
        base_image: Base generated image from DPA
        product_image: Product image to integrate
        product_category: Category of product for appropriate masking
        enhancement_type: Type of enhancement ("clothing", "placement", "auto")
        
    Returns:
        Enhanced image or None if failed
    """
    vto_generator = VirtualTryOnGenerator()
    
    # Determine enhancement strategy
    if enhancement_type == "auto":
        # Auto-detect based on product category
        clothing_categories = ["UPPER_BODY", "LOWER_BODY", "FULL_BODY", "DRESS", "OUTERWEAR"]
        if product_category.upper() in clothing_categories:
            enhancement_type = "clothing"
        else:
            enhancement_type = "placement"
    
    try:
        if enhancement_type == "clothing":
            # Use garment-based try-on
            results = vto_generator.generate_clothing_try_on(
                person_image=base_image,
                clothing_image=product_image,
                clothing_type=product_category.upper(),
                quality="premium"
            )
        else:
            # Use prompt-based placement
            placement_prompt = f"replace the product with the {product_category}"
            results = vto_generator.generate_product_placement(
                scene_image=base_image,
                product_image=product_image,
                placement_prompt=placement_prompt,
                quality="premium"
            )
        
        if results and len(results) > 0:
            return results[0]  # Return best result
        
        return None
        
    except Exception as e:
        logger.error(f"Error in DPA try-on enhancement: {e}")
        return None

# Product Category Mapping for VTO
PRODUCT_CATEGORY_MAPPING = {
    # Clothing categories
    "shirt": "UPPER_BODY",
    "t-shirt": "UPPER_BODY", 
    "blouse": "UPPER_BODY",
    "jacket": "OUTERWEAR",
    "coat": "OUTERWEAR",
    "sweater": "UPPER_BODY",
    "hoodie": "UPPER_BODY",
    "pants": "LOWER_BODY",
    "jeans": "LOWER_BODY",
    "shorts": "LOWER_BODY",
    "skirt": "LOWER_BODY",
    "dress": "DRESS",
    "suit": "FULL_BODY",
    
    # Accessories (use prompt-based)
    "watch": "ACCESSORY",
    "jewelry": "ACCESSORY",
    "bag": "ACCESSORY",
    "shoes": "FOOTWEAR",
    "hat": "HEADWEAR",
    "sunglasses": "EYEWEAR",
    
    # Electronics (use prompt-based)
    "phone": "ELECTRONICS",
    "laptop": "ELECTRONICS",
    "tablet": "ELECTRONICS",
    "headphones": "ELECTRONICS",
    
    # Default
    "default": "UPPER_BODY"
}

def get_vto_category(product_category: str) -> str:
    """Get appropriate VTO category for product."""
    return PRODUCT_CATEGORY_MAPPING.get(
        product_category.lower(), 
        PRODUCT_CATEGORY_MAPPING["default"]
    )
