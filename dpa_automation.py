#!/usr/bin/env python3
"""
DPA (Dynamic Product Ads) Automation System
Integrates with Nova Canvas & Reel Optimizer for AI-powered ad creation
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import boto3
from pathlib import Path
import logging
import os
import requests
from PIL import Image
import io

# Import from existing modules
from generation import generate_single_image, generate_video, optimize_prompt
from config import DEFAULT_BUCKET, DEFAULT_GUIDELINE
from virtual_try_on import VirtualTryOnGenerator, enhance_dpa_with_try_on, get_vto_category
from image_understanding import DPAImageAnalyzer, analyze_product_for_dpa

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import NovaCanvasResizer from standalone module
try:
    from canvas_resizer import NovaCanvasResizer
    logger.info("NovaCanvasResizer imported successfully")
except ImportError as e:
    NovaCanvasResizer = None
    logger.warning(f"NovaCanvasResizer not available: {e} - will use text-to-image generation")

@dataclass
class ProductData:
    """Product information for DPA"""
    product_id: str
    name: str
    description: str
    price: float
    currency: str = "USD"
    category: str = ""
    brand: str = ""
    image_url: str = ""
    availability: str = "in stock"
    condition: str = "new"
    custom_labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.custom_labels is None:
            self.custom_labels = {}

@dataclass
class AdCreative:
    """Ad creative configuration"""
    headline: str
    description: str
    call_to_action: str = "Shop Now"
    image_prompt: str = ""
    video_prompt: str = ""
    image_negative_prompt: str = ""  # Negative prompt for image generation
    video_negative_prompt: str = ""  # Negative prompt for video generation
    target_audience: str = ""
    ad_format: str = "single_image"  # single_image, carousel, video, collection
    ai_analysis: dict = None  # Store AI analysis results for VTO and other enhancements
    scene_prompts: list = None  # Multiple scene prompts for separate image generation
    scene_names: list = None  # Names for each scene for tracking
    downloaded_product_image: Image.Image = None  # Cached product image to avoid re-download
    original_image_path: str = None  # Path to saved original product image for analytics
    
@dataclass
class DPATemplate:
    """DPA template configuration"""
    template_id: str
    name: str
    platform: str  # facebook, google, amazon, etc.
    ad_format: str
    headline_template: str
    description_template: str
    image_style: str = "product_focused"
    video_style: str = "lifestyle"
    image_negative_style: str = ""  # Negative prompts for image generation
    video_negative_style: str = ""  # Negative prompts for video generation
    use_virtual_try_on: bool = False  # Enable virtual try-on enhancement
    vto_enhancement_type: str = "auto"  # "auto", "clothing", "placement"
    target_dimensions: tuple = (1080, 1080)
    
class DPAAutomator:
    """Main DPA automation class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.templates = self._load_templates()
        self.s3_client = boto3.client('s3')
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load DPA configuration"""
        default_config = {
            "output_directory": "generated_ads",
            "s3_bucket": DEFAULT_BUCKET,
            "s3_upload_enabled": True,  # Enable S3 upload by default
            "aws_region": "us-east-1",  # Default AWS region
            "image_quality": "premium",
            "video_duration": 6,
            "batch_size": 10,
            "platforms": {
                "facebook": {
                    "image_sizes": [(1080, 1080), (1200, 628), (1080, 1920)],
                    "video_sizes": [(1080, 1080), (1080, 1920)],
                    "max_headline_length": 40,
                    "max_description_length": 125
                },
                "google": {
                    "image_sizes": [(1200, 628), (300, 250), (728, 90)],
                    "video_sizes": [(1280, 720), (1920, 1080)],
                    "max_headline_length": 30,
                    "max_description_length": 90
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def _load_templates(self) -> List[DPATemplate]:
        """Load DPA templates"""
        return [
            DPATemplate(
                template_id="ecommerce_standard",
                name="E-commerce Standard",
                platform="facebook",
                ad_format="single_image",
                headline_template="{brand} {name} - {price} {currency}",
                description_template="Discover {name}. {description} Shop now and save!",
                image_style="clean product shot on white background",
                video_style="product showcase with lifestyle context"
            ),
            DPATemplate(
                template_id="lifestyle_video",
                name="Lifestyle Video",
                platform="facebook",
                ad_format="video",
                headline_template="Experience {name}",
                description_template="{description} Perfect for your lifestyle.",
                image_style="lifestyle setting with natural lighting",
                video_style="person using product in natural environment"
            ),
            DPATemplate(
                template_id="carousel_showcase",
                name="Carousel Showcase",
                platform="facebook",
                ad_format="carousel",
                headline_template="{name} Collection",
                description_template="Explore our {category} collection. {description}",
                image_style="multiple product angles and details",
                video_style="product rotation and feature highlights"
            )
        ]
    
    def create_ad_creative(self, product: ProductData, template: DPATemplate) -> AdCreative:
        """Generate ad creative from product data and template with automatic AI analysis"""
        
        # Generate headline
        headline = template.headline_template.format(
            name=product.name,
            brand=product.brand,
            price=product.price,
            currency=product.currency,
            category=product.category
        )
        
        # Generate description
        description = template.description_template.format(
            name=product.name,
            description=product.description,
            brand=product.brand,
            category=product.category
        )
        
        # Base prompts from template
        base_image_prompt = f"{template.image_style}, {product.name}, professional photography, high quality"
        video_prompt = f"{template.video_style}, {product.name}, {product.description}, engaging and dynamic"
        
        # Initialize with template negative prompts
        image_negative_prompt = template.image_negative_style
        video_negative_prompt = template.video_negative_style
        
        # Store multiple image prompts for different scenes
        image_prompts = [base_image_prompt]  # Start with base prompt
        scene_names = ["base_template"]  # Track scene names
        
        # Check if AI Suggest is enabled in template (default to True for backward compatibility)
        use_ai_suggest = getattr(template, 'use_ai_suggest', True)
        
        # Store downloaded product image for reuse
        downloaded_product_image = None
        
        # Automatic AI image analysis if product has image AND AI Suggest is enabled
        ai_analysis = None
        if hasattr(product, 'image_url') and product.image_url and use_ai_suggest:
            try:
                logger.info(f"🧠 Performing automatic AI analysis for {product.name}")
                
                # Download product image ONCE for both AI analysis and later use
                downloaded_product_image = self._download_product_image(product.image_url)
                if downloaded_product_image:
                    # Save original product image for analytics display
                    original_image_path = self._save_original_product_image(downloaded_product_image, product)
                    
                    # Perform comprehensive AI analysis
                    ai_analysis = analyze_product_for_dpa(
                        product_image=downloaded_product_image,
                        product_data={
                            'name': product.name,
                            'category': product.category,
                            'description': product.description
                        },
                        analysis_type="comprehensive"
                    )
                    
                    if ai_analysis and ai_analysis.get('success'):
                        logger.info(f"✅ AI analysis successful for {product.name}")
                        
                        # Create SEPARATE prompts for each AI-suggested background
                        if 'background_suggestions' in ai_analysis and ai_analysis['background_suggestions']:
                            background_scenes = ai_analysis['background_suggestions']
                            logger.info(f"🎨 Creating {len(background_scenes)} separate image prompts")
                            
                            # Replace base prompt with individual scene prompts
                            image_prompts = []
                            scene_names = []
                            
                            for i, scene in enumerate(background_scenes[:5]):  # Use up to 5 scenes
                                scene_prompt = f"{scene}, professional photography, high quality"
                                scene_prompt = scene_prompt.replace("*","")
                                image_prompts.append(scene_prompt)
                                
                                # Create clean scene name for tracking
                                scene_name = scene.split(',')[0].strip()  # Take first part of scene description
                                scene_name = self._clean_filename(scene_name)
                                scene_names.append(f"ai_scene_{i+1}_{scene_name}")
                        
                        # Apply AI-enhanced negative prompts
                        if 'image_generation_tips' in ai_analysis:
                            tips = ai_analysis['image_generation_tips']
                            ai_negatives = self._extract_negative_prompts_from_tips(tips)
                            if ai_negatives:
                                image_negative_prompt = f"{template.image_negative_style}, {ai_negatives}"
                        
                        logger.info(f"🚀 Created {len(image_prompts)} separate scene prompts for {product.name}")
                    else:
                        logger.warning(f"⚠️ AI analysis failed for {product.name}")
                        
            except Exception as e:
                logger.error(f"❌ Error in automatic AI analysis for {product.name}: {e}")
        elif not use_ai_suggest:
            # AI Suggest is disabled - use manual scene configuration
            logger.info(f"📝 AI Suggest disabled for {product.name}, using manual scene configuration")
            
            scene_method = getattr(template, 'scene_method', 'text_description')
            if scene_method == 'text_description':
                scene_description = getattr(template, 'scene_description', '')
                if scene_description:
                    logger.info(f"📝 Using manual scene description: {scene_description[:100]}...")
                    # Create single scene prompt from description
                    scene_prompt = f"{scene_description}, professional photography, high quality"
                    image_prompts = [scene_prompt]
                    scene_names = ["manual_scene_1"]
                else:
                    logger.warning("⚠️ No scene description provided in template")
            elif scene_method == 'reference_image':
                reference_image_name = getattr(template, 'reference_image_name', '')
                if reference_image_name:
                    logger.info(f"📸 Using reference image: {reference_image_name}")
                    # Use reference image for outpainting (implementation depends on your setup)
                    scene_prompt = f"Place product in reference scene from {reference_image_name}, professional photography, high quality"
                    image_prompts = [scene_prompt]
                    scene_names = ["reference_scene_1"]
                else:
                    logger.warning("⚠️ No reference image provided in template")
        else:
            # No image URL and AI Suggest enabled - fallback to basic prompt
            logger.info(f"ℹ️ No product image available for {product.name}, using basic template prompt")
        
        creative = AdCreative(
            headline=headline,
            description=description,
            image_prompt=base_image_prompt,  # Keep base prompt for compatibility
            video_prompt=video_prompt,
            image_negative_prompt=image_negative_prompt,
            video_negative_prompt=video_negative_prompt,
            ad_format=template.ad_format
        )
        
        # Store AI analysis results and ALL scene prompts for multi-scene generation
        if ai_analysis:
            creative.ai_analysis = ai_analysis
        
        # Store all scene prompts and names for multi-scene generation
        creative.scene_prompts = image_prompts  # All scene prompts will be used
        creative.scene_names = scene_names
        
        # Store downloaded product image and original image path for reuse
        creative.downloaded_product_image = downloaded_product_image
        if downloaded_product_image:
            creative.original_image_path = getattr(self, '_last_original_image_path', None)
        
        logger.info(f"✅ Creative generated with {len(image_prompts)} scene prompts for {product.name}")
        for i, (prompt, name) in enumerate(zip(image_prompts, scene_names)):
            logger.info(f"   Scene {i+1}: {name} - {prompt[:60]}...")
            
        return creative
    
    def _save_original_product_image(self, product_image: Image.Image, product: ProductData) -> str:
        """Save original product image for analytics display"""
        try:
            output_dir = Path(self.config["output_directory"])
            output_dir.mkdir(exist_ok=True)
            
            # Create filename for original product image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = f"original_{product.product_id}_{timestamp}.png"
            original_path = output_dir / original_filename
            
            # Save original product image
            product_image.save(original_path)
            logger.info(f"📷 Saved original product image: {original_path}")
            
            # Store path for later use
            self._last_original_image_path = str(original_path)
            
            return str(original_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save original product image: {e}")
            return None
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename by removing or replacing invalid characters"""
        import re
        
        # Remove or replace invalid filename characters
        # Replace common problematic characters
        filename = filename.replace('/', '_')
        filename = filename.replace('\\', '_')
        filename = filename.replace('*', '_')
        filename = filename.replace('?', '_')
        filename = filename.replace('<', '_')
        filename = filename.replace('>', '_')
        filename = filename.replace('|', '_')
        filename = filename.replace(':', '_')
        filename = filename.replace('"', '_')
        
        # Replace spaces and multiple underscores
        filename = filename.replace(' ', '_')
        filename = re.sub(r'_+', '_', filename)  # Replace multiple underscores with single
        
        # Remove leading/trailing underscores and convert to lowercase
        filename = filename.strip('_').lower()
        
        # Limit length to avoid filesystem issues
        if len(filename) > 50:
            filename = filename[:50].rstrip('_')
        
        # Ensure we have a valid filename
        if not filename or filename == '_':
            filename = 'scene'
            
        return filename
    
    def _download_product_image(self, image_url: str) -> Optional[Image.Image]:
        """Download product image from URL with enhanced error handling and RGB conversion"""
        try:
            if not image_url:
                return None
                
            logger.info(f"📷 Downloading product image from: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Open image from response content
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"✅ Successfully downloaded image: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"❌ Failed to download product image from {image_url}: {e}")
            return None
    
    def _extract_negative_prompts_from_tips(self, tips: List[str]) -> str:
        """Extract negative prompts from AI generation tips"""
        negative_keywords = []
        
        for tip in tips:
            tip_lower = tip.lower()
            # Look for "avoid" keywords in tips
            if 'avoid' in tip_lower:
                # Extract what to avoid
                avoid_part = tip_lower.split('avoid')[-1].strip()
                # Clean up the text
                avoid_part = avoid_part.replace('harsh', '').replace('shadows', 'harsh shadows')
                avoid_part = avoid_part.replace('poor', '').replace('lighting', 'poor lighting')
                negative_keywords.append(avoid_part.strip(', .'))
        
        return ', '.join(negative_keywords) if negative_keywords else ""
    
    def _enhance_with_ai_vto(
        self,
        base_image: Image.Image,
        product_image: Image.Image,
        product_category: str,
        enhancement_type: str = "auto",
        ai_mask_prompt: str = None
    ) -> Optional[Image.Image]:
        """Enhanced VTO with AI-analyzed mask prompts"""
        try:
            from virtual_try_on import VirtualTryOnGenerator
            
            vto_generator = VirtualTryOnGenerator()
            
            # Determine enhancement strategy
            if enhancement_type == "auto":
                clothing_categories = ["UPPER_BODY", "LOWER_BODY", "FULL_BODY", "DRESS", "OUTERWEAR"]
                if product_category.upper() in clothing_categories:
                    enhancement_type = "clothing"
                else:
                    enhancement_type = "placement"
            
            if enhancement_type == "clothing":
                # Use garment-based try-on with AI mask prompt if available
                results, _ = vto_generator.generate_product_try_on(
                    source_image=base_image,
                    product_image=product_image,
                    product_category=product_category.upper(),
                    mask_prompt=ai_mask_prompt,  # Use AI-suggested mask prompt
                    quality="premium"
                )
            else:
                # Use prompt-based placement with AI mask prompt
                placement_prompt = ai_mask_prompt or f"replace the product with the {product_category}"
                results = vto_generator.generate_product_placement(
                    scene_image=base_image,
                    product_image=product_image,
                    placement_prompt=placement_prompt,
                    quality="premium"
                )
            
            if results and len(results) > 0:
                logger.info(f"✅ AI-enhanced VTO successful with mask: {ai_mask_prompt}")
                return results[0]  # Return best result
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in AI-enhanced VTO: {e}")
            return None
    
    def _upload_to_s3(self, local_path: str, s3_key: str) -> Optional[str]:
        """Upload file to S3 and return S3 URL, with fallback to local path"""
        
        # Check if S3 upload is enabled
        if not self.config.get("s3_upload_enabled", True):
            logger.info("S3 upload disabled in configuration")
            return None
            
        try:
            bucket_name = self.config["s3_bucket"].replace("s3://", "")
            
            logger.info(f"Uploading {local_path} to {bucket_name}/{s3_key}")
            
            # Check if local file exists
            if not os.path.exists(local_path):
                logger.error(f"Local file does not exist: {local_path}")
                return None
            
            # Verify S3 client is available
            if not hasattr(self, 's3_client') or self.s3_client is None:
                logger.error("S3 client not initialized")
                return None
            
            # Test bucket access first
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
            except Exception as bucket_error:
                logger.error(f"Cannot access S3 bucket '{bucket_name}': {bucket_error}")
                logger.info("S3 upload disabled due to bucket access issues")
                # Disable S3 upload for this session to avoid repeated failures
                self.config["s3_upload_enabled"] = False
                return None
            
            # Upload file to S3 (private by default - more secure)
            self.s3_client.upload_file(
                local_path, 
                bucket_name, 
                s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            
            # Return S3 URL (will be converted to presigned URL when needed)
            s3_url = f"s3://{bucket_name}/{s3_key}"
            logger.info(f"Successfully uploaded to S3: {s3_url}")
            return s3_url
            
        except FileNotFoundError as e:
            logger.error(f"File not found for S3 upload: {local_path} - {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            logger.error(f"Bucket: {bucket_name}, Key: {s3_key}, Local path: {local_path}")
            logger.info("Continuing with local-only storage...")
            # Disable S3 upload for this session to avoid repeated failures
            self.config["s3_upload_enabled"] = False
            return None
    
    def generate_presigned_url(self, s3_url: str, expiration: int = 86400) -> Optional[str]:
        """
        Generate a presigned URL for an S3 object
        
        Args:
            s3_url: S3 URL in format s3://bucket/key
            expiration: Time in seconds for the presigned URL to remain valid (default: 24 hours)
            
        Returns:
            Presigned URL string or None if failed
        """
        try:
            # Parse S3 URL
            if not s3_url or not s3_url.startswith('s3://'):
                return None
                
            # Extract bucket and key from s3://bucket/key
            s3_parts = s3_url.replace('s3://', '').split('/', 1)
            if len(s3_parts) != 2:
                return None
                
            bucket_name, s3_key = s3_parts
            
            # Generate presigned URL
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for {s3_url} (expires in {expiration}s)")
            return presigned_url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {s3_url}: {e}")
            return None
    

    def optimize_creative(self, creative: AdCreative, product: ProductData) -> AdCreative:
        """Optimize ad creative using AI"""
        try:
            # Check if guideline file exists
            guideline_path = DEFAULT_GUIDELINE
            if not os.path.exists(guideline_path):
                logger.warning(f"Guideline file {guideline_path} not found. Skipping optimization.")
                return creative
            
            # Optimize image prompt using default guideline
            # optimize_prompt returns (prompt, negative_prompt) tuple
            optimized_image_prompt, image_negative_prompt = optimize_prompt(
                creative.image_prompt,
                guideline_path
            )
            
            # Optimize video prompt using default guideline
            # optimize_prompt returns (prompt, negative_prompt) tuple
            optimized_video_prompt, video_negative_prompt = optimize_prompt(
                creative.video_prompt,
                guideline_path
            )
            
            creative.image_prompt = optimized_image_prompt
            creative.video_prompt = optimized_video_prompt
            creative.image_negative_prompt = image_negative_prompt or ""
            creative.video_negative_prompt = video_negative_prompt or ""
            
            logger.info(f"Optimized creative for product {product.product_id}")
            return creative
            
        except Exception as e:
            logger.error(f"Failed to optimize creative: {e}")
            logger.info("Continuing with original prompts...")
            return creative
    
    def _validate_dimensions(self, width: int, height: int) -> tuple:
        """
        Validate and adjust dimensions for Nova Canvas compatibility
        Nova Canvas requires dimensions to be multiples of 64 and within specific ranges
        """
        # Nova Canvas dimension constraints
        MIN_DIM = 320
        MAX_DIM = 2048
        
        # Ensure dimensions are within valid range
        width = max(MIN_DIM, min(MAX_DIM, width))
        height = max(MIN_DIM, min(MAX_DIM, height))
        
        # Round to nearest multiple of 64 (Nova Canvas requirement)
        width = ((width + 31) // 64) * 64
        height = ((height + 31) // 64) * 64
        
        # Ensure they're still within range after rounding
        width = min(MAX_DIM, width)
        height = min(MAX_DIM, height)
        
        logger.info(f"Validated dimensions: {width}x{height}")
        return width, height

    def generate_ad_assets(self, product: ProductData, creative: AdCreative, 
                          template: DPATemplate, progress_callback=None) -> Dict[str, Any]:
        """Generate image and video assets for the ad using product images when available"""
        
        assets = {
            "product_id": product.product_id,
            "images": [],
            "videos": [],
            "metadata": {
                "template": template.template_id,
                "generated_at": datetime.utcnow().isoformat(),
                "creative": asdict(creative),
                "generation_method": "unknown"
            }
        }
        
        platform_config = self.config["platforms"].get(template.platform, {})
        
        # Use already downloaded product image from creative (avoid duplicate download)
        product_image = getattr(creative, 'downloaded_product_image', None)
        original_image_path = getattr(creative, 'original_image_path', None)
        
        # Add original image info to metadata
        if product_image and original_image_path:
            assets["metadata"]["original_product_image"] = original_image_path
            logger.info(f"📷 Using cached product image for {product.name}")
        elif hasattr(product, 'image_url') and product.image_url:
            logger.warning(f"⚠️ Product image not cached, downloading again for {product.name}")
            product_image = self._download_product_image(product.image_url)
            if product_image:
                original_image_path = self._save_original_product_image(product_image, product)
                assets["metadata"]["original_product_image"] = original_image_path
        
        try:
            # Generate images for different sizes
            if creative.ad_format in ["single_image", "carousel"]:
                image_sizes = platform_config.get("image_sizes", [(1280, 720)])
                
                for width, height in image_sizes:
                    # Validate dimensions for Nova Canvas compatibility
                    validated_width, validated_height = self._validate_dimensions(width, height)
                    logger.info(f"Generating image with validated dimensions: {validated_width}x{validated_height}")
                    
                    if product_image and NovaCanvasResizer:
                        # Use outpainting with product image for EACH SCENE
                        logger.info(f"Using outpainting with multi-scene prompts for {product.name}")
                        assets["metadata"]["generation_method"] = "outpainting_multi_scene"
                        
                        # Get all scene prompts for outpainting
                        scene_prompts = getattr(creative, 'scene_prompts', [creative.image_prompt])
                        scene_names = getattr(creative, 'scene_names', ['base_template'])
                        
                        logger.info(f"🎨 Outpainting {len(scene_prompts)} separate scenes for {product.name}")
                        
                        for scene_idx, (scene_prompt, scene_name) in enumerate(zip(scene_prompts, scene_names)):
                            logger.info(f"🖼️ Outpainting scene {scene_idx + 1}/{len(scene_prompts)}: {scene_name}")
                            
                            # Call progress callback if provided
                            if progress_callback:
                                progress_callback(f"🎨 Outpainting scene {scene_idx + 1}/{len(scene_prompts)}: {scene_name[:50]}...")
                            
                            try:
                                resizer = NovaCanvasResizer()
                                
                                # Create enhanced prompt for this scene's outpainting
                                outpainting_prompt = f"Professional advertising style, {scene_prompt}, commercial photography, high quality, marketing ready"
                                
                                # Resize and outpaint the product image with scene-specific prompt
                                result_image = resizer.resize_image(
                                    original_image=product_image,
                                    target_width=validated_width,
                                    target_height=validated_height,
                                    prompt=outpainting_prompt,
                                    negative_prompt=creative.image_negative_prompt,
                                    quality=self.config["image_quality"]
                                )
                                
                                if result_image:
                                    # Save the result image locally with scene-specific name
                                    output_dir = Path(self.config["output_directory"])
                                    output_dir.mkdir(exist_ok=True)
                                    
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    clean_scene_name = self._clean_filename(scene_name)
                                    filename = f"dpa_outpaint_{product.product_id}_{clean_scene_name}_{validated_width}x{validated_height}_{timestamp}.png"
                                    local_path = output_dir / filename
                                    
                                    result_image.save(local_path)
                                    logger.info(f"✅ Saved outpainted scene {scene_idx + 1}: {local_path}")
                                    
                                    # Upload to S3
                                    s3_key = f"dpa_generated/{filename}"
                                    s3_url = self._upload_to_s3(str(local_path), s3_key)
                                    
                                    # Add scene image to assets
                                    scene_asset = {
                                        "size": f"{validated_width}x{validated_height}",
                                        "s3_url": s3_url,
                                        "prompt": outpainting_prompt,
                                        "negative_prompt": creative.image_negative_prompt,
                                        "scene_prompt": scene_prompt,  # Original scene prompt
                                        "method": "outpainting",
                                        "image_type": "outpainted_scene",
                                        "scene_name": scene_name,
                                        "scene_index": scene_idx + 1,
                                        "total_scenes": len(scene_prompts)
                                    }
                                    
                                    # Auto-score the generated image
                                    try:
                                        # Try S3 URL first, fallback to local path
                                        image_source = s3_url if s3_url else str(local_path)
                                        
                                        if image_source and image_source != 'None':
                                            logger.info(f"🤖 Auto-scoring outpainted scene: {scene_name}")
                                            logger.info(f"   Using source: {image_source}")
                                            
                                            from image_understanding import DPAImageAnalyzer
                                            analyzer = DPAImageAnalyzer()
                                            
                                            # Score the image using flexible source (S3 URL or local path)
                                            scoring_results = analyzer.score_image_from_path_or_url(image_source)
                                            if scoring_results and scoring_results.get('success', True):
                                                scene_asset["ai_scoring"] = scoring_results
                                                logger.info(f"✅ Auto-scoring completed for scene: {scene_name}")
                                            else:
                                                error_msg = scoring_results.get('error', 'Unknown error') if scoring_results else 'No results'
                                                logger.warning(f"⚠️ Auto-scoring failed for scene: {scene_name} - {error_msg}")
                                        else:
                                            logger.warning(f"⚠️ Skipping auto-scoring for scene {scene_name}: No valid image source")
                                    except Exception as e:
                                        logger.error(f"❌ Auto-scoring error for scene {scene_name}: {e}")
                                    
                                    assets["images"].append(scene_asset)
                                    
                                    logger.info(f"✅ Outpainted scene {scene_idx + 1} completed: {scene_name}")
                                else:
                                    logger.warning(f"⚠️ Outpainting failed for scene {scene_idx + 1}: {scene_name}")
                                    
                            except Exception as e:
                                logger.error(f"❌ Outpainting error for scene {scene_name}: {e}")
                                
                                # Fallback to text-to-image for this scene
                                logger.info(f"🔄 Falling back to text-to-image for scene: {scene_name}")
                                image_result = generate_single_image(
                                    prompt=scene_prompt,
                                    negative_prompt=creative.image_negative_prompt,
                                    width=validated_width,
                                    height=validated_height,
                                    quality=self.config["image_quality"]
                                )
                                
                                if image_result:
                                    # Scene-specific filename for fallback
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    clean_scene_name = self._clean_filename(scene_name)
                                    fallback_filename = f"dpa_fallback_{product.product_id}_{clean_scene_name}_{validated_width}x{validated_height}_{timestamp}.png"
                                    
                                    # Copy to scene-specific filename
                                    output_dir = Path(self.config["output_directory"])
                                    fallback_path = output_dir / fallback_filename
                                    
                                    import shutil
                                    shutil.copy2(image_result, fallback_path)
                                    
                                    # Upload fallback image
                                    fallback_s3_key = f"dpa_generated/{fallback_filename}"
                                    fallback_s3_url = self._upload_to_s3(str(fallback_path), fallback_s3_key)
                                    
                                    fallback_asset = {
                                        "size": f"{validated_width}x{validated_height}",
                                        "s3_url": fallback_s3_url,
                                        "prompt": scene_prompt,
                                        "method": "text_to_image_fallback",
                                        "image_type": "scene_generated",
                                        "scene_name": scene_name,
                                        "scene_index": scene_idx + 1,
                                        "total_scenes": len(scene_prompts)
                                    }
                                    assets["images"].append(fallback_asset)
                                    
                                    logger.info(f"✅ Fallback scene {scene_idx + 1} completed: {scene_name}")
                        
                        # Update metadata with scene information
                        assets["metadata"]["total_scenes"] = len(scene_prompts)
                        assets["metadata"]["scene_names"] = scene_names
                        if hasattr(creative, 'ai_analysis') and creative.ai_analysis:
                            assets["metadata"]["has_ai_analysis"] = True
                            assets["metadata"]["ai_background_scenes"] = len(scene_prompts)
                    else:
                        # Use text-to-image generation with multiple scenes
                        logger.info(f"Using text-to-image generation for {product.name}")
                        assets["metadata"]["generation_method"] = "text_to_image_multi_scene"
                        
                        # Generate images for each scene prompt
                        scene_prompts = getattr(creative, 'scene_prompts', [creative.image_prompt])
                        scene_names = getattr(creative, 'scene_names', ['base_template'])
                        
                        logger.info(f"🎨 Generating {len(scene_prompts)} separate scene images for {product.name}")
                        
                        for scene_idx, (scene_prompt, scene_name) in enumerate(zip(scene_prompts, scene_names)):
                            logger.info(f"🖼️ Generating scene {scene_idx + 1}/{len(scene_prompts)}: {scene_name}")
                            
                            # Call progress callback if provided
                            if progress_callback:
                                progress_callback(f"🎨 Generating scene {scene_idx + 1}/{len(scene_prompts)}: {scene_name[:50]}...")
                            
                            image_result = generate_single_image(
                                prompt=scene_prompt,
                                negative_prompt=creative.image_negative_prompt,
                                width=validated_width,
                                height=validated_height,
                                quality=self.config["image_quality"]
                            )
                            
                            if image_result:
                                # Create scene-specific filename
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                clean_scene_name = self._clean_filename(scene_name)
                                scene_filename = f"dpa_{product.product_id}_{clean_scene_name}_{validated_width}x{validated_height}_{timestamp}.png"
                                
                                # Copy to scene-specific filename
                                output_dir = Path(self.config["output_directory"])
                                scene_path = output_dir / scene_filename
                                
                                # Copy the generated image to scene-specific path
                                import shutil
                                shutil.copy2(image_result, scene_path)
                                
                                # Upload scene image to S3
                                scene_s3_key = f"dpa_generated/{scene_filename}"
                                scene_s3_url = self._upload_to_s3(str(scene_path), scene_s3_key)
                                
                                # Add scene image to assets
                                scene_image_asset = {
                                    "size": f"{validated_width}x{validated_height}",
                                    "s3_url": scene_s3_url,
                                    "prompt": scene_prompt,
                                    "negative_prompt": creative.image_negative_prompt,
                                    "method": "text_to_image",
                                    "image_type": "scene_generated",
                                    "scene_name": scene_name,
                                    "scene_index": scene_idx + 1,
                                    "total_scenes": len(scene_prompts)
                                }
                                
                                # Auto-score the generated image
                                try:
                                    # Try S3 URL first, fallback to local path
                                    image_source = scene_s3_url if scene_s3_url else str(scene_path)
                                    
                                    if image_source and image_source != 'None':
                                        logger.info(f"🤖 Auto-scoring text-to-image scene: {scene_name}")
                                        logger.info(f"   Using source: {image_source}")
                                        
                                        from image_understanding import DPAImageAnalyzer
                                        analyzer = DPAImageAnalyzer()
                                        
                                        # Score the image using flexible source (S3 URL or local path)
                                        scoring_results = analyzer.score_image_from_path_or_url(image_source)
                                        if scoring_results and scoring_results.get('success', True):
                                            scene_image_asset["ai_scoring"] = scoring_results
                                            logger.info(f"✅ Auto-scoring completed for scene: {scene_name}")
                                        else:
                                            error_msg = scoring_results.get('error', 'Unknown error') if scoring_results else 'No results'
                                            logger.warning(f"⚠️ Auto-scoring failed for scene: {scene_name} - {error_msg}")
                                    else:
                                        logger.warning(f"⚠️ Skipping auto-scoring for scene {scene_name}: No valid image source")
                                except Exception as e:
                                    logger.error(f"❌ Auto-scoring error for scene {scene_name}: {e}")
                                
                                assets["images"].append(scene_image_asset)
                                
                                logger.info(f"✅ Scene {scene_idx + 1} generated: {scene_name}")
                                
                                # Check if virtual try-on enhancement is enabled for this scene
                                if template.use_virtual_try_on and product_image:
                                    logger.info(f"🎭 Applying VTO enhancement to scene: {scene_name}")
                                    try:
                                        # Load the generated scene image
                                        generated_scene_image = Image.open(scene_path)
                                        
                                        # Get VTO category for the product
                                        vto_category = get_vto_category(product.category)
                                        
                                        # Apply virtual try-on enhancement with AI mask prompts
                                        ai_mask_prompt = None
                                        if hasattr(creative, 'ai_analysis') and creative.ai_analysis:
                                            mask_prompts = creative.ai_analysis.get('mask_prompts', [])
                                            if mask_prompts:
                                                ai_mask_prompt = mask_prompts[0]
                                                logger.info(f"🎯 Using AI mask prompt: {ai_mask_prompt}")
                                        
                                        enhanced_scene_image = self._enhance_with_ai_vto(
                                            base_image=generated_scene_image,
                                            product_image=product_image,
                                            product_category=vto_category,
                                            enhancement_type=template.vto_enhancement_type,
                                            ai_mask_prompt=ai_mask_prompt
                                        )
                                        
                                        if enhanced_scene_image:
                                            # Save VTO-enhanced scene image
                                            clean_scene_name = self._clean_filename(scene_name)
                                            vto_scene_filename = f"dpa_vto_{product.product_id}_{clean_scene_name}_{validated_width}x{validated_height}_{timestamp}.png"
                                            vto_scene_path = output_dir / vto_scene_filename
                                            enhanced_scene_image.save(vto_scene_path)
                                            
                                            # Upload VTO scene image to S3
                                            vto_scene_s3_key = f"dpa_generated/{vto_scene_filename}"
                                            vto_scene_s3_url = self._upload_to_s3(str(vto_scene_path), vto_scene_s3_key)
                                            
                                            # Add VTO scene image to assets
                                            vto_scene_asset = {
                                                "size": f"{validated_width}x{validated_height}",
                                                "s3_url": vto_scene_s3_url,
                                                "prompt": scene_prompt,
                                                "negative_prompt": creative.image_negative_prompt,
                                                "method": "text_to_image_with_vto",
                                                "image_type": "vto_scene_enhanced",
                                                "scene_name": scene_name,
                                                "scene_index": scene_idx + 1,
                                                "total_scenes": len(scene_prompts),
                                                "enhancement": "virtual_try_on",
                                                "vto_category": vto_category,
                                                "base_scene_path": str(scene_path)
                                            }
                                            
                                            # Auto-score the VTO enhanced image
                                            try:
                                                # Try S3 URL first, fallback to local path
                                                image_source = vto_scene_s3_url if vto_scene_s3_url else str(vto_scene_path)
                                                
                                                if image_source and image_source != 'None':
                                                    logger.info(f"🤖 Auto-scoring VTO enhanced scene: {scene_name}")
                                                    logger.info(f"   Using source: {image_source}")
                                                    
                                                    from image_understanding import DPAImageAnalyzer
                                                    analyzer = DPAImageAnalyzer()
                                                    
                                                    # Score the image using flexible source (S3 URL or local path)
                                                    scoring_results = analyzer.score_image_from_path_or_url(image_source)
                                                    if scoring_results and scoring_results.get('success', True):
                                                        vto_scene_asset["ai_scoring"] = scoring_results
                                                        logger.info(f"✅ Auto-scoring completed for VTO scene: {scene_name}")
                                                    else:
                                                        error_msg = scoring_results.get('error', 'Unknown error') if scoring_results else 'No results'
                                                        logger.warning(f"⚠️ Auto-scoring failed for VTO scene: {scene_name} - {error_msg}")
                                                else:
                                                    logger.warning(f"⚠️ Skipping auto-scoring for VTO scene {scene_name}: No valid image source")
                                            except Exception as e:
                                                logger.error(f"❌ Auto-scoring error for VTO scene {scene_name}: {e}")
                                            
                                            assets["images"].append(vto_scene_asset)
                                            
                                            logger.info(f"✅ VTO scene {scene_idx + 1} enhanced: {scene_name}")
                                        else:
                                            logger.warning(f"⚠️ VTO enhancement failed for scene: {scene_name}")
                                            
                                    except Exception as e:
                                        logger.error(f"❌ VTO enhancement error for scene {scene_name}: {e}")
                            else:
                                logger.warning(f"⚠️ Failed to generate scene {scene_idx + 1}: {scene_name}")
                        
                        # Update metadata with scene information
                        assets["metadata"]["total_scenes"] = len(scene_prompts)
                        assets["metadata"]["scene_names"] = scene_names
                        if hasattr(creative, 'ai_analysis') and creative.ai_analysis:
                            assets["metadata"]["has_ai_analysis"] = True
                            assets["metadata"]["ai_background_scenes"] = len(scene_prompts)
            
            # Generate videos (unchanged for now)
            if creative.ad_format in ["video", "carousel"]:
                video_sizes = platform_config.get("video_sizes", [(1080, 1080)])
                
                for width, height in video_sizes:
                    video_result = generate_video(
                        prompt=creative.video_prompt,
                        bucket=self.config["s3_bucket"]
                    )
                    
                    if video_result:
                        assets["videos"].append({
                            "size": f"{width}x{height}",
                            "path": video_result,
                            "prompt": creative.video_prompt
                        })
            
            logger.info(f"Generated assets for product {product.product_id} using {assets['metadata']['generation_method']}")
            
        except Exception as e:
            logger.error(f"Failed to generate assets for {product.product_id}: {e}")
        
        return assets
    
    def process_product_catalog(self, catalog_path: str, template_id: str, 
                              output_format: str = "json") -> List[Dict]:
        """Process entire product catalog for DPA creation"""
        
        # Load product catalog
        if catalog_path.endswith('.csv'):
            df = pd.read_csv(catalog_path)
        elif catalog_path.endswith('.json'):
            df = pd.read_json(catalog_path)
        else:
            raise ValueError("Unsupported catalog format. Use CSV or JSON.")
        
        # Find template
        template = next((t for t in self.templates if t.template_id == template_id), None)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        results = []
        
        for _, row in df.iterrows():
            try:
                # Create product data
                product = ProductData(
                    product_id=str(row.get('product_id', '')),
                    name=str(row.get('name', '')),
                    description=str(row.get('description', '')),
                    price=float(row.get('price', 0)),
                    currency=str(row.get('currency', 'USD')),
                    category=str(row.get('category', '')),
                    brand=str(row.get('brand', '')),
                    image_url=str(row.get('image_url', '')),
                    availability=str(row.get('availability', 'in stock')),
                    condition=str(row.get('condition', 'new'))
                )
                
                # Create and optimize creative
                creative = self.create_ad_creative(product, template)
                creative = self.optimize_creative(creative, product)
                
                # Generate assets
                assets = self.generate_ad_assets(product, creative, template)
                
                results.append({
                    "product": asdict(product),
                    "creative": asdict(creative),
                    "assets": assets
                })
                
                logger.info(f"Processed product {product.product_id}")
                
            except Exception as e:
                logger.error(f"Failed to process product {row.get('product_id', 'unknown')}: {e}")
                continue
        
        # Save results
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"dpa_results_{template_id}_{timestamp}.{output_format}"
        
        if output_format == "json":
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        elif output_format == "csv":
            # Flatten results for CSV
            flattened = []
            for result in results:
                flat_result = {
                    **result["product"],
                    "headline": result["creative"]["headline"],
                    "description": result["creative"]["description"],
                    "image_prompt": result["creative"]["image_prompt"],
                    "video_prompt": result["creative"]["video_prompt"],
                    "num_images": len(result["assets"]["images"]),
                    "num_videos": len(result["assets"]["videos"])
                }
                flattened.append(flat_result)
            
            pd.DataFrame(flattened).to_csv(output_file, index=False)
        
        logger.info(f"Saved results to {output_file}")
        return results

def main():
    """Example usage"""
    automator = DPAAutomator()
    
    # Example: Process a product catalog
    # results = automator.process_product_catalog(
    #     catalog_path="product_catalog.csv",
    #     template_id="ecommerce_standard",
    #     output_format="json"
    # )
    
    print("DPA Automator initialized successfully!")
    print(f"Available templates: {[t.template_id for t in automator.templates]}")

if __name__ == "__main__":
    main()
