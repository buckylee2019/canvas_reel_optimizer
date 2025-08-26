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
        """Generate ad creative from product data and template"""
        
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
        
        # Create image prompt
        image_prompt = f"{template.image_style}, {product.name}, {product.description}, professional photography, high quality"
        
        # Create video prompt
        video_prompt = f"{template.video_style}, {product.name}, {product.description}, engaging and dynamic"
        
        return AdCreative(
            headline=headline,
            description=description,
            image_prompt=image_prompt,
            video_prompt=video_prompt,
            image_negative_prompt=template.image_negative_style,
            video_negative_prompt=template.video_negative_style,
            ad_format=template.ad_format
        )
    
    def _upload_to_s3(self, local_path: str, s3_key: str) -> Optional[str]:
        """Upload file to S3 and return S3 URL"""
        
        # Check if S3 upload is enabled
        if not self.config.get("s3_upload_enabled", False):
            logger.info("S3 upload disabled in configuration")
            return None
            
        try:
            bucket_name = self.config["s3_bucket"].replace("s3://", "")
            
            logger.info(f"Uploading {local_path} to {bucket_name}/{s3_key}")
            
            # Upload file to S3
            self.s3_client.upload_file(
                local_path, 
                bucket_name, 
                s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            
            # Return S3 URL
            s3_url = f"s3://{bucket_name}/{s3_key}"
            logger.info(f"Successfully uploaded to S3: {s3_url}")
            return s3_url
            
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            logger.info("Continuing with local-only storage...")
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
    
    def _download_product_image(self, image_url: str) -> Optional[Image.Image]:
        """Download product image from URL"""
        try:
            if not image_url:
                return None
                
            logger.info(f"Downloading product image from: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Open image from response content
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"Successfully downloaded image: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to download image from {image_url}: {e}")
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
                          template: DPATemplate) -> Dict[str, Any]:
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
        
        # Download product image if available
        product_image = None
        if hasattr(product, 'image_url') and product.image_url:
            product_image = self._download_product_image(product.image_url)
        
        try:
            # Generate images for different sizes
            if creative.ad_format in ["single_image", "carousel"]:
                image_sizes = platform_config.get("image_sizes", [(1280, 720)])
                
                for width, height in image_sizes:
                    # Validate dimensions for Nova Canvas compatibility
                    validated_width, validated_height = self._validate_dimensions(width, height)
                    logger.info(f"Generating image with validated dimensions: {validated_width}x{validated_height}")
                    
                    if product_image and NovaCanvasResizer:
                        # Use outpainting with product image
                        try:
                            resizer = NovaCanvasResizer()
                            
                            # Create enhanced prompt for outpainting
                            outpainting_prompt = f"Professional advertising style, {creative.image_prompt}, commercial photography, high quality, marketing ready"
                            
                            logger.info(f"Using outpainting with product image for {product.name}")
                            assets["metadata"]["generation_method"] = "outpainting"
                            
                            # Resize and outpaint the product image
                            result_image = resizer.resize_image(
                                original_image=product_image,
                                target_width=validated_width,
                                target_height=validated_height,
                                prompt=outpainting_prompt,
                                quality=self.config["image_quality"]
                            )
                            
                            if result_image:
                                # Save the result image locally
                                output_dir = Path(self.config["output_directory"])
                                output_dir.mkdir(exist_ok=True)
                                
                                filename = f"dpa_{product.product_id}_{validated_width}x{validated_height}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                                local_path = output_dir / filename
                                
                                result_image.save(local_path)
                                logger.info(f"Saved locally: {local_path}")
                                
                                # Upload to S3
                                s3_key = f"dpa_generated/{filename}"
                                s3_url = self._upload_to_s3(str(local_path), s3_key)
                                
                                assets["images"].append({
                                    "size": f"{validated_width}x{validated_height}",
                                    "local_path": str(local_path),
                                    "s3_url": s3_url,
                                    "prompt": outpainting_prompt,
                                    "method": "outpainting"
                                })
                                
                                logger.info(f"Saved outpainted image: {local_path}")
                            else:
                                logger.warning(f"Outpainting failed for {width}x{height}")
                                
                        except Exception as e:
                            logger.error(f"Outpainting failed: {e}, falling back to text-to-image")
                            # Fallback to text-to-image
                            image_result = generate_single_image(
                                prompt=creative.image_prompt,
                                negative_prompt=creative.image_negative_prompt,
                                width=width,
                                height=height,
                                quality=self.config["image_quality"]
                            )
                            
                            if image_result:
                                # Upload to S3
                                filename = Path(image_result).name
                                s3_key = f"dpa_generated/{filename}"
                                s3_url = self._upload_to_s3(image_result, s3_key)
                                
                                assets["images"].append({
                                    "size": f"{width}x{height}",
                                    "local_path": image_result,
                                    "s3_url": s3_url,
                                    "prompt": creative.image_prompt,
                                    "method": "text_to_image_fallback"
                                })
                                assets["metadata"]["generation_method"] = "text_to_image_fallback"
                    else:
                        # Use text-to-image generation (original method)
                        logger.info(f"Using text-to-image generation for {product.name}")
                        assets["metadata"]["generation_method"] = "text_to_image"
                        
                        image_result = generate_single_image(
                            prompt=creative.image_prompt,
                            negative_prompt=creative.image_negative_prompt,
                            width=validated_width,
                            height=validated_height,
                            quality=self.config["image_quality"]
                        )
                        
                        if image_result:
                            # Always save the base AI-generated image first
                            base_filename = Path(image_result).name
                            base_s3_key = f"dpa_generated/{base_filename}"
                            base_s3_url = self._upload_to_s3(image_result, base_s3_key)
                            
                            # Add base image to assets
                            base_image_asset = {
                                "size": f"{validated_width}x{validated_height}",
                                "local_path": image_result,
                                "s3_url": base_s3_url,
                                "prompt": creative.image_prompt,
                                "negative_prompt": creative.image_negative_prompt,
                                "method": "text_to_image",
                                "image_type": "base_generated"
                            }
                            assets["images"].append(base_image_asset)
                            logger.info(f"Base AI-generated image saved for {product.name}: {base_filename}")
                            
                            # Check if virtual try-on enhancement is enabled
                            if template.use_virtual_try_on and product_image:
                                logger.info(f"Applying virtual try-on enhancement for {product.name}")
                                try:
                                    # Load the generated image
                                    generated_image = Image.open(image_result)
                                    
                                    # Get VTO category for the product
                                    vto_category = get_vto_category(product.category)
                                    
                                    # Apply virtual try-on enhancement
                                    enhanced_image = enhance_dpa_with_try_on(
                                        base_image=generated_image,
                                        product_image=product_image,
                                        product_category=vto_category,
                                        enhancement_type=template.vto_enhancement_type
                                    )
                                    
                                    if enhanced_image:
                                        # Save enhanced image with clear naming
                                        enhanced_filename = f"dpa_vto_{product.product_id}_{validated_width}x{validated_height}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                                        enhanced_path = output_dir / enhanced_filename
                                        enhanced_image.save(enhanced_path)
                                        
                                        # Upload enhanced image to S3
                                        enhanced_s3_key = f"dpa_generated/{enhanced_filename}"
                                        enhanced_s3_url = self._upload_to_s3(str(enhanced_path), enhanced_s3_key)
                                        
                                        # Add enhanced image to assets
                                        enhanced_image_asset = {
                                            "size": f"{validated_width}x{validated_height}",
                                            "local_path": str(enhanced_path),
                                            "s3_url": enhanced_s3_url,
                                            "prompt": creative.image_prompt,
                                            "negative_prompt": creative.image_negative_prompt,
                                            "method": "text_to_image_with_vto",
                                            "image_type": "vto_enhanced",
                                            "enhancement": "virtual_try_on",
                                            "vto_category": vto_category,
                                            "base_image_path": image_result  # Reference to original
                                        }
                                        assets["images"].append(enhanced_image_asset)
                                        
                                        logger.info(f"VTO-enhanced image saved for {product.name}: {enhanced_filename}")
                                        
                                        # Update metadata to indicate both versions available
                                        assets["metadata"]["has_vto_enhancement"] = True
                                        assets["metadata"]["vto_category"] = vto_category
                                        assets["metadata"]["total_image_variants"] = len([img for img in assets["images"] if img.get("method", "").startswith("text_to_image")])
                                        
                                    else:
                                        logger.warning(f"Virtual try-on enhancement failed for {product.name}, only base image available")
                                        assets["metadata"]["vto_enhancement_failed"] = True
                                        
                                except Exception as e:
                                    logger.error(f"Virtual try-on enhancement error for {product.name}: {e}")
                                    assets["metadata"]["vto_enhancement_error"] = str(e)
                            else:
                                if template.use_virtual_try_on:
                                    logger.info(f"VTO enabled but no product image available for {product.name}")
                                    assets["metadata"]["vto_skipped_reason"] = "no_product_image"
            
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
