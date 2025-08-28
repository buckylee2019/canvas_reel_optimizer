#!/usr/bin/env python3
"""
Image Understanding Module for DPA
Analyzes product images to generate better mask prompts and background suggestions
"""

import base64
import json
import boto3
from PIL import Image
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class DPAImageAnalyzer:
    """Advanced image analyzer for DPA using Claude Sonnet for product-specific insights"""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize the image analyzer with AWS Bedrock client."""
        self.bedrock = boto3.client(service_name="bedrock-runtime", region_name=region_name)
        self.model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    
    def _image_to_base64(self, image_input) -> str:
        """Convert various image inputs to base64 string."""
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
            # It's an uploaded file or bytes
            if hasattr(image_input, 'getvalue'):
                image_bytes = image_input.getvalue()
            else:
                image_bytes = image_input
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def analyze_product_image(
        self,
        image_input,
        product_category: str,
        product_name: str = "",
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Analyze product image for DPA enhancement suggestions.
        
        Args:
            image_input: PIL Image, file path, or uploaded file
            product_category: Category of the product (shirt, phone, watch, etc.)
            product_name: Name of the product for context
            analysis_type: "mask", "background", "vto", or "comprehensive"
            
        Returns:
            Dictionary with analysis results and suggestions
        """
        try:
            image_b64 = self._image_to_base64(image_input)
            
            # Create product-specific analysis prompt
            prompt = self._create_analysis_prompt(product_category, product_name, analysis_type)
            
            # Prepare the request
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1500,
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
            response = self.bedrock.invoke_model(
                body=json.dumps(request_body),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response.get("body").read())
            suggestions = response_body.get("content", [{}])[0].get("text", "")
            
            # Parse the structured response
            parsed_results = self._parse_analysis_results(suggestions, analysis_type)
            logger.info("parsed_results: %s", parsed_results)
            return {
                "success": True,
                "product_category": product_category,
                "product_name": product_name,
                "analysis_type": analysis_type,
                "raw_response": suggestions,
                **parsed_results
            }
            
        except Exception as e:
            logger.error(f"Error analyzing product image: {e}")
            return {
                "success": False,
                "error": str(e),
                "product_category": product_category,
                "analysis_type": analysis_type
            }
    
    def _create_analysis_prompt(self, product_category: str, product_name: str, analysis_type: str) -> str:
        """Create product-specific analysis prompt based on category and type."""
        
        base_context = f"""
        You are analyzing a product image for Dynamic Product Advertising (DPA).
        Product Category: {product_category}
        Product Name: {product_name}
        
        This analysis will be used to:
        1. Generate better Virtual Try-On mask prompts
        2. Suggest appropriate background scenes for the product
        3. Optimize image generation for advertising
        """
        
        if analysis_type == "mask":
            return base_context + """
            TASK: Generate Virtual Try-On mask prompts for this product image.
            
            Analyze the product and suggest specific mask prompts that would work best for Virtual Try-On:
            
            For CLOTHING items, focus on:
            - Garment type and fit areas
            - Specific clothing regions (collar, sleeves, hem, etc.)
            - Layering considerations
            
            For ACCESSORIES, focus on:
            - Placement areas (wrist, neck, ears, etc.)
            - Size and positioning
            - Attachment points
            
            For ELECTRONICS, focus on:
            - Usage contexts (hand-held, desktop, worn)
            - Interaction areas
            - Placement surfaces
            
            Return 4-6 specific mask prompts, one per line:
            """
        
        elif analysis_type == "background":
            return base_context + """
            TASK: Suggest appropriate background scenes for this product in advertising.
            
            Analyze the product and suggest background scenes that would:
            - Complement the product's style and target audience
            - Create appealing lifestyle contexts
            - Enhance the product's perceived value
            - Work well for the target demographic
            
            Consider:
            - Product positioning (luxury, casual, professional, etc.)
            - Usage scenarios (home, office, outdoor, social, etc.)
            - Target audience lifestyle
            - Brand positioning
            
            Return 5-7 background scene suggestions, one per line:
            """
        
        elif analysis_type == "vto":
            return base_context + """
            TASK: Provide Virtual Try-On optimization suggestions for this product.
            
            Analyze the product for VTO enhancement and suggest:
            
            VTO CATEGORY:
            Determine the best VTO category (UPPER_BODY, LOWER_BODY, FULL_BODY, ACCESSORY, etc.)
            
            MASK STRATEGY:
            Recommend garment-based or prompt-based masking approach
            
            PLACEMENT TIPS:
            Suggest optimal placement and sizing considerations
            
            ENHANCEMENT TYPE:
            Recommend "clothing", "placement", or "auto" enhancement type
            
            Format as structured sections.
            """
        
        else:  # comprehensive
            return base_context + """
            TASK: Provide comprehensive analysis for DPA optimization.
            
            Analyze this product image and provide:
            
            PRODUCT ANALYSIS:
            - Product type and characteristics
            - Style and positioning (luxury, casual, professional, etc.)
            - Key visual features
            - Target audience indicators
            
            VTO RECOMMENDATIONS:
            - Best VTO category for this product
            - Recommended mask strategy (garment-based vs prompt-based)
            - Enhancement type (clothing/placement/auto)
            
            MASK PROMPTS:
            - 4-5 specific mask prompts for Virtual Try-On
            
            BACKGROUND SUGGESTIONS:
            - 2-3 background scenes that would complement this product for advertisement
            - Consider target audience and usage scenarios
            - example : 
                1. A modern advertisement of a sleek smartphone placed on a glossy pedestal, 
                spotlights highlighting its slim design and shiny black finish. 
                Minimalist futuristic background with glowing lines and a sense of innovation. 
                2. A lifestyle advertisement of a new smartphone lying on a wooden coffee table, 
                next to a cup of cappuccino and a laptop. 
                Warm natural light from a window, cozy and inviting mood. 
                The focus is on the smartphone screen glowing with a stylish wallpaper.
            
            IMAGE GENERATION TIPS:
            - Optimal image styles for this product category
            - Lighting and composition suggestions
            - Negative prompts to avoid common issues
            
            Format with clear section headers.
            """
    
    def _parse_analysis_results(self, response_text: str, analysis_type: str) -> Dict[str, Any]:
        """Parse Claude's analysis response into structured data."""
        lines = response_text.split('\n')
        results = {}
        
        if analysis_type == "comprehensive":
            current_section = None
            section_content = []
            
            for line in lines:
                line = line.strip()
                
                # Detect section headers
                if any(header in line.upper() for header in [
                    'PRODUCT ANALYSIS', 'VTO RECOMMENDATIONS', 'MASK PROMPTS', 
                    'BACKGROUND SUGGESTIONS', 'IMAGE GENERATION TIPS'
                ]):
                    # Save previous section
                    if current_section and section_content:
                        results[current_section] = self._clean_section_content(section_content)
                    
                    # Start new section
                    current_section = self._normalize_section_name(line)
                    section_content = []
                
                elif line and current_section:
                    # Add content to current section
                    cleaned_line = line.lstrip('•-*').strip()
                    if cleaned_line and not cleaned_line.startswith(('Analyze', 'Consider', 'Format')):
                        section_content.append(cleaned_line)
            
            # Save last section
            if current_section and section_content:
                results[current_section] = self._clean_section_content(section_content)
        
        else:
            # Single type analysis
            suggestions = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('TASK:', 'Analyze', 'Return', 'Consider', 'Format')):
                    cleaned_line = line.lstrip('•-*').strip()
                    if cleaned_line:
                        suggestions.append(cleaned_line)
            
            if analysis_type == "mask":
                results["mask_prompts"] = suggestions
            elif analysis_type == "background":
                results["background_suggestions"] = suggestions
            elif analysis_type == "vto":
                results["vto_recommendations"] = suggestions
        
        return results
    
    def _normalize_section_name(self, header: str) -> str:
        """Normalize section header to consistent key name."""
        header_upper = header.upper()
        if 'PRODUCT ANALYSIS' in header_upper:
            return 'product_analysis'
        elif 'VTO RECOMMENDATIONS' in header_upper:
            return 'vto_recommendations'
        elif 'MASK PROMPTS' in header_upper:
            return 'mask_prompts'
        elif 'BACKGROUND SUGGESTIONS' in header_upper:
            return 'background_suggestions'
        elif 'IMAGE GENERATION' in header_upper:
            return 'image_generation_tips'
        else:
            return header.lower().replace(' ', '_').replace(':', '')
    
    def _clean_section_content(self, content_lines: List[str]) -> List[str]:
        """Clean and filter section content."""
        cleaned = []
        for line in content_lines:
            # Remove common instruction phrases
            if not any(phrase in line.lower() for phrase in [
                'suggest', 'recommend', 'consider', 'analyze', 'provide', 'format'
            ]):
                cleaned.append(line)
        return cleaned
    
    def suggest_image_style(self, product_category: str, target_audience: str = "general") -> str:
        """Suggest optimal image style based on product category and audience."""
        
        style_mapping = {
            # Clothing
            "shirt": "lifestyle photography with person wearing product in casual setting",
            "dress": "elegant fashion photography with model in sophisticated environment", 
            "jeans": "casual lifestyle photography in urban or outdoor setting",
            "jacket": "stylish outdoor or urban photography showcasing versatility",
            
            # Accessories
            "watch": "luxury product photography with elegant lifestyle context",
            "jewelry": "sophisticated close-up photography with premium lighting",
            "bag": "lifestyle photography showing practical usage in daily scenarios",
            "shoes": "dynamic lifestyle photography showing style and comfort",
            
            # Electronics
            "phone": "modern tech lifestyle photography in contemporary setting",
            "laptop": "professional workspace photography showing productivity",
            "headphones": "lifestyle photography showing usage in various scenarios",
            "tablet": "versatile usage photography in home or work environment",
            
            # Default
            "default": "professional product photography with lifestyle context"
        }
        
        base_style = style_mapping.get(product_category.lower(), style_mapping["default"])
        
        # Adjust for target audience
        if target_audience.lower() in ["luxury", "premium"]:
            base_style = base_style.replace("photography", "luxury photography with premium lighting")
        elif target_audience.lower() in ["young", "trendy", "millennial"]:
            base_style = base_style.replace("photography", "trendy social media style photography")
        elif target_audience.lower() in ["professional", "business"]:
            base_style = base_style.replace("photography", "professional corporate photography")
        
        return base_style
    
    def suggest_negative_prompts(self, product_category: str) -> str:
        """Suggest category-specific negative prompts."""
        
        base_negative = "blurry, low quality, distorted, watermark, text overlay, poor lighting"
        
        category_specific = {
            # Clothing
            "shirt": ", wrinkled, ill-fitting, stained, faded colors",
            "dress": ", wrinkled, poor fit, cheap fabric, unflattering",
            "jeans": ", faded, torn inappropriately, poor fit, cheap denim",
            
            # Accessories  
            "watch": ", scratched, cheap plastic, poor craftsmanship",
            "jewelry": ", tarnished, fake gems, poor quality metal",
            "bag": ", worn out, cheap material, poor stitching",
            
            # Electronics
            "phone": ", cracked screen, outdated design, poor build quality",
            "laptop": ", thick bezels, poor screen quality, outdated design",
            "headphones": ", cheap plastic, poor build, tangled wires",
            
            # Default
            "default": ", cheap appearance, poor craftsmanship"
        }
        
        specific_negative = category_specific.get(product_category.lower(), category_specific["default"])
        return base_negative + specific_negative
    
    def score_image_from_path_or_url(self, image_source: str, product_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Score an image from either local path or S3 URL.
        
        Args:
            image_source: Local file path or S3 URL (s3://bucket/key) or HTTPS URL
            product_info: Optional product information
            
        Returns:
            Comprehensive scoring results from both models
        """
        try:
            # Validate input
            if not image_source or image_source == 'None' or not isinstance(image_source, str):
                logger.error(f"Invalid image source provided: {image_source}")
                return {
                    "success": False,
                    "error": f"Invalid image source: {image_source}",
                    "source": image_source
                }
            
            from PIL import Image
            from io import BytesIO
            
            # Handle different types of image sources
            if image_source.startswith('s3://'):
                # S3 URL - generate presigned URL and download
                logger.info(f"Processing S3 URL: {image_source}")
                
                try:
                    import boto3
                    s3_client = boto3.client('s3')
                    
                    # Parse S3 URL
                    s3_parts = image_source.replace('s3://', '').split('/', 1)
                    if len(s3_parts) != 2:
                        return {
                            "success": False,
                            "error": f"Invalid S3 URL format: {image_source}",
                            "source": image_source
                        }
                    
                    bucket_name, s3_key = s3_parts
                    
                    # Generate presigned URL (valid for 1 hour)
                    presigned_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': bucket_name, 'Key': s3_key},
                        ExpiresIn=3600
                    )
                    
                    # Download from presigned URL
                    import requests
                    response = requests.get(presigned_url, timeout=30)
                    response.raise_for_status()
                    
                    image = Image.open(BytesIO(response.content))
                    logger.info(f"Successfully loaded image from S3: {image.size}")
                    
                except Exception as e:
                    logger.error(f"Failed to load from S3 URL: {e}")
                    return {
                        "success": False,
                        "error": f"Failed to load from S3: {str(e)}",
                        "source": image_source
                    }
            
            elif image_source.startswith(('http://', 'https://')):
                # HTTP/HTTPS URL - download directly
                logger.info(f"Processing HTTP URL: {image_source}")
                
                try:
                    import requests
                    response = requests.get(image_source, timeout=30)
                    response.raise_for_status()
                    
                    image = Image.open(BytesIO(response.content))
                    logger.info(f"Successfully loaded image from URL: {image.size}")
                    
                except Exception as e:
                    logger.error(f"Failed to load from URL: {e}")
                    return {
                        "success": False,
                        "error": f"Failed to load from URL: {str(e)}",
                        "source": image_source
                    }
            
            else:
                # Assume it's a local file path
                logger.info(f"Processing local file: {image_source}")
                
                try:
                    if not os.path.exists(image_source):
                        return {
                            "success": False,
                            "error": f"Local file not found: {image_source}",
                            "source": image_source
                        }
                    
                    image = Image.open(image_source)
                    logger.info(f"Successfully loaded local image: {image.size}")
                    
                except Exception as e:
                    logger.error(f"Failed to load local file: {e}")
                    return {
                        "success": False,
                        "error": f"Failed to load local file: {str(e)}",
                        "source": image_source
                    }
            
            # Use the existing scoring system
            scorer = AdImageScorer()
            scoring_results = scorer.score_advertisement_image(image, product_info)
            
            # Add metadata about the source
            scoring_results['source'] = image_source
            scoring_results['image_size'] = image.size
            
            return scoring_results
            
        except Exception as e:
            logger.error(f"Error scoring image from {image_source}: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": image_source
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image from URL {s3_url}: {e}")
            return {
                "success": False,
                "error": f"Failed to download image: {str(e)}",
                "s3_url": s3_url
            }
        except Exception as e:
            logger.error(f"Error scoring image from URL {s3_url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "s3_url": s3_url
            }

# Integration functions for DPA

def analyze_product_for_dpa(
    product_image,
    product_data: Dict[str, Any],
    analysis_type: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Analyze product image for DPA optimization.
    
    Args:
        product_image: Product image (PIL Image, path, or uploaded file)
        product_data: Product information (name, category, etc.)
        analysis_type: Type of analysis to perform
        
    Returns:
        Analysis results with suggestions
    """
    analyzer = DPAImageAnalyzer()
    
    return analyzer.analyze_product_image(
        image_input=product_image,
        product_category=product_data.get('category', 'unknown'),
        product_name=product_data.get('name', ''),
        analysis_type=analysis_type
    )

def get_smart_template_suggestions(product_data: Dict[str, Any]) -> Dict[str, str]:
    """Get smart template suggestions based on product data."""
    analyzer = DPAImageAnalyzer()
    
    category = product_data.get('category', 'default')
    
    return {
        "image_style": analyzer.suggest_image_style(category),
        "negative_prompts": analyzer.suggest_negative_prompts(category),
        "vto_category": _map_to_vto_category(category)
    }

def _map_to_vto_category(product_category: str) -> str:
    """Map product category to VTO category."""
    mapping = {
        "shirt": "UPPER_BODY",
        "t-shirt": "UPPER_BODY",
        "blouse": "UPPER_BODY", 
        "jacket": "OUTERWEAR",
        "pants": "LOWER_BODY",
        "jeans": "LOWER_BODY",
        "dress": "DRESS",
        "watch": "ACCESSORY",
        "jewelry": "ACCESSORY",
        "bag": "ACCESSORY",
        "shoes": "FOOTWEAR",
        "phone": "ELECTRONICS",
        "laptop": "ELECTRONICS"
    }
    
    return mapping.get(product_category.lower(), "UPPER_BODY")


class AdImageScorer:
    """Score generated advertisement images using multiple AI models"""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize the ad image scorer with AWS Bedrock client."""
        self.bedrock = boto3.client(service_name="bedrock-runtime", region_name=region_name)
        self.claude_model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        self.nova_model_id = "us.amazon.nova-pro-v1:0"
    
    def _image_to_base64(self, image_input) -> str:
        """Convert various image inputs to base64 string."""
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
            # It's an uploaded file or bytes
            if hasattr(image_input, 'getvalue'):
                image_bytes = image_input.getvalue()
            else:
                image_bytes = image_input
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _create_scoring_prompt(self, product_info: Dict[str, Any] = None) -> str:
        """Create prompt for scoring advertisement images."""
        
        product_context = ""
        if product_info:
            product_context = f"""
Product Context:
- Name: {product_info.get('name', 'Unknown')}
- Category: {product_info.get('category', 'Unknown')}
- Description: {product_info.get('description', 'N/A')}
"""
        
        return f"""
You are an expert advertising and visual marketing analyst. Please evaluate this advertisement image across multiple dimensions and provide a comprehensive score.

{product_context}

Please analyze the image and provide scores (1-10 scale) for the following criteria:

**VISUAL QUALITY (1-10)**
- Image clarity and resolution
- Color balance and saturation
- Lighting and composition
- Overall visual appeal

**PRODUCT PRESENTATION (1-10)**
- Product visibility and prominence
- Product positioning and angle
- Product integration with background
- Brand/product recognition clarity

**ADVERTISING EFFECTIVENESS (1-10)**
- Eye-catching and attention-grabbing
- Emotional appeal and engagement
- Target audience appropriateness
- Call-to-action potential

**TECHNICAL EXECUTION (1-10)**
- Background integration quality
- Realistic lighting and shadows
- Absence of artifacts or distortions
- Professional finish quality

**IMAGE HARMONY (1-10)**
- Visual balance and composition harmony
- Color scheme coherence and consistency
- Element proportions and spatial relationships
- Overall aesthetic unity and flow

**IMAGE RATIONALITY (1-10)**
- Logical scene composition and context
- Absence of unrealistic or impossible elements
- Appropriate object placement and physics
- No unwanted or out-of-place artifacts

**OVERALL COMMERCIAL APPEAL (1-10)**
- Would this drive purchase intent?
- Suitable for e-commerce platforms?
- Competitive advantage potential
- Brand image enhancement

Please respond in the following JSON format:
{{
    "visual_quality": {{
        "score": X,
        "feedback": "Detailed explanation..."
    }},
    "product_presentation": {{
        "score": X,
        "feedback": "Detailed explanation..."
    }},
    "advertising_effectiveness": {{
        "score": X,
        "feedback": "Detailed explanation..."
    }},
    "technical_execution": {{
        "score": X,
        "feedback": "Detailed explanation..."
    }},
    "image_harmony": {{
        "score": X,
        "feedback": "Detailed explanation..."
    }},
    "image_rationality": {{
        "score": X,
        "feedback": "Detailed explanation..."
    }},
    "overall_commercial_appeal": {{
        "score": X,
        "feedback": "Detailed explanation..."
    }},
    "total_score": X,
    "strengths": ["strength1", "strength2", "strength3"],
    "improvements": ["improvement1", "improvement2", "improvement3"],
    "summary": "Overall assessment in 2-3 sentences"
}}

Provide honest, constructive feedback that would help improve future ad generation.
"""
    
    def score_with_claude(self, image_input, product_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Score advertisement image using Claude Sonnet 4."""
        try:
            image_b64 = self._image_to_base64(image_input)
            prompt = self._create_scoring_prompt(product_info)
            
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
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
            
            response = self.bedrock.invoke_model(
                body=json.dumps(request_body),
                modelId=self.claude_model_id,
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get("body").read())
            raw_response = response_body.get("content", [{}])[0].get("text", "")
            
            # Parse JSON response
            try:
                # Extract JSON from response (handle potential markdown formatting)
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = raw_response[json_start:json_end]
                    parsed_score = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse Claude JSON response: {e}")
                # Fallback: create basic structure
                parsed_score = {
                    "visual_quality": {"score": 7, "feedback": "Unable to parse detailed feedback"},
                    "product_presentation": {"score": 7, "feedback": "Unable to parse detailed feedback"},
                    "advertising_effectiveness": {"score": 7, "feedback": "Unable to parse detailed feedback"},
                    "technical_execution": {"score": 7, "feedback": "Unable to parse detailed feedback"},
                    "overall_commercial_appeal": {"score": 7, "feedback": "Unable to parse detailed feedback"},
                    "total_score": 35,
                    "strengths": ["Analysis available in raw response"],
                    "improvements": ["Check raw response for details"],
                    "summary": "Detailed analysis available in raw response"
                }
            
            return {
                "success": True,
                "model": "Claude Sonnet 4",
                "model_id": self.claude_model_id,
                "scores": parsed_score,
                "raw_response": raw_response
            }
            
        except Exception as e:
            logger.error(f"Error scoring with Claude: {e}")
            return {
                "success": False,
                "model": "Claude Sonnet 4",
                "error": str(e)
            }
    
    def score_with_nova(self, image_input, product_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Score advertisement image using Amazon Nova Pro."""
        try:
            image_b64 = self._image_to_base64(image_input)
            prompt = self._create_scoring_prompt(product_info)
            
            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {
                                        "bytes": image_b64
                                    }
                                }
                            },
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 2000,
                    "temperature": 0.1
                }
            }
            
            response = self.bedrock.invoke_model(
                body=json.dumps(request_body),
                modelId=self.nova_model_id,
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get("body").read())
            raw_response = response_body.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
            
            # Parse JSON response
            try:
                # Extract JSON from response (handle potential markdown formatting)
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = raw_response[json_start:json_end]
                    parsed_score = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse Nova JSON response: {e}")
                # Fallback: create basic structure
                parsed_score = {
                    "visual_quality": {"score": 7, "feedback": "Unable to parse detailed feedback"},
                    "product_presentation": {"score": 7, "feedback": "Unable to parse detailed feedback"},
                    "advertising_effectiveness": {"score": 7, "feedback": "Unable to parse detailed feedback"},
                    "technical_execution": {"score": 7, "feedback": "Unable to parse detailed feedback"},
                    "overall_commercial_appeal": {"score": 7, "feedback": "Unable to parse detailed feedback"},
                    "total_score": 35,
                    "strengths": ["Analysis available in raw response"],
                    "improvements": ["Check raw response for details"],
                    "summary": "Detailed analysis available in raw response"
                }
            
            return {
                "success": True,
                "model": "Amazon Nova Pro",
                "model_id": self.nova_model_id,
                "scores": parsed_score,
                "raw_response": raw_response
            }
            
        except Exception as e:
            logger.error(f"Error scoring with Nova: {e}")
            return {
                "success": False,
                "model": "Amazon Nova Pro",
                "error": str(e)
            }
    
    def score_advertisement_image(self, image_input, product_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Score advertisement image using both Claude and Nova models."""
        logger.info("🎯 Starting advertisement image scoring with dual models...")
        
        results = {
            "timestamp": "2025-08-26T14:23:06.523Z",
            "product_info": product_info or {},
            "claude_results": {},
            "nova_results": {},
            "comparison": {}
        }
        
        # Score with Claude Sonnet 4
        logger.info("📊 Scoring with Claude Sonnet 4...")
        claude_results = self.score_with_claude(image_input, product_info)
        results["claude_results"] = claude_results
        
        # Score with Nova Pro
        logger.info("📊 Scoring with Amazon Nova Pro...")
        nova_results = self.score_with_nova(image_input, product_info)
        results["nova_results"] = nova_results
        
        # Create comparison if both succeeded
        if claude_results.get("success") and nova_results.get("success"):
            results["comparison"] = self._compare_scores(claude_results, nova_results)
        
        logger.info("✅ Advertisement image scoring completed")
        return results
    
    def _compare_scores(self, claude_results: Dict, nova_results: Dict) -> Dict[str, Any]:
        """Compare scores from both models."""
        claude_scores = claude_results.get("scores", {})
        nova_scores = nova_results.get("scores", {})
        
        comparison = {
            "score_differences": {},
            "agreement_level": "unknown",
            "consensus_strengths": [],
            "consensus_improvements": [],
            "model_preferences": {}
        }
        
        # Compare individual scores
        score_categories = ["visual_quality", "product_presentation", "advertising_effectiveness", 
                          "technical_execution", "image_harmony", "image_rationality",
                          "overall_commercial_appeal"]
        
        total_diff = 0
        valid_comparisons = 0
        
        for category in score_categories:
            claude_score = claude_scores.get(category, {}).get("score", 0)
            nova_score = nova_scores.get(category, {}).get("score", 0)
            
            if claude_score and nova_score:
                diff = abs(claude_score - nova_score)
                comparison["score_differences"][category] = {
                    "claude": claude_score,
                    "nova": nova_score,
                    "difference": diff,
                    "higher_scorer": "Claude" if claude_score > nova_score else "Nova" if nova_score > claude_score else "Tie"
                }
                total_diff += diff
                valid_comparisons += 1
        
        # Calculate agreement level
        if valid_comparisons > 0:
            avg_diff = total_diff / valid_comparisons
            if avg_diff <= 1:
                comparison["agreement_level"] = "High Agreement"
            elif avg_diff <= 2:
                comparison["agreement_level"] = "Moderate Agreement"
            else:
                comparison["agreement_level"] = "Low Agreement"
        
        # Find consensus strengths and improvements
        claude_strengths = set(claude_scores.get("strengths", []))
        nova_strengths = set(nova_scores.get("strengths", []))
        comparison["consensus_strengths"] = list(claude_strengths.intersection(nova_strengths))
        
        claude_improvements = set(claude_scores.get("improvements", []))
        nova_improvements = set(nova_scores.get("improvements", []))
        comparison["consensus_improvements"] = list(claude_improvements.intersection(nova_improvements))
        
        # Calculate overall totals by summing individual category scores
        def calculate_total_score(scores_dict):
            """Calculate total score by summing individual category scores"""
            categories = ["visual_quality", "product_presentation", "advertising_effectiveness", 
                         "technical_execution", "image_harmony", "image_rationality", 
                         "overall_commercial_appeal"]
            total = 0
            for category in categories:
                if category in scores_dict:
                    score = scores_dict[category].get("score", 0)
                    total += score
            return total
        
        claude_total = calculate_total_score(claude_scores)
        nova_total = calculate_total_score(nova_scores)
        
        comparison["total_scores"] = {
            "claude": claude_total,
            "nova": nova_total,
            "average": (claude_total + nova_total) / 2 if claude_total and nova_total else 0,
            "difference": abs(claude_total - nova_total) if claude_total and nova_total else 0
        }
        
        return comparison


def score_advertisement_image(image_input, product_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Score an advertisement image using both Claude Sonnet 4 and Amazon Nova Pro.
    
    Args:
        image_input: Advertisement image (PIL Image, path, or uploaded file)
        product_info: Product information (name, category, description, etc.)
        
    Returns:
        Comprehensive scoring results from both models with comparison
    """
    scorer = AdImageScorer()
    return scorer.score_advertisement_image(image_input, product_info)
