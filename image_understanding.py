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
