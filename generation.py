import boto3
from botocore.config import Config
import base64
import time
import os
import json
import string
import random
import re
from PIL import Image
import io
import sys
import concurrent.futures
from functools import wraps

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    SYSTEM_TEXT_ONLY,
    SYSTEM_IMAGE_TEXT,
    SYSTEM_CANVAS,
    MODEL_OPTIONS,
    DEFAULT_BUCKET,
    DEFAULT_GUIDELINE,
    GENERATED_VIDEOS_DIR,
    REEL_MODEL_ID,
    CANVAS_SIZE,
    PRO_MODEL_ID,
    LITE_MODEL_ID)
from utils import (
    random_string_name,
    load_guideline,
    parse_prompt,
    process_image,
    download_video
)

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

def get_bedrock_client(region_name='us-east-1'):
    """Get bedrock client directly with boto3."""
    return boto3.client('bedrock-runtime', region_name=region_name)

# Legacy compatibility
client = get_bedrock_client()
bedrock_runtime = get_bedrock_client()

# Constants
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MAX_PROMPT_LENGTH = 512

def optimize_prompt(prompt, guideline_path, model_id=PRO_MODEL_ID, image=None):
    if image is not None:
        image = process_image(image)
    
    doc_bytes = load_guideline(guideline_path)
    
    if image is None:
        # Text-only prompt optimization
        system = [{"text": SYSTEM_TEXT_ONLY}]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "document": {
                            "format": "pdf",
                            "name": "DocumentPDFmessages",
                            "source": {"bytes": doc_bytes}
                        }
                    },
                    {"text": f"Please optimize: {prompt}"},
                ],
            }
        ]
    else:
        # Image + text prompt optimization
        system = [{"text": SYSTEM_IMAGE_TEXT}]
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "document": {
                            "format": "pdf",
                            "name": "DocumentPDFmessages",
                            "source": {"bytes": doc_bytes}
                        }
                    },
                    {"image": {"format": "png", "source": {"bytes": img_bytes}}},
                    {"text": f"Please optimize: {prompt}"},
                ],
            }
        ]

    # Configure inference parameters
    inf_params = {"maxTokens": 512, "topP": 0.9, "temperature": 0.8}
    
    # Get response
    response = client.converse_stream(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inf_params
    )
    
    # Collect response text
    text = ""
    stream = response.get("stream")
    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                text += event["contentBlockDelta"]["delta"]["text"]
    
    optimized = parse_prompt(text)
    length = len(optimized)
    return optimized, f"{length} chars" + (" (Too Long!)" if length > MAX_PROMPT_LENGTH else "")

def optimize_canvas_prompt(prompt, model_id=LITE_MODEL_ID):
    """Optimize prompt for image generation using Canvas model"""
    system = [{"text": SYSTEM_CANVAS}]
    messages = [
        {
            "role": "user",
            "content": [{"text": f"Please optimize: {prompt}"}],
        }
    ]
    
    # Configure inference parameters
    inf_params = {"maxTokens": 512, "topP": 0.9, "temperature": 0.8}
    
    # Get response
    response = client.converse_stream(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inf_params
    )
    
    # Collect response text
    text = ""
    stream = response.get("stream")
    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                text += event["contentBlockDelta"]["delta"]["text"]
    
    # Extract both prompt and negative prompt
    prompt = parse_prompt(text, r"<prompt>(.*?)</prompt>")
    try:
        negative_prompt = parse_prompt(text, r"<negative_prompt>(.*?)</negative_prompt>")
    except ValueError:
        negative_prompt = ""
    
    return prompt, negative_prompt

def generate_image_pair(original_prompt, optimized_prompt, negative_prompt="", quality="standard", num_images=1, height=720, width=1280, seed=0, cfg_scale=6.5):
    """Generate images sequentially to avoid connection issues"""
    # Generate images sequentially instead of in parallel to avoid connection limits
    print("Generating original image...")
    original_images = generate_single_image(
        original_prompt,
        negative_prompt,
        quality,
        num_images,
        height,
        width,
        seed,
        cfg_scale
    )
    
    # Small delay between requests
    time.sleep(0.5)
    
    print("Generating optimized image...")
    optimized_images = generate_single_image(
        optimized_prompt,
        negative_prompt,
        quality,
        num_images,
        height,
        width,
        seed,
        cfg_scale
    )
    
    return original_images, optimized_images

@retry_with_backoff(max_retries=3, base_delay=1.0)
def generate_single_image(prompt, negative_prompt="", quality="standard", num_images=1, height=720, width=1280, seed=0, cfg_scale=6.5):
    """Generate image using Nova Canvas model with retry logic"""
    # Prepare text-to-image parameters
    text_params = {"text": prompt}
    if negative_prompt and negative_prompt.strip():  # Only add if not empty
        text_params["negativeText"] = negative_prompt
    
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": text_params,
        "imageGenerationConfig": {
            "numberOfImages": int(num_images),  # Ensure integer
            "height": int(height),
            "width": int(width),
            "cfgScale": float(cfg_scale),
            "seed":  random.randint(0,858993459) if int(seed) == -1 else int(seed),
            "quality": quality
        }
    })
    
    try:
        # Use connection manager
        bedrock_client = get_bedrock_client()
        
        response = bedrock_client.invoke_model(
            body=body,
            modelId='amazon.nova-canvas-v1:0',
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        
        # Create temporary directory for saving images
        local_dir = './generated_images'
        os.makedirs(local_dir, exist_ok=True)
        image_paths = []
        
        # Save each image to a temporary file and collect paths
        for i, base64_image in enumerate(response_body.get("images", [])):
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes))
            rand_name = random_string_name()
            path = f"{local_dir}/generated_{rand_name}.png"
            image.save(path)
            image_paths.append(path)
        
        # Return single path if only one image requested, otherwise return list
        if num_images == 1 and image_paths:
            return image_paths[0]
        else:
            return image_paths if image_paths else None
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None
        
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None

def generate_video(prompt, bucket, image=None, seed=0):
    print(f"Starting video generation with prompt: {prompt[:50]}...")
    print(f"Bucket: {bucket}")
    print(f"Image provided: {image is not None}")
    print(f"Image type: {type(image) if image is not None else 'None'}")
    print(f"Seed: {seed}")
    
    if image is not None:
        print(f"Original image size: {image.size}")
        image = process_image(image)
        print(f"Image processed successfully: {image.size}")
        print(f"Image mode: {image.mode}")
    
    # Prepare model input
    model_input = {
        "taskType": "TEXT_VIDEO",
        "textToVideoParams": {
            "text": prompt,
        },
        "videoGenerationConfig": {
            "durationSeconds": 6,
            "fps": 24,
            "dimension": "1280x720",
            "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed),
        },
    }
    
    # Add image if provided
    if image is not None:
        print("Adding image to model input...")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        print(f"Image encoded to base64, length: {len(img_base64)}")
        
        model_input['textToVideoParams']['images'] = [{
            "format": "png",
            "source": {
                "bytes": img_base64
            }
        }]
        print("Image successfully added to model input")
    else:
        print("No image provided - generating text-only video")
    print(bucket)
    # Start async video generation
    try:
        print("Starting async video generation...")
        print(f"Using model ID: {REEL_MODEL_ID}")
        print(f"Model input keys: {list(model_input.keys())}")
        
        # Clean bucket name - remove s3:// prefix if present
        clean_bucket = bucket.replace("s3://", "") if bucket.startswith("s3://") else bucket
        print(f"Using clean bucket name: {clean_bucket}")
        
        invocation = bedrock_runtime.start_async_invoke(
            modelId=REEL_MODEL_ID,
            modelInput=model_input,
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": f"s3://{clean_bucket}"
                }
            }
        )
        print(f"Invocation started successfully: {invocation['invocationArn']}")
    except Exception as e:
        print(f"Error starting async invocation: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Wait for completion
    print("Waiting for video generation to complete...")
    while True:
        try:
            response = bedrock_runtime.get_async_invoke(
                invocationArn=invocation['invocationArn']
            )
            status = response["status"]
            print(f"Video generation status: {status}")
            if status != 'InProgress':
                break
            time.sleep(10)
        except Exception as e:
            print(f"Error checking async invocation status: {str(e)}")
            return None
    
    if status != 'Completed':
        print(f"Video generation failed with status: {status}")
        if 'failureMessage' in response:
            print(f"Failure message: {response['failureMessage']}")
        return None
    
    # Download video
    try:
        output_uri = f"{response['outputDataConfig']['s3OutputDataConfig']['s3Uri']}/output.mp4"
        print(f"Downloading video from: {output_uri}")
        local_path = download_video(output_uri,GENERATED_VIDEOS_DIR)
        print(f"Video downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None
    
    return local_path


def generate_video_with_model(prompt, bucket, model_id, image=None, seed=0, duration="5s"):
    """
    Generate video using specified model (Nova Reel or Luma Ray)
    """
    print(f"Starting video generation with model: {model_id}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Bucket: {bucket}")
    print(f"Image provided: {image is not None}")
    print(f"Image type: {type(image) if image is not None else 'None'}")
    print(f"Seed: {seed}")
    print(f"Duration: {duration}")
    
    # Get the appropriate region for the model
    from config import MODEL_REGIONS
    region = MODEL_REGIONS.get(model_id, "us-east-1")
    print(f"Using region: {region}")
    
    # Create region-specific bedrock client with default credentials
    session_regional = boto3.Session(region_name=region)
    bedrock_runtime_regional = session_regional.client(
        'bedrock-runtime',
        config=Config(
            region_name=region,
            retries={'max_attempts': 10, 'mode': 'adaptive'},
            max_pool_connections=50
        )
    )
    
    if image is not None:
        print(f"Original image size: {image.size}")
        image = process_image(image)
        print(f"Image processed successfully: {image.size}")
        print(f"Image mode: {image.mode}")
    
    # Prepare model input based on model type
    if model_id.startswith("luma.ray"):
        # Luma Ray format - based on official AWS documentation
        model_input = {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "duration": duration,  # "5s" or "9s"
            "resolution": "720p",
            "loop": False
        }
        
        # For Luma Ray image-to-video, use keyframes structure
        if image is not None:
            print("Adding image to Luma Ray model input...")
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')  # Use JPEG for Luma Ray
            img_bytes = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            print(f"Image encoded to base64, length: {len(img_base64)}")
            
            # Luma Ray uses 'keyframes' with 'frame0' for image-to-video
            model_input['keyframes'] = {
                "frame0": {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_base64
                    }
                }
            }
            print("Image successfully added to Luma Ray model input as keyframe")
        else:
            print("Generating text-only video with Luma Ray")
            
    else:
        # Nova Reel format
        model_input = {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {
                "text": prompt,
            },
            "videoGenerationConfig": {
                "durationSeconds": 6,
                "fps": 24,
                "dimension": "1280x720",
                "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed),
            },
        }
        
        # Add image if provided (for Nova Reel image-to-video)
        if image is not None:
            print("Adding image to Nova Reel model input...")
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            print(f"Image encoded to base64, length: {len(img_base64)}")
            
            model_input['textToVideoParams']['images'] = [{
                "format": "png",
                "source": {
                    "bytes": img_base64
                }
            }]
            print("Image successfully added to Nova Reel model input")
        else:
            print("No image provided - generating text-only video with Nova Reel")
    
    print(bucket)
    # Start async video generation
    try:
        print("Starting async video generation...")
        print(f"Using model ID: {model_id}")
        print(f"Model input keys: {list(model_input.keys())}")
        
        # Clean bucket name - remove s3:// prefix if present
        clean_bucket = bucket.replace("s3://", "") if bucket.startswith("s3://") else bucket
        print(f"Using clean bucket name: {clean_bucket}")
        
        invocation = bedrock_runtime_regional.start_async_invoke(
            modelId=model_id,
            modelInput=model_input,
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": f"s3://{clean_bucket}"
                }
            }
        )
        print(f"Invocation started successfully: {invocation['invocationArn']}")
    except Exception as e:
        print(f"Error starting async invocation: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Wait for completion
    print("Waiting for video generation to complete...")
    while True:
        try:
            response = bedrock_runtime_regional.get_async_invoke(
                invocationArn=invocation['invocationArn']
            )
            status = response["status"]
            print(f"Video generation status: {status}")
            if status != 'InProgress':
                break
            time.sleep(10)
        except Exception as e:
            print(f"Error checking async invocation status: {str(e)}")
            return None
    
    if status != 'Completed':
        print(f"Video generation failed with status: {status}")
        if 'failureMessage' in response:
            print(f"Failure message: {response['failureMessage']}")
        return None
    
    # Download video
    try:
        output_uri = f"{response['outputDataConfig']['s3OutputDataConfig']['s3Uri']}/output.mp4"
        print(f"Downloading video from: {output_uri}")
        local_path = download_video(output_uri,GENERATED_VIDEOS_DIR)
        print(f"Video downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None
    
    return local_path



def generate_comparison_videos_with_model(original_prompt, optimized_prompt, bucket, model_id, image=None, seed=0, duration="5s"):
    # Create separate copies of the image for thread safety
    image_original = None
    image_optimized = None
    
    if image is not None:
        print("Creating thread-safe image copies for comparison videos...")
        try:
            # Create a copy for the original video
            image_original = image.copy()
            # Create a copy for the optimized video  
            image_optimized = image.copy()
            print(f"Image copies created successfully: {image_original.size}, {image_optimized.size}")
        except Exception as e:
            print(f"Error creating image copies: {str(e)}")
            # Fallback: create fresh images from the original
            import io
            import base64
            try:
                # Convert to bytes and back to create fresh copies
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                
                image_original = Image.open(io.BytesIO(img_bytes))
                image_optimized = Image.open(io.BytesIO(img_bytes))
                print(f"Fresh image copies created from bytes: {image_original.size}, {image_optimized.size}")
            except Exception as e2:
                print(f"Error creating fresh image copies: {str(e2)}")
                # If all else fails, generate without images
                image_original = None
                image_optimized = None
    
    # Create a thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both video generation tasks with separate image copies
        future_original = executor.submit(generate_video_with_model, original_prompt, bucket, model_id, image_original, seed, duration)
        future_optimized = executor.submit(generate_video_with_model, optimized_prompt, bucket, model_id, image_optimized, seed, duration)
        
        # Wait for both tasks to complete
        original_video = future_original.result()
        optimized_video = future_optimized.result()
    
    return original_video, optimized_video


# Keep the original function name for backward compatibility
def generate_comparison_videos(original_prompt, optimized_prompt, bucket, image=None, seed=0):
    """
    Backward compatibility wrapper - uses Nova Reel by default
    """
    from config import REEL_MODEL_ID
    return generate_comparison_videos_with_model(original_prompt, optimized_prompt, bucket, REEL_MODEL_ID, image, seed)


# Keep backward compatibility for single video generation too
def generate_video(prompt, bucket, image=None, seed=0):
    """
    Backward compatibility wrapper for the original generate_video function - uses Nova Reel by default
    """
    from config import REEL_MODEL_ID
    return generate_video_with_model(prompt, bucket, REEL_MODEL_ID, image, seed)