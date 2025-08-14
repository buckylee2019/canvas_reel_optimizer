import boto3
import json
import base64
import time
import os
import io
import re
import magic
import random
import string
from datetime import datetime
from PIL import Image
from io import BytesIO
from botocore.config import Config
from moviepy import VideoFileClip, CompositeVideoClip, TextClip
from json import JSONDecodeError
import sys
import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SHOT_SYSTEM,SYSTEM_TEXT_ONLY,SYSTEM_IMAGE_TEXT,DEFAULT_GUIDELINE,LITE_MODEL_ID,CONTINUOUS_SHOT_SYSTEM
from utils import random_string_name
from generation import optimize_canvas_prompt



class ReelGenerator:
    def __init__(self,model_id = LITE_MODEL_ID,region='us-east-1', bucket_name='s3://bedrock-video-generation-us-east-1-jlvyiv'):
        """Initialize ReelGenerator with AWS credentials and configuration."""
        config = Config(
            connect_timeout=1000,
            read_timeout=1000,
        )
        if not bucket_name.startswith('s3://'):
            raise ValueError("Invalid S3 URI format")
        path_parts = bucket_name[5:].split('/', 1)
        self.s3_bucket =  path_parts[0]
        self.session = boto3.session.Session(region_name=region)
        self.bedrock_runtime = self.session.client(service_name='bedrock-runtime')
        self.MODEL_ID =model_id
        print(f"Bucket Name: {self.s3_bucket}, region:{region}")

    def _parse_json(self, pattern: str, text: str) -> str:
        """Parse text using regex pattern."""
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
            return text.strip()
        else:
            raise JSONDecodeError("No match found", text, 0)
    
    def _split_caption(self,text: str) -> list:
        """Split caption text into parts."""
        delimiters = [',', '，', '。', '.', ';', '；', '\n', '\t']
        pattern = '|'.join(map(re.escape, delimiters))
        return [part for part in re.split(pattern, text) if part]

    def generate_shots(self, story: str, system_prompt: str) -> dict:
        """Generate shot descriptions from a story."""
        with open(DEFAULT_GUIDELINE, "rb") as file:
            doc_bytes = file.read()
        system = [{"text": system_prompt}]
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
                    {"text": f"Please generate shots for User input:{story}"}],
            },
            {
                "role": "assistant",
                "content": [{"text": "```json"}],
            }
        ]

        response = self.bedrock_runtime.converse_stream(
            modelId=self.MODEL_ID,
            messages=messages,
            system=system,
            inferenceConfig={"maxTokens": 2000, "topP": 0.9, "temperature": 0.8}
        )

        text = ""
        stream = response.get("stream")
        if stream:
            for event in stream:
                if "contentBlockDelta" in event:
                    text += event["contentBlockDelta"]["delta"]["text"]
        
        return json.loads(text[:-3])

    def generate_image(self, body: dict) -> list:
        """Generate images using Amazon Nova Canvas model."""
        accept = "application/json"
        content_type = "application/json"

        response = self.bedrock_runtime.invoke_model(
            body=json.dumps(body),
            modelId='amazon.nova-canvas-v1:0',
            accept=accept,
            contentType=content_type
        )
        
        response_body = json.loads(response.get("body").read())
        image_bytes_list = []
        
        if "images" in response_body:
            for base64_image in response_body["images"]:
                base64_bytes = base64_image.encode('ascii')
                image_bytes = base64.b64decode(base64_bytes)
                image_bytes_list.append(image_bytes)

        return image_bytes_list

    def generate_variations(self,reference_image_paths,prompt,negative_prompt,save_filepath,seed:int = 0,cfg_scale:float = 6.5,similarity_strength:float = 0.8):
        # Load all reference images as base64.
        images = []
        for path in reference_image_paths:
            with open(path, "rb") as image_file:
                images.append(base64.b64encode(image_file.read()).decode("utf-8"))

        # Configure the inference parameters.
        inference_params = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "images": images, # Images to use as reference
                "text": prompt, 
                "similarityStrength": similarity_strength,  # Range: 0.2 to 1.0
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,  # Number of variations to generate. 1 to 5.
                "quality": "standard",  # Allowed values are "standard" and "premium"
                "width": 1280,  # See README for supported output resolutions
                "height": 720,  # See README for supported output resolutions
                "cfgScale": cfg_scale,  # How closely the prompt will be followed
                "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed)
            },
        }
        if len(negative_prompt):
            inference_params['imageVariationParams']["negativeText"] = negative_prompt
            
        try:
            image_bytes_ret = self.generate_image( inference_params)
            for idx,image_bytes in enumerate(image_bytes_ret):
                image = Image.open(io.BytesIO(image_bytes))
                image.save(save_filepath)
                print(f"image saved to {save_filepath}")
            return image_bytes_ret
        except Exception as err:
            raise ValueError(f"generate_variations:{str(err)}")

    def generate_text2img(self, prompt: str, negative_prompt: str, save_filepath: str,seed:int =0,cfg_scale:float = 6.5) -> str:
        """Generate image from text prompt."""
        inference_params = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 720,
                "width": 1280,
                "cfgScale": cfg_scale,
                "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed)
            }
        }
        if len(negative_prompt):
            inference_params['textToImageParams']["negativeText"] = negative_prompt

        try:
            image_bytes_list = self.generate_image(inference_params)
            for image_bytes in image_bytes_list:
                image = Image.open(BytesIO(image_bytes))
                image.save(save_filepath)
                print(f"image saved to {save_filepath}")
            return save_filepath
        except Exception as err:
            raise ValueError(f"generate_text2img:{str(err)}")

    def optimize_reel_prompt(self, system_prompt: str, user_prompt: str, ref_image: str, doc_bytes: bytes) -> str:
        """Optimize reel prompt using reference image."""
        with open(ref_image, "rb") as f:
            image = f.read()
        mime_type = magic.Magic(mime=True).from_file(ref_image)

        system = [{"text": system_prompt}]
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
                    {"image": {"format": mime_type.split('/')[1], "source": {"bytes": image}}},
                    {"text": f"Please generate shots for User input:{user_prompt}"},
                ],
            }
        ]

        response = self.bedrock_runtime.converse_stream(
            modelId=self.MODEL_ID,
            messages=messages,
            system=system,
            inferenceConfig={"maxTokens": 2000, "topP": 0.9, "temperature": 0.5}
        )

        text = ""
        stream = response.get("stream")
        if stream:
            for event in stream:
                if "contentBlockDelta" in event:
                    text += event["contentBlockDelta"]["delta"]["text"]
        
        return self._parse_json(r"<prompt>(.*?)</prompt>", text)

    def generate_video(self, text_prompt: str, ref_image: str = None,seed:int = 0) -> dict:
        """Generate video from text prompt and optional reference image."""
        model_input = {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {
                "text": text_prompt,
            },
            "videoGenerationConfig": {
                "durationSeconds": 6,
                "fps": 24,
                "dimension": "1280x720",
                "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed),
            },
        }

        if ref_image:
            with open(ref_image, "rb") as f:
                image = f.read()
                input_image_base64 = base64.b64encode(image).decode("utf-8")
                model_input['textToVideoParams']['images'] = [
                    {
                        "format": magic.Magic(mime=True).from_file(ref_image).split('/')[1],
                        "source": {"bytes": input_image_base64}
                    }
                ]

        try:
            invocation = self.bedrock_runtime.start_async_invoke(
                modelId="amazon.nova-reel-v1:1",
                modelInput=model_input,
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": f"s3://{self.s3_bucket}"
                    }
                }
            )
            return invocation
        except Exception as e:
            print(f"Error generating video: {str(e)}")
            return None

    def fetch_job_status(self, invocation_arns: list) -> list:
        """Fetch status of video generation jobs."""
        final_responses = []
        for invocation in invocation_arns:
            while True:
                response = self.bedrock_runtime.get_async_invoke(invocationArn=invocation)
                status = response["status"]
                if status != 'InProgress':
                    final_responses.append(response)
                    break
                time.sleep(5)
        return final_responses

    def download_video_from_s3(self, s3_uri: str, local_path: str) -> str:
        """Download generated video from S3."""
        try:
            s3_client = self.session.client('s3')
            
            if not s3_uri.startswith('s3://'):
                raise ValueError("Invalid S3 URI format")
            
            path_parts = s3_uri[5:].split('/', 1)
            if len(path_parts) != 2:
                raise ValueError("Invalid S3 URI format")
            
            bucket_name = path_parts[0]
            s3_key = path_parts[1]
            
            os.makedirs(local_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            fname = timestamp + ''.join(random_string_name()) + '.mp4'
            local_file_path = os.path.join(local_path, fname)
            
            s3_client.download_file(bucket_name, s3_key, local_file_path)
            return local_file_path
            
        except Exception as e:
            print(f"Error downloading file: {str(e)}")
            return None

    def stitch_videos(self, video1_path: str, video2_path: str, output_path: str) -> str:
        """Stitch two videos together."""
        print(f"      🔗 Starting individual video stitching...")
        print(f"         Input 1: {video1_path}")
        print(f"         Input 2: {video2_path}")
        print(f"         Output: {output_path}")
        
        try:
            # Validate input files
            if not os.path.exists(video1_path):
                print(f"         ❌ Video 1 does not exist: {video1_path}")
                return None
            if not os.path.exists(video2_path):
                print(f"         ❌ Video 2 does not exist: {video2_path}")
                return None
            
            # Get file sizes
            size1 = os.path.getsize(video1_path)
            size2 = os.path.getsize(video2_path)
            print(f"         📊 Input sizes: {size1:,} bytes + {size2:,} bytes")
            
            # Load video clips
            print(f"         📹 Loading video clips...")
            clip1 = VideoFileClip(video1_path)
            print(f"         ✅ Clip 1 loaded: {clip1.duration:.2f}s, {clip1.size}")
            
            clip2 = VideoFileClip(video2_path)
            print(f"         ✅ Clip 2 loaded: {clip2.duration:.2f}s, {clip2.size}")
            
            # Create composite
            print(f"         🎬 Creating composite video...")
            final_clip = CompositeVideoClip([
                clip1,
                clip2.with_start(clip1.duration),
            ])
            
            total_duration = clip1.duration + clip2.duration
            print(f"         📏 Total duration: {total_duration:.2f}s")
            
            # Write output
            print(f"         💾 Writing output video...")
            final_clip.write_videofile(output_path, verbose=False, logger=None)
            
            # Cleanup
            print(f"         🧹 Cleaning up clips...")
            clip1.close()
            clip2.close()
            final_clip.close()
            
            # Verify output
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                print(f"         ✅ Stitching completed successfully!")
                print(f"         📊 Output size: {output_size:,} bytes")
                return output_path
            else:
                print(f"         ❌ Output file was not created")
                return None
                
        except Exception as e:
            print(f"         ❌ Error during stitching: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def add_timed_captions(self, video_path: str, output_path: str, captions: list, font: str = './yahei.ttf') -> None:
        """Add timed captions to video."""
        video = VideoFileClip(video_path)
        txt_clips = []
        
        for caption in captions:
            text, start_time, end_time = caption
            txt_clip = TextClip(
                text=text,
                font_size=50,
                color='white',
                font=font,
                text_align='center',
                margin=(20, 20)
            )
            txt_clip = txt_clip.with_position('bottom').with_start(start_time).with_end(end_time)
            txt_clips.append(txt_clip)
        
        final_video = CompositeVideoClip([video] + txt_clips)
        final_video.write_videofile(output_path)
        
        video.close()
        final_video.close()



def generate_shots(reel_gen:ReelGenerator,story:str,num_shot:int=3,is_continues_shot = False):
    # Generate shots
    system = CONTINUOUS_SHOT_SYSTEM if is_continues_shot else SHOT_SYSTEM
    shots = reel_gen.generate_shots(story, system.replace("<num_shot>",str(num_shot)))
    return shots

def generate_shot_image(reel_gen:ReelGenerator,shots:dict,timestamp:str, seed:int=0,cfg_scale:float = 6.5, similarity_strength:float = 0.8, is_continues_shot = False):
    # Create directories for outputs
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join('shot_images', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if reference image already exists (uploaded by user)
    reference_shot_path = os.path.join(output_dir, 'shot_0.png')
    has_reference_image = os.path.exists(reference_shot_path)
    
    if has_reference_image:
        # Verify it's actually a reference image (not a placeholder)
        try:
            from PIL import Image
            ref_img = Image.open(reference_shot_path)
            # Check if it's your screenshot size (1374, 912) or similar user upload
            if ref_img.size == (1374, 912) or ref_img.size[0] > 1000:
                print(f"✅ Found valid reference image at {reference_shot_path}: {ref_img.size}")
                print(f"🎬 Will use reference image as first shot")
            else:
                print(f"⚠️  Found image at {reference_shot_path} but size {ref_img.size} seems too small for reference")
                has_reference_image = False
        except Exception as e:
            print(f"❌ Error checking reference image: {str(e)}")
            has_reference_image = False
    
    # Generate images for each shot
    image_files = []
    prompts = []
    for idx, shot in enumerate(shots['shots']):
        save_path = os.path.join(output_dir, f'shot_{idx}.png')
        
        # Skip generating first image if valid reference image exists
        if idx == 0 and has_reference_image:
            print(f"🎬 Using reference image as shot_0.png (skipping AI generation)")
            image_files.append(save_path)  # Reference image is already saved there
            prompts.append("Reference image (user uploaded)")
            continue
        
        # optimize prompt for canvas
        prompt,negative_prompt = optimize_canvas_prompt(shot['caption'])
        
        if not image_files:  # First image (no reference)
            print(f"🎨 Generating first image with AI (no reference found)")
            ret = reel_gen.generate_text2img(prompt,negative_prompt, save_path,seed,cfg_scale)
        else:
            print(f"🎨 Generating image {idx} with variations from previous images")
            ret = reel_gen.generate_variations(image_files,prompt,negative_prompt,save_path,seed,cfg_scale,similarity_strength)
        if ret:
            image_files.append(save_path)
            prompts.append(prompt)

        if is_continues_shot: #continues_shot only generates the first image
            break
        time.sleep(10)  # Rate limiting
    
    print(f"📋 Final image files: {[os.path.basename(f) for f in image_files]}")
    return image_files

def generate_reel_prompts(reel_gen:ReelGenerator, shots:dict,image_files:list, skip:bool = True):
    # Read PDF document for prompt optimization
    with open(DEFAULT_GUIDELINE, "rb") as file:
        doc_bytes = file.read()
    
    # Optimize prompts for video generation
    reel_prompts = []
    for shot, ref_img in zip(shots['shots'], image_files):
        if skip:
            user_prompt = f"{shot['prompt']} {shot['cinematography']}"
            reel_prompts.append(user_prompt)
        else:
            optimized_prompt = reel_gen.optimize_reel_prompt(
                system_prompt=SYSTEM_IMAGE_TEXT,
                user_prompt=f"{shot['caption']} {shot['cinematography']}",
                ref_image=ref_img,
                doc_bytes=doc_bytes
            )
            reel_prompts.append(optimized_prompt)
        
    return reel_prompts

def generate_shot_vidoes(reel_gen:ReelGenerator,image_files:list,reel_prompts:list):        
    # Generate videos
    invocation_arns = []
    for prompt, image_file in zip(reel_prompts, image_files):
        invocation = reel_gen.generate_video(prompt, image_file)
        if invocation:
            invocation_arns.append(invocation['invocationArn'])
    
    # Wait for video generation to complete
    final_responses = reel_gen.fetch_job_status(invocation_arns)

    # Download generated videos
    video_files = []
    for response in final_responses:
        output_uri = response['outputDataConfig']['s3OutputDataConfig']['s3Uri'] + '/output.mp4'
        video_file = reel_gen.download_video_from_s3(output_uri, './generated_videos')
        if video_file:
            video_files.append(video_file)
    return video_files

def sistch_vidoes(reel_gen:ReelGenerator,video_files:list,shots:dict,timestamp:str):      
    print(f"\n🎬 STARTING VIDEO STITCHING PROCESS")
    print(f"=" * 50)
    print(f"📝 Input parameters:")
    print(f"   Video files count: {len(video_files)}")
    print(f"   Shots count: {len(shots.get('shots', []))}")
    print(f"   Timestamp: {timestamp}")
    
    # Validate input video files
    print(f"\n📹 Validating input video files:")
    valid_video_files = []
    for i, video_file in enumerate(video_files):
        print(f"   {i+1}. {video_file}")
        if os.path.exists(video_file):
            file_size = os.path.getsize(video_file)
            print(f"      ✅ File exists, size: {file_size:,} bytes")
            
            # Check if it's a valid video file
            try:
                import subprocess
                result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_file], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"      ✅ Valid video file")
                    valid_video_files.append(video_file)
                else:
                    print(f"      ❌ Invalid video file (ffprobe failed)")
            except Exception as e:
                print(f"      ⚠️  Could not validate video: {str(e)}")
                # Still add to valid files, might work
                valid_video_files.append(video_file)
        else:
            print(f"      ❌ File does not exist!")
    
    print(f"\n📊 Validation results:")
    print(f"   Total input videos: {len(video_files)}")
    print(f"   Valid videos: {len(valid_video_files)}")
    
    if len(valid_video_files) < 2:
        print(f"❌ ERROR: Need at least 2 valid videos for stitching, got {len(valid_video_files)}")
        return None, None
    
    # Update video_files to only include valid ones
    video_files = valid_video_files
    
    # Stitch videos together
    print(f"\n🔗 Starting video stitching process...")
    final_video = None
    caption_video_file = None  # Initialize caption_video_file
    prefix = random_string_name()
    
    # Create output directory
    output_dir = os.path.join('generated_videos', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    
    # Stitch videos in pairs
    print(f"\n🎞️  Stitching {len(video_files)} videos...")
    for idx in range(len(video_files) - 1):
        output_path = os.path.join(output_dir, f'{prefix}_stitched_{idx}.mp4')
        
        if not final_video:
            # First stitching operation
            video1 = video_files[idx]
            video2 = video_files[idx + 1]
            print(f"   Step {idx+1}: Stitching first pair")
            print(f"      Video 1: {os.path.basename(video1)}")
            print(f"      Video 2: {os.path.basename(video2)}")
            print(f"      Output: {os.path.basename(output_path)}")
            
            try:
                final_video = reel_gen.stitch_videos(video1, video2, output_path)
                if final_video and os.path.exists(final_video):
                    file_size = os.path.getsize(final_video)
                    print(f"      ✅ Stitching successful, output size: {file_size:,} bytes")
                else:
                    print(f"      ❌ Stitching failed - no output file created")
                    return None, None
            except Exception as e:
                print(f"      ❌ Stitching failed with error: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None
        else:
            # Subsequent stitching operations
            next_video = video_files[idx + 1]
            print(f"   Step {idx+1}: Stitching with next video")
            print(f"      Current result: {os.path.basename(final_video)}")
            print(f"      Next video: {os.path.basename(next_video)}")
            print(f"      Output: {os.path.basename(output_path)}")
            
            try:
                new_final_video = reel_gen.stitch_videos(final_video, next_video, output_path)
                if new_final_video and os.path.exists(new_final_video):
                    file_size = os.path.getsize(new_final_video)
                    print(f"      ✅ Stitching successful, output size: {file_size:,} bytes")
                    final_video = new_final_video
                else:
                    print(f"      ❌ Stitching failed - no output file created")
                    return None, None
            except Exception as e:
                print(f"      ❌ Stitching failed with error: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None
    
    print(f"\n🎯 Video stitching completed!")
    if final_video:
        final_size = os.path.getsize(final_video)
        print(f"   Final stitched video: {final_video}")
        print(f"   Final video size: {final_size:,} bytes")
    
    # Add captions
    print(f"\n📝 Adding captions to final video...")
    if final_video:
        duration = 6  # Duration per shot
        captions = []
        
        print(f"   Processing {len(shots.get('shots', []))} shots for captions...")
        for idx, shot in enumerate(shots.get('shots', [])):
            print(f"   Shot {idx+1}: {shot.get('caption', 'No caption')[:50]}...")
            desc_arr = reel_gen._split_caption(shot['caption'])
            print(f"      Split into {len(desc_arr)} caption segments")
            
            for idy, sub_desc in enumerate(desc_arr):
                if sub_desc:  # Only add non-empty captions
                    start_time = idx * duration + (idy * duration / len(desc_arr))
                    end_time = idx * duration + ((idy + 1) * duration / len(desc_arr))
                    captions.append((sub_desc, start_time, end_time))
                    print(f"         Caption {idy+1}: '{sub_desc[:30]}...' ({start_time:.1f}s - {end_time:.1f}s)")
        
        print(f"   Total captions to add: {len(captions)}")
        
        if captions:
            caption_video_file = os.path.splitext(final_video)[0] + "_caption.mp4"
            print(f"   Caption output file: {caption_video_file}")
            
            try:
                reel_gen.add_timed_captions(final_video, caption_video_file, captions)
                if os.path.exists(caption_video_file):
                    caption_size = os.path.getsize(caption_video_file)
                    print(f"   ✅ Captions added successfully, size: {caption_size:,} bytes")
                    print(f"   Final video with captions saved to: {caption_video_file}")
                else:
                    print(f"   ❌ Caption file was not created")
            except Exception as e:
                print(f"   ❌ Error adding captions: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ⚠️  No captions to add")
    else:
        print(f"   ❌ No final video to add captions to")
    
    print(f"\n🎉 STITCHING PROCESS COMPLETED")
    print(f"   Final video: {final_video if final_video else 'None'}")
    print(f"   Caption video: {caption_video_file if caption_video_file else 'None'}")
    print(f"=" * 50)
    
    return final_video, caption_video_file

def extract_last_frame(video_path: str, output_path: str):
    """
    Extracts the last frame of a video file.

    Args:
        video_path (str): The local path to the video to extract the last frame from.
        output_path (str): The local path to save the extracted frame to.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Move to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    # Read the last frame
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if ret:
        # Save the last frame as an image
        cv2.imwrite(output_path, frame)
        # print(f"Last frame saved as {output_path}")
    else:
        print("Error: Could not read the last frame.")

    # Release the video capture object
    cap.release()