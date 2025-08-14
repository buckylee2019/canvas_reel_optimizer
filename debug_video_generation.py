#!/usr/bin/env python3
"""
Debug video generation issues
"""

import os
import sys
from PIL import Image

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_video_generation_flow():
    """Debug the video generation flow"""
    print("🎬 DEBUGGING VIDEO GENERATION FLOW")
    print("=" * 50)
    
    # Check recent shot_images directories
    print("📝 Step 1: Checking recent shot_images...")
    shot_images_dir = 'shot_images'
    if os.path.exists(shot_images_dir):
        subdirs = [d for d in os.listdir(shot_images_dir) if os.path.isdir(os.path.join(shot_images_dir, d))]
        subdirs.sort(reverse=True)  # Most recent first
        
        if subdirs:
            latest_dir = os.path.join(shot_images_dir, subdirs[0])
            print(f"✅ Latest shot_images directory: {latest_dir}")
            
            # Check images in latest directory
            image_files = [f for f in os.listdir(latest_dir) if f.endswith('.png')]
            image_files.sort()
            
            print(f"📸 Images found: {len(image_files)}")
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(latest_dir, img_file)
                try:
                    img = Image.open(img_path)
                    file_size = os.path.getsize(img_path)
                    print(f"   {i+1}. {img_file}: {img.size}, {file_size:,} bytes")
                    
                    # Check if this looks like a reference image (your screenshot)
                    if img_file == 'shot_0.png' and img.size == (1374, 912):
                        print(f"      🎯 This appears to be your reference image!")
                    elif img_file == 'shot_0.png':
                        print(f"      ⚠️  This might be a generated image, not your reference")
                        
                except Exception as e:
                    print(f"   {i+1}. {img_file}: Error reading - {str(e)}")
            
            return latest_dir, image_files
        else:
            print("❌ No shot_images directories found")
            return None, []
    else:
        print("❌ shot_images directory doesn't exist")
        return None, []

def debug_video_files():
    """Debug generated video files"""
    print(f"\n📝 Step 2: Checking generated videos...")
    videos_dir = 'generated_videos'
    
    if os.path.exists(videos_dir):
        # Get all video files
        video_files = []
        for root, dirs, files in os.walk(videos_dir):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    video_files.append(video_path)
        
        video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # Most recent first
        
        print(f"🎥 Video files found: {len(video_files)}")
        for i, video_file in enumerate(video_files[:10]):  # Show latest 10
            file_size = os.path.getsize(video_file)
            mtime = os.path.getmtime(video_file)
            import datetime
            mod_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   {i+1}. {os.path.basename(video_file)}: {file_size:,} bytes, modified: {mod_time}")
            
        return video_files[:5]  # Return 5 most recent
    else:
        print("❌ generated_videos directory doesn't exist")
        return []

def debug_video_generation_parameters():
    """Debug the parameters passed to video generation"""
    print(f"\n📝 Step 3: Checking video generation parameters...")
    
    # Simulate what happens in generate_shot_vidoes
    latest_dir, image_files = debug_video_generation_flow()
    
    if latest_dir and image_files:
        print(f"\n🔍 Video Generation Analysis:")
        print(f"   Image files that would be passed to video generation:")
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(latest_dir, img_file)
            print(f"   {i+1}. {img_path}")
            
            # Check if file exists and is readable
            if os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        data = f.read(100)  # Read first 100 bytes
                    print(f"      ✅ File readable, starts with: {data[:20].hex()}")
                except Exception as e:
                    print(f"      ❌ File read error: {str(e)}")
            else:
                print(f"      ❌ File doesn't exist!")
        
        # Check if shot_0.png is your reference image
        shot_0_path = os.path.join(latest_dir, 'shot_0.png')
        if os.path.exists(shot_0_path):
            img = Image.open(shot_0_path)
            if img.size == (1374, 912):
                print(f"\n✅ shot_0.png appears to be your reference image (correct size)")
            else:
                print(f"\n⚠️  shot_0.png size is {img.size}, expected (1374, 912) for reference")
        
        return True
    else:
        print("❌ No image files to analyze")
        return False

def debug_video_stitching_paths():
    """Debug video stitching path issues"""
    print(f"\n📝 Step 4: Checking video stitching paths...")
    
    video_files = debug_video_files()
    
    if len(video_files) >= 2:
        print(f"\n🔍 Video Stitching Analysis:")
        print(f"   Videos that would be stitched:")
        
        for i, video_path in enumerate(video_files[:3]):  # Check first 3
            print(f"   {i+1}. {video_path}")
            
            # Check if file exists and is readable
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                print(f"      ✅ File exists, size: {file_size:,} bytes")
                
                # Check if it's a valid video file
                try:
                    import subprocess
                    result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        print(f"      ✅ Valid video file")
                    else:
                        print(f"      ❌ Invalid video file")
                except Exception as e:
                    print(f"      ⚠️  Could not validate video: {str(e)}")
            else:
                print(f"      ❌ File doesn't exist!")
        
        return True
    else:
        print("❌ Not enough video files for stitching analysis")
        return False

if __name__ == "__main__":
    print("🧪 VIDEO GENERATION DEBUG")
    print("=" * 40)
    
    # Debug 1: Image generation flow
    success1 = debug_video_generation_parameters()
    
    # Debug 2: Video files
    success2 = len(debug_video_files()) > 0
    
    # Debug 3: Video stitching
    success3 = debug_video_stitching_paths()
    
    print(f"\n📊 Debug Results:")
    print(f"   Image Generation Flow: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Video Files Found: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"   Video Stitching Ready: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    if success1 and success2 and success3:
        print(f"\n💡 Likely Issues:")
        print(f"   1. Check if shot_0.png is actually your reference image")
        print(f"   2. Verify video generation is using the correct image files")
        print(f"   3. Check video file paths in stitching function")
    else:
        print(f"\n⚠️  Issues found. Check the analysis above.")
