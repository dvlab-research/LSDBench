import os
import sys
import subprocess
import cv2
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm
from PIL import Image
import argparse
import re

def get_new_height_width(height, width, longest_side_size=512):
    if height > width:
        new_height = longest_side_size
        new_width = int(new_height * width / height)  # floor value
    else:
        new_width = longest_side_size
        new_height = int(new_width * height / width)  # floor value
    return new_height, new_width

def downsample_video(input_video_path, output_video_path, target_fps=None):
    print(f"Downsampling video: {input_video_path} to {output_video_path}")
    if os.path.exists(output_video_path):
        print('Downsampled video exists. Exiting now...')
        return
        
    try:
        print("Attempting to use FFmpeg for fast downsampling...")
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", 
                      "stream=width,height,r_frame_rate,duration", "-of", "csv=p=0", input_video_path]
        
        try:
            probe_output = subprocess.check_output(probe_cmd, universal_newlines=True).strip().split(',')
            width, height, fps_fraction, duration = probe_output
            original_fps = eval(fps_fraction)
            width, height = int(width), int(height)
            new_height, new_width = get_new_height_width(height, width, longest_side_size=512)
            total_duration = float(duration)
        except:
            video_reader = VideoReader(input_video_path, ctx=cpu(0), num_threads=0)
            original_fps = video_reader.get_avg_fps()
            first_frame = video_reader[0].asnumpy()
            height, width, _ = first_frame.shape
            new_height, new_width = get_new_height_width(height, width, longest_side_size=512)
            total_duration = len(video_reader) / original_fps
            del video_reader
        
        temp_output_path = output_video_path.split('.')[0] + '.temp.mp4'
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        
            
        # Use target_fps if provided, otherwise use original_fps
        output_fps = target_fps if target_fps is not None else original_fps
            
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", input_video_path,
            "-vf", f"fps={output_fps},scale={new_width}:{new_height}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-g", "60",
            "-bf", "2",
            "-an",
            "-pix_fmt", "yuv420p",
            temp_output_path
        ]
        
        with tqdm(total=total_duration, desc=f"Processing {os.path.basename(input_video_path)}", leave=False) as pbar:
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stderr_output = []
            for line in process.stderr:
                stderr_output.append(line)
                match = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
                if match:
                    time_str = match.group(1)
                    h, m, s = map(float, time_str.split(':'))
                    current_time = h * 3600 + m * 60 + s
                    pbar.update(current_time - pbar.n)
            process.wait()
        
        if process.returncode == 0:
            os.replace(temp_output_path, output_video_path)
            print(f"Successfully downsampled video using FFmpeg")
            return
        else:
            error_msg = '\n'.join(stderr_output)
            print(f"FFmpeg method failed with return code {process.returncode}.")
            print(f"FFmpeg error output:\n{error_msg}")
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            raise RuntimeError(f"FFmpeg failed with return code {process.returncode}. Error: {error_msg}")
            
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"FFmpeg method failed: {str(e)}. Falling back to Python implementation.")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        raise
    

def process_video(filename, source_dir, target_dir, target_fps=None):
    if '_' not in filename and filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        input_video_path = os.path.join(source_dir, filename)
        output_video_path = os.path.join(target_dir, filename)
        #print(f"Processing {filename}...")
         
        downsample_video(input_video_path, output_video_path, target_fps)

def main():
    parser = argparse.ArgumentParser(description="Downsample videos from a source directory to a target directory.")
    parser.add_argument('source_dir', type=str, help='The directory containing the source videos. (e.g. /dataset/ego4d/v2/full_scale/)')
    parser.add_argument('target_dir', type=str, help='The directory to save the downsampled videos.')
    parser.add_argument('video_ids_file', type=str, help='The file containing video IDs to process.')
    parser.add_argument('--target-fps', type=float, help='Target frame rate for the output videos (optional)', default=None)
    
    args = parser.parse_args()
    
    source_dir = args.source_dir
    target_dir = args.target_dir
    video_ids_file = args.video_ids_file
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    with open(video_ids_file, 'r') as f:
        video_ids = {line.strip() for line in f if line.strip()}
    
    filenames = [f for f in os.listdir(source_dir) if f.split('.')[0] in video_ids and f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    print(f"Found {len(filenames)} videos to process.")
    
    if not filenames:
        print("No videos found to process. Exiting.")
        return
    
    print(f"Processing {len(filenames)} videos sequentially...")
    with tqdm(total=len(filenames), desc="Overall Progress") as pbar:
        for filename in filenames:
            try:
                process_video(filename, source_dir, target_dir, args.target_fps)
            except Exception as e:
                print(f"An error occurred during processing {filename}: {e}")
            pbar.update(1)
    
    print("Processing completed successfully.")

if __name__ == "__main__":
    main()