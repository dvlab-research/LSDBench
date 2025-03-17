import os
import sys
import cv2
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm

import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from decord import VideoReader, cpu
import decord
import logging
import time
from typing import Dict, List, Optional, Union, Tuple

# Convert time strings to seconds
def time_to_seconds(time_str: str) -> int:
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# Normalize time segments format
def normalize_time_segment(segment: Union[Tuple[str, str], Dict[str, str]]) -> Tuple[int, int]:
    if isinstance(segment, tuple):
        start_time, end_time = segment
    else:
        start_time = segment['start']
        end_time = segment['end']
    return (time_to_seconds(start_time), time_to_seconds(end_time))

class SamplingManager:
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        self.sampler = VideoFrameSampler(clip_model_name=clip_model_name)

    def convert_indices_to_segments(self, phase1_result: str, max_duration: int = 60, frame_indices: List[int] = None, fps: float = None, min_side_length: int = 30) -> List[Dict[str, str]]:
        """
        Convert index list from phase 1 to time segments list
        Args:
            phase1_result: String containing index list like "[1, 2, 3]" or "6, 8, 10, 12" - indices of segments to select
            max_duration: Maximum total duration in minutes
            frame_indices: List of frame indices that define key frames
            fps: Frames per second of the video, required if frame_indices is provided
        Returns:
            List[Dict[str, str]]: List of time segments with start and end times
        """
        # If frame_indices provided, use frame-based segmentation
        if frame_indices is not None and fps is not None:
            if len(frame_indices) < 2:
                return [{"start": "00:00:00", "end": "00:01:00"}]
            
            # 1. First create all possible segments based on frame_indices
            timestamps = [frame_idx / fps for frame_idx in frame_indices]
            midpoints = [(timestamps[i] + timestamps[i+1]) / 2 for i in range(len(timestamps)-1)]
            midpoints = [0] + midpoints + [timestamps[-1] + (timestamps[-1] - midpoints[-1])]
            
            all_segments = []
            for i in range(len(timestamps)):
                start_seconds = max(0, min(midpoints[i], timestamps[i] - min_side_length))
                end_seconds = min(midpoints[-1], max(timestamps[i] + min_side_length, midpoints[i+1]))
                
                start_time = time.strftime('%H:%M:%S', time.gmtime(start_seconds))
                end_time = time.strftime('%H:%M:%S', time.gmtime(end_seconds))
                
                all_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "duration": end_seconds - start_seconds,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "index": i  # Add index to track position
                })
                # print(f"start: {start_time}, end: {end_time}, duration: {end_seconds - start_seconds}")
            
            # 2. Extract selected indices from phase1_result
            try:
                phase1_result = phase1_result.replace("image", "")
                phase1_result = phase1_result.replace("Image", "")
                import re
                indices_str = re.findall(r'\[(.*?)\]', phase1_result)
                if indices_str:
                    numbers_str = indices_str[0]
                else:
                    numbers_str = re.findall(r'\d+(?:\s*,\s*\d+)*', phase1_result)[0]
                
                selected_indices = sorted([int(idx.strip())-1 for idx in numbers_str.split(',') if idx.strip()])
                if not selected_indices:
                    return [all_segments[0]]
            except (ValueError, IndexError):
                return [all_segments[0]]
            
            # 3. Select segments based on indices while respecting max_duration
            selected_segments = []
            total_duration = 0
            
            for idx in selected_indices:
                if idx >= len(all_segments):
                    continue
                    
                segment = all_segments[idx].copy()
                if total_duration + segment["duration"] > max_duration * 60:
                    break
                    
                selected_segments.append(segment)
                total_duration += segment["duration"]
            
            # Ensure we have at least one segment
            if not selected_segments:
                return [{"start": all_segments[0]["start"], "end": all_segments[0]["end"]}]
            
            # 4. Merge adjacent segments using index property
            merged_segments = []
            current_segment = selected_segments[0]
            
            for next_segment in selected_segments[1:]:
                # Check if segments are adjacent using their indices
                if abs(next_segment["index"] - current_segment["index"]) == 1:
                    # Merge segments by updating end time
                    current_segment["end"] = next_segment["end"]
                    current_segment["end_seconds"] = next_segment["end_seconds"]
                    current_segment["duration"] = current_segment["end_seconds"] - current_segment["start_seconds"]
                    current_segment["index"] = next_segment["index"]
                else:
                    # Add current segment to merged list and start a new one
                    merged_segments.append({
                        "start": current_segment["start"],
                        "end": current_segment["end"]
                    })
                    current_segment = next_segment
            
            # Add the last segment
            merged_segments.append({
                "start": current_segment["start"],
                "end": current_segment["end"]
            })
                
            return merged_segments
        
            
    
    def sample_frames(self, 
                     video_reader,
                     sampling_config: Dict,
                     time_segments: Optional[List] = None) -> List[int]:
        """
        Sample frames according to the configuration strategy
        
        Args:
            video_reader: VideoReader instance
            sampling_config: Sampling configuration dictionary
            time_segments: Optional list of time segments
            
        Returns:
            List[int]: List of sampled frame indices
        """
        strategy = sampling_config.get("sampling_strategy", "fixed")
        print(f"strategy: {strategy}")
        
        if strategy == "fps":
            return self._sample_fps(video_reader, sampling_config[f"{strategy}_config"], time_segments)
        elif strategy == "fixed":
            return self._sample_fixed(video_reader, sampling_config[f"{strategy}_config"], time_segments)
        elif strategy == "sgfs":
            return self._sample_sgfs(video_reader, sampling_config[f"{strategy}_config"], time_segments)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _sample_fps(self, video_reader, config: Dict, time_segments: Optional[List]) -> List[int]:
        fps = config.get("fps", 1.0)
        if not time_segments:
            return self.sampler.fps_sampling(video_reader, fps)
        
        # First get all frame indices from segments
        all_segment_indices = []
        video_fps = video_reader.get_avg_fps()
        for segment in time_segments:
            start_frame = int(time_to_seconds(segment['start']) * video_fps)
            end_frame = int(time_to_seconds(segment['end']) * video_fps)
            all_segment_indices.extend(range(start_frame, end_frame))
        
        # Calculate interval based on target fps
        interval = int(video_fps / fps)
        # Sample from concatenated indices
        sampled_indices = all_segment_indices[::interval]
        print("fps, interval:", fps, interval)
        return sorted(sampled_indices)
    
    def _sample_fixed(self, video_reader, config: Dict, time_segments: Optional[List]) -> List[int]:
        num_frames = config.get("num_frames", 8)
        if not time_segments:
            return self.sampler.fixed_sampling(video_reader, num_frames)
        
        # First get all frame indices from segments
        all_segment_indices = []
        video_fps = video_reader.get_avg_fps()
        for segment in time_segments:
            start_frame = int(time_to_seconds(segment['start']) * video_fps)
            end_frame = int(time_to_seconds(segment['end']) * video_fps)
            all_segment_indices.extend(range(start_frame, end_frame))
        
        # Perform uniform sampling on concatenated indices
        if len(all_segment_indices) < num_frames:
            print("len(all_segment_indices) < num_frames:", len(all_segment_indices), num_frames)
            return sorted(all_segment_indices)  # Return all frames if fewer than requested
        
        indices = np.linspace(0, len(all_segment_indices) - 1, num_frames, dtype=int)
        print("num_frames:", num_frames)
        return sorted([all_segment_indices[i] for i in indices])
    
    def _sample_sgfs(self, video_reader, config: Dict, time_segments: Optional[List]) -> List[int]:
        num_frames = config.get("num_frames")
        keep_ratio = config.get("keep_ratio")
        initial_frames = config.get("initial_frames")
        initial_fps = config.get("initial_fps")
        length_penalty = config.get("length_penalty", 0.0)
        length_penalty_exponent = config.get("length_penalty_exponent", 1.0)
        print(f"num_frames: {num_frames}, keep_ratio: {keep_ratio}, initial_frames: {initial_frames}, initial_fps: {initial_fps}, length_penalty: {length_penalty}, length_penalty_exponent: {length_penalty_exponent}")
        
        if not time_segments:
            return self.sampler.sgfs_sampling(
                video_reader,
                num_samples=num_frames,
                keep_ratio=keep_ratio,
                initial_frames=initial_frames,
                initial_fps=initial_fps,
                length_penalty=length_penalty,
                length_penalty_exponent=length_penalty_exponent
            )
        
        assert False, "Not implemented yet!"
    
    @staticmethod
    def _filter_indices_by_segment(indices: List[int], segment: Dict, fps: float) -> List[int]:
        """Filter frame indices within the specified time segment"""
        start_frame = int(time_to_seconds(segment['start']) * fps)
        end_frame = int(time_to_seconds(segment['end']) * fps)
        # print(f"start_frame: {start_frame}, end_frame: {end_frame}")
        return [idx for idx in indices if start_frame <= idx < end_frame]
    
    @staticmethod
    def _get_segment_duration(segment: Dict) -> float:
        """Get the duration of a time segment in seconds"""
        start_seconds = time_to_seconds(segment['start'])
        end_seconds = time_to_seconds(segment['end'])
        return end_seconds - start_seconds

class VideoFrameSampler:
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the video frame sampler with CLIP model.

        Args:
            clip_model_name (str): Name of the CLIP model to use
            device (str): Device to run the model on
        """
        self.device = device

        if clip_model_name is not None:
            self.model = AutoModel.from_pretrained(
                clip_model_name,
                attn_implementation="flash_attention_2", 
                device_map="cuda",
                torch_dtype=torch.float16
            ).eval().to(device)
            if torch.__version__ >= "2.0.0":
                self.model = torch.compile(self.model)  # Compile model for faster inference
            self.processor = AutoProcessor.from_pretrained(clip_model_name)

            # Set model to eval mode
            self.model.eval()
        
        self.logger = logging.getLogger(__name__)

    def encode_images(self, images, max_batch_size=64):
        """Encode images using model's vision encoder.
        
        Args:
            images: List of images or single image
            max_batch_size (int): Maximum batch size for processing
            
        Returns:
            torch.Tensor: Encoded image features
            
        Raises:
            ValueError: If images input is invalid
            RuntimeError: If encoding fails
        """
        try:
            # Convert single image to list
            if not isinstance(images, list):
                images = [images]
            
            # Process in batches
            encoded_features = []
            for i in range(0, len(images), max_batch_size):
                batch = images[i:i + max_batch_size]
                
                # Process batch
                with torch.inference_mode():
                    # Prepare inputs
                    inputs = self.processor(
                        images=batch,
                        return_tensors="pt",
                        padding=True
                    ).to(self.model.device)
                    
                    # Get predictions
                    batch_embeds = self.model.get_image_features(**inputs) 
                    
                    encoded_features.append(batch_embeds)
            
            # Concatenate all batches
            if len(encoded_features) > 1:
                encoded_features = torch.cat(encoded_features, dim=0)
            else:
                encoded_features = encoded_features[0]
            
            return encoded_features
            
        except ValueError as e:
            self.logger.error(f"Invalid input format: {str(e)}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Error during image encoding: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in encode_images: {str(e)}")
            raise

    def _get_video_reader(self, video_input):
        """Helper method to get VideoReader from either path or reader.
        
        Args:
            video_input: Either a string path to video file or a VideoReader instance
            
        Returns:
            VideoReader: A decord VideoReader instance
        """
        if isinstance(video_input, str):
            return VideoReader(video_input, ctx=decord.cpu(), num_threads=0)
        elif isinstance(video_input, VideoReader):
            return video_input
        else:
            raise ValueError("video_input must be either a path string or VideoReader instance")

    def sgfs_sampling(
        self, 
        video_input,
        num_samples: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        initial_frames: Optional[int] = None,
        initial_fps: Optional[float] = None,
        length_penalty: float = 0.0,
        length_penalty_exponent: float = 1.0,
    ):
        """Sample frames from video using sgfs.
        
        Args:
            video_input: Either a path string to video file or a VideoReader instance
            num_samples (int, optional): Number of frames to sample
            keep_ratio (float, optional): Ratio of frames to keep (0.0-1.0)
            initial_frames (int, optional): Number of initial uniform samples
            initial_fps (float, optional): FPS for initial sampling
            length_penalty (float): Penalty for temporal distance between frames
            length_penalty_exponent (float): Exponent for the length penalty
            
        Returns:
            List of frame indices
        """
        try:
            # Load video
            start_time = time.time()
            vr = self._get_video_reader(video_input)
            num_frames = len(vr)
            video_fps = vr.get_avg_fps()

            # Determine initial sampling method
            if initial_fps is not None:
                # FPS-based initial sampling
                interval = round(video_fps / initial_fps)
                initial_idx = np.arange(0, num_frames, interval, dtype=int)
            else:
                # Frame count-based initial sampling
                if initial_frames is None:
                    initial_frames = min(num_frames, num_samples * 2 if num_samples else int(num_frames * 0.5))
                initial_idx = np.linspace(0, num_frames - 1, initial_frames, dtype=int)

            # Get initial frames
            initial_frames = vr.get_batch(initial_idx).asnumpy()
            print(f"Video loading time: {time.time() - start_time:.2f}s")

            # Determine target number of samples
            if num_samples is None and keep_ratio is None:
                raise ValueError("Either num_samples or keep_ratio must be specified")
            if keep_ratio is not None:
                if not 0 < keep_ratio <= 1:
                    raise ValueError("keep_ratio must be between 0 and 1")
                num_samples = max(1, int(len(initial_idx) * keep_ratio))
            else:
                num_samples = min(num_samples, len(initial_idx))

            # Encode frames with CLIP
            start_time = time.time()
            with torch.inference_mode():
                initial_frames_features = self.encode_images(initial_frames)
                
                # Normalize features
                flattened_features = initial_frames_features.view(len(initial_idx), -1)
                normalized_features = torch.nn.functional.normalize(flattened_features, p=2, dim=1).float()
                
                # Compute similarity matrix
                similarity_matrix = torch.mm(normalized_features, normalized_features.t()).cpu().numpy()
            print(f"Feature extraction time: {time.time() - start_time:.2f}s")
            _similarity_matrix = np.zeros((len(initial_idx) + 1, len(initial_idx) + 1))
            _similarity_matrix[1:, 1:] = similarity_matrix

            start_time = time.time()
            # Dynamic programming for frame selection
            f = np.full((len(initial_idx) + 1, num_samples + 1), np.inf)
            route_trace = np.full((len(initial_idx) + 1, num_samples + 1), -1, dtype=int)
            f[0, 0] = 0.0 # 1 frame selected for the first 1 frames
            route_trace[0, 0] = -1

            # Compute length penalties
            length_penalties = np.array([(abs(i - k) / len(initial_idx) * length_penalty) ** length_penalty_exponent 
                                for i in range(len(initial_idx)) 
                                for k in range(len(initial_idx))])
            length_penalties = length_penalties.reshape(len(initial_idx), len(initial_idx))
            _length_penalties = np.zeros((len(initial_idx) + 1, len(initial_idx) + 1))
            _length_penalties[1:, 1:] = length_penalties

            # Dynamic programming
            for j in range(1, num_samples + 1):
                for i in range(j, len(initial_idx) + 1):
                    tmp = f[j-1:i, j-1] + _similarity_matrix[j-1:i, i] - _length_penalties[j-1:i, i]
                    min_idx = np.argmin(tmp)
                    f[i, j] = tmp[min_idx]
                    route_trace[i, j] = j-1 + min_idx

            # Backtrack to get selected frames
            selected_frames = []
            i = len(initial_idx)
            while i > 0 and len(selected_frames) < num_samples:
                selected_frames.append(initial_idx[i - 1])
                i = route_trace[i, num_samples - len(selected_frames) + 1]
            selected_frames = selected_frames[::-1]
            print(f"Dynamic programming time: {time.time() - start_time:.2f}s")

            if len(selected_frames) < num_samples:
                raise ValueError(f"Only got {len(selected_frames)} frames, needed {num_samples}")
            
            torch.cuda.empty_cache()

            return selected_frames

        except Exception as e:
            self.logger.error(f"Error in sgfs_sampling: {str(e)}")
            raise

    def fps_sampling(
        self,
        video_input,
        target_fps: float,
    ) -> List[int]:
        """Sample frames based on target FPS.
        
        Args:
            video_input: Either a path string to video file or a VideoReader instance
            target_fps (float): Target frames per second
            
        Returns:
            List[int]: List of frame indices
        """
        try:
            # Load video
            vr = self._get_video_reader(video_input)
            num_frames = len(vr)
            original_fps = vr.get_avg_fps()
            
            # Calculate frame interval
            interval = int(original_fps / target_fps)
            if interval < 1:
                raise ValueError(f"Target FPS ({target_fps}) is higher than video FPS ({original_fps})")
            
            # Generate frame indices
            frame_indices = np.arange(0, num_frames, interval, dtype=int)
            return frame_indices.tolist()
            
        except Exception as e:
            self.logger.error(f"Error in fps_sampling: {str(e)}")
            raise e

    def fixed_sampling(
        self,
        video_input,
        num_samples: int,
    ) -> List[int]:
        """Sample frames uniformly from video.
        
        Args:
            video_input: Either a path string to video file or a VideoReader instance
            num_samples (int): Number of frames to sample
            
        Returns:
            List[int]: List of frame indices
        """
        try:
            # Load video
            vr = self._get_video_reader(video_input)
            num_frames = len(vr)
            
            if num_samples > num_frames:
                raise ValueError(f"Requested {num_samples} samples but video only has {num_frames} frames")
            
            # Get uniform samples
            indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
            return indices.tolist()
            
        except Exception as e:
            self.logger.error(f"Error in fixed_sampling: {str(e)}")
            raise e

