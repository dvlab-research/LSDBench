import yaml
from pathlib import Path
import os
import time
import tempfile
from typing import Union, List, Tuple, Dict
import base64
from io import BytesIO

from decord import VideoReader
from PIL import Image
from colorama import Fore, Style
from .video_utils import SamplingManager

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
import torch
from .vison_process import process_vision_info

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

def extract_frames(vr: VideoReader, final_fps: float = None, num_frames: int = None, time_segments: List = None) -> List[int]:
    """
    Extract frame indices based on either FPS or fixed frame count sampling
    Args:
        vr: VideoReader object
        final_fps: Target frames per second for sampling (mutually exclusive with num_frames)
        num_frames: Fixed number of frames to extract (mutually exclusive with final_fps)
        time_segments: List of time segments to process
    Returns:
        List of frame indices to extract
    """
    if final_fps is not None and num_frames is not None:
        raise ValueError("Cannot specify both final_fps and num_frames")
    if final_fps is None and num_frames is None:
        raise ValueError("Must specify either final_fps or num_frames")

    fps = vr.get_avg_fps()
    total_frames = len(vr)
    
    # Process all segments or use entire video if no segments specified
    if not time_segments:
        time_segments = [{"start": "00:00:00", "end": "00:00:00"}]
    
    # Collect frames from all segments
    all_frame_indices = []
    for segment in time_segments:
        start_seconds, end_seconds = normalize_time_segment(segment)
        start_frame = int(start_seconds * fps)
        end_frame = int(end_seconds * fps) if end_seconds > 0 else total_frames
        
        if final_fps:
            # FPS-based sampling
            interval_seconds = 1.0 / final_fps
            interval_frames = int(interval_seconds * fps)
            segment_indices = list(range(start_frame, end_frame, max(1, interval_frames)))
        else:
            # Fixed frame count sampling
            segment_length = end_frame - start_frame
            if segment_length <= num_frames:
                segment_indices = list(range(start_frame, end_frame))
            else:
                # Evenly distribute frames across segment
                interval = segment_length / num_frames
                segment_indices = [start_frame + int(i * interval) for i in range(num_frames)]
        
        all_frame_indices.extend(segment_indices)
    
    return sorted(list(set(all_frame_indices)))


class RHSQwen25VL:
    def __init__(
        self,
        config_path: str,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        device_map: str = "auto",
        use_flash_attention_2: bool = True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        self.prompt_dir = self.config['prompt_dir']

        # Initialize device
        self._device = torch.device(device)
        self.device_map = device_map

        # Initialize model
        if use_flash_attention_2:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=self.device_map
            ).eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained, 
                torch_dtype="auto", 
                device_map=self.device_map
            ).eval()

        # Initialize processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            pretrained, 
            max_pixels=max_pixels, 
            min_pixels=min_pixels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

        # Save configuration
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        # Video config
        self.clip_model_name = self.config['video_config']["clip_model_name"]
        self.sampling_manager = SamplingManager(clip_model_name=self.clip_model_name)

    def generate(self,
                question: str,
                video_path: str,
                **kwargs,
                ) -> str:
        question, options = self._split_question_and_options(question)
        
        with torch.inference_mode():
            # Phase 1: Use sparse sampling with multiple images approach
            prefix1 = open(os.path.join(self.prompt_dir, self.config['phase1_config']["prompt_file"]), "r").read()
            question1 = prefix1 + '\n' + "Question:" + question
            phase1_result, time_segments = self.ask_question_multi_frames(question1, video_path, sampling_config=self.config['phase1_config'], return_segments=True)   
            
            torch.cuda.empty_cache()

            # Phase 2: Use dense sampling for predicted target segments
            phase2_fps = self.config['phase2_config']["fps"]
            prefix2 = open(os.path.join(self.prompt_dir, self.config['phase2_config']["prompt_file"]), "r").read()
            question2 = prefix2 + '\n' + question + '\n' + options
            phase2_result = self.ask_question_video(question2, video_path, time_segments, fps=phase2_fps)

            torch.cuda.empty_cache()

        return phase2_result

    def ask_question_multi_frames(self, 
                                  question: str, 
                                  video_path: str, 
                                  final_fps: float = None, 
                                  num_frames: int = None, 
                                  sampling_config: Dict = None,
                                  time_segments: List = None, 
                                  add_vision_id: bool = False,
                                  return_segments: bool = False) -> str:
        """Ask questions using multiple images approach"""
        vr = VideoReader(video_path, num_threads=1)
        if sampling_config is None:
            frame_indices = extract_frames(vr, final_fps, num_frames, time_segments)
        else:
            frame_indices = self.sampling_manager.sample_frames(vr, sampling_config, time_segments)
        
        # Batch convert frames to PIL Images
        images = []
        frame_indices = [idx for idx in frame_indices if idx < len(vr)]
        if frame_indices:
            frames = vr.get_batch(frame_indices).asnumpy()
            images = [Image.fromarray(frame) for frame in frames]
        # Build message
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # Add image content
        image_content = []
        for i, img in enumerate(images):
            base64_image = img.convert("RGB")
            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")
            image_content.append({"type": "text", "text": f"image {i+1}:"})
            image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
            
        message.append({"role": "user", "content": image_content + [{"type": "text", "text": question}]})
        
        # Use model to generate response
        response = self._generate_response(message, add_vision_id=add_vision_id)
        response_time_segments = self.sampling_manager.convert_indices_to_segments(response, frame_indices=frame_indices, fps=vr.get_avg_fps())
        if return_segments:
            return response, response_time_segments
        else:
            return response

    def ask_question_sampled_video(self, 
                                  question: str, 
                                  video_path: str, 
                                  final_fps: float = None, 
                                  num_frames: int = None, 
                                  sampling_config: Dict = None,
                                  time_segments: List = None, 
                                  add_vision_id: bool = False) -> str:
        """Ask questions using custom sampled video approach"""
        vr = VideoReader(video_path, num_threads=1)
        if sampling_config is None:
            frame_indices = extract_frames(vr, final_fps, num_frames, time_segments)
        else:
            frame_indices = self.sampling_manager.sample_frames(vr, sampling_config, time_segments)
        
        # Batch convert frames to PIL Images
        frame_indices = [idx for idx in frame_indices if idx < len(vr)]
        frames = []
        if frame_indices:
            batch_frames = vr.get_batch(frame_indices).asnumpy()
            frames = [Image.fromarray(frame) for frame in batch_frames]
        
        # Build message
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # Build video content
        message.append({
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,  # Pass list of PIL Images directly
                },
                {"type": "text", "text": question}
            ]
        })
        
        # Use model to generate response
        response = self._generate_response(message, add_vision_id=add_vision_id)
        return response
    
    def ask_question_video(self, 
                                question: str, 
                                video_path: str, 
                                time_segments: List[Dict[str, str]] = None, 
                                fps: float = None, 
                                num_frames: int = None, 
                                **kwargs) -> str:
        """Ask questions using complete video approach, using qwen2.5vl's original sampling"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist")
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        message.append(
            {"role": "user", "content": [
                {
                    "type": "video", 
                    "video": video_path,
                }, 
                {
                    "type": "text", "text": question
                }
            ]}
        )
        if fps is not None:
            message[-1]["content"][0]["fps"] = fps
        if num_frames is not None:
            message[-1]["content"][0]["nframes"] = num_frames
        if time_segments is not None:
            message[-1]["content"][0]["time_segments"] = time_segments

        response= self._generate_response(message)

        
        return response

    def _generate_response(self, message: List[Dict], add_vision_id: bool = False) -> str:
        """Generate response using Qwen model"""
        text = self.processor.apply_chat_template(message, 
                                                    tokenize=False, 
                                                    add_generation_prompt=True, 
                                                    add_vision_id=add_vision_id
                                                    )

        image_inputs, video_inputs = process_vision_info(message)
        
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self._device)

        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.001,
            top_k=1,
        )

        response = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response
            

    def _split_question_and_options(self, text: str) -> tuple[str, str]:
        """
        Split multiple choice question text into question and options
        Args:
            text: Complete text containing question and options
        Returns:
            tuple: (question, options text)
        """
        # Find the position of the first option (A), A., A, etc.)
        option_markers = ['A)', 'A.', 'A、', 'A．']
        split_pos = -1
        
        for marker in option_markers:
            pos = text.find(marker)
            if pos != -1:
                split_pos = pos
                break
        
        if split_pos == -1:
            return text, ""
        
        # Separate question and options text
        question = text[:split_pos].strip()
        options = text[split_pos:].strip()
            
        return question, options
    
    def ask_question_text(self, context:str) -> str:
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        message.append({"role": "user", "content": context})
        response = self._generate_response(message)
        return response
    

class BaseQwen25VL(RHSQwen25VL):

    def generate(self, question: str, video_path: str, **kwargs) -> str:
        
        with torch.inference_mode():
            # Global sampling method
            prefix1 = open(os.path.join(self.prompt_dir, self.config['phase1_config']["prompt_file"]), "r").read()
            question1 = prefix1 + '\n' + question
            if self.config['phase1_config']["sampling_strategy"] == "global_dynamic":
                phase1_result = self.ask_question_video(question1, video_path)
            else:
                phase1_result = self.ask_question_sampled_video(question1, video_path, sampling_config=self.config['phase1_config'])

            torch.cuda.empty_cache()

        return phase1_result
    
class OracleQwen25VL(RHSQwen25VL):

    def generate(self, question: str, video_path: str, time_segments: List[Dict[str, str]], **kwargs) -> str:
        
        with torch.inference_mode():
            # Phase 2: Only input target segments, using fps sampling
            prefix2 = open(os.path.join(self.prompt_dir, self.config['phase2_config']["prompt_file"]), "r").read()
            question2 = prefix2 + '\n' + question

            phase2_result = self.ask_question_video(question2, video_path, time_segments=time_segments, fps=self.config['phase2_config']["fps"])

            torch.cuda.empty_cache()

        return phase2_result