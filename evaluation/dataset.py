import json
from pathlib import Path
from loguru import logger
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class LSDBenchDataset(Dataset):
    """LSDBench dataset implementation using PyTorch Dataset"""
    def __init__(
        self,
        data_path: str,
        video_dir: str,
    ):
        self.data_path = Path(data_path)
        self.video_dir = Path(video_dir)
        self.samples = self._load_data()
        
        logger.info(f"Loaded {len(self.samples)} samples from {self.data_path}")
        
    def _load_data(self) -> List[Dict]:
        """Load dataset from json file"""
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
            
        samples = []
        for idx, item in enumerate(raw_data):
            try:
                question = item["question"]
                options = item["options"]
                options_str = ""
                for option, option_text in options.items():
                    options_str += f"{option}. {option_text}\n"
                question = question + "\n" + options_str
                sample = dict(
                    id=str(idx),
                    question=question,
                    video_id=item['video_id'],
                    correct_answer=item['correct_answer'], # ['A', 'B', 'C', 'D']
                    target_segment=item['time_range'] # {start: 'hh:mm:ss', end: 'hh:mm:ss'}
                )
                # Verify video exists
                if self._get_video_path(sample['video_id']):
                    samples.append(sample)
                else:
                    logger.warning(f"Video not found for sample {idx}, skipping")
            except Exception as e:
                logger.error(f"Error loading sample {idx}: {str(e)}")
                raise
                
        return samples
    
    def _get_video_path(self, video_id: str) -> Optional[Path]:
        """Get video path with fallback extensions"""
        base_path = self.video_dir / video_id
        for ext in ['.mp4', '.MP4', '.mkv']:
            path = base_path.with_suffix(ext)
            if path.exists():
                return path
            
        raise FileNotFoundError(f"Video not found for video_id: {video_id}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Dict, str]:
        """
        Returns:
            tuple: (sample, video_path)
        """
        sample = self.samples[idx]
        video_path = str(self._get_video_path(sample['video_id']))
        return sample, video_path