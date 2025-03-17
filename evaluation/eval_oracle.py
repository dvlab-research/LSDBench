import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import time
import uuid
import argparse
from dataset import LSDBenchDataset
from models.model_qwen_rhs import OracleQwen25VL
from utils import extract_characters_regex, parse_model_args


def parse_args():
    parser = argparse.ArgumentParser(description='LSDBench Video QA Evaluation')
    
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the LSDBench annotation file')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing the video files')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name, for this setting, it should be "qwen_2stage_oracle"')
    parser.add_argument('--model_args', type=str, required=True,
                        help='Model arguments in format "key1=val1,key2=val2"')
    parser.add_argument('--output_dir', type=str, default='eval_logs',
                        help='Directory to save evaluation results')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize dataset
    dataset = LSDBenchDataset(
        data_path=args.data_path,
        video_dir=args.video_dir
    )
    
    # Initialize model
    if args.model_name == "qwen_2stage_oracle":
        model = OracleQwen25VL(**parse_model_args(args.model_args))
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:6]
    log_file = output_dir / f"eval_{timestamp}_{unique_id}.log"
    logger.add(log_file)
    
    # Log configuration details
    logger.info(f"\n{'='*50}")
    logger.info("Configuration Details:")
    logger.info(f"Model Class: {model.__class__.__name__}")
    logger.info(f"Model Args: {args.model_args}")
    logger.info(f"{'='*50}\n")
    
    # Initialize metrics
    total_samples = len(dataset)
    correct_predictions = 0
    results = []
    
    logger.info("Starting evaluation...")
    logger.info(f"Total samples: {total_samples}")
    
    # Main evaluation loop
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        try:
            sample, video_path = dataset[idx]
            target_segment = sample['target_segment']
            
            # Generate response
            response = model.generate(
                question=sample['question'],
                video_path=video_path,
                time_segments=[target_segment]
            )
            
            # Process prediction
            pred_answer = extract_characters_regex(response)
            is_correct = pred_answer == sample['correct_answer']
            
            # Update metrics
            if is_correct:
                correct_predictions += 1
            
            # Store result
            result = {
                'id': sample['id'],
                'video_id': sample['video_id'],
                'question': sample['question'],
                'prediction': pred_answer,
                'ground_truth': sample['correct_answer'],
                'is_correct': is_correct,
                'response': response
            }
            results.append(result)
            
            # Calculate current accuracy
            current_accuracy = (correct_predictions / (idx + 1)) * 100
            
            # Log progress
            logger.info(f"\nSample {idx + 1}/{total_samples}")
            logger.info(f"Question: {sample['question']}")
            logger.info(f"Prediction: {pred_answer}")
            logger.info(f"Ground Truth: {sample['correct_answer']}")
            logger.info(f"Current Accuracy: {current_accuracy:.2f}%")
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            raise
    
    # Calculate final metrics
    final_accuracy = (correct_predictions / total_samples) * 100
    
    # Log final results
    logger.info("\nEvaluation Complete!")
    logger.info(f"Final Accuracy: {final_accuracy:.2f}%")
    logger.info(f"Correct Predictions: {correct_predictions}/{total_samples}")
    
    # Save results
    results_file = output_dir / f"results_{timestamp}_{unique_id}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'model_class': model.__class__.__name__,
                'model_args': args.model_args
            },
            'metrics': {
                'final_accuracy': final_accuracy,
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
            },
            'results': results
        }, f, indent=2)

if __name__ == "__main__":
    main()
