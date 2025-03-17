# LSDBench: Long-video Sampling Dilemma Benchmark


A benchmark that focuses on the sampling dilemma in long-video tasks. Through well-designed tasks, it evaluates the sampling efficiency of long-video VLMs.

## Sampling Dilemma

<div align=center>
<img width="95%" src="assets/teaser.png"/>
</div>

***(Left)** In Q1, identifying a camera wearer's visited locations requires analyzing the entire video. However, key frames are sparse, so sampling one frame per minute often provides enough information. In contrast, Q2 examines the packing order during checkout, requiring high-resolution sampling to capture rapid actions. **(Right)** **Sampling Dilemma** emerges in tasks like Q2: a low sampling density fails to provide sufficient visual cues for accurate answers, while a high sampling density results in redundant frames, significantly slowing inference speed. This challenge underscores the need for adaptive sampling strategies, especially for tasks with high necessary sampling density.*


## Introduction

Large Vision-Language Models (LVLMs) have shown impressive capabilities in video understanding. However, processing long videos efficiently remains a challenge due to the "Sampling Dilemma": low-density sampling risks missing critical information, while high-density sampling introduces redundancy. Our work introduces **LSDBench**, a novel benchmark designed to evaluate LVLMs on long-video tasks by constructing high Necessary Sampling Density (NSD, the minimum sampling density required to accurately answer a given question) questions. We also provide the code for our proposed **Reasoning-Driven Hierarchical Sampling (RHS)** framework and **Semantic-Guided Frame Selector (SGFS)** to address this dilemma.

## Key Features

<div align=center>
<img width="50%" src="assets/accuracy_vs_frames.png"/>
</div>


*   **LSDBench Dataset:** A benchmark with questions characterized by high Necessary Sampling Density (NSD) requirements and videos lasting for hours.  Focuses on dense action sequences within short segments of long videos.
*   **Reasoning-Driven Hierarchical Sampling (RHS):** A two-stage framework that improves long-video processing efficiency by focusing the VLM on important segments.
*   **Semantic-Guided Frame Selector (SGFS):** A lightweight module that selects frames with higher visual information content without any question prior.

## LSDBench


The LSDBench dataset is designed to evaluate the sampling efficiency of long-video VLMs. It consists of multiple-choice question-answer pairs based on hour-long videos, focusing on short-duration actions with high Necessary Sampling Density (NSD).

*   **Number of QA Pairs:** 1304
*   **Number of Videos:** 400
*   **Average Video Length:** 45.39 minutes (ranging from 20.32 to 115.32 minutes)
*   **Average Target Segment Duration:** 3 minutes

### 1. Video Download (Ego4D)

Due to the Ego4D dataset license restrictions, you must obtain an Ego4D license and download the videos directly from the Ego4D website using the [Ego4D CLI tool](https://github.com/facebookresearch/Ego4d/tree/main/ego4d/cli). We also provide a quick guide to download the videos:


1.  **Apply for an Ego4D License:**

    *   Visit the website and submit the license application: https://ego4ddataset.com/ego4d-license/
    *   Wait for your license application to be approved. This process may take some time.

2.  **Install and Configure the AWS CLI:**

    *   The Ego4D datasets are hosted on Amazon S3 and require AWS credentials to access.
    *   You can install the AWS CLI using either the [official installation guide](https://aws.amazon.com/cli/) or this simple script:
        ```bash
        bash lsdbench/install_aws_cli.sh
        source ~/.bashrc  # or source ~/.zshrc if using zsh
        ```
    *   Open a command line and type `aws configure` (or `aws configure --profile ego4d` if you prefer to use a named profile).
    *   Enter your AWS access key ID and secret access key when prompted. You can leave the default region blank. These keys are provided when your Ego4D license is approved.

3.  **Install the Ego4D CLI:**

    ```bash
    pip install ego4d
    ```

4.  **Download the Videos using the Ego4D CLI:**

    *   Use the following command to download the full-scale videos:

        ```bash
        ego4d --output_directory="path/to/your/ego4d_data" --datasets full_scale --video_uid_file "lsdbench/video_ids.txt" 
        ```

        Replace the following placeholders:

        *   `path/to/your/ego4d_data`: The local directory where you want to store the downloaded videos.

### 2. Video Preprocessing

We provide a script to preprocess(downsample) the videos for the LSDBench dataset.

```bash
python lsdbench/preprocess_videos.py  "VIDEO_DIR" "DOWNSAMPLED_VIDEO_DIR" "lsdbench/video_ids.txt" [--target-fps FPS]
```

Replace the following placeholders:

*   `VIDEO_DIR`: The directory where the downloaded videos are stored.
*   `DOWNSAMPLED_VIDEO_DIR`: The directory where the preprocessed videos will be stored.
*   `--target-fps FPS`: (Optional) Target frame rate for the output videos. By default, it maintains the original frame rate, which may result in longer loading times. You can set a lower target value (not recommended below 2 fps) to reduce video size and loading time.

### 3. Environment Setup

First clone the repository:

```bash
git clone https://github.com/taintaintainu/LSDBench.git
cd LSDBench
```

Before running the evaluation, you need to set up a Python virtual environment with the required dependencies:

```bash
conda create -n lsdbench python=3.9.2
conda activate lsdbench
pip install -r evaluation/requirements.txt
pip install flash-attn --no-build-isolation
```

### 4. Evaluation

**Example 1: Evaluate RHS-Qwen2.5-VL on LSDBench**
```bash
python evaluation/eval.py \
 --data_path lsdbench/mc_qa_annotations_1300.json \
 --video_dir lsdbench/downsampled_videos \
 --model_name qwen_2stage_sgfs \
 --model_args config_path=evaluation/models/configs/qwen_2stage_sgfs.yaml,use_flash_attention_2=True
```

**Example 2: Evaluate Qwen2.5-VL on LSDBench in Oracle Setting**
```bash
python evaluation/eval_oracle.py \
 --data_path lsdbench/mc_qa_annotations_1300.json \
 --video_dir lsdbench/downsampled_videos \
 --model_name qwen_2stage_oracle \
 --model_args config_path=evaluation/models/configs/qwen_2stage_oracle_1fps.yaml,use_flash_attention_2=True
```

## Citation

```bibtex
@article{qu2025lsdbench,
  title      = {Does Your Vision-Language Model Get Lost in the Long Video Sampling Dilemma?},
  author     = {Qu, Tianyuan and Tang, Longxiang and Peng, Bohao and Yang, Senqiao and Yu, Bei and Jia, Jiaya},
  journal    = {arXiv preprint arXiv:2503.12496},
  year       = {2025}
}
```