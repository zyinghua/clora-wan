#!/usr/bin/env python3
"""
ViCLIP-based quantitative evaluation script for CLora-Wan ablation study.

Computes video-text similarity scores between generated videos and their 
corresponding prompts (both base and ablation), then reports comparisons 
across different DiT block configurations.
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_viclip_model(device="cuda"):
    """Load ViCLIP model from HuggingFace."""
    import sys
    import importlib.util
    import types
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Loading ViCLIP-L-14 model...")
    
    # Download model and get local path
    model_path = snapshot_download("OpenGVLab/ViCLIP-L-14-hf")
    
    # Create a virtual package namespace for imports
    dummy_pkg = types.ModuleType("viclip_pkg")
    dummy_pkg.__path__ = [model_path]
    sys.modules["viclip_pkg"] = dummy_pkg
    
    # Load configuration module first
    config_module_path = os.path.join(model_path, "configuration_viclip.py")
    spec = importlib.util.spec_from_file_location("viclip_pkg.configuration_viclip", config_module_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["viclip_pkg.configuration_viclip"] = config_module
    spec.loader.exec_module(config_module)
    
    # Load the viclip module
    viclip_module_path = os.path.join(model_path, "viclip.py")
    spec = importlib.util.spec_from_file_location("viclip_pkg.viclip", viclip_module_path)
    viclip_module = importlib.util.module_from_spec(spec)
    sys.modules["viclip_pkg.viclip"] = viclip_module
    spec.loader.exec_module(viclip_module)
    
    # Load config.json and set absolute tokenizer path
    config_json_path = os.path.join(model_path, "config.json")
    with open(config_json_path, 'r') as f:
        config_dict = json.load(f)
    config_dict['tokenizer_path'] = os.path.join(model_path, "bpe_simple_vocab_16e6.txt.gz")
    
    # Create proper PreTrainedConfig instance
    config = config_module.Config(**config_dict)
    
    # Instantiate model on CPU first
    model = viclip_module.ViCLIP(config)
    
    # Load weights from safetensors
    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    
    tokenizer = model.tokenizer
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model, tokenizer


def extract_frames_from_video(video_path, max_frames=None):
    """Extract frames from video file."""
    video = cv2.VideoCapture(video_path)
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if success:
            frames.append(frame)
        else:
            break
    video.release()
    
    if max_frames and len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
    
    return frames


# Preprocessing constants (CLIP normalization statistics)
V_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 3)
V_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 3)


def normalize(data):
    """Normalize frame data."""
    return (data / 255.0 - V_MEAN) / V_STD


def frames_to_tensor(frames, num_frames=8, target_size=(224, 224), device=torch.device('cuda')):
    """Convert video frames to tensor for ViCLIP, sampling with stride to span whole video."""
    # Sample num_frames with stride to span the entire video
    total_frames = len(frames)
    if total_frames <= num_frames:
        # Use all frames if video is shorter than num_frames
        sampled_frames = frames
        # Pad by repeating last frame if needed
        while len(sampled_frames) < num_frames:
            sampled_frames = sampled_frames + [sampled_frames[-1]]
    else:
        # Calculate stride to span the whole video
        stride = total_frames // num_frames
        indices = [i * stride for i in range(num_frames)]
        sampled_frames = [frames[i] for i in indices]
    
    # Resize and normalize (BGR to RGB conversion)
    processed = [cv2.resize(x[:, :, ::-1], target_size) for x in sampled_frames]
    processed = [np.expand_dims(normalize(x), axis=(0, 1)) for x in processed]
    
    vid_tube = np.concatenate(processed, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    
    return vid_tube


def compute_similarity(video_tensor, text, model, tokenizer, text_feat_cache=None):
    """Compute cosine similarity between video and text."""
    if text_feat_cache is None:
        text_feat_cache = {}
    
    # Get video features
    vid_feat = model.get_vid_features(video_tensor)
    
    # Get text features (with caching)
    if text not in text_feat_cache:
        text_feat_cache[text] = model.get_text_features(text, tokenizer, {})
    text_feat = text_feat_cache[text]
    
    # Compute cosine similarity
    vid_feat = vid_feat / vid_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    similarity = (vid_feat @ text_feat.T).item()
    
    return similarity, text_feat_cache


# ============================================================================
# ABLATION_TASKS: Modify this list to control which videos are evaluated.
# Only videos matching tasks in this list will be processed.
# Comment out or remove tasks you don't want to evaluate.
# ============================================================================
ABLATION_TASKS = [
    # CONTENT
    # {"base": "A video of a frog jumping.", "ablation": "A video of a toad jumping."},
    {"base": "A video of a monkey climbing.", "ablation": "A video of a squirrel climbing."},
    {"base": "A video of a lion roaring.", "ablation": "A video of a bear roaring."},
    #{"base": "A video of a rocket launching.", "ablation": "A video of a spaceship launching."},
    {"base": "A video of a spider crawling.", "ablation": "A video of a crab crawling."},
    {"base": "A video of a dog running.", "ablation": "A video of a cat running."},
    {"base": "A video of a car driving.", "ablation": "A video of a truck driving."},
    {"base": "A video of a plane flying.", "ablation": "A video of a helicopter flying."},
    {"base": "A video of a sailboat cruising.", "ablation": "A video of a speedboat cruising."},
    {"base": "A video of a fish swimming.", "ablation": "A video of a turtle swimming."},
    # {"base": "A video of a man reading.", "ablation": "A video of a woman reading."},
    {"base": "A video of an astronaut walking.", "ablation": "A video of a robot walking."},
    #{"base": "A video of a train moving.", "ablation": "A video of a bus moving."},
    {"base": "A video of a penguin walking.", "ablation": "A video of a duck walking."},
    # {"base": "A video of a sword swinging.", "ablation": "A video of a stick swinging."},
    # STYLE
    {"base": "A video of a black chair spinning.", "ablation": "A video of a white chair spinning."},
    {"base": "A video of a red light flashing.", "ablation": "A video of a blue light flashing."},
    {"base": "A video of a white bird flying.", "ablation": "A video of a black bird flying."},
    {"base": "A video of a pink flower blooming.", "ablation": "A video of a white flower blooming."},
    {"base": "A video of a black cat sleeping.", "ablation": "A video of a brown cat sleeping."},
    {"base": "A video of a white cup falling.", "ablation": "A video of a black cup falling."},
    {"base": "A video of a blue sedan driving.", "ablation": "A video of a red sedan driving."},
    {"base": "A video of a pink rose blooming.", "ablation": "A video of a yellow rose blooming."},
    {"base": "A video of a white flag waving in the wind.", "ablation": "A video of a black flag waving in the wind."},
    {"base": "A video of a blue flag waving.", "ablation": "A video of a red flag waving."},
    # MOTION
    # {"base": "A video of a ball rolling left.", "ablation": "A video of a ball rolling right."},
    {"base": "A video of a car driving forward.", "ablation": "A video of a car driving backward."},
    # {"base": "A video of a monkey walking.", "ablation": "A video of a monkey climbing."},
    # {"base": "A video of a door swinging open.", "ablation": "A video of a door swinging closed."},
    {"base": "A video of a person swimming forward.", "ablation": "A video of a person floating still."},
    # {"base": "A video of a man walking forward toward the camera.", "ablation": "A video of a man walking backward away from the camera."},
    # {"base": "A video shot of a bird walking.", "ablation": "A video shot of a bird flying."},
    {"base": "A video of a person jumping up.", "ablation": "A video of a person falling down."},
    # {"base": "A video of a train moving left.", "ablation": "A video of a train moving right."},
    {"base": "A video of a ball bouncing up.", "ablation": "A video of a ball rolling forward."},
    {"base": "A video of a dog standing still.", "ablation": "A video of a dog running fast."},
    {"base": "A video of a man jumping up.", "ablation": "A video of a man crouching down."},
    {"base": "A video of a baby crawling on a floor.", "ablation": "A video of a baby walking on a floor."},
    {"base": "A video of a man walking forward toward the camera.", "ablation": "A video of a man walking backward away from the camera."},
    {"base": "A video of a woman walking along a beach.", "ablation": "A video of a woman running along a beach."},
    {"base": "A video shot of a bird walking.", "ablation": "A video shot of a bird flying."},
]


def format_filename_string(text):
    """Cleans a prompt to be used as filename (mirrors ablation script)."""
    clean = re.sub(r'[^\w\s]', '', text.lower())
    return "_".join(clean.split())[:50]


def build_prompt_lookup():
    """Build lookup table from truncated filename parts to full prompts."""
    lookup = {}
    for task in ABLATION_TASKS:
        base_key = format_filename_string(task["base"])
        ablation_key = format_filename_string(task["ablation"])
        combined_key = f"{base_key}-{ablation_key}"
        lookup[combined_key] = (task["base"], task["ablation"])
    return lookup


PROMPT_LOOKUP = build_prompt_lookup()


def parse_filename_to_prompts(filename):
    """
    Parse video filename to extract base and ablation prompts.
    Uses lookup table to handle truncated filenames correctly.
    """
    stem = Path(filename).stem
    
    # Try exact match first
    if stem in PROMPT_LOOKUP:
        return PROMPT_LOOKUP[stem]
    
    # Try prefix matching for truncated filenames
    for key, (base, ablation) in PROMPT_LOOKUP.items():
        if stem.startswith(key[:len(stem)]) or key.startswith(stem):
            return base, ablation
    
    # Fallback: parse from filename directly
    parts = stem.split('-')
    
    for i in range(1, len(parts)):
        potential_ablation = '-'.join(parts[i:])
        if potential_ablation.startswith('a_video'):
            base_clean = '-'.join(parts[:i])
            ablation_clean = potential_ablation
            
            base_prompt = base_clean.replace('_', ' ').strip()
            ablation_prompt = ablation_clean.replace('_', ' ').strip()
            
            base_prompt = base_prompt[0].upper() + base_prompt[1:] + "."
            ablation_prompt = ablation_prompt[0].upper() + ablation_prompt[1:] + "."
            
            return base_prompt, ablation_prompt
    
    # Last resort fallback
    mid = len(parts) // 2
    base_clean = '-'.join(parts[:mid])
    ablation_clean = '-'.join(parts[mid:])
    
    base_prompt = base_clean.replace('_', ' ').strip()
    ablation_prompt = ablation_clean.replace('_', ' ').strip()
    base_prompt = base_prompt[0].upper() + base_prompt[1:] + "."
    ablation_prompt = ablation_prompt[0].upper() + ablation_prompt[1:] + "."
    
    return base_prompt, ablation_prompt


def get_expected_filenames():
    """Get set of expected filename stems from ABLATION_TASKS."""
    expected = set()
    for task in ABLATION_TASKS:
        base_key = format_filename_string(task["base"])
        ablation_key = format_filename_string(task["ablation"])
        expected.add(f"{base_key}-{ablation_key}")
    return expected


def collect_video_files(ablation_dir, filter_by_tasks=True):
    """Collect video files organized by category and block config.
    
    Args:
        ablation_dir: Directory containing ablation outputs
        filter_by_tasks: If True, only include videos matching ABLATION_TASKS
    """
    videos = defaultdict(lambda: defaultdict(list))
    expected_filenames = get_expected_filenames() if filter_by_tasks else None
    
    for category in ['content', 'motion', 'style']:
        category_dir = Path(ablation_dir) / category
        if not category_dir.exists():
            continue
            
        for block_dir in sorted(category_dir.iterdir()):
            if not block_dir.is_dir():
                continue
            
            block_config = block_dir.name  # e.g., "5-1", "10-2"
            
            for video_file in sorted(block_dir.glob("*.mp4")):
                # Filter by ABLATION_TASKS if enabled
                if filter_by_tasks and expected_filenames:
                    stem = video_file.stem
                    # Check if this video matches any expected filename (handle truncation)
                    match = any(
                        stem == exp or stem.startswith(exp) or exp.startswith(stem)
                        for exp in expected_filenames
                    )
                    if not match:
                        continue
                videos[category][block_config].append(video_file)
    
    return videos


def evaluate_videos(ablation_dir, output_dir, device="cuda"):
    """Main evaluation function."""
    
    # Load model
    model, tokenizer = load_viclip_model(device)
    
    # Collect videos
    print(f"\nCollecting videos from {ablation_dir}...")
    videos = collect_video_files(ablation_dir)
    
    total_videos = sum(
        len(files) 
        for category_videos in videos.values() 
        for files in category_videos.values()
    )
    print(f"Found {total_videos} videos across {len(videos)} categories")
    
    # Results storage
    results = []
    text_feat_cache = {}
    
    # Process all videos
    with torch.no_grad():
        for category in ['content', 'motion', 'style']:
            if category not in videos:
                continue
                
            print(f"\n{'='*60}")
            print(f"Processing {category.upper()} category")
            print(f"{'='*60}")
            
            for block_config in sorted(videos[category].keys(), 
                                       key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]))):
                print(f"\n  Block config: {block_config}")
                
                for video_path in tqdm(videos[category][block_config], 
                                       desc=f"    {block_config}", leave=False):
                    
                    # Parse prompts from filename
                    base_prompt, ablation_prompt = parse_filename_to_prompts(video_path.name)
                    
                    # Extract 8 uniformly sampled frames (ViCLIP requirement)
                    frames = extract_frames_from_video(str(video_path), max_frames=8)
                    if len(frames) == 0:
                        print(f"    Warning: No frames in {video_path.name}")
                        continue
                    
                    video_tensor = frames_to_tensor(frames, device=device)
                    
                    # Compute similarity to ablation prompt only
                    sim_ablation, text_feat_cache = compute_similarity(
                        video_tensor, ablation_prompt, model, tokenizer, text_feat_cache
                    )
                    
                    # Parse block config
                    stride, block_id = block_config.split('-')
                    
                    results.append({
                        'category': category,
                        'stride': int(stride),
                        'block_id': int(block_id),
                        'block_config': block_config,
                        'video_file': video_path.name,
                        'ablation_prompt': ablation_prompt,
                        'sim_ablation': sim_ablation,
                        'num_frames': len(frames),
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / "viclip_scores_detailed.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Generate summary statistics
    generate_summary(df, output_dir)
    
    return df


def generate_summary(df, output_dir):
    """Generate summary statistics and comparison tables."""
    
    output_dir = Path(output_dir)
    
    # Summary by category and block config
    summary = df.groupby(['category', 'stride', 'block_id']).agg({
        'sim_ablation': ['mean', 'std', 'count'],
        'num_frames': ['mean'],
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    summary_path = output_dir / "viclip_summary_by_block.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary by block saved to: {summary_path}")
    
    # Generate markdown report
    report_lines = [
        "# ViCLIP Ablation Evaluation Report",
        "",
        "## Overview",
        f"- Total videos evaluated: {len(df)}",
        f"- Categories: {', '.join(df['category'].unique())}",
        f"- Stride configurations: {sorted(df['stride'].unique())}",
        f"- Average frames per video: {df['num_frames'].mean():.1f}",
        "",
        "## Metric",
        "- **sim_ablation**: Similarity between video and ablation prompt",
        "- Higher score = video content more closely matches the ablation prompt",
        "",
    ]
    
    # Results by category
    for category in ['content', 'motion', 'style']:
        cat_df = df[df['category'] == category]
        if len(cat_df) == 0:
            continue
            
        report_lines.extend([
            f"## {category.upper()} Ablation Results",
            "",
        ])
        
        # Pivot table for this category
        for stride in sorted(cat_df['stride'].unique()):
            stride_df = cat_df[cat_df['stride'] == stride]
            
            report_lines.append(f"### Stride {stride}")
            report_lines.append("")
            report_lines.append("| Block ID | Sim(Ablation) | Std | n |")
            report_lines.append("|----------|---------------|-----|---|")
            
            for block_id in sorted(stride_df['block_id'].unique()):
                block_df = stride_df[stride_df['block_id'] == block_id]
                
                sim_ablation_mean = block_df['sim_ablation'].mean()
                sim_ablation_std = block_df['sim_ablation'].std()
                n = len(block_df)
                
                report_lines.append(
                    f"| {block_id} | {sim_ablation_mean:.4f} | {sim_ablation_std:.4f} | {n} |"
                )
            
            report_lines.append("")
        
        # Category summary
        report_lines.append(f"**{category.upper()} Summary:**")
        report_lines.append(f"- Mean sim_ablation: {cat_df['sim_ablation'].mean():.4f} (±{cat_df['sim_ablation'].std():.4f})")
        report_lines.append("")
    
    # Cross-category comparison
    report_lines.extend([
        "## Cross-Category Comparison",
        "",
        "### Mean Ablation Similarity by Block (Stride 5)",
        "",
        "| Block | Content | Motion | Style |",
        "|-------|---------|--------|-------|",
    ])
    
    stride5_df = df[df['stride'] == 5]
    for block_id in sorted(stride5_df['block_id'].unique()):
        row = f"| {block_id} |"
        for cat in ['content', 'motion', 'style']:
            block_cat_df = stride5_df[(stride5_df['block_id'] == block_id) & 
                                       (stride5_df['category'] == cat)]
            if len(block_cat_df) > 0:
                row += f" {block_cat_df['sim_ablation'].mean():.4f} |"
            else:
                row += " N/A |"
        report_lines.append(row)
    
    report_lines.append("")
    
    # Overall statistics
    report_lines.extend([
        "## Overall Statistics",
        "",
        f"- Mean sim_ablation: {df['sim_ablation'].mean():.4f} (±{df['sim_ablation'].std():.4f})",
        "",
        "### Interpretation",
        "- **Higher sim_ablation**: Video content more closely matches the ablation prompt",
        "- Blocks with **higher scores** show stronger response to ablation prompts",
        "- Compare scores across blocks to identify which DiT blocks control each semantic aspect",
    ])
    
    report_path = output_dir / "viclip_evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print("\nMean Ablation Similarity by Block:")
    print("-" * 50)
    
    pivot = df.pivot_table(
        values='sim_ablation', 
        index=['stride', 'block_id'], 
        columns='category',
        aggfunc='mean'
    ).round(4)
    print(pivot.to_string())
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ablation videos using ViCLIP video-text similarity"
    )
    parser.add_argument(
        "--ablation_dir", 
        type=str, 
        default="ablation_runs2",
        help="Directory containing ablation video outputs"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="viclip_evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run evaluation on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_videos(
        ablation_dir=args.ablation_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
