#!/usr/bin/env python3
"""
Extract frames from ablation_runs2 videos for preliminary qualitative analysis.
For each video file name across 6 blocks (5-1 through 5-6):
- Extract 3 frames per block (initial + 2 with equal stride)
Total: 6 blocks * 3 frames = 18 frames per video file name
"""

import subprocess
import os
from pathlib import Path

# Video file names and their categories
VIDEO_INFO = {
    "a_video_of_a_pink_rose_blooming-a_video_of_a_yellow_rose_blooming.mp4": "style",
    "a_video_of_a_white_flag_waving_in_the_wind-a_video_of_a_black_flag_waving_in_the_wind.mp4": "style",
    "a_video_of_a_lion_roaring-a_video_of_a_bear_roaring.mp4": "content",
    "a_video_of_a_dog_running-a_video_of_a_cat_running.mp4": "content",
    "a_video_of_a_baby_crawling_on_a_floor-a_video_of_a_baby_walking_on_a_floor.mp4": "motion",
    "a_video_shot_of_a_bird_walking-a_video_shot_of_a_bird_flying.mp4": "motion",
}

BLOCKS = ["5-1", "5-2", "5-3", "5-4", "5-5", "5-6"]
BASE_DIR = Path("/users/erluo/scratch/clora-wan/ablation_runs2")
OUTPUT_DIR = Path("/users/erluo/scratch/clora-wan/prelim_qual")

# Frame indices to extract (0, 40, 80 for 81-frame videos with equal stride)
FRAME_INDICES = [0, 40, 80]


def extract_frame(video_path: Path, frame_idx: int, output_path: Path):
    """Extract a single frame from a video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", str(video_path),
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-vframes", "1",
        "-q:v", "2",  # high quality
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error extracting frame {frame_idx} from {video_path}:")
        print(result.stderr)
        return False
    return True


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for video_name, category in VIDEO_INFO.items():
        print(f"\nProcessing: {video_name}")
        
        # Create subfolder for this video (use name without extension)
        video_subfolder = OUTPUT_DIR / video_name.replace(".mp4", "")
        video_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Extract 3 frames from each block
        for block in BLOCKS:
            video_path = BASE_DIR / category / block / video_name
            if not video_path.exists():
                print(f"  WARNING: {video_path} not found!")
                continue
            
            print(f"  Extracting frames from {block}...")
            
            # Extract initial frame (frame 0)
            frame0_path = video_subfolder / f"{block}_frame0.png"
            extract_frame(video_path, FRAME_INDICES[0], frame0_path)
            
            # Extract frame at index 40
            frame1_path = video_subfolder / f"{block}_frame1.png"
            extract_frame(video_path, FRAME_INDICES[1], frame1_path)
            
            # Extract frame at index 80
            frame2_path = video_subfolder / f"{block}_frame2.png"
            extract_frame(video_path, FRAME_INDICES[2], frame2_path)
        
        # Verify frame count
        extracted_frames = list(video_subfolder.glob("*.png"))
        print(f"  Total frames extracted: {len(extracted_frames)}")
    
    print(f"\nDone! Output saved to: {OUTPUT_DIR}")
    
    # Summary
    print("\nSummary:")
    for subfolder in sorted(OUTPUT_DIR.iterdir()):
        if subfolder.is_dir():
            frame_count = len(list(subfolder.glob("*.png")))
            print(f"  {subfolder.name}: {frame_count} frames")


if __name__ == "__main__":
    main()
