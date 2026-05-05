#!/usr/bin/env python3
"""
Preprocess videos for LoRA training.
- Resizes to 832x480
- Trims to 5 seconds
- Saves to lora_videos folder
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_video_duration(video_path: Path) -> float:
    """Get the duration of a video in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr}")
    return float(result.stdout.strip())


def preprocess_video(
    input_path: Path,
    output_path: Path,
    width: int = 832,
    height: int = 480,
    duration: float = 5.0,
    center_crop: bool = True
) -> bool:
    """
    Preprocess a single video: resize and trim.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        width: Target width
        height: Target height
        duration: Target duration in seconds
        center_crop: If True, center crop to target aspect ratio; if False, pad with black bars
    
    Returns:
        True if successful, False otherwise
    """
    target_aspect = width / height
    
    if center_crop:
        # Scale and center crop to exact dimensions
        filter_complex = (
            f"scale=w='if(gt(a,{target_aspect}),{height}*a,{width})':h='if(gt(a,{target_aspect}),{height},{width}/a)',"
            f"crop={width}:{height}"
        )
    else:
        # Scale and pad to fit (letterbox/pillarbox)
        filter_complex = (
            f"scale=w='min({width},a*{height})':h='min({height},{width}/a)',"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", str(input_path),
        "-t", str(duration),  # Limit duration
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",  # High quality
        "-an",  # Remove audio (not needed for training)
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for LoRA training")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("lora_videos_unprocessed"),
        help="Input directory containing raw videos"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lora_videos"),
        help="Output directory for processed videos"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Target width (default: 832)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Target height (default: 480)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Target duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        help="Pad videos instead of center cropping to preserve full content"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without processing"
    )
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    videos = [
        f for f in args.input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if not videos:
        print(f"No video files found in '{args.input_dir}'")
        sys.exit(0)
    
    print(f"Found {len(videos)} videos to process")
    print(f"Target: {args.width}x{args.height}, {args.duration}s")
    print(f"Mode: {'Pad' if args.pad else 'Center crop'}")
    print(f"Output: {args.output_dir}")
    print("-" * 50)
    
    success_count = 0
    fail_count = 0
    
    for i, video in enumerate(sorted(videos), 1):
        output_path = args.output_dir / video.name
        
        if args.dry_run:
            print(f"[{i}/{len(videos)}] Would process: {video.name}")
            continue
        
        print(f"[{i}/{len(videos)}] Processing: {video.name}")
        
        try:
            original_duration = get_video_duration(video)
            print(f"  Original duration: {original_duration:.2f}s")
        except Exception as e:
            print(f"  Warning: Could not get duration: {e}")
            original_duration = None
        
        if preprocess_video(
            video,
            output_path,
            width=args.width,
            height=args.height,
            duration=args.duration,
            center_crop=not args.pad
        ):
            success_count += 1
            print(f"  Saved: {output_path}")
        else:
            fail_count += 1
            print(f"  FAILED: {video.name}")
    
    if not args.dry_run:
        print("-" * 50)
        print(f"Done! Processed {success_count}/{len(videos)} videos successfully")
        if fail_count > 0:
            print(f"Failed: {fail_count} videos")


if __name__ == "__main__":
    main()
