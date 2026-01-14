"""
Extract frames from video at specified interval
Usage: python extract_frames.py
"""

import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir=None, interval_ms=4, start_time=0, end_time=None):
    """
    Extract frames from video at specified interval
    

        video_path: Path to video file
        output_dir: Directory to save frames (default: frames/)
        interval_ms: Interval between frames in milliseconds
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (None = end of video)
    """
    print(f"{'='*60}")
    print(f"Video Frame Extractor")
    print(f"{'='*60}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Set output directory
    if output_dir is None:
        video_name = Path(video_path).stem
        output_dir = f"frames_{video_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error opening video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Video Info:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Interval: {interval_ms} ms ({interval_ms/1000:.3f} seconds)")
    
    # Calculate frame interval
    # interval_ms is in milliseconds, convert to frame number
    frame_interval = int(fps * (interval_ms / 1000.0))
    if frame_interval < 1:
        frame_interval = 1
    
    print(f"   Frame interval: Every {frame_interval} frame(s)")
    
    # Set start position
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Set end position
    end_frame = total_frames
    if end_time is not None:
        end_frame = int(end_time * fps)
    
    print(f"\n🔄 Extracting frames...")
    print(f"   Start frame: {start_frame}")
    print(f"   End frame: {end_frame}")
    
    frame_count = 0
    saved_count = 0
    current_frame = start_frame
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame at interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{current_frame:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"   Saved {saved_count} frames...")
        
        frame_count += 1
        current_frame += 1
    
    cap.release()
    
    print(f"\n✅ Extraction complete!")
    print(f"   Total frames extracted: {saved_count}")
    print(f"   Saved to: {output_dir}/")
    
    # Estimate output size
    if saved_count > 0:
        sample_file = os.path.join(output_dir, f"frame_{start_frame:06d}.jpg")
        if os.path.exists(sample_file):
            file_size = os.path.getsize(sample_file) / (1024 * 1024)  # MB
            total_size = file_size * saved_count
            print(f"   Estimated total size: {total_size:.2f} MB")

def main():
    """Main function"""
    # Video path
    video_path = "ownpics/mi0.1.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print("\nPlease check the video path.")
        return
    
    print("🎬 Video Frame Extractor")
    print("="*60)
    
    # Get user preferences
    print(f"\nVideo: {video_path}")
    
    try:
        interval_input = input("Enter interval in milliseconds (default: 4): ").strip()
        interval_ms = float(interval_input) if interval_input else 4.0
        
        start_input = input("Start time in seconds (default: 0): ").strip()
        start_time = float(start_input) if start_input else 0.0
        
        end_input = input("End time in seconds (default: full video, press Enter): ").strip()
        end_time = float(end_input) if end_input else None
        
        output_input = input("Output directory (default: frames_mi0.1): ").strip()
        output_dir = output_input if output_input else None
        
    except ValueError:
        print("⚠️  Invalid input, using defaults...")
        interval_ms = 4.0
        start_time = 0.0
        end_time = None
        output_dir = None
    
    # Extract frames
    extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        interval_ms=interval_ms,
        start_time=start_time,
        end_time=end_time
    )

if __name__ == "__main__":
    main()
