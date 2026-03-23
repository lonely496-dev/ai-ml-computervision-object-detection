#!/usr/bin/env python
"""
Thin wrapper script for src/face_detector.py.
Allows calling from project root while preserving backward compatibility.

Usage:
    python run_face_detection.py video.mp4 --output result.mp4
    python run_face_detection.py image.jpg --output result.jpg --display
    python run_face_detection.py video.mp4 --analyze
"""
import sys
import os

# Add src/ to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Import the main detection function and run with command-line args
    import argparse
    from src.face_detector import (
        detect_faces_in_video,
        detect_faces_in_image,
        analyze_video_faces
    )

    parser = argparse.ArgumentParser(description="Face Detection in Videos/Images")
    parser.add_argument("input", help="Input video or image file")
    parser.add_argument("--output", help="Output file path (save annotated result)")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Detection confidence threshold (0-1)")
    parser.add_argument("--display", action="store_true", help="Display result in window")
    parser.add_argument("--analyze", action="store_true", help="Show video statistics only")

    args = parser.parse_args()

    # Check if input is video or image
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))

    try:
        if is_video:
            print(f"🎬 Processing video: {args.input}")
            
            if args.analyze:
                print("📊 Analyzing video (statistics only, no output saved)...\n")
                stats = analyze_video_faces(args.input, args.confidence)
                print("\n" + "="*60)
                print("📊 VIDEO FACE ANALYSIS RESULTS")
                print("="*60)
                print(f"  Total frames:       {stats['total_frames']}")
                print(f"  Frames with faces:  {stats['frames_with_faces']}")
                print(f"  Total detections:   {stats['face_detections_total']}")
                print(f"  Avg per frame:      {stats['face_per_frame_avg']:.2f}")
                print(f"  Confidence min:     {stats['confidence_stats']['min']:.2f}")
                print(f"  Confidence max:     {stats['confidence_stats']['max']:.2f}")
                print(f"  Confidence avg:     {stats['confidence_stats']['avg']:.2f}")
                print("="*60)
            else:
                print(f"Processing video with confidence threshold: {args.confidence}\n")
                detect_faces_in_video(
                    args.input,
                    output_path=args.output,
                    confidence_threshold=args.confidence,
                    display=args.display
                )
                if args.output:
                    print(f"\n✅ Face-detected video saved: {args.output}")
        else:
            print(f"📷 Processing image: {args.input}")
            print(f"Confidence threshold: {args.confidence}\n")
            
            detect_faces_in_image(
                args.input,
                output_path=args.output,
                confidence_threshold=args.confidence,
                display=args.display
            )
            
            if args.output:
                print(f"\n✅ Face-detected image saved: {args.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    print("\n✓ Complete!")
