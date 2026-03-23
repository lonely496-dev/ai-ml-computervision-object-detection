"""
run_realtime.py
---------------
Real-time pose detection + classification on webcam feed or video file.
Displays FPS, detected pose label, and landmark overlay.

Usage
─────
    # Webcam
    python run_realtime.py

    # Video file
    python run_realtime.py --source media/running.mp4

    # Save output video
    python run_realtime.py --source media/running.mp4 --save

    # Low-power mode (smaller model + smaller input + frame skipping)
    python run_realtime.py --low_power --frame_step 2
"""

import argparse
import os
import sys
from time import time

import cv2
import mediapipe as mp

from pose_detector   import detect_pose
from pose_classifier import classify_pose

mp_pose = mp.solutions.pose


def run(source,
        save: bool = False,
        output_path: str = "output/realtime_output.mp4",
        low_power: bool = False,
        frame_step: int = 1) -> None:
    """
    Run pose detection + classification on a video source.

    Args:
        source      : 0 for webcam, or a path string for video file.
        save        : Whether to write annotated output to disk.
        output_path : File path for saved video.
        low_power   : Use lighter model and smaller input.
        frame_step  : Process every N-th frame (1 = every frame).
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    # Video writer (optional)
    writer = None
    if save:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps_out, (w, h))
        print(f"Saving output to: {output_path}")

    # pick a lighter model on low-power machines
    model_complexity = 0 if low_power else 1

    pose = mp_pose.Pose(
        static_image_mode        = False,
        model_complexity         = model_complexity,
        smooth_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

    cv2.namedWindow("Pose Detection — Press ESC to quit",
                    cv2.WINDOW_NORMAL)

    prev_time = 0
    frame_count = 0
    last_annotated = None

    print("Running... Press ESC to stop.\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # Flip for natural selfie-view (webcam only)
        if source == 0:
            frame = cv2.flip(frame, 1)

        # Resize: keep aspect ratio, use a smaller height on low-power mode
        h, w, _ = frame.shape
        target_h = 360 if low_power else 640
        frame = cv2.resize(frame, (int(w * target_h / h), target_h))

        # Frame skipping: process every N-th frame (saves CPU). Reuse last
        # annotated frame for intermediate displays to reduce processing.
        frame_count += 1
        if frame_count % max(1, frame_step) == 0:
            annotated, landmarks = detect_pose(frame, pose, draw=True)
            if landmarks:
                annotated, label = classify_pose(landmarks, annotated)
            last_annotated = annotated
            display_frame = annotated
        else:
            display_frame = last_annotated if last_annotated is not None else frame

        # FPS overlay
        curr_time = time()
        if (curr_time - prev_time) > 0:
            fps = 1.0 / (curr_time - prev_time)
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 65),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        prev_time = curr_time

        cv2.imshow("Pose Detection — Press ESC to quit", display_frame)

        if writer:
            writer.write(display_frame)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

    cap.release()
    pose.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time pose detection on webcam or video."
    )
    parser.add_argument(
        "--source", default=0,
        help="Webcam index (default: 0) or path to video file."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated output video."
    )
    parser.add_argument(
        "--output", default="output/realtime_output.mp4",
        help="Output video path when --save is used."
    )
    parser.add_argument(
        "--low_power", action="store_true",
        help="Run in low-power mode: smaller input, lighter model."
    )
    parser.add_argument(
        "--frame_step", type=int, default=1,
        help="Process every N-th frame (1 = every frame). Use higher values to save CPU."
    )
    args = parser.parse_args()

    # Convert numeric string → int for webcam index
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    run(source, args.save, args.output, args.low_power, args.frame_step)


if __name__ == "__main__":
    main()
