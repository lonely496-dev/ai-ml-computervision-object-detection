"""
run_on_image.py
---------------
Run pose detection + classification on a single image or a folder of images.

Usage
─────
    python run_on_image.py --input media/sample.jpg
    python run_on_image.py --input media/ --save
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp

from pose_detector   import detect_pose
from pose_classifier import classify_pose

mp_pose = mp.solutions.pose


def process_image(image_path: str,
                  pose,
                  save: bool = False,
                  output_dir: str = "output") -> None:
    """Detect and classify pose in a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    # Detect landmarks
    output_image, landmarks = detect_pose(image, pose, draw=True)

    # Classify pose
    output_image, label = classify_pose(landmarks, output_image)

    print(f"  {os.path.basename(image_path):30s} → {label}")

    # Display
    cv2.imshow("Pose Detection", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save
    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(
            output_dir, "pose_" + os.path.basename(image_path)
        )
        cv2.imwrite(out_path, output_image)
        print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pose detection + classification on images."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to an image file or folder of images."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated output images."
    )
    parser.add_argument(
        "--output_dir", default="output",
        help="Directory to save results (default: output/)."
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum detection confidence (default: 0.5)."
    )
    args = parser.parse_args()

    pose = mp_pose.Pose(
        static_image_mode       = True,
        model_complexity        = 2,
        min_detection_confidence = args.confidence,
    )

    if os.path.isfile(args.input):
        print("\nProcessing image...")
        process_image(args.input, pose, args.save, args.output_dir)

    elif os.path.isdir(args.input):
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [
            f for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in image_exts
        ]
        if not files:
            print("[ERROR] No images found in directory.")
            return

        print(f"\nProcessing {len(files)} image(s) in '{args.input}'...\n")
        for fname in sorted(files):
            process_image(
                os.path.join(args.input, fname),
                pose, args.save, args.output_dir
            )
    else:
        print(f"[ERROR] Path not found: {args.input}")

    pose.close()


if __name__ == "__main__":
    main()
