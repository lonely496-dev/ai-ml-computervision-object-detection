"""
face_detector.py
----------------
Face detection module using MediaPipe Face Detection.
Detects faces in images, videos, and webcam feeds.
Can output annotated images/videos with face bounding boxes.
"""

import cv2
import numpy as np
import mediapipe as mp


# ── MediaPipe Face Detection setup ────────────────────────────────────────────
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# ── Face drawing style ───────────────────────────────────────────────────────
FACE_BOX_COLOR = (0, 255, 0)      # Green bounding box
FACE_TEXT_COLOR = (0, 255, 0)     # Green text
CONFIDENCE_TEXT_COLOR = (200, 200, 200)  # Light gray


def detect_faces(image: np.ndarray,
                 face_detector,
                 draw: bool = False,
                 confidence_threshold: float = 0.5) -> tuple:
    """
    Detect faces in an image using MediaPipe Face Detection.

    Args:
        image                  : BGR input image (numpy array).
        face_detector          : Initialized mp.solutions.face_detection.FaceDetection instance.
        draw                   : If True, draw bounding boxes on the output image.
        confidence_threshold   : Minimum confidence to consider a face detected (0-1).

    Returns:
        output_image : Image with face boxes drawn (if draw=True).
        faces        : List of detected faces with info:
                      {
                          'bbox': (x_min, y_min, x_max, y_max),
                          'confidence': float,
                          'keypoints': [(x, y), (x, y), ...],  # 6 keypoints per face
                      }
    """
    output_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)

    height, width, _ = image.shape
    faces = []

    if results.detections:
        for detection in results.detections:
            confidence = detection.score[0]

            # Skip low-confidence detections
            if confidence < confidence_threshold:
                continue

            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * width)
            y_min = int(bbox.ymin * height)
            x_max = int((bbox.xmin + bbox.width) * width)
            y_max = int((bbox.ymin + bbox.height) * height)

            # Clamp to image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            # Extract keypoints (nose, left eye, right eye, left ear, right ear, mouth)
            keypoints = []
            if detection.location_data.relative_keypoints:
                for keypoint in detection.location_data.relative_keypoints:
                    kp_x = int(keypoint.x * width)
                    kp_y = int(keypoint.y * height)
                    keypoints.append((kp_x, kp_y))

            # Store face info
            face_info = {
                'bbox': (x_min, y_min, x_max, y_max),
                'confidence': confidence,
                'keypoints': keypoints,
            }
            faces.append(face_info)

            # Draw on image if requested
            if draw:
                # Draw bounding box
                cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max),
                              FACE_BOX_COLOR, 2)

                # Draw keypoints
                for kp in keypoints:
                    cv2.circle(output_image, kp, 4, FACE_TEXT_COLOR, -1)

                # Draw confidence
                conf_text = f"Conf: {confidence:.2f}"
                cv2.putText(output_image, conf_text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIDENCE_TEXT_COLOR, 2)

    return output_image, faces


def detect_faces_in_video(video_path: str,
                         output_path: str = None,
                         confidence_threshold: float = 0.5,
                         display: bool = False) -> list:
    """
    Detect faces in a video file and optionally save annotated video.

    Args:
        video_path             : Path to input video file.
        output_path            : Path to save annotated video (None = don't save).
        confidence_threshold   : Minimum confidence (0-1).
        display                : If True, show frames in real-time (slower).

    Returns:
        all_faces : List of lists, where each inner list contains faces detected in that frame.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer if output path specified
    out_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_faces = []
    frame_count = 0

    with mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = more robust to different faces
            min_detection_confidence=confidence_threshold
    ) as face_detector:

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Detect faces
            annotated_frame, faces = detect_faces(
                frame, face_detector,
                draw=True,
                confidence_threshold=confidence_threshold
            )

            all_faces.append(faces)

            # Write to output video
            if out_writer:
                out_writer.write(annotated_frame)

            # Display if requested
            if display:
                cv2.imshow('Face Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress indicator
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"({frame_count/total_frames*100:.1f}%)")

    cap.release()
    if out_writer:
        out_writer.release()

    print(f"\n✓ Video processing complete!")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with faces: {sum(1 for f in all_faces if f)}")

    if output_path:
        print(f"  Output saved: {output_path}")

    return all_faces


def detect_faces_in_image(image_path: str,
                          output_path: str = None,
                          confidence_threshold: float = 0.5,
                          display: bool = False) -> tuple:
    """
    Detect faces in a single image file.

    Args:
        image_path             : Path to input image.
        output_path            : Path to save annotated image (None = don't save).
        confidence_threshold   : Minimum confidence (0-1).
        display                : If True, display the annotated image.

    Returns:
        output_image : Annotated image.
        faces        : List of detected faces.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image file: {image_path}")

    with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=confidence_threshold
    ) as face_detector:

        output_image, faces = detect_faces(
            image, face_detector,
            draw=True,
            confidence_threshold=confidence_threshold
        )

    if output_path:
        cv2.imwrite(output_path, output_image)
        print(f"✓ Annotated image saved: {output_path}")

    if display:
        cv2.imshow('Face Detection', output_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_image, faces


def analyze_video_faces(video_path: str,
                       confidence_threshold: float = 0.5) -> dict:
    """
    Analyze a video and return statistics about detected faces.

    Args:
        video_path           : Path to video file.
        confidence_threshold : Minimum confidence (0-1).

    Returns:
        stats : Dictionary with analysis results:
               {
                   'total_frames': int,
                   'frames_with_faces': int,
                   'face_per_frame_avg': float,
                   'unique_faces_approx': int,
                   'confidence_stats': {'min': float, 'max': float, 'avg': float},
               }
    """
    all_faces = detect_faces_in_video(
        video_path,
        output_path=None,
        confidence_threshold=confidence_threshold,
        display=False
    )

    total_frames = len(all_faces)
    frames_with_faces = sum(1 for frame_faces in all_faces if frame_faces)
    total_face_detections = sum(len(frame_faces) for frame_faces in all_faces)

    confidences = []
    for frame_faces in all_faces:
        for face in frame_faces:
            confidences.append(face['confidence'])

    stats = {
        'total_frames': total_frames,
        'frames_with_faces': frames_with_faces,
        'face_detections_total': total_face_detections,
        'face_per_frame_avg': total_face_detections / total_frames if total_frames > 0 else 0,
        'unique_faces_approx': frames_with_faces,  # Rough estimate
        'confidence_stats': {
            'min': min(confidences) if confidences else 0,
            'max': max(confidences) if confidences else 0,
            'avg': sum(confidences) / len(confidences) if confidences else 0,
        }
    }

    return stats


if __name__ == "__main__":
    # Example usage:
    # python face_detector.py --input video.mp4 --output result.mp4
    import argparse

    parser = argparse.ArgumentParser(description="Face Detection in Videos/Images")
    parser.add_argument("--input", required=True, help="Input video or image file")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Detection confidence threshold (0-1)")
    parser.add_argument("--display", action="store_true", help="Display result")
    parser.add_argument("--analyze", action="store_true", help="Show statistics only")

    args = parser.parse_args()

    # Check file type
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))

    if is_video:
        if args.analyze:
            stats = analyze_video_faces(args.input, args.confidence)
            print("\n📊 Video Face Analysis:")
            print(f"  Total frames:       {stats['total_frames']}")
            print(f"  Frames with faces:  {stats['frames_with_faces']}")
            print(f"  Total detections:   {stats['face_detections_total']}")
            print(f"  Avg per frame:      {stats['face_per_frame_avg']:.2f}")
            print(f"  Confidence range:   {stats['confidence_stats']['min']:.2f} - "
                  f"{stats['confidence_stats']['max']:.2f}")
            print(f"  Avg confidence:     {stats['confidence_stats']['avg']:.2f}")
        else:
            detect_faces_in_video(
                args.input,
                output_path=args.output,
                confidence_threshold=args.confidence,
                display=args.display
            )
    else:
        detect_faces_in_image(
            args.input,
            output_path=args.output,
            confidence_threshold=args.confidence,
            display=args.display
        )

    print("✓ Done!")
