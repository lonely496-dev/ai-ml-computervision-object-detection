"""
pose_detector.py
----------------
Core module for real-time 3D pose detection using MediaPipe.
Handles landmark detection on images, videos, and webcam feeds.
"""

import math
import cv2
import numpy as np
import mediapipe as mp


# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


# ── Landmark drawing style ────────────────────────────────────────────────────
LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(255, 255, 255), thickness=3, circle_radius=3
)
CONNECTION_STYLE = mp_drawing.DrawingSpec(
    color=(49, 125, 237), thickness=2, circle_radius=2
)


def detect_pose(image: np.ndarray,
                pose,
                draw: bool = False,
                display: bool = False):
    """
    Perform pose landmark detection on an image.

    Args:
        image   : BGR input image (numpy array).
        pose    : Initialized mp.solutions.pose.Pose instance.
        draw    : If True, draw landmarks on the output image.
        display : If True, display side-by-side original vs output.

    Returns:
        output_image : Image with landmarks drawn (if draw=True).
        landmarks    : List of (x, y, z) tuples in pixel coordinates.
    """
    output_image = image.copy()
    image_rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results      = pose.process(image_rgb)

    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        if draw:
            mp_drawing.draw_landmarks(
                image            = output_image,
                landmark_list    = results.pose_landmarks,
                connections      = mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec   = LANDMARK_STYLE,
                connection_drawing_spec = CONNECTION_STYLE,
            )

        for lm in results.pose_landmarks.landmark:
            landmarks.append((
                int(lm.x * width),
                int(lm.y * height),
                lm.z * width,
            ))

    if display:
        # Import matplotlib lazily — it's large and not needed in normal runs
        import matplotlib.pyplot as plt

        plt.figure(figsize=[18, 9])
        plt.subplot(1, 2, 1)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Pose Detected")
        plt.axis("off")

        if results.pose_world_landmarks:
            mp_drawing.plot_landmarks(
                results.pose_world_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )
        plt.tight_layout()
        plt.show()
        return

    return output_image, landmarks
