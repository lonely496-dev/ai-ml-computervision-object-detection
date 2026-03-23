"""
pose_classifier.py
------------------
Classifies yoga / exercise poses from detected MediaPipe landmarks
by computing joint angles and applying geometric heuristics.

Supported poses
───────────────
  • Warrior II  (Virabhadrasana II)
  • T Pose      (reference / bind pose)
  • Tree Pose   (Vrikshasana)
"""

import math
import cv2
import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def calculate_angle(landmark1: tuple,
                    landmark2: tuple,
                    landmark3: tuple) -> float:
    """
    Calculate the angle (in degrees) at landmark2 formed by the
    rays landmark2→landmark1 and landmark2→landmark3.

    Args:
        landmark1 : (x, y, z) of the first point.
        landmark2 : (x, y, z) of the vertex point.
        landmark3 : (x, y, z) of the third point.

    Returns:
        Angle in degrees, range [0, 360).
    """
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
    )

    if angle < 0:
        angle += 360

    return angle


# ─────────────────────────────────────────────────────────────────────────────
# Pose angle extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pose_angles(landmarks: list) -> dict:
    """
    Compute all joint angles needed for pose classification.

    Args:
        landmarks : List of (x, y, z) tuples from detect_pose().

    Returns:
        Dictionary of joint-name → angle (degrees).
    """
    L = mp_pose.PoseLandmark  # shorthand

    return {
        "left_elbow":    calculate_angle(
            landmarks[L.LEFT_SHOULDER.value],
            landmarks[L.LEFT_ELBOW.value],
            landmarks[L.LEFT_WRIST.value],
        ),
        "right_elbow":   calculate_angle(
            landmarks[L.RIGHT_SHOULDER.value],
            landmarks[L.RIGHT_ELBOW.value],
            landmarks[L.RIGHT_WRIST.value],
        ),
        "left_shoulder": calculate_angle(
            landmarks[L.LEFT_ELBOW.value],
            landmarks[L.LEFT_SHOULDER.value],
            landmarks[L.LEFT_HIP.value],
        ),
        "right_shoulder": calculate_angle(
            landmarks[L.RIGHT_HIP.value],
            landmarks[L.RIGHT_SHOULDER.value],
            landmarks[L.RIGHT_ELBOW.value],
        ),
        "left_knee":     calculate_angle(
            landmarks[L.LEFT_HIP.value],
            landmarks[L.LEFT_KNEE.value],
            landmarks[L.LEFT_ANKLE.value],
        ),
        "right_knee":    calculate_angle(
            landmarks[L.RIGHT_HIP.value],
            landmarks[L.RIGHT_KNEE.value],
            landmarks[L.RIGHT_ANKLE.value],
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pose classification rules
# ─────────────────────────────────────────────────────────────────────────────

def _arms_extended(angles: dict) -> bool:
    """Both elbows ~180° and both shoulders ~90°."""
    return (
        165 < angles["left_elbow"]    < 195
        and 165 < angles["right_elbow"]   < 195
        and 80  < angles["left_shoulder"] < 110
        and 80  < angles["right_shoulder"] < 110
    )


def classify_pose(landmarks: list,
                  output_image: np.ndarray,
                  display: bool = False):
    """
    Classify the yoga pose of the person in the image.

    Args:
        landmarks    : List of (x, y, z) from detect_pose().
        output_image : Image to annotate with the pose label.
        display      : If True, display the annotated image.

    Returns:
        output_image : Annotated image.
        label        : Detected pose label string.
    """
    label = "Unknown Pose"
    color = (0, 0, 255)      # red = unknown

    if not landmarks:
        cv2.putText(output_image, label, (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        return output_image, label

    angles = extract_pose_angles(landmarks)

    lk = angles["left_knee"]
    rk = angles["right_knee"]

    # ── Warrior II ────────────────────────────────────────────────────────────
    # Arms extended + one knee straight (~180°) + other knee bent (~90–120°)
    if _arms_extended(angles):
        straight_knee = (165 < lk < 195) or (165 < rk < 195)
        bent_knee     = (90  < lk < 120) or (90  < rk < 120)

        if straight_knee and bent_knee:
            label = "Warrior II Pose"

        # ── T Pose ────────────────────────────────────────────────────────────
        # Arms extended + both knees straight
        elif (160 < lk < 195) and (160 < rk < 195):
            label = "T Pose"

    # ── Tree Pose ─────────────────────────────────────────────────────────────
    # One knee straight + other knee deeply bent (~25–45° or ~315–335°)
    one_straight = (165 < lk < 195) or (165 < rk < 195)
    one_bent     = (315 < lk < 335) or (25  < rk < 45)

    if one_straight and one_bent:
        label = "Tree Pose"

    # ── Annotate ──────────────────────────────────────────────────────────────
    if label != "Unknown Pose":
        color = (0, 255, 0)   # green = recognised

    cv2.putText(output_image, label, (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    if display:
        # Lazy import to avoid heavy startup overhead when display isn't used
        import matplotlib.pyplot as plt

        plt.figure(figsize=[8, 8])
        plt.imshow(output_image[:, :, ::-1])
        plt.title(f"Classified: {label}")
        plt.axis("off")
        plt.show()
        return

    return output_image, label
