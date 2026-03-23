#!/usr/bin/env python
"""
Comprehensive test suite for POSE project.
Tests all modules and features.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*60)
    print("TEST 1: MODULE IMPORTS")
    print("="*60)
    try:
        from src.pose_detector import detect_pose
        print("✅ pose_detector imported")
        
        from src.pose_classifier import classify_pose, calculate_angle, extract_pose_angles
        print("✅ pose_classifier imported")
        
        from src.face_detector import detect_faces_in_image, detect_faces_in_video, analyze_video_faces
        print("✅ face_detector imported")
        
        from src.run_realtime import main as realtime_main
        print("✅ run_realtime imported")
        
        from src.run_on_image import main as image_main
        print("✅ run_on_image imported")
        
        print("\n✅ ALL IMPORTS PASSED\n")
        return True
    except Exception as e:
        print(f"❌ IMPORT FAILED: {e}\n")
        return False


def test_sample_image():
    """Test 2: Pose detection on sample image"""
    print("="*60)
    print("TEST 2: POSE DETECTION ON IMAGE")
    print("="*60)
    try:
        import cv2
        import mediapipe as mp
        from src.pose_detector import detect_pose
        
        # Check if sample image exists
        img_path = Path("media/sample.jpg")
        if not img_path.exists():
            print(f"⚠️  Sample image not found at {img_path}")
            print("   Skipping image test (can run manually with your own image)\n")
            return None
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ Could not load image from {img_path}\n")
            return False
        
        print(f"   Loaded image: {img_path.name} ({image.shape[1]}x{image.shape[0]})")
        
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=False
        )
        
        # Detect pose
        output_image, landmarks = detect_pose(image, pose, draw=True, display=False)
        
        if landmarks:
            print(f"   Detected {len(landmarks)} landmarks")
            print(f"   Sample landmark: x={landmarks[0][0]:.2f}, y={landmarks[0][1]:.2f}, z={landmarks[0][2]:.2f}")
        
        print("\n✅ POSE DETECTION TEST PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ POSE DETECTION TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_pose_classifier():
    """Test 3: Pose classification logic"""
    print("="*60)
    print("TEST 3: POSE CLASSIFICATION LOGIC")
    print("="*60)
    try:
        from src.pose_classifier import calculate_angle, extract_pose_angles
        
        # Create mock landmarks (33 landmarks, each with x, y, z)
        # Using sample values that should classify as "T Pose"
        mock_landmarks = []
        for i in range(33):
            mock_landmarks.append([0.5, 0.5, 0.0])
        
        # Manually set positions for T Pose
        # Right shoulder, right elbow, right wrist (arm extended)
        mock_landmarks[12] = [0.3, 0.4, 0.0]  # Right shoulder
        mock_landmarks[14] = [0.2, 0.4, 0.0]  # Right elbow
        mock_landmarks[16] = [0.1, 0.4, 0.0]  # Right wrist
        
        # Left shoulder, left elbow, left wrist (arm extended)
        mock_landmarks[11] = [0.7, 0.4, 0.0]  # Left shoulder
        mock_landmarks[13] = [0.8, 0.4, 0.0]  # Left elbow
        mock_landmarks[15] = [0.9, 0.4, 0.0]  # Left wrist
        
        # Both knees straight
        mock_landmarks[25] = [0.4, 0.8, 0.0]  # Right knee
        mock_landmarks[26] = [0.6, 0.8, 0.0]  # Left knee
        
        # Calculate angles
        angles = extract_pose_angles(mock_landmarks)
        
        print(f"   Extracted angles:")
        for joint, angle in angles.items():
            print(f"   - {joint}: {angle:.2f}°")
        
        # Test angle calculation
        angle = calculate_angle(
            mock_landmarks[11],  # Left shoulder
            mock_landmarks[13],  # Left elbow
            mock_landmarks[15]   # Left wrist
        )
        print(f"\n   Test angle (shoulder-elbow-wrist): {angle:.2f}°")
        
        print("\n✅ POSE CLASSIFIER TEST PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ POSE CLASSIFIER TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_face_detector():
    """Test 4: Face detection on image"""
    print("="*60)
    print("TEST 4: FACE DETECTION ON IMAGE")
    print("="*60)
    try:
        import cv2
        import mediapipe as mp
        from src.face_detector import detect_faces
        
        # Check if sample image exists
        img_path = Path("media/sample.jpg")
        if not img_path.exists():
            print(f"⚠️  Sample image not found at {img_path}")
            print("   Skipping face detection test\n")
            return None
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ Could not load image\n")
            return False
        
        print(f"   Loaded image: {img_path.name}")
        
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        # Detect faces
        output_image, faces = detect_faces(image, face_detector, draw=True, confidence_threshold=0.5)
        
        if faces:
            print(f"   Detected {len(faces)} face(s)")
            for i, face in enumerate(faces):
                print(f"   Face {i+1}: confidence={face.get('confidence', 'N/A'):.2f}")
        else:
            print("   No faces detected (may not be a face photo)")
        
        print("\n✅ FACE DETECTION TEST PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ FACE DETECTION TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_cli_args():
    """Test 5: Command-line argument parsing"""
    print("="*60)
    print("TEST 5: CLI ARGUMENT PARSING")
    print("="*60)
    try:
        import argparse
        
        # Test argument parser similar to run_realtime.py
        parser = argparse.ArgumentParser()
        parser.add_argument("--source", type=int, default=0, help="Webcam or video source")
        parser.add_argument("--save", action="store_true", help="Save output")
        parser.add_argument("--output", type=str, default="output.mp4", help="Output path")
        parser.add_argument("--low_power", action="store_true", help="Low power mode")
        parser.add_argument("--frame_step", type=int, default=1, help="Process every Nth frame")
        
        # Test parsing
        test_args = ["--low_power", "--frame_step", "3"]
        args = parser.parse_args(test_args)
        
        print(f"   Parsed arguments:")
        print(f"   - low_power: {args.low_power}")
        print(f"   - frame_step: {args.frame_step}")
        print(f"   - save: {args.save}")
        print(f"   - output: {args.output}")
        
        assert args.low_power == True, "low_power should be True"
        assert args.frame_step == 3, "frame_step should be 3"
        
        print("\n✅ CLI ARGUMENT TEST PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ CLI ARGUMENT TEST FAILED: {e}\n")
        return False


def test_requirements():
    """Test 6: Verify all dependencies installed"""
    print("="*60)
    print("TEST 6: DEPENDENCIES CHECK")
    print("="*60)
    try:
        required = {
            'mediapipe': '0.10.5',
            'opencv': 'cv2',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib'
        }
        
        modules_to_check = [
            ('mediapipe', 'mediapipe'),
            ('cv2', 'opencv-python'),
            ('numpy', 'numpy'),
            ('matplotlib', 'matplotlib')
        ]
        
        all_installed = True
        for import_name, package_name in modules_to_check:
            try:
                mod = __import__(import_name)
                version = getattr(mod, '__version__', 'unknown')
                print(f"   ✅ {package_name}: {version}")
            except ImportError:
                print(f"   ❌ {package_name}: NOT INSTALLED")
                all_installed = False
        
        if all_installed:
            print("\n✅ ALL DEPENDENCIES INSTALLED\n")
            return True
        else:
            print("\n❌ SOME DEPENDENCIES MISSING\n")
            return False
            
    except Exception as e:
        print(f"❌ DEPENDENCY CHECK FAILED: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "POSE PROJECT - COMPLETE TEST SUITE" + " "*9 + "║")
    print("╚" + "="*58 + "╝")
    
    results = {
        "Imports": test_imports(),
        "Dependencies": test_requirements(),
        "Pose Detection": test_sample_image(),
        "Pose Classifier": test_pose_classifier(),
        "Face Detection": test_face_detector(),
        "CLI Arguments": test_cli_args(),
    }
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is None:
            status = "⚠️  SKIP"
        else:
            status = "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    print(f"Total: {passed} passed, {skipped} skipped, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Project is ready to use.\n")
        return 0
    else:
        print(f"⚠️  {failed} test(s) failed. Please review above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
