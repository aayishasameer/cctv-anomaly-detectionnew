#!/usr/bin/env python3
"""
MediaPipe Pose Estimation for Behavior Analysis
Detects: bending, picking, running, fighting, suspicious hand movement
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional

# MediaPipe Pose landmark indices
class PoseLandmark:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26


class PoseEstimator:
    """Full-body pose estimation using MediaPipe for behavior analysis"""
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=light, 1=full, 2=heavy
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.available = True
            print("✅ MediaPipe Pose initialized")
        except Exception as e:
            print(f"⚠️  MediaPipe Pose not available: {e}")
            self.pose = None
            self.available = False
    
    def detect_pose(self, frame: np.ndarray, person_bbox: Optional[List[float]] = None) -> Optional[Dict]:
        """
        Detect pose in frame, optionally cropped to person bounding box.
        
        Args:
            frame: BGR image
            person_bbox: Optional [x1, y1, x2, y2] to crop person region for better accuracy
        
        Returns:
            Dict with landmarks, angles, and behavior flags, or None if no pose detected
        """
        if not self.available or self.pose is None:
            return None
        
        h, w, _ = frame.shape
        
        # Crop to person region if provided (improves accuracy for crowded scenes)
        if person_bbox is not None:
            x1, y1, x2, y2 = [int(c) for c in person_bbox]
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            rgb_frame = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_h, crop_w = crop.shape[:2]
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop_w, crop_h = w, h
            x1, y1 = 0, 0
        
        try:
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            landmarks = results.pose_landmarks.landmark
            
            # Convert to pixel coordinates (relative to crop, then offset if cropped)
            def get_landmark(idx: int) -> Tuple[float, float]:
                lm = landmarks[idx]
                px = lm.x * crop_w + x1
                py = lm.y * crop_h + y1
                return (px, py)
            
            # Extract key points
            left_shoulder = get_landmark(PoseLandmark.LEFT_SHOULDER)
            right_shoulder = get_landmark(PoseLandmark.RIGHT_SHOULDER)
            left_hip = get_landmark(PoseLandmark.LEFT_HIP)
            right_hip = get_landmark(PoseLandmark.RIGHT_HIP)
            left_knee = get_landmark(PoseLandmark.LEFT_KNEE)
            right_knee = get_landmark(PoseLandmark.RIGHT_KNEE)
            left_wrist = get_landmark(PoseLandmark.LEFT_WRIST)
            right_wrist = get_landmark(PoseLandmark.RIGHT_WRIST)
            left_elbow = get_landmark(PoseLandmark.LEFT_ELBOW)
            right_elbow = get_landmark(PoseLandmark.RIGHT_ELBOW)
            nose = get_landmark(PoseLandmark.NOSE)
            
            # Compute pose angles
            torso_angle = self._compute_torso_angle(
                left_shoulder, right_shoulder, left_hip, right_hip
            )
            left_leg_angle = self._compute_angle(left_hip, left_knee, (left_knee[0], left_hip[1]))
            right_leg_angle = self._compute_angle(right_hip, right_knee, (right_knee[0], right_hip[1]))
            
            # Compute pose-based behavior flags
            is_bending = self._detect_bending(torso_angle, left_leg_angle, right_leg_angle)
            hands_raised = self._detect_hands_raised(
                nose, left_wrist, right_wrist, left_shoulder, right_shoulder
            )
            arms_extended = self._detect_arms_extended(
                left_shoulder, left_elbow, left_wrist,
                right_shoulder, right_elbow, right_wrist
            )
            
            return {
                'landmarks': {
                    'left_shoulder': left_shoulder,
                    'right_shoulder': right_shoulder,
                    'left_hip': left_hip,
                    'right_hip': right_hip,
                    'left_knee': left_knee,
                    'right_knee': right_knee,
                    'left_wrist': left_wrist,
                    'right_wrist': right_wrist,
                    'nose': nose,
                },
                'angles': {
                    'torso_angle': torso_angle,  # degrees from vertical
                    'left_leg_angle': left_leg_angle,
                    'right_leg_angle': right_leg_angle,
                },
                'behaviors': {
                    'is_bending': is_bending,
                    'hands_raised': hands_raised,
                    'arms_extended': arms_extended,
                },
                'raw_landmarks': landmarks,
            }
        except Exception as e:
            return None
    
    def _compute_angle(self, p1: Tuple[float, float], vertex: Tuple[float, float], 
                       p2: Tuple[float, float]) -> float:
        """Compute angle at vertex in degrees"""
        v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]])
        v2 = np.array([p2[0] - vertex[0], p2[1] - vertex[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    def _compute_torso_angle(self, ls: Tuple, rs: Tuple, lh: Tuple, rh: Tuple) -> float:
        """Compute torso angle from vertical (0=standing, 90=bent forward)"""
        shoulder_center = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip_center = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        
        # Vertical vector (down)
        vertical = (0, 1)
        torso_vec = (hip_center[0] - shoulder_center[0], hip_center[1] - shoulder_center[1])
        
        torso_len = np.sqrt(torso_vec[0]**2 + torso_vec[1]**2) + 1e-6
        cos_angle = np.dot(torso_vec, vertical) / torso_len
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_from_vertical = np.degrees(np.arccos(cos_angle))
        
        return angle_from_vertical
    
    def _detect_bending(self, torso_angle: float, left_leg: float, right_leg: float) -> bool:
        """Detect if person is bending (e.g., picking up object)"""
        # Bending: torso forward (angle from vertical > 30°) and legs may be bent
        return torso_angle > 30
    
    def _detect_hands_raised(self, nose: Tuple, lw: Tuple, rw: Tuple, 
                             ls: Tuple, rs: Tuple) -> bool:
        """Detect if hands are raised (suspicious / fighting)"""
        # Hands above shoulders
        left_raised = lw[1] < ls[1]
        right_raised = rw[1] < rs[1]
        return left_raised or right_raised
    
    def _detect_arms_extended(self, ls: Tuple, le: Tuple, lw: Tuple,
                              rs: Tuple, re: Tuple, rw: Tuple) -> bool:
        """Detect if arms are extended (reaching / picking)"""
        def arm_extended(shoulder, elbow, wrist):
            arm_len = np.linalg.norm(np.array(wrist) - np.array(shoulder))
            upper_len = np.linalg.norm(np.array(elbow) - np.array(shoulder))
            lower_len = np.linalg.norm(np.array(wrist) - np.array(elbow))
            total_len = upper_len + lower_len + 1e-6
            # Extended = arm mostly straight
            return arm_len / total_len > 0.8
        
        return arm_extended(ls, le, lw) or arm_extended(rs, re, rw)
    
    def get_pose_features(self, pose_data: Optional[Dict]) -> Dict:
        """
        Extract pose features for behavior analysis.
        Returns dict with normalized values for VAE/behavior integration.
        """
        if pose_data is None:
            return {
                'torso_angle_norm': 0.0,
                'is_bending': 0.0,
                'hands_raised': 0.0,
                'arms_extended': 0.0,
            }
        
        angles = pose_data['angles']
        behaviors = pose_data['behaviors']
        
        # Normalize torso angle to 0-1 (0=standing, 1=fully bent ~90°)
        torso_norm = min(angles['torso_angle'] / 90.0, 1.0)
        
        return {
            'torso_angle_norm': torso_norm,
            'is_bending': 1.0 if behaviors['is_bending'] else 0.0,
            'hands_raised': 1.0 if behaviors['hands_raised'] else 0.0,
            'arms_extended': 1.0 if behaviors['arms_extended'] else 0.0,
        }
    
    def draw_pose(self, frame: np.ndarray, pose_data: Optional[Dict], 
                  color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw pose skeleton on frame"""
        if pose_data is None or 'landmarks' not in pose_data:
            return frame
        
        landmarks = pose_data['landmarks']
        # Draw key points
        for name, (px, py) in landmarks.items():
            cx, cy = int(px), int(py)
            cv2.circle(frame, (cx, cy), 4, color, -1)
        
        # Draw skeleton lines
        pts = landmarks
        skeleton = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
        ]
        for (a, b) in skeleton:
            if a in pts and b in pts:
                pt1 = (int(pts[a][0]), int(pts[a][1]))
                pt2 = (int(pts[b][0]), int(pts[b][1]))
                cv2.line(frame, pt1, pt2, color, 2)
        
        return frame
