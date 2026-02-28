#!/usr/bin/env python3
"""
Complete CCTV System with Global ReID and Real-Time Anomaly Visualization
Shows every person with global ID, cross-camera tracking, and 3-color anomaly visualization
"""

import cv2
import numpy as np
from ultralytics import YOLO
from vae_anomaly_detector import AnomalyDetector
from person_reid_system import GlobalPersonTracker
from adaptive_zone_learning import ActivityZoneLearner
from pose_estimator import PoseEstimator
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import time
import json
import os

class CompleteCCTVSystem:
    """Complete CCTV system with global ReID and real-time anomaly visualization"""
    
    def __init__(self, camera_id: str = "cam1", model_path: str = "models/vae_anomaly_detector.pth"):
        print(f"🚀 Initializing Complete CCTV System for {camera_id}")
        print("=" * 60)
        
        self.camera_id = camera_id
        
        # Initialize core components
        print("📹 Loading YOLO person detection...")
        self.yolo_model = YOLO("yolov8s.pt")  # 's' model for better small-object detection
        
        print("🧠 Loading VAE anomaly detector...")
        self.anomaly_detector = AnomalyDetector(model_path)
        try:
            self.anomaly_detector.load_model()
            print("✅ VAE anomaly detector loaded")
        except FileNotFoundError:
            print("❌ VAE model not found! Please train first.")
            raise
        
        print("🔍 Initializing Person ReID system...")
        self.reid_tracker = GlobalPersonTracker()
        
        print("🤚 Initializing hand detection...")
        self.hand_detector = self._init_hand_detector()
        
        print("🦴 Initializing pose estimation...")
        self.pose_estimator = PoseEstimator()
        
        print("🎯 Loading adaptive interaction zones...")
        self.zone_detector = None  # Will be initialized with video dimensions
        
        # Tracking and anomaly data
        self.person_data = {}  # global_id -> comprehensive person data
        self.anomaly_histories = {}  # global_id -> anomaly history
        
        # Visualization parameters
        self.colors = {
            'normal': (0, 255, 0),      # Green - Normal behavior
            'suspicious': (0, 165, 255), # Orange - Suspicious behavior  
            'anomaly': (0, 0, 255)      # Red - Anomalous behavior
        }
        
        # Anomaly thresholds for 3-color system
        self.anomaly_thresholds = {
            'suspicious': 0.3,  # Above this = suspicious (orange)
            'anomaly': 0.7      # Above this = anomaly (red)
        }
        
        # Smoothing parameters
        self.anomaly_window_size = 15  # Frames for anomaly smoothing
        self.min_track_length = 10     # Minimum frames before showing anomaly
        
        print("✅ Complete CCTV System initialized successfully!")
    
    def _init_hand_detector(self):
        """Initialize MediaPipe hand detection"""
        try:
            mp_hands = mp.solutions.hands
            return mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=6,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except AttributeError:
            print("⚠️  MediaPipe hand detection not available, using fallback")
            return None
    
    def _init_zone_detector(self, width: int, height: int):
        """Initialize adaptive zone detector with video dimensions"""
        try:
            from stealing_detection_system import AdaptiveZoneDetector
            self.zone_detector = AdaptiveZoneDetector(width, height)
            print(f"🎯 Loaded {len(self.zone_detector.interaction_zones)} learned interaction zones")
        except Exception as e:
            print(f"⚠️  Zone detector not available: {e}")
            print("🎯 Using fallback mode without interaction zones")
            self.zone_detector = None
    
    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        """Detect hands in frame"""
        if self.hand_detector is None:
            return []  # Return empty list if hand detection not available
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.process(rgb_frame)
            
            hands_info = []
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    h, w, _ = frame.shape
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    
                    handedness = "Right"
                    if idx < len(results.multi_handedness):
                        handedness = results.multi_handedness[idx].classification[0].label
                    
                    hands_info.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'center': [center_x, center_y],
                        'handedness': handedness,
                        'landmarks': hand_landmarks
                    })
            
            return hands_info
        except Exception as e:
            print(f"⚠️  Hand detection error: {e}")
            return []
    
    def get_person_hands(self, person_bbox: np.ndarray, all_hands: List[Dict]) -> List[Dict]:
        """Get hands belonging to a specific person"""
        person_hands = []
        px1, py1, px2, py2 = person_bbox
        
        for hand in all_hands:
            hx, hy = hand['center']
            
            # Check if hand is within person's bounding box (with tolerance)
            tolerance = 50
            if (px1 - tolerance <= hx <= px2 + tolerance and 
                py1 - tolerance <= hy <= py2 + tolerance):
                person_hands.append(hand)
        
        return person_hands
    
    def analyze_person_behavior(self, global_id: int, local_track_id: int, 
                               person_bbox: List[float], person_hands: List[Dict],
                               pose_data: Optional[Dict],
                               frame_idx: int, fps: int) -> Dict:
        """Comprehensive person behavior analysis"""
        
        timestamp = frame_idx / fps
        
        # Initialize person data if new
        if global_id not in self.person_data:
            self.person_data[global_id] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'cameras_seen': {self.camera_id},
                'local_tracks': {self.camera_id: local_track_id},
                'positions': [],
                'anomaly_scores': [],
                'behaviors': [],
                'interactions': [],
                'total_detections': 0
            }
            self.anomaly_histories[global_id] = []
        
        person_info = self.person_data[global_id]
        
        # Update basic info
        person_info['last_seen'] = timestamp
        person_info['cameras_seen'].add(self.camera_id)
        person_info['local_tracks'][self.camera_id] = local_track_id
        person_info['total_detections'] += 1
        
        # Update position history
        center_x = (person_bbox[0] + person_bbox[2]) / 2
        center_y = (person_bbox[1] + person_bbox[3]) / 2
        person_info['positions'].append([center_x, center_y, timestamp])
        
        # Keep only recent positions
        if len(person_info['positions']) > 100:
            person_info['positions'] = person_info['positions'][-100:]
        
        # 1. BEHAVIORAL ANOMALY DETECTION
        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(
            local_track_id, person_bbox, frame_idx
        )
        
        # Store anomaly score
        person_info['anomaly_scores'].append(anomaly_score)
        if len(person_info['anomaly_scores']) > 50:
            person_info['anomaly_scores'] = person_info['anomaly_scores'][-50:]
        
        # 2. INTERACTION ANALYSIS
        interaction_score = 0.0
        if self.zone_detector and person_hands:
            zone_interactions = self.zone_detector.detect_hand_interaction(
                person_hands, person_bbox
            )
            if zone_interactions['has_interaction']:
                interaction_score = zone_interactions['interaction_score']
                person_info['interactions'].append({
                    'timestamp': timestamp,
                    'score': interaction_score,
                    'zones': zone_interactions['interaction_zones']
                })
        
        # 3. MOTION ANALYSIS
        motion_score = 0.0
        if len(person_info['positions']) > 5:
            recent_positions = np.array([p[:2] for p in person_info['positions'][-5:]])
            
            # Calculate speed
            if len(recent_positions) > 1:
                distances = np.linalg.norm(np.diff(recent_positions, axis=0), axis=1)
                avg_speed = np.mean(distances)
                
                # Normalize speed (higher speed = higher score)
                motion_score = min(avg_speed / 20.0, 1.0)  # Normalize to 0-1
        
        # 4. POSE-BASED BEHAVIOR ANALYSIS
        pose_score = 0.0
        pose_behaviors = {}
        if self.pose_estimator.available and pose_data is not None:
            pose_features = self.pose_estimator.get_pose_features(pose_data)
            pose_behaviors = pose_features
            # Bending + arms extended near zone = picking (suspicious)
            if pose_features['is_bending'] and pose_features['arms_extended'] and interaction_score > 0:
                pose_score = 0.5
            # Hands raised = potentially aggressive
            elif pose_features['hands_raised']:
                pose_score = 0.3
            # Bending alone = could be picking
            elif pose_features['is_bending']:
                pose_score = 0.2
        
        # 5. COMBINED ANOMALY SCORE
        # Weight different factors (pose adds to suspicion)
        combined_score = (
            0.5 * anomaly_score +       # VAE anomaly (50%)
            0.25 * interaction_score +  # Zone interactions (25%)
            0.1 * motion_score +        # Motion patterns (10%)
            0.15 * pose_score           # Pose-based (15%)
        )
        
        # 6. TEMPORAL SMOOTHING
        self.anomaly_histories[global_id].append(combined_score)
        if len(self.anomaly_histories[global_id]) > self.anomaly_window_size:
            self.anomaly_histories[global_id] = self.anomaly_histories[global_id][-self.anomaly_window_size:]
        
        # Calculate smoothed anomaly score
        if len(self.anomaly_histories[global_id]) >= self.min_track_length:
            smoothed_score = np.percentile(self.anomaly_histories[global_id],70)
        else:
            smoothed_score = 0.0  # Not enough data yet
        
        # 7. DETERMINE BEHAVIOR CATEGORY
        if smoothed_score >= self.anomaly_thresholds['anomaly']:
            behavior_category = 'anomaly'
            behavior_text = 'ANOMALY'
        elif smoothed_score >= self.anomaly_thresholds['suspicious']:
            behavior_category = 'suspicious'
            behavior_text = 'SUSPICIOUS'
        else:
            behavior_category = 'normal'
            behavior_text = 'NORMAL'
        
        # 8. ADDITIONAL BEHAVIOR DETAILS
        duration = timestamp - person_info['first_seen']
        is_loitering = duration > 10.0 and len(person_info['cameras_seen']) == 1
        has_interactions = len(person_info['interactions']) > 0
        is_multi_camera = len(person_info['cameras_seen']) > 1
        
        return {
            'global_id': global_id,
            'local_track_id': local_track_id,
            'behavior_category': behavior_category,
            'behavior_text': behavior_text,
            'anomaly_score': smoothed_score,
            'raw_anomaly_score': combined_score,
            'duration': duration,
            'cameras_seen': len(person_info['cameras_seen']),
            'total_detections': person_info['total_detections'],
            'details': {
                'is_loitering': is_loitering,
                'has_interactions': has_interactions,
                'is_multi_camera': is_multi_camera,
                'motion_score': motion_score,
                'interaction_score': interaction_score,
                'vae_score': anomaly_score,
                'pose_score': pose_score,
                'pose_behaviors': pose_behaviors
            }
        }
    
    def draw_person_visualization(self, frame: np.ndarray, person_bbox: np.ndarray, 
                                analysis: Dict, person_hands: List[Dict]) -> np.ndarray:
        """Draw comprehensive person visualization"""
        x1, y1, x2, y2 = person_bbox.astype(int)
        global_id = analysis['global_id']
        anomaly_score = analysis['anomaly_score']

        # Choose color based on behavior category
        color = self.colors[analysis['behavior_category']]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Small label: ID + anomaly score
        label = f"ID:{global_id} {anomaly_score:.2f}"

        cv2.putText(frame, label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)

        return frame
    
    def create_dashboard(self, frame_idx, total_frames, active_persons, anomaly_counts, fps_current):
        """Create separate dashboard window"""

        dashboard = np.zeros((400, 600, 3), dtype=np.uint8)

        y = 40
        gap = 35

        cv2.putText(dashboard, "CCTV SYSTEM DASHBOARD",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        y += gap

        cv2.putText(dashboard, f"Frame: {frame_idx}/{total_frames}",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += gap

        cv2.putText(dashboard, f"Active Persons: {active_persons}",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += gap

        cv2.putText(dashboard, f"Normal: {anomaly_counts['normal']}",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += gap

        cv2.putText(dashboard, f"Suspicious: {anomaly_counts['suspicious']}",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
        y += gap

        cv2.putText(dashboard, f"Anomalies: {anomaly_counts['anomaly']}",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        y += gap

        # ReID statistics
        reid_stats = self.reid_tracker.get_tracking_statistics()

        cv2.putText(dashboard, f"Global Persons: {reid_stats['total_global_persons']}",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        y += gap

        cv2.putText(dashboard, f"ReID Matches: {reid_stats['reid_matches']}",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        y += gap

        cv2.putText(dashboard, f"FPS: {fps_current:.2f}",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        return dashboard
    
    def draw_system_info(self, frame: np.ndarray, frame_idx: int, total_frames: int, 
                        active_persons: int, anomaly_counts: Dict) -> np.ndarray:
        """Draw system information overlay"""
        
        h, w = frame.shape[:2]
        
        # Main system info
        info_lines = [
            f"CCTV System - Camera: {self.camera_id}",
            f"Frame: {frame_idx}/{total_frames}",
            f"Active Persons: {active_persons}",
            f"Global ReID: {'ON' if self.reid_tracker else 'OFF'}"
        ]
        
        # Draw info background
        info_bg_height = len(info_lines) * 30 + 20
        cv2.rectangle(frame, (10, 10), (400, info_bg_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, info_bg_height), (255, 255, 255), 2)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 40 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Anomaly statistics
        stats_lines = [
            f"Normal: {anomaly_counts.get('normal', 0)}",
            f"Suspicious: {anomaly_counts.get('suspicious', 0)}",
            f"Anomalies: {anomaly_counts.get('anomaly', 0)}"
        ]
        
        stats_colors = [self.colors['normal'], self.colors['suspicious'], self.colors['anomaly']]
        
        # Draw stats background
        stats_bg_height = len(stats_lines) * 25 + 20
        stats_y = info_bg_height + 20
        cv2.rectangle(frame, (10, stats_y), (250, stats_y + stats_bg_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, stats_y), (250, stats_y + stats_bg_height), (255, 255, 255), 2)
        
        # Draw stats text
        for i, (line, color) in enumerate(zip(stats_lines, stats_colors)):
            cv2.putText(frame, line, (20, stats_y + 30 + i*22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Legend
        legend_y = stats_y + stats_bg_height + 20
        legend_items = [
            ("Green: Normal Behavior", self.colors['normal']),
            ("Orange: Suspicious Behavior", self.colors['suspicious']),
            ("Red: Anomalous Behavior", self.colors['anomaly'])
        ]
        
        for i, (text, color) in enumerate(legend_items):
            cv2.putText(frame, text, (10, legend_y + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        
        # ReID statistics (if available)
        if self.reid_tracker:
            reid_stats = self.reid_tracker.get_tracking_statistics()
            reid_text = f"ReID: {reid_stats['total_global_persons']} persons, {reid_stats['reid_matches']} matches"
            cv2.putText(frame, reid_text, (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def draw_interaction_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw learned interaction zones"""
        
        if not self.zone_detector:
            return frame
        
        for zone in self.zone_detector.interaction_zones:
            x1, y1, x2, y2 = [int(coord) for coord in zone['bbox']]
            
            # Color based on zone density
            if zone['density'] > 0.4:
                zone_color = (0, 255, 255)  # High activity - Yellow
            elif zone['density'] > 0.2:
                zone_color = (255, 255, 0)  # Medium activity - Cyan
            else:
                zone_color = (128, 128, 128)  # Low activity - Gray
            
            # Draw zone boundary
            cv2.rectangle(frame, (x1, y1), (x2, y2), zone_color, 1)
            
            # Zone label
            label = f"{zone['id']} (D:{zone['density']:.2f})"
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, zone_color, 1)
        
        return frame
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """Process video with complete CCTV system"""
        
        print(f"\n🎬 Processing Video: {os.path.basename(video_path)}")
        print("=" * 60)
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"📹 Camera ID: {self.camera_id}")
        print(f"🔍 ReID Enabled: Yes")
        print(f"🎯 Anomaly Visualization: 3-Color System")
        
        # Initialize zone detector with video dimensions
        self._init_zone_detector(width, height)
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {output_path}")
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Person detection and tracking
                results = self.yolo_model.track(
                    source=frame,
                    tracker="botsort_improved.yaml",
                    persist=True,
                    classes=[0],  # person only
                    conf=0.25,    # Lower threshold to detect small/crouched persons
                    imgsz=640,    # Higher resolution for small objects
                    verbose=False
                )
                
                # Hand detection
                hands = self.detect_hands(frame)
                
                # Process frame
                annotated_frame = frame.copy()
                
                # Draw interaction zones
               # annotated_frame = self.draw_interaction_zones(annotated_frame)
                
                # Process person detections
                anomaly_counts = {'normal': 0, 'suspicious': 0, 'anomaly': 0}
                active_persons = 0
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        if conf < 0.25:
                            continue
                        
                        active_persons += 1
                        
                        # Global ReID processing
                        global_id = self.reid_tracker.update_global_tracking(
                            self.camera_id, track_id, frame, box.tolist(), conf, timestamp
                        )
                        
                        # Get person's hands
                        person_hands = self.get_person_hands(box, hands)
                        
                        # Detect pose for this person (crop to bbox for accuracy)
                        pose_data = None
                        if self.pose_estimator.available:
                            pose_data = self.pose_estimator.detect_pose(frame, box.tolist())
                        
                        # Analyze person behavior
                        analysis = self.analyze_person_behavior(
                            global_id, track_id, box.tolist(), person_hands, pose_data, frame_idx, fps
                        )
                        
                        # Count anomaly categories
                        anomaly_counts[analysis['behavior_category']] += 1
                        
                        # Draw pose skeleton (optional, for debugging)
                        if pose_data and self.pose_estimator.available:
                            color = self.colors[analysis['behavior_category']]
                            annotated_frame = self.pose_estimator.draw_pose(
                                annotated_frame, pose_data, color
                            )
                        
                        # Draw person visualization
                        annotated_frame = self.draw_person_visualization(
                            annotated_frame, box, analysis, person_hands
                        )
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    elapsed = time.time() - start_time
                    fps_current = frame_idx / elapsed if elapsed > 0 else 0

                    dashboard_frame = self.create_dashboard(
                        frame_idx,
                        total_frames,
                        active_persons,
                        anomaly_counts,
                        fps_current
                    )

                    cv2.imshow('CCTV Video', annotated_frame)
                    cv2.imshow('CCTV Dashboard', dashboard_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏹️  Processing stopped by user")
                        break
                    elif key == ord(' '):
                        print("⏸️  Paused - Press any key to continue")
                        cv2.waitKey(0)
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_idx / elapsed
                    progress = (frame_idx / total_frames) * 100
                    
                    print(f"📊 Progress: {progress:.1f}% | FPS: {fps_current:.1f} | "
                          f"Persons: {active_persons} | Anomalies: {anomaly_counts['anomaly']}")
                
                # Periodic cleanup
                if frame_idx % 1000 == 0:
                    self.reid_tracker.cleanup_old_tracks(timestamp)
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time
        reid_stats = self.reid_tracker.get_tracking_statistics()
        
        print(f"\n🎯 PROCESSING COMPLETE")
        print("=" * 50)
        print(f"📊 Frames processed: {frame_idx}")
        print(f"⚡ Average FPS: {avg_fps:.1f}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🌍 Global persons tracked: {reid_stats['total_global_persons']}")
        print(f"🔄 ReID matches: {reid_stats['reid_matches']}")
        print(f"📈 ReID match rate: {reid_stats['reid_match_rate']:.2%}")
        
        # Save ReID data
        self.reid_tracker.save_reid_data(f"reid_data_{self.camera_id}.pkl")
        
        if output_path:
            print(f"💾 Output saved to: {output_path}")
        
        return {
            'frames_processed': frame_idx,
            'avg_fps': avg_fps,
            'reid_statistics': reid_stats,
            'total_persons': len(self.person_data)
        }

def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete CCTV System with Global ReID and Anomaly Visualization')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path (optional)')
    parser.add_argument('--camera-id', '-c', default='cam1', help='Camera ID for ReID')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input video not found: {args.input}")
        return
    
    # Set default output path if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"complete_cctv_output_{args.camera_id}_{base_name}.mp4"
    
    print("🚀 COMPLETE CCTV SYSTEM")
    print("=" * 60)
    print("Features:")
    print("✅ Global Person ReID across camera angles")
    print("✅ Real-time anomaly detection with VAE")
    print("✅ 3-Color behavior visualization (Green/Orange/Red)")
    print("✅ Anomaly score display with progress bars")
    print("✅ Hand detection and interaction analysis")
    print("✅ Adaptive zone learning from normal behavior")
    print("✅ Multi-camera tracking statistics")
    print("=" * 60)
    
    try:
        # Initialize complete system
        system = CompleteCCTVSystem(camera_id=args.camera_id)
        
        # Process video
        results = system.process_video(
            video_path=args.input,
            output_path=args.output,
            display=not args.no_display
        )
        
        print(f"\n🏆 COMPLETE CCTV SYSTEM PROCESSING SUCCESSFUL!")
        
    except FileNotFoundError as e:
        print(f"❌ Required model not found: {e}")
        print("Please ensure VAE model is trained:")
        print("python train_vae_model.py")
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()