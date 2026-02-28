#!/usr/bin/env python3
"""
Enhanced Stealing Detection System with Person Re-Identification
Combines behavioral anomaly detection with object interaction analysis and global person tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
from vae_anomaly_detector import AnomalyDetector
from person_reid_system import GlobalPersonTracker
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import time
import json
import os

class HandDetector:
    """Hand detection using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        """Detect hands in frame and return hand information"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands_info = []
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand bounding box
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Get hand center
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                # Get handedness
                handedness = "Right" if idx < len(results.multi_handedness) else "Unknown"
                if idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                
                hands_info.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'center': [center_x, center_y],
                    'handedness': handedness,
                    'landmarks': hand_landmarks
                })
        
        return hands_info

class AdaptiveZoneDetector:
    """Adaptive zone detector using learned interaction zones from normal behavior"""
    
    def __init__(self, frame_width: int, frame_height: int, zones_path: str = "models/learned_interaction_zones.pkl"):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zones_path = zones_path
        
        # Load learned zones or create fallback zones
        self.interaction_zones = self._load_learned_zones()
        
        # Product interaction tracking
        self.interaction_history = {}
        
    def _load_learned_zones(self) -> List[Dict]:
        """Load learned interaction zones from normal behavior analysis"""
        
        if os.path.exists(self.zones_path):
            try:
                import pickle
                with open(self.zones_path, 'rb') as f:
                    zone_data = pickle.load(f)
                
                zones = zone_data['zones']
                print(f"✅ Loaded {len(zones)} learned interaction zones")
                
                # Add sensitivity based on zone density
                for zone in zones:
                    # Higher density zones get higher sensitivity
                    zone['sensitivity'] = 1.0 + (zone['density'] * 0.5)
                
                return zones
                
            except Exception as e:
                print(f"⚠️  Error loading learned zones: {e}")
                return self._create_fallback_zones()
        else:
            print(f"⚠️  Learned zones not found at {self.zones_path}")
            print("   Run 'python adaptive_zone_learning.py' first to learn zones")
            return self._create_fallback_zones()
    
    def _create_fallback_zones(self) -> List[Dict]:
        """Create fallback zones when learned zones are not available"""
        print("🔄 Using fallback interaction zones")
        
        zones = []
        
        # Create basic zones based on common retail layouts
        zones.append({
            'id': 'fallback_left',
            'center': [self.frame_width * 0.2, self.frame_height * 0.5],
            'bbox': [0, int(self.frame_height * 0.2), 
                    int(self.frame_width * 0.4), int(self.frame_height * 0.8)],
            'sensitivity': 1.0,
            'density': 0.3,
            'point_count': 10,
            'area': self.frame_width * 0.4 * self.frame_height * 0.6
        })
        
        zones.append({
            'id': 'fallback_right',
            'center': [self.frame_width * 0.8, self.frame_height * 0.5],
            'bbox': [int(self.frame_width * 0.6), int(self.frame_height * 0.2),
                    self.frame_width, int(self.frame_height * 0.8)],
            'sensitivity': 1.0,
            'density': 0.3,
            'point_count': 10,
            'area': self.frame_width * 0.4 * self.frame_height * 0.6
        })
        
        zones.append({
            'id': 'fallback_center',
            'center': [self.frame_width * 0.5, self.frame_height * 0.6],
            'bbox': [int(self.frame_width * 0.3), int(self.frame_height * 0.4),
                    int(self.frame_width * 0.7), int(self.frame_height * 0.8)],
            'sensitivity': 0.8,
            'density': 0.2,
            'point_count': 5,
            'area': self.frame_width * 0.4 * self.frame_height * 0.4
        })
        
        return zones
    
    def is_in_interaction_zone(self, point: List[float]) -> Optional[Dict]:
        """Check if point is in any learned interaction zone"""
        x, y = point
        
        for zone in self.interaction_zones:
            x1, y1, x2, y2 = zone['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone
        
        return None
    
    def detect_hand_interaction(self, hands: List[Dict], person_bbox: List[float]) -> Dict:
        """Detect if hands are interacting with learned interaction zones"""
        interactions = {
            'has_interaction': False,
            'interaction_zones': [],
            'interaction_score': 0.0,
            'hand_positions': [],
            'zone_details': []
        }
        
        for hand in hands:
            hand_center = hand['center']
            interactions['hand_positions'].append(hand_center)
            
            # Check if hand is in learned interaction zone
            zone = self.is_in_interaction_zone(hand_center)
            if zone:
                interactions['has_interaction'] = True
                interactions['interaction_zones'].append(zone['id'])
                interactions['interaction_score'] += zone['sensitivity']
                
                # Add detailed zone information
                zone_detail = {
                    'zone_id': zone['id'],
                    'zone_density': zone['density'],
                    'zone_center': zone['center'],
                    'hand_distance_to_center': np.linalg.norm(
                        np.array(hand_center) - np.array(zone['center'])
                    )
                }
                interactions['zone_details'].append(zone_detail)
        
        return interactions


class ShelfZoneDetector(AdaptiveZoneDetector):
    """Alias for AdaptiveZoneDetector with shelf-specific interface for test compatibility."""
    
    def detect_hand_shelf_interaction(self, hands: List[Dict], person_bbox: List[float]) -> Dict:
        """Alias for detect_hand_interaction - same interface for shelf interaction detection."""
        return self.detect_hand_interaction(hands, person_bbox)


class StealingDetectionSystem:
    """Multi-level stealing detection system with ReID integration"""
    
    def __init__(self, model_path: str = "models/vae_anomaly_detector.pth", 
                 enable_reid: bool = True, camera_id: str = "cam1"):
        print("Initializing Enhanced Stealing Detection System with ReID...")
        
        # Camera identification
        self.camera_id = camera_id
        
        # Load existing components
        self.yolo_model = YOLO("yolov8s.pt")  # 's' model for better small-object detection
        self.anomaly_detector = AnomalyDetector(model_path)
        
        try:
            self.anomaly_detector.load_model()
            print("✓ Behavioral anomaly detector loaded")
        except FileNotFoundError:
            print("❌ Anomaly detection model not found!")
            raise
        
        # Initialize ReID system
        self.enable_reid = enable_reid
        if enable_reid:
            self.reid_tracker = GlobalPersonTracker()
            print("✓ Person ReID system initialized")
        else:
            self.reid_tracker = None
            print("⚠️  ReID system disabled")
        
        # Initialize hand detection
        self.hand_detector = HandDetector()
        print("✓ Hand detection initialized")
        
        # Tracking data with ReID integration
        self.track_histories = {}
        self.track_stealing_scores = {}
        self.track_interaction_history = {}
        self.global_person_data = {}  # global_id -> comprehensive data
        
        # Stealing detection parameters
        self.loitering_threshold = 5.0  # seconds
        self.interaction_threshold = 3   # minimum interactions
        self.suspicious_duration = 8.0   # seconds of suspicious behavior
        
        # Colors for different threat levels
        self.colors = {
            'normal': (0, 255, 0),           # Green
            'suspicious': (0, 165, 255),     # Orange  
            'high_risk': (0, 100, 255),      # Dark Orange
            'stealing': (0, 0, 255),         # Red
            'confirmed_theft': (128, 0, 128) # Purple
        }
    
    def analyze_stealing_behavior(self, track_id: int, global_id: int, person_bbox: List[float], 
                                hands: List[Dict], zone_interactions: Dict,
                                frame_idx: int, fps: int) -> Dict:
        """Comprehensive stealing behavior analysis with ReID integration"""
        
        timestamp = frame_idx / fps
        
        # Use global ID for consistent tracking across cameras
        tracking_id = global_id if global_id is not None else track_id
        
        # Initialize tracking for new person (using global ID)
        if tracking_id not in self.track_histories:
            self.track_histories[tracking_id] = {
                'first_seen': timestamp,
                'positions': [],
                'interactions': [],
                'suspicious_periods': [],
                'behavioral_anomalies': [],
                'cameras_seen': {self.camera_id},
                'local_track_ids': {self.camera_id: track_id}
            }
            self.track_stealing_scores[tracking_id] = {
                'behavioral_score': 0.0,
                'interaction_score': 0.0,
                'temporal_score': 0.0,
                'reid_consistency_score': 0.0,
                'final_score': 0.0
            }
        
        history = self.track_histories[tracking_id]
        scores = self.track_stealing_scores[tracking_id]
        
        # Update camera and local track information
        history['cameras_seen'].add(self.camera_id)
        history['local_track_ids'][self.camera_id] = track_id
        
        # Update position history
        center_x = (person_bbox[0] + person_bbox[2]) / 2
        center_y = (person_bbox[1] + person_bbox[3]) / 2
        history['positions'].append([center_x, center_y, timestamp])
        
        # 1. BEHAVIORAL ANOMALY ANALYSIS
        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(
            track_id, person_bbox, frame_idx  # Use local track_id for VAE
        )
        
        if is_anomaly:
            history['behavioral_anomalies'].append({
                'timestamp': timestamp,
                'score': anomaly_score,
                'type': 'movement_anomaly',
                'camera': self.camera_id
            })
        
        scores['behavioral_score'] = anomaly_score
        
        # 2. HAND-ZONE INTERACTION ANALYSIS (Using Learned Zones)
        if zone_interactions['has_interaction']:
            interaction_data = {
                'timestamp': timestamp,
                'zones': zone_interactions['interaction_zones'],
                'score': zone_interactions['interaction_score'],
                'hand_positions': zone_interactions['hand_positions'],
                'zone_details': zone_interactions['zone_details'],
                'camera': self.camera_id
            }
            history['interactions'].append(interaction_data)
        
        # Calculate interaction score with learned zone weighting
        recent_interactions = [i for i in history['interactions'] 
                             if timestamp - i['timestamp'] < 10.0]
        
        # Weight interactions by zone density (higher density = more significant)
        weighted_interaction_score = 0.0
        for interaction in recent_interactions:
            base_score = 0.2
            if 'zone_details' in interaction:
                for zone_detail in interaction['zone_details']:
                    density_weight = 1.0 + zone_detail['zone_density']
                    weighted_interaction_score += base_score * density_weight
            else:
                weighted_interaction_score += base_score
        
        scores['interaction_score'] = weighted_interaction_score
        
        # 3. TEMPORAL PATTERN ANALYSIS
        duration_in_area = timestamp - history['first_seen']
        
        # Loitering detection
        if len(history['positions']) > 10:
            recent_positions = np.array([p[:2] for p in history['positions'][-10:]])
            position_variance = np.var(recent_positions, axis=0)
            is_loitering = np.mean(position_variance) < 1000  # Low movement variance
            
            if is_loitering and duration_in_area > self.loitering_threshold:
                scores['temporal_score'] += 0.3
        
        # Suspicious duration
        if duration_in_area > self.suspicious_duration:
            scores['temporal_score'] += 0.2
        
        # 4. REID CONSISTENCY ANALYSIS (New)
        if self.enable_reid and global_id is not None:
            # Multi-camera consistency bonus
            num_cameras = len(history['cameras_seen'])
            if num_cameras > 1:
                scores['reid_consistency_score'] += 0.1 * (num_cameras - 1)
            
            # Get ReID statistics if available
            if self.reid_tracker:
                person_info = self.reid_tracker.get_person_info(global_id)
                if person_info:
                    total_detections = person_info.get('total_detections', 1)
                    avg_quality = np.mean(person_info.get('quality_scores', [0.5]))
                    
                    # Quality bonus for high-quality ReID features
                    if avg_quality > 0.7:
                        scores['reid_consistency_score'] += 0.1
                    
                    # Persistence bonus for long-tracked persons
                    if total_detections > 50:
                        scores['reid_consistency_score'] += 0.1
        
        # 5. CALCULATE FINAL STEALING RISK SCORE (Updated weights)
        scores['final_score'] = (
            0.35 * min(scores['behavioral_score'], 2.0) +      # Behavioral anomalies (35%)
            0.35 * min(scores['interaction_score'], 2.0) +     # Hand-zone interactions (35%)
            0.20 * min(scores['temporal_score'], 2.0) +        # Temporal patterns (20%)
            0.10 * min(scores['reid_consistency_score'], 2.0)  # ReID consistency (10%)
        )
        
        # 6. DETERMINE THREAT LEVEL
        threat_level = self._determine_threat_level(scores['final_score'], history)
        
        return {
            'track_id': track_id,
            'global_id': global_id,
            'threat_level': threat_level,
            'scores': scores,
            'duration': duration_in_area,
            'interaction_count': len(recent_interactions),
            'anomaly_count': len(history['behavioral_anomalies']),
            'cameras_seen': len(history['cameras_seen']),
            'details': {
                'is_loitering': duration_in_area > self.loitering_threshold,
                'has_interactions': len(recent_interactions) > 0,
                'is_behaviorally_anomalous': is_anomaly,
                'recent_interactions': len(recent_interactions),
                'multi_camera_tracking': len(history['cameras_seen']) > 1,
                'reid_enabled': self.enable_reid and global_id is not None
            }
        }
    
    def _determine_threat_level(self, final_score: float, history: Dict) -> str:
        """Determine threat level based on comprehensive analysis"""
        
        # Get recent activity
        current_time = history['positions'][-1][2] if history['positions'] else 0
        recent_interactions = [i for i in history['interactions'] 
                             if current_time - i['timestamp'] < 5.0]
        recent_anomalies = [a for a in history['behavioral_anomalies']
                          if current_time - a['timestamp'] < 5.0]
        
        # Multi-criteria threat assessment
        if (final_score > 1.5 and 
            len(recent_interactions) >= 2 and 
            len(recent_anomalies) >= 1):
            return 'confirmed_theft'
        
        elif (final_score > 1.2 and 
              (len(recent_interactions) >= 2 or len(recent_anomalies) >= 2)):
            return 'stealing'
        
        elif (final_score > 0.8 and 
              (len(recent_interactions) >= 1 or len(recent_anomalies) >= 1)):
            return 'high_risk'
        
        elif final_score > 0.4:
            return 'suspicious'
        
        else:
            return 'normal'
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """Process video with enhanced stealing detection and ReID"""
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        print(f"Camera ID: {self.camera_id}")
        print(f"ReID enabled: {self.enable_reid}")
        
        # Initialize adaptive zone detector
        zone_detector = AdaptiveZoneDetector(width, height)
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        stealing_detections = []
        reid_stats = {'total_persons': 0, 'reid_matches': 0, 'new_persons': 0}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Person detection and tracking
                results = self.yolo_model.track(
                    source=frame,
                    tracker="botsort.yaml",
                    persist=True,
                    classes=[0],  # person only
                    conf=0.4,
                    verbose=False
                )
                
                # Hand detection
                hands = self.hand_detector.detect_hands(frame)
                
                # Process frame
                annotated_frame = frame.copy()
                
                # Draw learned interaction zones
                for zone in zone_detector.interaction_zones:
                    x1, y1, x2, y2 = [int(coord) for coord in zone['bbox']]
                    # Use different colors based on zone density
                    if zone['density'] > 0.4:
                        zone_color = (0, 255, 255)  # High activity - Yellow
                    elif zone['density'] > 0.2:
                        zone_color = (255, 255, 0)  # Medium activity - Cyan
                    else:
                        zone_color = (128, 128, 128)  # Low activity - Gray
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), zone_color, 1)
                    
                    # Add zone label with learned statistics
                    label = f"{zone['id']} (D:{zone['density']:.2f})"
                    cv2.putText(annotated_frame, label, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, zone_color, 1)
                
                # Draw hands
                for hand in hands:
                    x1, y1, x2, y2 = hand['bbox']
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(annotated_frame, f"Hand-{hand['handedness']}", 
                              (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # Process person detections
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        if conf < 0.25:
                            continue
                        
                        # ReID processing
                        global_id = None
                        if self.enable_reid and self.reid_tracker:
                            global_id = self.reid_tracker.update_global_tracking(
                                self.camera_id, track_id, frame, box.tolist(), conf, timestamp
                            )
                            reid_stats['total_persons'] += 1
                        
                        # Detect hand-interaction with learned zones
                        person_hands = self._get_person_hands(box, hands)
                        zone_interactions = zone_detector.detect_hand_interaction(
                            person_hands, box.tolist()
                        )
                        
                        # Analyze stealing behavior with ReID integration
                        analysis = self.analyze_stealing_behavior(
                            track_id, global_id, box.tolist(), person_hands, 
                            zone_interactions, frame_idx, fps
                        )
                        
                        # Choose color based on threat level
                        color = self.colors[analysis['threat_level']]
                        
                        # Draw person bounding box
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Create detailed label with ReID info
                        threat_level = analysis['threat_level'].upper()
                        final_score = analysis['scores']['final_score']
                        duration = analysis['duration']
                        
                        if global_id is not None:
                            label = f"L:{track_id} G:{global_id} {threat_level}"
                        else:
                            label = f"ID:{track_id} {threat_level}"
                        
                        if final_score > 0:
                            label += f" ({final_score:.2f})"
                        
                        # Add duration and camera info for suspicious cases
                        if analysis['threat_level'] != 'normal':
                            label += f" {duration:.1f}s"
                            if analysis['cameras_seen'] > 1:
                                label += f" C:{analysis['cameras_seen']}"
                        
                        # Draw label with background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), 
                                    color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Add interaction indicators with learned zone info
                        if zone_interactions['has_interaction']:
                            interaction_text = "LEARNED ZONE INTERACTION!"
                            if zone_interactions['zone_details']:
                                zone_detail = zone_interactions['zone_details'][0]
                                interaction_text += f" (D:{zone_detail['zone_density']:.2f})"
                            
                            cv2.putText(annotated_frame, interaction_text, 
                                      (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 255), 2)
                        
                        # Add ReID indicator
                        if self.enable_reid and global_id is not None:
                            reid_text = f"ReID: Global ID {global_id}"
                            if analysis['details']['multi_camera_tracking']:
                                reid_text += " (Multi-Cam)"
                            
                            cv2.putText(annotated_frame, reid_text, 
                                      (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.4, (255, 255, 0), 1)
                        
                        # Log high-risk detections
                        if analysis['threat_level'] in ['stealing', 'confirmed_theft']:
                            detection_data = {
                                'frame': frame_idx,
                                'track_id': track_id,
                                'global_id': global_id,
                                'threat_level': analysis['threat_level'],
                                'scores': analysis['scores'],
                                'timestamp': timestamp,
                                'camera_id': self.camera_id,
                                'details': analysis['details']
                            }
                            stealing_detections.append(detection_data)
                
                # Add comprehensive info panel
                info_lines = [
                    f"Frame: {frame_idx}/{total_frames} | Camera: {self.camera_id}",
                    f"Hands: {len(hands)} | Stealing Alerts: {len(stealing_detections)}",
                    f"Active Tracks: {len(self.track_histories)}"
                ]
                
                if self.enable_reid and self.reid_tracker:
                    reid_stats_current = self.reid_tracker.get_tracking_statistics()
                    info_lines.append(f"Global Persons: {reid_stats_current['total_global_persons']} | ReID Matches: {reid_stats_current['reid_matches']}")
                
                for i, line in enumerate(info_lines):
                    cv2.putText(annotated_frame, line, (10, 30 + i*25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add legend
                legend_y = 150
                legend_items = [
                    ("Green: Normal", self.colors['normal']),
                    ("Orange: Suspicious", self.colors['suspicious']),
                    ("Dark Orange: High Risk", self.colors['high_risk']),
                    ("Red: Stealing Detected", self.colors['stealing']),
                    ("Purple: Confirmed Theft", self.colors['confirmed_theft'])
                ]
                
                for i, (text, color) in enumerate(legend_items):
                    cv2.putText(annotated_frame, text, (10, legend_y + i*18), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                
                # Add ReID legend
                if self.enable_reid:
                    cv2.putText(annotated_frame, "L:Local G:Global C:Cameras", 
                              (10, legend_y + len(legend_items)*18 + 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Enhanced Stealing Detection with ReID', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)
                
                frame_idx += 1
                
                # Periodic cleanup of old tracks
                if frame_idx % 1000 == 0 and self.enable_reid and self.reid_tracker:
                    self.reid_tracker.cleanup_old_tracks(timestamp)
                
                # Progress update
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Stealing Alerts: {len(stealing_detections)}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Get final ReID statistics
        final_reid_stats = {}
        if self.enable_reid and self.reid_tracker:
            final_reid_stats = self.reid_tracker.get_tracking_statistics()
            # Save ReID data
            self.reid_tracker.save_reid_data(f"reid_data_{self.camera_id}.pkl")
        
        # Print comprehensive results
        print(f"\n=== Enhanced Stealing Detection with ReID Results ===")
        print(f"Total frames processed: {frame_idx}")
        print(f"Total stealing alerts: {len(stealing_detections)}")
        print(f"Unique persons tracked: {len(self.track_histories)}")
        
        if self.enable_reid:
            print(f"\n=== ReID Statistics ===")
            print(f"Global persons identified: {final_reid_stats.get('total_global_persons', 0)}")
            print(f"ReID matches: {final_reid_stats.get('reid_matches', 0)}")
            print(f"Match rate: {final_reid_stats.get('reid_match_rate', 0):.2%}")
            print(f"Multi-camera persons: {len([p for p in self.track_histories.values() if len(p.get('cameras_seen', set())) > 1])}")
        
        # Analyze stealing detections by threat level
        threat_counts = {}
        for detection in stealing_detections:
            level = detection['threat_level']
            threat_counts[level] = threat_counts.get(level, 0) + 1
        
        if threat_counts:
            print(f"\nThreat Level Breakdown:")
            for level, count in threat_counts.items():
                print(f"  {level.upper()}: {count}")
        
        if output_path:
            print(f"Enhanced output saved to: {output_path}")
        
        return {
            'stealing_detections': stealing_detections,
            'threat_counts': threat_counts,
            'total_tracks': len(self.track_histories),
            'reid_statistics': final_reid_stats
        }
    
    def _get_person_hands(self, person_bbox: np.ndarray, all_hands: List[Dict]) -> List[Dict]:
        """Get hands that belong to a specific person based on proximity"""
        person_hands = []
        px1, py1, px2, py2 = person_bbox
        person_center_x = (px1 + px2) / 2
        person_center_y = (py1 + py2) / 2
        
        for hand in all_hands:
            hx, hy = hand['center']
            
            # Check if hand is within person's bounding box area (with some tolerance)
            tolerance = 50
            if (px1 - tolerance <= hx <= px2 + tolerance and 
                py1 - tolerance <= hy <= py2 + tolerance):
                person_hands.append(hand)
        
        return person_hands

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced stealing detection system with ReID')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--camera-id', '-c', default='cam1', help='Camera ID for ReID')
    parser.add_argument('--disable-reid', action='store_true', help='Disable ReID system')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found!")
        return
    
    # Initialize stealing detection system with ReID
    try:
        detector = StealingDetectionSystem(
            enable_reid=not args.disable_reid,
            camera_id=args.camera_id
        )
    except FileNotFoundError:
        return
    
    # Process video
    display = not args.no_display
    results = detector.process_video(args.input, args.output, display)
    
    print(f"\nStealing detection with ReID completed successfully!")
    
    # Print ReID-specific results
    if not args.disable_reid and 'reid_statistics' in results:
        reid_stats = results['reid_statistics']
        print(f"\n🔍 ReID Performance:")
        print(f"  Global persons: {reid_stats.get('total_global_persons', 0)}")
        print(f"  ReID matches: {reid_stats.get('reid_matches', 0)}")
        print(f"  Match rate: {reid_stats.get('reid_match_rate', 0):.2%}")
        print(f"  Avg detections per person: {reid_stats.get('avg_detections_per_person', 0):.1f}")

if __name__ == "__main__":
    main()