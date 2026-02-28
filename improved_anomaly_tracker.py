#!/usr/bin/env python3
"""
Improved Anomaly Detection Tracker with Better ID Consistency and Reduced False Positives
"""

import cv2
import numpy as np
from ultralytics import YOLO
from vae_anomaly_detector import AnomalyDetector
import os
import argparse
from typing import Dict, Tuple, List
from collections import defaultdict
import time

class ImprovedAnomalyTracker:
    """Improved tracker with better ID consistency and anomaly filtering"""
    
    def __init__(self, model_path: str = "models/vae_anomaly_detector.pth"):
        print("Initializing Improved Anomaly Tracker...")
        
        # Load YOLO model with higher confidence
        self.yolo_model = YOLO("yolov8s.pt")  # 's' model for better small-object detection
        
        # Load anomaly detector
        self.anomaly_detector = AnomalyDetector(model_path)
        try:
            self.anomaly_detector.load_model()
            print("✓ Anomaly detection model loaded successfully")
        except FileNotFoundError:
            print("❌ Anomaly detection model not found!")
            raise
        
        # Enhanced tracking parameters - FIXED for better detection
        self.track_anomaly_scores = {}
        self.track_anomaly_history = {}
        self.track_confidence_history = {}
        self.track_position_history = {}
        self.track_last_seen = {}
        
        # FIXED: More responsive anomaly detection parameters
        self.track_exponential_scores = {}  # Exponentially weighted anomaly scores
        self.decay_factor = 0.8             # Faster decay for quicker response
        self.anomaly_threshold_frames = 8   # REDUCED: Faster anomaly confirmation
        self.anomaly_confirmation_ratio = 0.6  # REDUCED: 60% confirmation (was 75%)
        self.min_track_length = 12          # REDUCED: Much faster response (was 25)
        self.confidence_threshold = 0.25    # Lower threshold to detect small/crouched persons
        
        # Context-aware detection parameters
        self.zone_sensitivity = {
            'entrance': 1.2,    # Higher sensitivity near entrances
            'exit': 1.2,        # Higher sensitivity near exits  
            'center': 1.0,      # Normal sensitivity in center
            'corner': 0.8       # Lower sensitivity in corners
        }
        
        # FIXED: ID consistency tracking with better tolerance
        self.track_id_mapping = {}
        self.next_stable_id = 1
        self.position_tolerance = 60       # INCREASED: Better ID consistency (was 40)
        
        # Colors
        self.normal_color = (0, 255, 0)         # Green
        self.anomaly_color = (0, 0, 255)       # Red  
        self.warning_color = (0, 165, 255)     # Orange
        self.low_confidence_color = (128, 128, 128)  # Gray
        
    def get_stable_track_id(self, current_id: int, bbox: List[float], confidence: float) -> int:
        """Map inconsistent track IDs to stable IDs based on position"""
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        current_pos = np.array([center_x, center_y])
        
        # Check if this is a known track ID
        if current_id in self.track_id_mapping:
            stable_id = self.track_id_mapping[current_id]
            
            # Update position history
            if stable_id in self.track_position_history:
                self.track_position_history[stable_id].append(current_pos)
                # Keep only recent positions
                if len(self.track_position_history[stable_id]) > 10:
                    self.track_position_history[stable_id] = self.track_position_history[stable_id][-10:]
            
            return stable_id
        
        # Check if this position matches an existing stable track
        for stable_id, positions in self.track_position_history.items():
            if len(positions) > 0:
                last_pos = positions[-1]
                distance = np.linalg.norm(current_pos - last_pos)
                
                if distance < self.position_tolerance:
                    # Map this track ID to existing stable ID
                    self.track_id_mapping[current_id] = stable_id
                    self.track_position_history[stable_id].append(current_pos)
                    return stable_id
        
        # Create new stable ID
        new_stable_id = self.next_stable_id
        self.next_stable_id += 1
        self.track_id_mapping[current_id] = new_stable_id
        self.track_position_history[new_stable_id] = [current_pos]
        
        return new_stable_id
    
    def is_valid_detection(self, bbox: List[float], confidence: float) -> bool:
        """FIXED: Less strict validation for better detection"""
        
        # FIXED: Lower confidence check
        if confidence < self.confidence_threshold:  # Now 0.4 instead of 0.6
            return False
        
        # FIXED: More lenient size check
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        if area < 500 or area > 80000:  # RELAXED: More lenient size range
            return False
        
        # FIXED: More lenient aspect ratio check
        aspect_ratio = height / width if width > 0 else 0
        if aspect_ratio < 1.0 or aspect_ratio > 5.0:  # RELAXED: More lenient ratios
            return False
        
        return True
    
    def get_zone_sensitivity(self, bbox: List[float], frame_width: int, frame_height: int) -> float:
        """Get context-aware sensitivity based on position in frame"""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Normalize coordinates
        norm_x = center_x / frame_width
        norm_y = center_y / frame_height
        
        # Define zones
        if norm_x < 0.2 or norm_x > 0.8:  # Near edges (entrance/exit areas)
            if norm_y < 0.3 or norm_y > 0.7:  # Corners
                return self.zone_sensitivity['corner']
            else:
                return self.zone_sensitivity['entrance']
        elif norm_y < 0.2 or norm_y > 0.8:  # Top/bottom edges
            return self.zone_sensitivity['exit']
        else:  # Center area
            return self.zone_sensitivity['center']
    
    def advanced_anomaly_smoothing(self, stable_id: int, is_anomaly: bool, anomaly_score: float, 
                                 bbox: List[float], frame_width: int, frame_height: int) -> Tuple[bool, str]:
        """Advanced temporal smoothing with exponential decay and context awareness"""
        
        # Initialize tracking for new IDs
        if stable_id not in self.track_anomaly_history:
            self.track_anomaly_history[stable_id] = []
            self.track_confidence_history[stable_id] = []
            self.track_exponential_scores[stable_id] = 0.0
        
        # Get context-aware sensitivity
        zone_sensitivity = self.get_zone_sensitivity(bbox, frame_width, frame_height)
        adjusted_score = anomaly_score * zone_sensitivity
        
        # Update exponential weighted score
        current_exp_score = self.track_exponential_scores[stable_id]
        self.track_exponential_scores[stable_id] = (
            self.decay_factor * current_exp_score + 
            (1 - self.decay_factor) * adjusted_score
        )
        
        # Add current detection to history
        self.track_anomaly_history[stable_id].append(is_anomaly)
        self.track_confidence_history[stable_id].append(adjusted_score)
        
        # Keep only recent history
        max_history = self.anomaly_threshold_frames * 2
        if len(self.track_anomaly_history[stable_id]) > max_history:
            self.track_anomaly_history[stable_id] = self.track_anomaly_history[stable_id][-max_history:]
            self.track_confidence_history[stable_id] = self.track_confidence_history[stable_id][-max_history:]
        
        # Check if track is long enough for reliable detection
        track_length = len(self.track_anomaly_history[stable_id])
        if track_length < self.min_track_length:
            return False, "NORMAL"  # FIXED: Show as NORMAL instead of TRACKING
        
        # Get recent window for analysis
        recent_window = self.track_anomaly_history[stable_id][-self.anomaly_threshold_frames:]
        recent_scores = self.track_confidence_history[stable_id][-self.anomaly_threshold_frames:]
        
        if len(recent_window) < self.anomaly_threshold_frames:
            return False, "NORMAL"
        
        # Multiple criteria for anomaly confirmation
        anomaly_count = sum(recent_window)
        anomaly_ratio = anomaly_count / len(recent_window)
        avg_recent_score = np.mean(recent_scores)
        exp_score = self.track_exponential_scores[stable_id]
        
        # FIXED: Much more responsive anomaly confirmation
        is_confirmed_anomaly = (
            (anomaly_ratio >= self.anomaly_confirmation_ratio and avg_recent_score > 0.4) or  # LOWERED threshold
            (exp_score > 0.6 and anomaly_ratio > 0.3) or  # LOWERED: More sensitive
            (avg_recent_score > 1.0)  # LOWERED: Detect high scores faster
        )
        
        if is_confirmed_anomaly:
            return True, "ANOMALY"
        elif anomaly_ratio > 0.2 or exp_score > 0.4:  # LOWERED: Earlier warning
            return False, "WARNING"
        else:
            return False, "NORMAL"
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """Process video with improved tracking and anomaly detection"""
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        anomaly_detections = []
        id_switches = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run tracking with improved config
                results = self.yolo_model.track(
                    source=frame,
                    tracker="botsort_improved.yaml",
                    persist=True,
                    classes=[0],  # person only
                    conf=self.confidence_threshold,
                    imgsz=640,    # Higher resolution for small objects
                    verbose=False
                )
                
                # Process detections
                annotated_frame = frame.copy()
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        
                        # Filter low-quality detections
                        if not self.is_valid_detection(box.tolist(), conf):
                            continue
                        
                        # Get stable track ID
                        stable_id = self.get_stable_track_id(track_id, box.tolist(), conf)
                        
                        # Detect anomaly
                        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(
                            stable_id, box.tolist(), frame_idx
                        )
                        
                        # Apply advanced anomaly filtering
                        is_confirmed_anomaly, status = self.advanced_anomaly_smoothing(
                            stable_id, is_anomaly, anomaly_score, box.tolist(), width, height
                        )
                        
                        # FIXED: Choose color based on status (removed TRACKING)
                        if is_confirmed_anomaly:
                            color = self.anomaly_color      # Red for anomalies
                        elif status == "WARNING":
                            color = self.warning_color      # Orange for warnings
                        else:
                            color = self.normal_color       # Green for normal (no gray tracking)
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Create informative label
                        if track_id != stable_id:
                            id_switches += 1
                        
                        label = f"ID:{stable_id} ({track_id}) {status}"
                        if anomaly_score > 0:
                            label += f" {anomaly_score:.2f}"
                        
                        # Add confidence
                        label += f" C:{conf:.2f}"
                        
                        # Draw label
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), 
                                    color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Log confirmed anomalies
                        if is_confirmed_anomaly:
                            anomaly_detections.append({
                                'frame': frame_idx,
                                'stable_id': stable_id,
                                'original_id': track_id,
                                'bbox': box.tolist(),
                                'score': anomaly_score,
                                'confidence': conf,
                                'timestamp': frame_idx / fps
                            })
                
                # Add comprehensive info
                info_lines = [
                    f"Frame: {frame_idx}/{total_frames}",
                    f"Anomalies: {len(anomaly_detections)}",
                    f"ID Switches: {id_switches}",
                    f"Stable IDs: {len(self.track_position_history)}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(annotated_frame, line, (10, 30 + i*25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # FIXED: Updated legend (removed gray tracking)
                legend_y = 150
                legend_items = [
                    ("Green: Normal Behavior", self.normal_color),
                    ("Orange: Suspicious (Warning)", self.warning_color), 
                    ("Red: ANOMALY DETECTED!", self.anomaly_color)
                ]
                
                for i, (text, color) in enumerate(legend_items):
                    cv2.putText(annotated_frame, text, (10, legend_y + i*20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Improved Anomaly Detection', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 200 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Anomalies: {len(anomaly_detections)} | ID Switches: {id_switches}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Print comprehensive summary
        print(f"\n=== Improved Processing Complete ===")
        print(f"Total frames processed: {frame_idx}")
        print(f"Total anomalies detected: {len(anomaly_detections)}")
        print(f"Total ID switches: {id_switches}")
        print(f"Unique stable IDs: {len(self.track_position_history)}")
        print(f"ID switch rate: {id_switches/frame_idx*100:.2f}% per frame")
        
        if anomaly_detections:
            print(f"\nAnomaly Summary (Top 10):")
            for i, detection in enumerate(anomaly_detections[:10]):
                print(f"  {i+1}. Stable ID {detection['stable_id']} at {detection['timestamp']:.1f}s "
                      f"(Score: {detection['score']:.3f}, Conf: {detection['confidence']:.2f})")
        
        if output_path:
            print(f"Improved output saved to: {output_path}")
        
        return {
            'anomaly_detections': anomaly_detections,
            'id_switches': id_switches,
            'stable_ids': len(self.track_position_history),
            'total_frames': frame_idx
        }

def main():
    parser = argparse.ArgumentParser(description='Improved anomaly detection with better tracking')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--model', '-m', default='models/vae_anomaly_detector.pth', 
                       help='Path to trained VAE model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found!")
        return
    
    # Initialize improved tracker
    try:
        tracker = ImprovedAnomalyTracker(args.model)
    except FileNotFoundError:
        return
    
    # Process video
    display = not args.no_display
    results = tracker.process_video(args.input, args.output, display)
    
    print(f"\nImproved processing completed successfully!")
    print(f"Improvements: {results['id_switches']} ID switches, {results['stable_ids']} stable tracks")

if __name__ == "__main__":
    main()