#!/usr/bin/env python3
"""
Adaptive Zone Learning for Theft Detection
Automatically learns interaction zones from normal behavior videos
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json

class ActivityZoneLearner:
    """Learn interaction zones from normal behavior videos"""
    
    def __init__(self):
        self.yolo_model = YOLO("yolov8s.pt")  # 's' model for better small-object detection
        
        # Zone learning parameters
        self.low_speed_threshold = 2.0  # pixels per frame
        self.min_interaction_duration = 30  # frames (1 second at 30fps)
        self.clustering_eps = 50  # DBSCAN epsilon for spatial clustering
        self.min_samples = 5  # Minimum samples per cluster
        
        # Learned zones storage
        self.interaction_zones = []
        self.zone_statistics = {}
        
        # Data collection
        self.interaction_points = []
        self.interaction_metadata = []
        
    def extract_interaction_points(self, video_path: str) -> List[Dict]:
        """Extract low-speed interaction points from normal behavior video"""
        
        print(f"📹 Analyzing normal behavior: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_idx = 0
        track_histories = {}
        video_interactions = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Track people
                results = self.yolo_model.track(
                    source=frame,
                    tracker="botsort.yaml",
                    persist=True,
                    classes=[0],  # person only
                    conf=0.25,    # Lower threshold to detect small/crouched persons
                    imgsz=640,
                    verbose=False
                )
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, track_ids):
                        # Calculate person center
                        center_x = (box[0] + box[2]) / 2
                        center_y = (box[1] + box[3]) / 2
                        
                        # Initialize track history
                        if track_id not in track_histories:
                            track_histories[track_id] = {
                                'positions': [],
                                'timestamps': [],
                                'low_speed_periods': []
                            }
                        
                        history = track_histories[track_id]
                        history['positions'].append([center_x, center_y])
                        history['timestamps'].append(frame_idx / fps)
                        
                        # Calculate speed if we have previous positions
                        if len(history['positions']) > 1:
                            prev_pos = np.array(history['positions'][-2])
                            curr_pos = np.array(history['positions'][-1])
                            speed = np.linalg.norm(curr_pos - prev_pos)
                            
                            # Detect low-speed interaction
                            if speed < self.low_speed_threshold:
                                history['low_speed_periods'].append({
                                    'position': [center_x, center_y],
                                    'timestamp': frame_idx / fps,
                                    'frame': frame_idx,
                                    'speed': speed
                                })
                
                frame_idx += 1
                
                if frame_idx % 500 == 0:
                    print(f"  Processed {frame_idx} frames...")
        
        finally:
            cap.release()
        
        # Extract sustained interactions (low-speed periods)
        for track_id, history in track_histories.items():
            low_speed_periods = history['low_speed_periods']
            
            if len(low_speed_periods) < self.min_interaction_duration:
                continue
            
            # Group consecutive low-speed periods
            interaction_groups = self._group_consecutive_interactions(low_speed_periods)
            
            for group in interaction_groups:
                if len(group) >= self.min_interaction_duration:
                    # Calculate interaction zone center
                    positions = [p['position'] for p in group]
                    center = np.mean(positions, axis=0)
                    
                    interaction_data = {
                        'center': center.tolist(),
                        'duration': len(group) / fps,
                        'track_id': track_id,
                        'video': os.path.basename(video_path),
                        'start_time': group[0]['timestamp'],
                        'end_time': group[-1]['timestamp'],
                        'position_variance': np.var(positions, axis=0).tolist(),
                        'avg_speed': np.mean([p['speed'] for p in group])
                    }
                    
                    video_interactions.append(interaction_data)
                    self.interaction_points.append(center.tolist())
                    self.interaction_metadata.append(interaction_data)
        
        print(f"  ✅ Found {len(video_interactions)} interaction zones")
        return video_interactions
    
    def _group_consecutive_interactions(self, low_speed_periods: List[Dict]) -> List[List[Dict]]:
        """Group consecutive low-speed periods into interaction sessions"""
        
        if not low_speed_periods:
            return []
        
        groups = []
        current_group = [low_speed_periods[0]]
        
        for i in range(1, len(low_speed_periods)):
            current = low_speed_periods[i]
            previous = low_speed_periods[i-1]
            
            # Check if consecutive (within reasonable time gap)
            time_gap = current['timestamp'] - previous['timestamp']
            if time_gap < 2.0:  # Less than 2 seconds gap
                current_group.append(current)
            else:
                # Start new group
                if len(current_group) >= self.min_interaction_duration:
                    groups.append(current_group)
                current_group = [current]
        
        # Add final group
        if len(current_group) >= self.min_interaction_duration:
            groups.append(current_group)
        
        return groups
    
    def learn_zones_from_videos(self, normal_video_paths: List[str]) -> Dict:
        """Learn interaction zones from multiple normal behavior videos"""
        
        print("🧠 Learning Interaction Zones from Normal Behavior")
        print("=" * 60)
        
        # Clear previous data
        self.interaction_points = []
        self.interaction_metadata = []
        
        # Extract interactions from all videos
        all_interactions = []
        for video_path in normal_video_paths:
            if os.path.exists(video_path):
                interactions = self.extract_interaction_points(video_path)
                all_interactions.extend(interactions)
            else:
                print(f"⚠️  Video not found: {video_path}")
        
        if not self.interaction_points:
            print("❌ No interaction points found!")
            return {}
        
        print(f"\n📊 Total interaction points collected: {len(self.interaction_points)}")
        
        # Cluster interaction points to find zones
        zones = self._cluster_interaction_zones()
        
        # Analyze zone characteristics
        zone_analysis = self._analyze_learned_zones(all_interactions)
        
        # Save learned zones
        self._save_learned_zones(zones, zone_analysis)
        
        return {
            'zones': zones,
            'analysis': zone_analysis,
            'total_interactions': len(self.interaction_points),
            'videos_processed': len(normal_video_paths)
        }
    
    def _cluster_interaction_zones(self) -> List[Dict]:
        """Cluster interaction points to identify distinct zones"""
        
        print("\n🎯 Clustering interaction points...")
        
        # Normalize interaction points
        points = np.array(self.interaction_points)
        scaler = StandardScaler()
        normalized_points = scaler.fit_transform(points)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=self.clustering_eps / 100,  # Normalize for scaled data
            min_samples=self.min_samples
        ).fit(normalized_points)
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"  📍 Found {n_clusters} interaction zones")
        print(f"  🔇 Noise points: {n_noise}")
        
        # Create zone definitions
        zones = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_points = points[cluster_mask]
            
            # Calculate zone properties
            center = np.mean(cluster_points, axis=0)
            std = np.std(cluster_points, axis=0)
            
            # Define zone boundaries (mean ± 2*std for 95% coverage)
            x_min = max(0, center[0] - 2 * std[0])
            x_max = center[0] + 2 * std[0]
            y_min = max(0, center[1] - 2 * std[1])
            y_max = center[1] + 2 * std[1]
            
            zone = {
                'id': f'learned_zone_{cluster_id}',
                'center': center.tolist(),
                'bbox': [x_min, y_min, x_max, y_max],
                'std': std.tolist(),
                'point_count': int(np.sum(cluster_mask)),
                'density': float(np.sum(cluster_mask) / len(points)),
                'area': (x_max - x_min) * (y_max - y_min)
            }
            
            zones.append(zone)
        
        # Sort zones by density (most active first)
        zones.sort(key=lambda x: x['density'], reverse=True)
        
        return zones
    
    def _analyze_learned_zones(self, all_interactions: List[Dict]) -> Dict:
        """Analyze characteristics of learned zones"""
        
        print("\n📈 Analyzing zone characteristics...")
        
        analysis = {
            'total_interactions': len(all_interactions),
            'avg_interaction_duration': np.mean([i['duration'] for i in all_interactions]),
            'interaction_duration_std': np.std([i['duration'] for i in all_interactions]),
            'zone_coverage': 0.0,
            'interaction_patterns': {}
        }
        
        if all_interactions:
            durations = [i['duration'] for i in all_interactions]
            speeds = [i['avg_speed'] for i in all_interactions]
            
            analysis.update({
                'duration_stats': {
                    'min': float(np.min(durations)),
                    'max': float(np.max(durations)),
                    'mean': float(np.mean(durations)),
                    'std': float(np.std(durations)),
                    'median': float(np.median(durations))
                },
                'speed_stats': {
                    'min': float(np.min(speeds)),
                    'max': float(np.max(speeds)),
                    'mean': float(np.mean(speeds)),
                    'std': float(np.std(speeds))
                }
            })
        
        return analysis
    
    def _save_learned_zones(self, zones: List[Dict], analysis: Dict):
        """Save learned zones to file"""
        
        os.makedirs("models", exist_ok=True)
        
        zone_data = {
            'zones': zones,
            'analysis': analysis,
            'learning_params': {
                'low_speed_threshold': self.low_speed_threshold,
                'min_interaction_duration': self.min_interaction_duration,
                'clustering_eps': self.clustering_eps,
                'min_samples': self.min_samples
            },
            'metadata': {
                'total_points': len(self.interaction_points),
                'learning_timestamp': np.datetime64('now').astype(str)
            }
        }
        
        # Save as JSON
        with open("models/learned_interaction_zones.json", 'w') as f:
            json.dump(zone_data, f, indent=2)
        
        # Save as pickle for faster loading
        with open("models/learned_interaction_zones.pkl", 'wb') as f:
            pickle.dump(zone_data, f)
        
        print(f"✅ Learned zones saved to models/learned_interaction_zones.json")
    
    def visualize_learned_zones(self, sample_video_path: str, output_path: str = "learned_zones_visualization.jpg"):
        """Visualize learned zones on a sample frame"""
        
        if not self.interaction_zones:
            print("❌ No zones learned yet!")
            return
        
        # Get a sample frame
        cap = cv2.VideoCapture(sample_video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ Could not read sample video: {sample_video_path}")
            return
        
        # Draw zones on frame
        viz_frame = frame.copy()
        
        for i, zone in enumerate(self.interaction_zones):
            x1, y1, x2, y2 = [int(coord) for coord in zone['bbox']]
            center = [int(coord) for coord in zone['center']]
            
            # Draw zone boundary
            color = (0, 255, 0)  # Green
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(viz_frame, tuple(center), 5, (0, 0, 255), -1)
            
            # Add zone label
            label = f"Zone {i+1} ({zone['point_count']} interactions)"
            cv2.putText(viz_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add title
        title = f"Learned Interaction Zones ({len(self.interaction_zones)} zones)"
        cv2.putText(viz_frame, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(output_path, viz_frame)
        print(f"✅ Zone visualization saved to: {output_path}")
    
    def load_learned_zones(self, zones_path: str = "models/learned_interaction_zones.pkl") -> bool:
        """Load previously learned zones"""
        
        if not os.path.exists(zones_path):
            print(f"❌ Learned zones file not found: {zones_path}")
            return False
        
        try:
            with open(zones_path, 'rb') as f:
                zone_data = pickle.load(f)
            
            self.interaction_zones = zone_data['zones']
            self.zone_statistics = zone_data['analysis']
            
            print(f"✅ Loaded {len(self.interaction_zones)} learned zones")
            return True
            
        except Exception as e:
            print(f"❌ Error loading zones: {e}")
            return False

def main():
    """Main function to learn zones from normal videos"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Learn interaction zones from normal behavior videos')
    parser.add_argument('--normal-videos', '-n', nargs='+', required=True,
                       help='Paths to normal behavior videos')
    parser.add_argument('--visualize', '-v', help='Sample video for visualization')
    parser.add_argument('--output-viz', '-o', default='learned_zones.jpg',
                       help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Initialize zone learner
    learner = ActivityZoneLearner()
    
    # Learn zones from normal videos
    results = learner.learn_zones_from_videos(args.normal_videos)
    
    if results:
        print(f"\n🎯 ZONE LEARNING RESULTS:")
        print(f"=" * 40)
        print(f"📍 Zones learned: {len(results['zones'])}")
        print(f"📊 Total interactions: {results['total_interactions']}")
        print(f"🎬 Videos processed: {results['videos_processed']}")
        
        # Print zone details
        for i, zone in enumerate(results['zones']):
            print(f"\n  Zone {i+1}: {zone['id']}")
            print(f"    Center: ({zone['center'][0]:.1f}, {zone['center'][1]:.1f})")
            print(f"    Interactions: {zone['point_count']}")
            print(f"    Density: {zone['density']:.3f}")
            print(f"    Area: {zone['area']:.0f} pixels²")
        
        # Create visualization if requested
        if args.visualize:
            learner.visualize_learned_zones(args.visualize, args.output_viz)
    
    print(f"\n🏆 Zone learning complete!")
    print(f"📁 Zones saved to: models/learned_interaction_zones.json")
    print(f"🔄 Use these zones in the enhanced stealing detection system")

if __name__ == "__main__":
    main()