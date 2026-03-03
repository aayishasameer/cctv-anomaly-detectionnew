#!/usr/bin/env python3
"""
Person Re-Identification System for Multi-Camera Tracking
Assigns global IDs to persons across different camera angles and views
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pickle
import os
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
import torchreid


class PersonReIDExtractor:
    """Professional Person ReID feature extractor using OSNet"""

    def __init__(self, device="auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto" else "cpu"
        )

        print("🔄 Loading OSNet ReID model...")

        # Build OSNet model pretrained on Market1501
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )

        self.model.to(self.device)
        self.model.eval()

        print("✅ OSNet ReID model loaded successfully")

        # Preprocessing (OSNet standard)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, person_crop: np.ndarray) -> np.ndarray:
        if person_crop is None or person_crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        try:
            # Convert BGR → RGB
            person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            input_tensor = self.transform(person_crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(input_tensor)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                features = features.cpu().numpy().flatten()

            return features.astype(np.float32)

        except Exception as e:
            print(f"⚠️ ReID extraction error: {e}")
            return np.zeros(512, dtype=np.float32)

class GlobalPersonTracker:
    """Global person tracker with ReID for multi-camera scenarios"""
    
    def __init__(self, reid_model_path: str = "models/person_reid_model.pth"):
        # Initialize ReID feature extractor
        self.reid_extractor = PersonReIDExtractor(device="auto")
        
        # Global tracking data
        self.global_persons = {}  # global_id -> person_data
        self.camera_tracks = defaultdict(dict)  # camera_id -> {local_id -> global_id}
        self.next_global_id = 1
        
        # ReID parameters - More conservative to prevent false matches
        self.similarity_threshold = 0.75  # Higher threshold to prevent false matches
        self.max_time_gap = 30.0  # Maximum time gap for re-identification (seconds)
        self.min_feature_quality = 0.6  # Higher quality threshold
        self.strict_matching = True  # Enable strict matching mode
        
        # Feature gallery for each global person
        self.feature_gallery = defaultdict(list)  # global_id -> [features]
        self.max_gallery_size = 10  # Maximum features per person
        
        # Tracking statistics
        self.reid_matches = 0
        self.new_persons = 0
        self.total_tracks = 0
    
    def extract_person_crop(self, frame: np.ndarray, bbox: List[float], 
                           padding: int = 10) -> np.ndarray:
        """Extract person crop from frame with padding"""
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Extract crop
        person_crop = frame[y1:y2, x1:x2]
        
        return person_crop
    
    def assess_crop_quality(self, person_crop: np.ndarray) -> float:
        """Assess the quality of person crop for ReID"""
        
        if person_crop is None or person_crop.size == 0:
            return 0.0
        
        h, w = person_crop.shape[:2]
        
        # Size quality (prefer larger crops)
        size_score = min(1.0, (h * w) / (128 * 256))
        
        # Aspect ratio quality (prefer person-like ratios)
        aspect_ratio = h / w if w > 0 else 0
        aspect_score = 1.0 - abs(aspect_ratio - 2.0) / 2.0  # Ideal ratio ~2.0
        aspect_score = max(0.0, aspect_score)
        
        # Blur detection (simple variance of Laplacian)
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY) if len(person_crop.shape) == 3 else person_crop
        blur_score = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0)
        
        # Combined quality score
        quality = (size_score * 0.4 + aspect_score * 0.3 + blur_score * 0.3)
        
        return quality
    
    def find_best_match(self, query_features: np.ndarray, 
                       camera_id: str, timestamp: float, 
                       current_bbox: List[float] = None) -> Optional[int]:
        """Find best matching global person using ReID features with strict validation"""
        
        if len(self.feature_gallery) == 0:
            return None
        
        best_match_id = None
        best_similarity = 0.0
        
        for global_id, gallery_features in self.feature_gallery.items():
            if len(gallery_features) == 0:
                continue
            
            # Check time constraint
            person_data = self.global_persons.get(global_id, {})
            last_seen = person_data.get('last_seen', 0)
            
            if timestamp - last_seen > self.max_time_gap:
                continue
            
            # Skip if person is currently active in the same camera
            # This prevents assigning same global ID to multiple people in same camera
            if camera_id in person_data.get('cameras', set()):
                recent_detection_gap = timestamp - last_seen
                if recent_detection_gap < 5.0:  # Less than 5 seconds ago
                    continue
            
            # Calculate similarity with gallery features
            gallery_array = np.array(gallery_features)
            similarities = cosine_similarity([query_features], gallery_array)[0]
            max_similarity = np.max(similarities)
            avg_similarity = np.mean(similarities)
            
            # Use both max and average similarity for better matching
            combined_similarity = 0.7 * max_similarity + 0.3 * avg_similarity
            
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_match_id = global_id
        
        # Strict matching: require high similarity AND consistency
        if self.strict_matching:
            if best_similarity >= self.similarity_threshold:
                # Additional validation: check if this would create conflicts
                if self._validate_match(best_match_id, camera_id, timestamp):
                    return best_match_id
            return None
        else:
            # Return match if above threshold
            if best_similarity >= self.similarity_threshold:
                return best_match_id
        
        return None
    
    def _validate_match(self, global_id: int, camera_id: str, timestamp: float) -> bool:
        """Validate that a match doesn't create conflicts"""
        
        if global_id not in self.global_persons:
            return True
        
        person_data = self.global_persons[global_id]
        
        # Check if person was recently seen in same camera
        if camera_id in person_data.get('cameras', set()):
            time_gap = timestamp - person_data.get('last_seen', 0)
            if time_gap < 2.0:  # Less than 2 seconds - likely same detection
                return True
            elif time_gap < 10.0:  # 2-10 seconds - suspicious, could be different person
                return False
        
        return True
    
    def update_global_tracking(self, camera_id: str, local_track_id: int, 
                             frame: np.ndarray, bbox: List[float], 
                             confidence: float, timestamp: float) -> int:
        """Update global tracking with new detection and conflict resolution"""
        
        self.total_tracks += 1
        
        # Extract person crop
        person_crop = self.extract_person_crop(frame, bbox)
        
        # Assess crop quality
        crop_quality = self.assess_crop_quality(person_crop)
        
        if crop_quality < self.min_feature_quality:
            # Low quality crop, use existing mapping if available
            if local_track_id in self.camera_tracks[camera_id]:
                existing_global_id = self.camera_tracks[camera_id][local_track_id]
                # Update timestamp for existing track
                if existing_global_id in self.global_persons:
                    self.global_persons[existing_global_id]['last_seen'] = timestamp
                    self.global_persons[existing_global_id]['total_detections'] += 1
                return existing_global_id
            else:
                # Create new global ID with placeholder
                global_id = self.next_global_id
                self.next_global_id += 1
                self.camera_tracks[camera_id][local_track_id] = global_id
                self.global_persons[global_id] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'cameras': {camera_id},
                    'local_tracks': {camera_id: local_track_id},
                    'total_detections': 1,
                    'quality_scores': [crop_quality]
                }
                self.new_persons += 1
                return global_id
        
        # Extract ReID features
        reid_features = self.reid_extractor.extract_features(person_crop)
        
        # Check if this local track already has a global ID
        if local_track_id in self.camera_tracks[camera_id]:
            global_id = self.camera_tracks[camera_id][local_track_id]
            
            # Validate that this global ID is not being used by another active track in same camera
            if self._check_id_conflict(camera_id, local_track_id, global_id, timestamp):
                # Conflict detected - create new global ID
                print(f"⚠️  ID conflict detected for camera {camera_id}, local {local_track_id}, global {global_id}")
                global_id = self.next_global_id
                self.next_global_id += 1
                self.camera_tracks[camera_id][local_track_id] = global_id
                
                # Create new global person entry
                self.global_persons[global_id] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'cameras': {camera_id},
                    'local_tracks': {camera_id: local_track_id},
                    'total_detections': 1,
                    'quality_scores': [crop_quality]
                }
                
                # Initialize feature gallery
                self.feature_gallery[global_id] = [reid_features]
                self.new_persons += 1
                
                return global_id
            
            # Update existing global person
            if global_id in self.global_persons:
                person_data = self.global_persons[global_id]
                person_data['last_seen'] = timestamp
                person_data['cameras'].add(camera_id)
                person_data['total_detections'] += 1
                person_data['quality_scores'].append(crop_quality)
                
                # Update feature gallery if quality is good
                if crop_quality >= self.min_feature_quality:
                    self.feature_gallery[global_id].append(reid_features)
                    
                    # Limit gallery size
                    if len(self.feature_gallery[global_id]) > self.max_gallery_size:
                        self.feature_gallery[global_id] = self.feature_gallery[global_id][-self.max_gallery_size:]
            
            return global_id
        
        # New local track - try to match with existing global persons
        best_match_id = self.find_best_match(reid_features, camera_id, timestamp, bbox)
        
        if best_match_id is not None:
            # Additional validation to prevent false matches
            if self._validate_reid_match(best_match_id, camera_id, timestamp, reid_features):
                # Match found - assign existing global ID
                global_id = best_match_id
                self.camera_tracks[camera_id][local_track_id] = global_id
                
                # Update global person data
                person_data = self.global_persons[global_id]
                person_data['last_seen'] = timestamp
                person_data['cameras'].add(camera_id)
                person_data['local_tracks'][camera_id] = local_track_id
                person_data['total_detections'] += 1
                person_data['quality_scores'].append(crop_quality)
                
                # Add to feature gallery
                self.feature_gallery[global_id].append(reid_features)
                if len(self.feature_gallery[global_id]) > self.max_gallery_size:
                    self.feature_gallery[global_id] = self.feature_gallery[global_id][-self.max_gallery_size:]
                
                self.reid_matches += 1
                
            else:
                # Match validation failed - create new person
                best_match_id = None
        
        if best_match_id is None:
            # No match found or validation failed - create new global person
            global_id = self.next_global_id
            self.next_global_id += 1
            
            self.camera_tracks[camera_id][local_track_id] = global_id
            
            # Create new global person entry
            self.global_persons[global_id] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'cameras': {camera_id},
                'local_tracks': {camera_id: local_track_id},
                'total_detections': 1,
                'quality_scores': [crop_quality]
            }
            
            # Initialize feature gallery
            self.feature_gallery[global_id] = [reid_features]
            
            self.new_persons += 1
        
        return global_id
    
    def _check_id_conflict(self, camera_id: str, local_track_id: int, 
                          global_id: int, timestamp: float) -> bool:
        """Check if there's an ID conflict in the same camera"""
        
        # Check if any other local track in same camera has same global ID
        for other_local_id, other_global_id in self.camera_tracks[camera_id].items():
            if (other_local_id != local_track_id and 
                other_global_id == global_id and
                other_global_id in self.global_persons):
                
                # Check if the other track is still active
                other_last_seen = self.global_persons[other_global_id]['last_seen']
                if timestamp - other_last_seen < 5.0:  # Active within last 5 seconds
                    return True  # Conflict detected
        
        return False
    
    def _validate_reid_match(self, global_id: int, camera_id: str, 
                           timestamp: float, query_features: np.ndarray) -> bool:
        """Additional validation for ReID matches"""
        
        if global_id not in self.global_persons:
            return False
        
        person_data = self.global_persons[global_id]
        
        # Check temporal consistency
        last_seen = person_data.get('last_seen', 0)
        time_gap = timestamp - last_seen
        
        # If person was seen very recently in same camera, it's suspicious
        if camera_id in person_data.get('cameras', set()) and time_gap < 1.0:
            return False
        
        # Check feature consistency with multiple gallery features
        if global_id in self.feature_gallery and len(self.feature_gallery[global_id]) >= 3:
            gallery_features = np.array(self.feature_gallery[global_id])
            similarities = cosine_similarity([query_features], gallery_features)[0]
            
            # Require consistent high similarity with multiple features
            high_sim_count = np.sum(similarities > 0.75)
            if high_sim_count < len(similarities) * 0.6:  # At least 60% high similarity
                return False
        
        return True
    
    def get_global_id(self, camera_id: str, local_track_id: int) -> Optional[int]:
        """Get global ID for a local track"""
        return self.camera_tracks[camera_id].get(local_track_id)
    
    def get_last_known_id(self, camera_id: str, local_track_id: int) -> int:
        """Get last known global ID for a local track, or create new if not found"""
        if local_track_id in self.camera_tracks[camera_id]:
            return self.camera_tracks[camera_id][local_track_id]
        else:
            # Create new global ID if not found
            global_id = self.next_global_id
            self.next_global_id += 1
            self.camera_tracks[camera_id][local_track_id] = global_id
            return global_id
    
    def get_person_info(self, global_id: int) -> Dict:
        """Get information about a global person"""
        return self.global_persons.get(global_id, {})
    
    def get_tracking_statistics(self) -> Dict:
        """Get tracking statistics"""
        return {
            'total_global_persons': len(self.global_persons),
            'total_tracks_processed': self.total_tracks,
            'reid_matches': self.reid_matches,
            'new_persons_created': self.new_persons,
            'reid_match_rate': self.reid_matches / max(1, self.total_tracks),
            'active_cameras': len(self.camera_tracks),
            'avg_detections_per_person': np.mean([p['total_detections'] for p in self.global_persons.values()]) if self.global_persons else 0
        }
    
    def cleanup_old_tracks(self, current_timestamp: float, max_age: float = 300.0):
        """Clean up old tracks that haven't been seen recently"""
        
        to_remove = []
        
        for global_id, person_data in self.global_persons.items():
            if current_timestamp - person_data['last_seen'] > max_age:
                to_remove.append(global_id)
        
        # Remove old tracks
        for global_id in to_remove:
            del self.global_persons[global_id]
            if global_id in self.feature_gallery:
                del self.feature_gallery[global_id]
            
            # Remove from camera tracks
            for camera_id, tracks in self.camera_tracks.items():
                tracks_to_remove = [local_id for local_id, gid in tracks.items() if gid == global_id]
                for local_id in tracks_to_remove:
                    del tracks[local_id]
        
        if to_remove:
            print(f"🧹 Cleaned up {len(to_remove)} old global tracks")
    
    def save_reid_data(self, filepath: str = "models/reid_tracking_data.pkl"):
        """Save ReID tracking data"""
        
        data = {
            'global_persons': dict(self.global_persons),
            'camera_tracks': dict(self.camera_tracks),
            'feature_gallery': dict(self.feature_gallery),
            'next_global_id': self.next_global_id,
            'statistics': self.get_tracking_statistics()
        }
        
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 ReID tracking data saved to {filepath}")
    
    def load_reid_data(self, filepath: str = "models/reid_tracking_data.pkl") -> bool:
        """Load ReID tracking data"""
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.global_persons = data.get('global_persons', {})
            self.camera_tracks = defaultdict(dict, data.get('camera_tracks', {}))
            self.feature_gallery = defaultdict(list, data.get('feature_gallery', {}))
            self.next_global_id = data.get('next_global_id', 1)
            
            print(f"📂 ReID tracking data loaded from {filepath}")
            print(f"   Global persons: {len(self.global_persons)}")
            print(f"   Active cameras: {len(self.camera_tracks)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading ReID data: {e}")
            return False

def main():
    """Test the ReID system"""
    
    print("🔍 Testing Person Re-Identification System")
    print("=" * 50)
    
    # Initialize global tracker
    tracker = GlobalPersonTracker()
    
    # Test with dummy data
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Simulate detections from multiple cameras
    detections = [
        ("cam1", 1, [100, 100, 200, 300], 0.8, 1.0),
        ("cam1", 2, [300, 150, 400, 350], 0.9, 2.0),
        ("cam2", 1, [120, 110, 220, 310], 0.7, 3.0),  # Should match cam1_1
        ("cam2", 2, [350, 200, 450, 400], 0.8, 4.0),  # Should match cam1_2
        ("cam3", 1, [80, 90, 180, 290], 0.6, 5.0),    # Should match cam1_1
    ]
    
    print("🎬 Processing test detections...")
    
    for camera_id, local_id, bbox, conf, timestamp in detections:
        global_id = tracker.update_global_tracking(
            camera_id, local_id, dummy_frame, bbox, conf, timestamp
        )
        
        print(f"  📹 {camera_id} | Local ID: {local_id} | Global ID: {global_id}")
    
    # Print statistics
    stats = tracker.get_tracking_statistics()
    print(f"\n📊 ReID System Statistics:")
    print(f"  🌍 Global persons: {stats['total_global_persons']}")
    print(f"  🔄 Total tracks: {stats['total_tracks_processed']}")
    print(f"  ✅ ReID matches: {stats['reid_matches']}")
    print(f"  🆕 New persons: {stats['new_persons_created']}")
    print(f"  📈 Match rate: {stats['reid_match_rate']:.2%}")
    
    # Save test data
    tracker.save_reid_data("test_reid_data.pkl")
    
    print(f"\n✅ ReID system test completed!")

if __name__ == "__main__":
    main()