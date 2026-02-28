#!/usr/bin/env python3
"""
Test Enhanced Stealing Detection System
"""

import cv2
import numpy as np
from stealing_detection_system import StealingDetectionSystem
import time
import os

def test_stealing_detection(video_path: str, max_frames: int = 500):
    """Test the enhanced stealing detection system"""
    
    print("🛡️ Testing Enhanced Stealing Detection System")
    print("=" * 60)
    
    # Initialize stealing detection system
    try:
        detector = StealingDetectionSystem()
        print("✅ Enhanced stealing detection system loaded!")
    except FileNotFoundError:
        print("❌ Model not found!")
        return
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"📊 Resolution: {width}x{height}, FPS: {fps}")
    print(f"⏱️  Test duration: {max_frames/fps:.1f} seconds")
    
    print(f"\n🛡️ STEALING DETECTION FEATURES:")
    print(f"  ✅ Level 1: Behavioral anomaly detection (VAE)")
    print(f"  ✅ Level 2: Hand-shelf interaction detection")
    print(f"  ✅ Level 3: Temporal pattern analysis")
    print(f"  ✅ Level 4: Multi-criteria threat assessment")
    print(f"  ✅ Level 5: Confirmed theft detection")
    
    frame_idx = 0
    threat_counts = {
        'normal': 0,
        'suspicious': 0,
        'high_risk': 0,
        'stealing': 0,
        'confirmed_theft': 0
    }
    
    hand_detections = 0
    shelf_interactions = 0
    behavioral_anomalies = 0
    start_time = time.time()
    
    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Hand detection test
            hands = detector.hand_detector.detect_hands(frame)
            if hands:
                hand_detections += len(hands)
            
            # Person detection and tracking
            results = detector.yolo_model.track(
                source=frame,
                tracker="botsort.yaml",
                persist=True,
                classes=[0],
                conf=0.25,    # Lower threshold to detect small/crouched persons
                verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Initialize shelf detector once (ShelfZoneDetector = AdaptiveZoneDetector)
                from stealing_detection_system import ShelfZoneDetector
                shelf_detector = ShelfZoneDetector(width, height)
                
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    if conf < 0.25:
                        continue
                    
                    # Get person's hands
                    person_hands = detector._get_person_hands(box, hands)
                    
                    # Test shelf interactions
                    shelf_interaction = shelf_detector.detect_hand_shelf_interaction(
                        person_hands, box.tolist()
                    )
                    
                    if shelf_interaction['has_interaction']:
                        shelf_interactions += 1
                    
                    # Test behavioral anomaly
                    is_anomaly, anomaly_score = detector.anomaly_detector.detect_anomaly(
                        track_id, box.tolist(), frame_idx
                    )
                    
                    if is_anomaly:
                        behavioral_anomalies += 1
                    
                    # Get global_id from ReID if enabled
                    global_id = None
                    if detector.enable_reid and detector.reid_tracker:
                        timestamp = frame_idx / fps
                        global_id = detector.reid_tracker.update_global_tracking(
                            detector.camera_id, track_id, frame, box.tolist(), conf, timestamp
                        )
                    
                    # Test comprehensive stealing analysis
                    analysis = detector.analyze_stealing_behavior(
                        track_id, global_id, box.tolist(), person_hands,
                        shelf_interaction, frame_idx, fps
                    )
                    
                    # Count threat levels
                    threat_level = analysis['threat_level']
                    threat_counts[threat_level] += 1
                    
                    # Log high-priority detections
                    if threat_level in ['stealing', 'confirmed_theft']:
                        print(f"🚨 Frame {frame_idx}: {threat_level.upper()} detected!")
                        print(f"   ID: {track_id}, Score: {analysis['scores']['final_score']:.3f}")
                        print(f"   Duration: {analysis['duration']:.1f}s")
                        print(f"   Interactions: {analysis['interaction_count']}")
                        print(f"   Anomalies: {analysis['anomaly_count']}")
                    
                    elif threat_level == 'high_risk' and frame_idx % 50 == 0:
                        print(f"⚠️  Frame {frame_idx}: HIGH RISK (ID: {track_id}, Score: {analysis['scores']['final_score']:.3f})")
            
            frame_idx += 1
            
            # Progress update
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_idx / elapsed if elapsed > 0 else 0
                progress = (frame_idx / max_frames) * 100
                
                print(f"📊 Progress: {progress:.1f}% | FPS: {fps_current:.1f}")
                print(f"   Hands: {hand_detections} | Interactions: {shelf_interactions} | Anomalies: {behavioral_anomalies}")
    
    finally:
        cap.release()
    
    # Final comprehensive results
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    total_detections = sum(threat_counts.values())
    
    print(f"\n🎯 ENHANCED STEALING DETECTION RESULTS:")
    print(f"=" * 50)
    print(f"📊 Frames processed: {frame_idx}")
    print(f"🎭 Total person detections: {total_detections}")
    print(f"👋 Hand detections: {hand_detections}")
    print(f"🛒 Shelf interactions: {shelf_interactions}")
    print(f"🚨 Behavioral anomalies: {behavioral_anomalies}")
    print(f"⚡ Average FPS: {avg_fps:.1f}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    print(f"\n🛡️ THREAT LEVEL BREAKDOWN:")
    print(f"=" * 30)
    for level, count in threat_counts.items():
        if count > 0:
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            emoji = {
                'normal': '✅',
                'suspicious': '🟡', 
                'high_risk': '🟠',
                'stealing': '🔴',
                'confirmed_theft': '🟣'
            }[level]
            print(f"  {emoji} {level.upper()}: {count} ({percentage:.1f}%)")
    
    # Success indicators
    print(f"\n🎉 SYSTEM CAPABILITIES VERIFIED:")
    success_indicators = []
    
    if hand_detections > 0:
        success_indicators.append("✅ Hand detection working!")
    if shelf_interactions > 0:
        success_indicators.append("✅ Shelf interaction detection working!")
    if behavioral_anomalies > 0:
        success_indicators.append("✅ Behavioral anomaly detection working!")
    if threat_counts['stealing'] > 0 or threat_counts['confirmed_theft'] > 0:
        success_indicators.append("✅ Stealing detection working!")
    if threat_counts['high_risk'] > 0:
        success_indicators.append("✅ Risk assessment working!")
    
    for indicator in success_indicators:
        print(f"  {indicator}")
    
    # Performance assessment
    print(f"\n📈 PERFORMANCE ASSESSMENT:")
    if avg_fps > 15:
        print(f"  ✅ Real-time performance: {avg_fps:.1f} FPS (Good)")
    elif avg_fps > 10:
        print(f"  ⚠️  Near real-time: {avg_fps:.1f} FPS (Acceptable)")
    else:
        print(f"  ❌ Slow processing: {avg_fps:.1f} FPS (Needs optimization)")
    
    # Detection effectiveness
    if total_detections > 0:
        high_priority_rate = (threat_counts['stealing'] + threat_counts['confirmed_theft']) / total_detections * 100
        if high_priority_rate > 0:
            print(f"  🎯 High-priority detection rate: {high_priority_rate:.1f}%")
        
        interaction_rate = (shelf_interactions / total_detections * 100) if total_detections > 0 else 0
        print(f"  🛒 Shelf interaction rate: {interaction_rate:.1f}%")
    
    print(f"\n🏆 STEALING DETECTION SYSTEM TEST COMPLETE!")

def test_quick_stealing_demo():
    """Quick demo test with a short video segment"""
    
    print("🚀 Quick Stealing Detection Demo")
    print("=" * 40)
    
    # Test with a shorter segment for quick validation
    test_video = "working/test_anomaly/Shoplifting020_x264.mp4"
    
    if os.path.exists(test_video):
        test_stealing_detection(test_video, max_frames=200)
    else:
        print(f"❌ Test video not found: {test_video}")
        print("Available videos:")
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    print(f"  {os.path.join(root, file)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test stealing detection system')
    parser.add_argument('--video', '-v', help='Video path to test')
    parser.add_argument('--frames', '-f', type=int, default=500, help='Max frames to process')
    parser.add_argument('--quick', '-q', action='store_true', help='Run quick demo')
    
    args = parser.parse_args()
    
    if args.quick:
        test_quick_stealing_demo()
    elif args.video:
        if os.path.exists(args.video):
            test_stealing_detection(args.video, args.frames)
        else:
            print(f"❌ Video not found: {args.video}")
    else:
        # Default quick test
        test_quick_stealing_demo()