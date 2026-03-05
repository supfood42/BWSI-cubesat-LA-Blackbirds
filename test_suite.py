"""
CubeSat Software Test Suite - Windows Compatible
Run this to test crater detection, mission FSM, and downlink without Raspberry Pi hardware.

Requirements:
- opencv-python (cv2)
- numpy
- matplotlib

Install: pip install opencv-python numpy matplotlib
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime

# Add project directory to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# Import test components
try:
    from crater_detection import CraterDetector, visualize_detections
    print("✓ crater_detection imported")
except Exception as e:
    print(f"✗ crater_detection import failed: {e}")
    sys.exit(1)

try:
    from mission_control import MissionStateMachine, MissionState
    print("✓ mission_control imported")
except Exception as e:
    print(f"✗ mission_control import failed: {e}")
    sys.exit(1)

try:
    from downlink import DataFilter
    print("✓ downlink imported")
except Exception as e:
    print(f"✗ downlink import failed: {e}")
    sys.exit(1)


def test_crater_detection():
    """Test crater detection on real crater images."""
    print("\n" + "="*70)
    print("TEST 1: CRATER DETECTION")
    print("="*70)
    
    image_dir = os.path.join(PROJECT_DIR, 'Pictures')
    if not os.path.exists(image_dir):
        print("✗ Pictures directory not found")
        return False
    
    test_images = [
        'PIA13642~orig.jpg',      # Mars crater image
        '22574_PIA23304.jpg',      # Another crater image
    ]
    
    detector = CraterDetector(confidence_threshold=0.3)
    results = {}
    
    for img_name in test_images:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"  ⊘ {img_name} not found, skipping")
            continue
        
        print(f"\n  Testing: {img_name}")
        try:
            # Load and preprocess
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"    ✗ Failed to load image")
                continue
            
            print(f"    Image shape: {img.shape}")
            
            # Enhance contrast (simple preprocessing)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img)
            
            # Detect craters
            craters = detector.detect_craters(img_enhanced, method='hough')
            
            print(f"    Craters detected: {len(craters)}")
            
            if craters:
                print(f"    Top detection:")
                top = craters[0]
                print(f"      Location: ({top.center_x}, {top.center_y})")
                print(f"      Radius: {top.radius}px")
                print(f"      Confidence: {top.confidence:.3f}")
                print(f"      Area fraction: {top.area_fraction:.4f}")
                
                results[img_name] = len(craters)
                print(f"    ✓ Success")
            else:
                print(f"    ⊘ No craters detected (may need threshold adjustment)")
                results[img_name] = 0
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results[img_name] = None
    
    return results


def test_mission_fsm():
    """Test mission state machine."""
    print("\n" + "="*70)
    print("TEST 2: MISSION STATE MACHINE")
    print("="*70)
    
    try:
        fsm = MissionStateMachine()
        print(f"\n  Initial state: {fsm.get_state().value}")
        
        # Boot sequence
        print(f"\n  Transitioning: BOOT → SCIENCE_OBSERVATION")
        fsm.transition(MissionState.SCIENCE_OBSERVATION)
        print(f"    Current state: {fsm.get_state().value}")
        
        # Simulate missions
        print(f"\n  Simulating 3 observation cycles...")
        for i in range(3):
            fsm.record_image_captured()
            fsm.transition(MissionState.IMAGE_PROCESSING)
            fsm.record_image_processed(num_craters=2 + i)
            fsm.transition(MissionState.DOWNLINK)
            fsm.record_data_ready()
            fsm.transition(MissionState.SCIENCE_OBSERVATION)
            print(f"    Cycle {i+1}: captured, processed {2+i} craters, prepared downlink")
        
        # Summary
        print(f"\n  {fsm.mission_summary()}")
        print(f"    ✓ Success")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_downlink_filtering():
    """Test bandwidth-aware downlink filtering."""
    print("\n" + "="*70)
    print("TEST 3: DOWNLINK DATA FILTERING")
    print("="*70)
    
    try:
        # Simulate detected features
        sample_features = [
            {'center': (100, 150), 'radius_px': 20, 'confidence': 0.95},
            {'center': (250, 180), 'radius_px': 18, 'confidence': 0.87},
            {'center': (320, 200), 'radius_px': 15, 'confidence': 0.62},
            {'center': (400, 120), 'radius_px': 12, 'confidence': 0.45},
        ]
        
        print(f"\n  Input features: {len(sample_features)} craters detected")
        
        for strategy in ['minimal', 'balanced', 'complete']:
            filter_obj = DataFilter(bandwidth_priority=strategy)
            
            packet = filter_obj.create_downlink_packet(
                image_id='TEST_IMG_001',
                capture_time=datetime.now().isoformat() + 'Z',
                image_shape=(480, 640, 3),
                features=sample_features,
                compress=True
            )
            
            dl_time = filter_obj.estimate_downlink_time([packet])
            
            print(f"\n  Strategy: {strategy.upper()}")
            print(f"    Features transmitted: {packet.features_detected}/{len(sample_features)}")
            print(f"    Packet size: {packet.compressed_size_bytes} bytes")
            print(f"    Est. downlink time: {dl_time:.3f}s @ 9.6 kbps")
        
        print(f"\n    ✓ Success")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_image_processing():
    """Test full image processing pipeline."""
    print("\n" + "="*70)
    print("TEST 4: IMAGE PROCESSING PIPELINE")
    print("="*70)
    
    try:
        from image_process import ImageProcessor
        
        image_dir = os.path.join(PROJECT_DIR, 'Pictures')
        test_image = os.path.join(image_dir, 'PIA13642~orig.jpg')
        
        if not os.path.exists(test_image):
            print(f"  ⊘ Test image not found")
            return False
        
        print(f"\n  Loading: PIA13642~orig.jpg")
        processor = ImageProcessor()
        result = processor.process_image(test_image, detect_craters=True)
        
        print(f"    Image shape: {result['image_shape']}")
        print(f"    Image loaded: ✓")
        print(f"    Preprocessing: ✓")
        print(f"    Crater detection: ✓")
        print(f"    Craters detected: {result['craters_detected']}")
        
        if result['features']:
            print(f"    Feature metadata: {len(result['features'])} entries")
            print(f"    ✓ Success")
            return True
        else:
            print(f"    ⊘ No features detected")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "█"*70)
    print("   CubeSat Software Test Suite - Windows Compatibility")
    print("█"*70)
    
    results = {}
    
    # Test 1: Crater Detection
    results['Crater Detection'] = test_crater_detection()
    
    # Test 2: Mission FSM
    results['Mission Control FSM'] = test_mission_fsm()
    
    # Test 3: Downlink Filtering
    results['Downlink Filtering'] = test_downlink_filtering()
    
    # Test 4: Image Processing
    results['Image Processing'] = test_image_processing()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results.values() if r and r is not True)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:.<50} {status}")
    
    print("\n" + "█"*70)
    print(f"  {sum(1 for r in results.values() if r)}/{total} tests completed")
    print("█"*70 + "\n")


if __name__ == '__main__':
    main()
