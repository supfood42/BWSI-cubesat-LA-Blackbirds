"""
Example Mission Workflow
Demonstrates how crater_detection, image_process, mission_control, and downlink
work together for a complete CubeSat imaging observation.

Run this to see the full mission pipeline in action.
"""

import cv2
from image_process import ImageProcessor
from downlink import DataFilter
from mission_control import MissionStateMachine, MissionState, MissionEvent
from datetime import datetime
import json


def example_mission_workflow():
    """
    Complete example mission: capture → process → detect → downlink.
    Uses sample crater image if available.
    """
    print("\n" + "="*60)
    print("    CubeSat Imaging Mission - Example Workflow")
    print("="*60 + "\n")
    
    # Initialize spacecraft systems
    fsm = MissionStateMachine()
    processor = ImageProcessor()
    downlink_filter = DataFilter(bandwidth_priority='balanced')
    
    # Define callbacks for state transitions
    def on_observation_start(status):
        print("  >>> Cameras powered on, IMU monitoring active")
    
    def on_processing_start(status):
        print("  >>> Beginning onboard image analysis...")
    
    def on_downlink_start(status):
        print("  >>> Preparing data packets for ground station")
    
    fsm.register_callback(MissionState.SCIENCE_OBSERVATION, on_observation_start)
    fsm.register_callback(MissionState.IMAGE_PROCESSING, on_processing_start)
    fsm.register_callback(MissionState.DOWNLINK, on_downlink_start)
    
    # MISSION SEQUENCE
    print("[1] BOOT AND INITIALIZATION")
    print("-" * 60)
    print("  Spacecraft systems: nominal")
    print("  Payload status: ready")
    print("  Memory available: 256 MB")
    print("  Downlink rate: 9.6 kbps\n")
    
    # Transition to science
    fsm.transition(MissionState.SCIENCE_OBSERVATION, 
                  MissionEvent("boot_sequence_complete"))
    print()
    
    # Simulate image captures
    print("[2] SCIENCE OBSERVATION - IMAGE ACQUISITION")
    print("-" * 60)
    
    test_images = [
        'Pictures/PIA13642~orig.jpg',
        'Pictures/22574_PIA23304.jpg',
    ]
    
    downlink_packets = []
    
    for img_idx, image_path in enumerate(test_images, 1):
        try:
            print(f"\n  Image {img_idx} acquisition:")
            print(f"    File: {image_path}...", end='')
            
            # PROCESSING STATE
            fsm.transition(MissionState.IMAGE_PROCESSING)
            fsm.record_image_captured()
            
            # Process image
            result = processor.process_image(image_path, detect_craters=True)
            print(f" OK")
            print(f"    Resolution: {result['image_shape']}")
            print(f"    Features detected: {result['craters_detected']}")
            
            # Update FSM with processing results
            fsm.record_image_processed(num_craters=result['craters_detected'])
            
            # Create downlink packet
            timestamp = datetime.now().isoformat() + 'Z'
            packet = downlink_filter.create_downlink_packet(
                image_id=f'IMG_{img_idx:03d}',
                capture_time=timestamp,
                image_shape=result['image_shape'],
                features=result['features'],
                compress=True
            )
            downlink_packets.append(packet)
            
            print(f"    Packet size: {packet.compressed_size_bytes} bytes")
            
        except Exception as e:
            print(f" ERROR: {e}")
            continue
    
    if not downlink_packets:
        print("\n  No images processed. Exiting.")
        return
    
    print(f"\n  Processed {len(downlink_packets)} images with "
          f"{sum(p.features_detected for p in downlink_packets)} total features\n")
    
    # DOWNLINK STATE
    print("[3] DATA DOWNLINK")
    print("-" * 60)
    
    fsm.transition(MissionState.DOWNLINK)
    
    for packet in downlink_packets:
        fsm.record_data_ready(num_features=packet.features_detected)
    
    # Generate manifest
    manifest = downlink_filter.generate_manifest(downlink_packets)
    
    print(f"  Total packets ready: {manifest['total_packets']}")
    print(f"  Total features: {manifest['total_features_detected']}")
    print(f"  Total payload: {manifest['total_size_bytes']} bytes")
    print(f"  Est. downlink time: {manifest['estimated_downlink_time_sec']:.2f}s")
    
    # Show packet details
    print("\n  Packet Details:")
    for i, packet in enumerate(downlink_packets, 1):
        print(f"    Packet {i}: {packet.features_detected} features, "
              f"{packet.compressed_size_bytes} bytes")
        if packet.features:
            for j, feature in enumerate(packet.features[:2], 1):  # Show first 2
                if isinstance(feature, dict):
                    print(f"      - Feature {j}: x={feature.get('x', 'N/A')}, "
                          f"y={feature.get('y', 'N/A')}, "
                          f"r={feature.get('r', 'N/A')}px, "
                          f"confidence={feature.get('c', 'N/A')}")
            if len(packet.features) > 2:
                print(f"      ... and {len(packet.features) - 2} more")
    
    # Back to observation
    print()
    fsm.transition(MissionState.SCIENCE_OBSERVATION, 
                  MissionEvent("downlink_complete"))
    
    # Mission summary
    print("\n[4] MISSION STATUS")
    print("-" * 60)
    print(fsm.mission_summary())
    
    # Save manifest to file
    manifest_file = 'downlink_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDownlink manifest saved to: {manifest_file}")
    
    print("\n" + "="*60)
    print("    Mission Example Complete")
    print("="*60 + "\n")


if __name__ == '__main__':
    example_mission_workflow()
