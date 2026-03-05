"""
CubeSat Downlink Data Filter
Manages bandwidth-efficient transmission by prioritizing high-confidence features
and metadata over full image data.

Goal: Maximize science return while respecting limited downlink bandwidth.
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class DownlinkPacket:
    """Minimal data packet for ground transmission."""
    image_id: str              # Unique image identifier
    capture_time: str          # ISO timestamp
    image_size: tuple          # (height, width) in pixels
    features_detected: int     # Number of features (craters)
    features: List[Dict]       # Feature metadata
    processing_status: str     # 'success', 'partial', 'failed'
    compressed_size_bytes: int # Estimated transmission size
    
    def to_json(self) -> str:
        """Serialize to compact JSON for transmission."""
        return json.dumps(asdict(self))
    
    def size_estimate(self) -> float:
        """Estimate transmission size in bytes."""
        return len(self.to_json())


class DataFilter:
    """
    Intelligent data filtering for limited-bandwidth downlink.
    Prioritizes high-confidence detections and reduces redundancy.
    """
    
    def __init__(self, bandwidth_priority: str = 'balanced'):
        """
        Initialize filter with bandwidth strategy.
        
        Parameters:
            bandwidth_priority: 'minimal', 'balanced', or 'complete'
              - minimal: only highest confidence detections
              - balanced: moderate quality threshold
              - complete: transmit all detections
        """
        self.bandwidth_priority = bandwidth_priority
        self.confidence_thresholds = {
            'minimal': 0.8,
            'balanced': 0.5,
            'complete': 0.0
        }
    
    def filter_features(self, features: List[Dict]) -> List[Dict]:
        """
        Filter features by confidence threshold based on bandwidth priority.
        
        Parameters:
            features: list of feature dictionaries from crater detection
            
        Returns:
            Filtered list of features to transmit
        """
        threshold = self.confidence_thresholds.get(self.bandwidth_priority, 0.5)
        return [f for f in features if f.get('confidence', 0) >= threshold]
    
    def compress_feature_metadata(self, feature: Dict) -> Dict:
        """
        Compress feature metadata to minimal essential data.
        
        Removes non-critical information to reduce packet size.
        
        Parameters:
            feature: feature dict with full metadata
            
        Returns:
            Compressed feature dict
        """
        compressed = {
            'x': feature.get('center', (0, 0))[0],     # Center X in pixels
            'y': feature.get('center', (0, 0))[1],     # Center Y in pixels
            'r': feature.get('radius_px', 0),           # Radius in pixels
            'c': round(feature.get('confidence', 0), 2) # Confidence (0.0-1.0)
        }
        return compressed
    
    def create_downlink_packet(self, image_id: str, capture_time: str,
                              image_shape: tuple, features: List[Dict],
                              processing_status: str = 'success',
                              compress: bool = True) -> DownlinkPacket:
        """
        Create optimized downlink packet.
        
        Parameters:
            image_id: unique identifier for image
            capture_time: ISO timestamp of capture
            image_shape: (height, width) of image
            features: detected feature list
            processing_status: processing result status
            compress: whether to compress metadata
            
        Returns:
            DownlinkPacket ready for transmission
        """
        # Filter by confidence
        filtered_features = self.filter_features(features)
        
        # Compress metadata if requested
        if compress:
            filtered_features = [self.compress_feature_metadata(f) 
                                for f in filtered_features]
        
        packet = DownlinkPacket(
            image_id=image_id,
            capture_time=capture_time,
            image_size=image_shape[:2],  # (height, width)
            features_detected=len(filtered_features),
            features=filtered_features,
            processing_status=processing_status,
            compressed_size_bytes=0  # Will be calculated
        )
        
        packet.compressed_size_bytes = int(packet.size_estimate())
        return packet
    
    def estimate_downlink_time(self, packets: List[DownlinkPacket],
                              bitrate_kbps: float = 9.6) -> float:
        """
        Estimate downlink time for packets.
        
        Parameters:
            packets: list of DownlinkPackets
            bitrate_kbps: communication link bitrate (default: 9.6 kbps for CubeSat)
            
        Returns:
            Estimated transmission time in seconds
        """
        total_bits = sum(p.compressed_size_bytes * 8 for p in packets)
        time_seconds = total_bits / (bitrate_kbps * 1000)
        return time_seconds
    
    def prioritize_packets(self, packets: List[DownlinkPacket],
                          max_total_size: Optional[int] = None) -> List[DownlinkPacket]:
        """
        Prioritize packets for transmission under bandwidth constraints.
        
        Packets with more detections are ranked higher.
        
        Parameters:
            packets: list of DownlinkPackets
            max_total_size: maximum total size in bytes (optional)
            
        Returns:
            Prioritized list of packets that fit within constraints
        """
        # Sort by number of features detected (descending)
        sorted_packets = sorted(packets, 
                               key=lambda p: p.features_detected, 
                               reverse=True)
        
        if max_total_size is None:
            return sorted_packets
        
        # Select packets that fit within size constraint
        selected = []
        total_size = 0
        for packet in sorted_packets:
            if total_size + packet.compressed_size_bytes <= max_total_size:
                selected.append(packet)
                total_size += packet.compressed_size_bytes
        
        return selected
    
    def generate_manifest(self, packets: List[DownlinkPacket]) -> Dict:
        """
        Generate data manifest for ground station.
        
        Provides statistics about downlink content.
        
        Parameters:
            packets: list of DownlinkPackets
            
        Returns:
            Manifest dictionary
        """
        total_packets = len(packets)
        total_features = sum(p.features_detected for p in packets)
        total_size = sum(p.compressed_size_bytes for p in packets)
        
        return {
            'manifest_version': '1.0',
            'total_packets': total_packets,
            'total_features_detected': total_features,
            'total_size_bytes': total_size,
            'estimated_downlink_time_sec': self.estimate_downlink_time(packets),
            'bandwidth_priority': self.bandwidth_priority,
            'packets': [asdict(p) for p in packets]
        }


def example_downlink():
    """Example downlink filtering workflow."""
    print("\n=== CubeSat Downlink Data Filter Example ===\n")
    
    # Simulated feature detections
    sample_features = [
        {'center': (100, 150), 'radius_px': 20, 'confidence': 0.95},
        {'center': (250, 180), 'radius_px': 18, 'confidence': 0.87},
        {'center': (320, 200), 'radius_px': 15, 'confidence': 0.62},
        {'center': (400, 120), 'radius_px': 12, 'confidence': 0.45},
    ]
    
    # Create filters with different strategies
    for strategy in ['minimal', 'balanced', 'complete']:
        print(f"Strategy: {strategy.upper()}")
        filter_obj = DataFilter(bandwidth_priority=strategy)
        
        packet = filter_obj.create_downlink_packet(
            image_id='IMG_001',
            capture_time='2026-03-04T12:30:00Z',
            image_shape=(480, 640, 3),
            features=sample_features,
            compress=True
        )
        
        print(f"  Features transmitted: {packet.features_detected}")
        print(f"  Packet size: {packet.compressed_size_bytes} bytes")
        print(f"  Estimated downlink time: {filter_obj.estimate_downlink_time([packet]):.2f}s")
        print()


if __name__ == '__main__':
    example_downlink()
