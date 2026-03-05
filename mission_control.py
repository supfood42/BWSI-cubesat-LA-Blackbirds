"""
CubeSat Mission Control State Machine
Models spacecraft operational states and transitions for imaging payloads.

States:
  - BOOT: Initialization and hardware check
  - SAFE_MODE: Safe configuration, minimal operations
  - SCIENCE_OBSERVATION: Active image capture
  - IMAGE_PROCESSING: Onboard analysis
  - DOWNLINK: Data transmission to ground
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional
import time
from datetime import datetime


class MissionState(Enum):
    """Spacecraft operational states."""
    BOOT = "BOOT"
    SAFE_MODE = "SAFE_MODE"
    SCIENCE_OBSERVATION = "SCIENCE_OBSERVATION"
    IMAGE_PROCESSING = "IMAGE_PROCESSING"
    DOWNLINK = "DOWNLINK"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class MissionEvent:
    """Event triggering state transition."""
    name: str
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)
    
    def __repr__(self):
        return f"MissionEvent({self.name} @ {datetime.fromtimestamp(self.timestamp).isoformat()})"


@dataclass
class MissionStatus:
    """Current mission status and telemetry."""
    current_state: MissionState
    images_captured: int = 0
    images_processed: int = 0
    craters_detected: int = 0
    data_ready_for_downlink: int = 0
    last_event: Optional[MissionEvent] = None
    state_entry_time: float = field(default_factory=time.time)
    
    def time_in_state(self) -> float:
        """Time spent in current state (seconds)."""
        return time.time() - self.state_entry_time
    
    def __repr__(self):
        time_str = f"{self.time_in_state():.1f}s"
        return (f"MissionStatus({self.current_state.value}, "
                f"captured={self.images_captured}, "
                f"processed={self.images_processed}, "
                f"craters={self.craters_detected}, "
                f"time_in_state={time_str})")


class MissionStateMachine:
    """
    Spacecraft state machine for imaging mission.
    Manages operational phases and data flow.
    """
    
    def __init__(self):
        self.state = MissionState.BOOT
        self.status = MissionStatus(current_state=self.state)
        self.transitions = {}
        self.callbacks = {}
        self._setup_transitions()
    
    def _setup_transitions(self):
        """Define valid state transitions."""
        self.transitions = {
            MissionState.BOOT: [MissionState.SAFE_MODE, MissionState.SCIENCE_OBSERVATION],
            MissionState.SAFE_MODE: [MissionState.SCIENCE_OBSERVATION, MissionState.SHUTDOWN],
            MissionState.SCIENCE_OBSERVATION: [MissionState.IMAGE_PROCESSING, MissionState.SAFE_MODE],
            MissionState.IMAGE_PROCESSING: [MissionState.DOWNLINK, MissionState.SCIENCE_OBSERVATION],
            MissionState.DOWNLINK: [MissionState.SCIENCE_OBSERVATION, MissionState.SAFE_MODE],
            MissionState.SHUTDOWN: []
        }
    
    def register_callback(self, state: MissionState, callback: Callable):
        """
        Register callback to execute on entering state.
        
        Parameters:
            state: MissionState to trigger callback
            callback: function to execute
        """
        if state not in self.callbacks:
            self.callbacks[state] = []
        self.callbacks[state].append(callback)
    
    def transition(self, target_state: MissionState, event: Optional[MissionEvent] = None) -> bool:
        """
        Attempt state transition.
        
        Parameters:
            target_state: desired next state
            event: event triggering transition
            
        Returns:
            True if transition successful, False otherwise
        """
        if target_state not in self.transitions.get(self.state, []):
            print(f"Invalid transition {self.state.value} -> {target_state.value}")
            return False
        
        # Exit current state
        print(f"[{self.state.value}] Exiting state (time in state: {self.status.time_in_state():.1f}s)")
        
        # Perform transition
        self.state = target_state
        self.status.current_state = target_state
        self.status.state_entry_time = time.time()
        self.status.last_event = event
        
        # Enter new state
        print(f"[{self.state.value}] Entering state")
        if self.state in self.callbacks:
            for callback in self.callbacks[self.state]:
                callback(self.status)
        
        return True
    
    def record_image_captured(self):
        """Log image capture event."""
        self.status.images_captured += 1
        print(f"  Image captured (total: {self.status.images_captured})")
    
    def record_image_processed(self, num_craters: int = 0):
        """Log image processing and crater detection."""
        self.status.images_processed += 1
        self.status.craters_detected += num_craters
        print(f"  Image processed (total: {self.status.images_processed})")
        if num_craters > 0:
            print(f"  Craters detected: {num_craters}")
    
    def record_data_ready(self, num_features: int = 1):
        """Log data ready for downlink."""
        self.status.data_ready_for_downlink += num_features
        print(f"  Data ready for downlink (total: {self.status.data_ready_for_downlink})")
    
    def get_status(self) -> MissionStatus:
        """Get current mission status."""
        return self.status
    
    def get_state(self) -> MissionState:
        """Get current state."""
        return self.state
    
    def mission_summary(self) -> str:
        """Generate mission summary."""
        return (
            f"Mission Summary:\n"
            f"  Current State: {self.state.value}\n"
            f"  Images Captured: {self.status.images_captured}\n"
            f"  Images Processed: {self.status.images_processed}\n"
            f"  Total Craters Detected: {self.status.craters_detected}\n"
            f"  Data Ready for Downlink: {self.status.data_ready_for_downlink}\n"
            f"  Time in Current State: {self.status.time_in_state():.1f}s"
        )


def example_mission():
    """Example mission workflow."""
    fsm = MissionStateMachine()
    
    # Define callbacks
    def on_enter_observation(status):
        print("  >> Starting observation mode, IMU monitoring active")
    
    def on_enter_processing(status):
        print("  >> Processing captured images...")
    
    def on_enter_downlink(status):
        print("  >> Preparing data for ground station transmission")
    
    fsm.register_callback(MissionState.SCIENCE_OBSERVATION, on_enter_observation)
    fsm.register_callback(MissionState.IMAGE_PROCESSING, on_enter_processing)
    fsm.register_callback(MissionState.DOWNLINK, on_enter_downlink)
    
    # Example mission sequence
    print("\n=== CubeSat Mission Timeline ===\n")
    
    # Boot sequence
    fsm.transition(MissionState.SCIENCE_OBSERVATION, 
                  MissionEvent("boot_complete"))
    
    # Simulate captures and processing
    for i in range(3):
        print(f"\n--- Cycle {i+1} ---")
        fsm.record_image_captured()
        time.sleep(0.5)
        
        fsm.transition(MissionState.IMAGE_PROCESSING)
        fsm.record_image_processed(num_craters=2 + i)
        
        fsm.transition(MissionState.DOWNLINK)
        fsm.record_data_ready(num_features=2 + i)
        
        fsm.transition(MissionState.SCIENCE_OBSERVATION)
    
    print(f"\n\n{fsm.mission_summary()}\n")


if __name__ == '__main__':
    example_mission()
