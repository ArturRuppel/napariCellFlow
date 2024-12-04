import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SegmentationState:
    """Maintains the state of segmentation processing"""
    # Core data
    full_stack: Optional[np.ndarray] = None  # Complete 3D stack of masks
    current_frame: int = 0  # Currently active frame
    total_frames: Optional[int] = None  # Total number of frames in stack

    # Processing state
    is_processing: bool = False  # Lock for operations
    processed_frames: Dict[int, bool] = None  # Track which frames have been processed

    # Metadata
    original_shape: Optional[Tuple[int, ...]] = None  # Original data dimensions
    last_error: Optional[str] = None  # Last error message if any
    metadata: Dict[str, Any] = None  # Additional metadata

    def __post_init__(self):
        """Initialize dependent fields after creation"""
        self.processed_frames = {}
        self.metadata = {}

    def initialize_stack(self, shape: Tuple[int, ...]) -> None:
        """Initialize an empty stack with given shape"""
        if len(shape) < 2:
            raise ValueError("Shape must be at least 2D")

        if len(shape) == 2:
            # Single frame - create 3D stack with one frame
            self.full_stack = np.zeros((1, *shape), dtype=np.uint16)
            self.total_frames = 1
        else:
            # 3D stack
            self.full_stack = np.zeros(shape, dtype=np.uint16)
            self.total_frames = shape[0]

        self.original_shape = shape
        self.processed_frames = {i: False for i in range(self.total_frames)}

    def update_frame(self, frame_index: int, mask: np.ndarray) -> None:
        """Update a single frame in the stack"""
        if self.full_stack is None:
            raise ValueError("Stack not initialized")

        if frame_index >= self.total_frames:
            raise ValueError(f"Frame index {frame_index} out of bounds")

        if mask.shape != self.full_stack.shape[1:]:
            raise ValueError(f"Mask shape {mask.shape} doesn't match stack shape {self.full_stack.shape[1:]}")

        self.full_stack[frame_index] = mask
        self.processed_frames[frame_index] = True
        self.current_frame = frame_index

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Get a specific frame from the stack"""
        if self.full_stack is None or frame_index >= self.total_frames:
            return None
        return self.full_stack[frame_index]

    def is_frame_processed(self, frame_index: int) -> bool:
        """Check if a specific frame has been processed"""
        return self.processed_frames.get(frame_index, False)

    def all_frames_processed(self) -> bool:
        """Check if all frames have been processed"""
        return all(self.processed_frames.values())

    def get_processing_progress(self) -> Tuple[int, int]:
        """Get progress as (processed_frames, total_frames)"""
        if not self.processed_frames:
            return (0, 0)
        processed = sum(1 for v in self.processed_frames.values() if v)
        return (processed, self.total_frames)


class SegmentationStateManager:
    """Manages segmentation state and coordinates updates"""

    def __init__(self):
        self.state = SegmentationState()
        self._state_change_callbacks = []

    def register_callback(self, callback):
        """Register a callback for state changes"""
        self._state_change_callbacks.append(callback)

    def _notify_state_change(self):
        """Notify all registered callbacks of state change"""
        for callback in self._state_change_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

    def initialize_processing(self, data_shape: Tuple[int, ...]) -> None:
        """Initialize for new processing job"""
        try:
            self.state = SegmentationState()
            self.state.initialize_stack(data_shape)
            self._notify_state_change()
        except Exception as e:
            logger.error(f"Failed to initialize processing: {e}")
            raise

    def update_frame_result(self, frame_index: int, mask: np.ndarray, metadata: Dict[str, Any] = None) -> None:
        """Update results for a single frame"""
        try:
            self.state.update_frame(frame_index, mask)
            if metadata:
                self.state.metadata[f"frame_{frame_index}"] = metadata
            self._notify_state_change()
        except Exception as e:
            logger.error(f"Failed to update frame result: {e}")
            self.state.last_error = str(e)
            raise

    def start_processing(self) -> None:
        """Mark the start of processing"""
        if self.state.is_processing:
            raise RuntimeError("Processing already in progress")
        self.state.is_processing = True
        self._notify_state_change()

    def finish_processing(self) -> None:
        """Mark processing as complete"""
        self.state.is_processing = False
        self._notify_state_change()

    def get_results(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Get the current results and metadata"""
        return self.state.full_stack, self.state.metadata