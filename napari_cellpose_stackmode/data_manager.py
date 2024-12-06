import logging
from pathlib import Path
from typing import Optional, Tuple, Union
from threading import Lock
import numpy as np
import tifffile
from .debug_logging import log_operation

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data across different components of the application with robust frame handling."""

    def __init__(self):
        self._updating = False
        self._lock = Lock()
        self._preprocessed_data = None
        self._segmentation_data = None
        self._tracked_data = None
        self._num_frames = None
        self._frame_states = {}  # Track state of individual frames
        self._initialized = False

    def initialize_stack(self, num_frames: int) -> None:
        """Initialize the stack with proper dimensionality and frame tracking."""
        with self._lock:
            try:
                logger.info(f"Initializing stack with {num_frames} frames")
                self._num_frames = num_frames
                self._frame_states = {i: {'modified': False, 'last_update': None} for i in range(num_frames)}
                self._initialized = True
                logger.debug("Stack initialization complete")
            except Exception as e:
                logger.error(f"Failed to initialize stack: {e}")
                raise ValueError(f"Stack initialization failed: {str(e)}")

    @property
    @log_operation
    def preprocessed_data(self) -> Optional[np.ndarray]:
        """Get the preprocessed data."""
        with self._lock:
            return self._preprocessed_data

    @preprocessed_data.setter
    @log_operation
    def preprocessed_data(self, data: Optional[np.ndarray]):
        """Set the preprocessed data with validation."""
        if self._updating:
            return

        with self._lock:
            try:
                self._updating = True
                if data is not None:
                    if not isinstance(data, np.ndarray):
                        raise ValueError("Preprocessed data must be a numpy array")
                    # Update number of frames if not already set
                    if self._num_frames is None:
                        self._num_frames = data.shape[0] if data.ndim == 3 else 1
                self._preprocessed_data = data
            finally:
                self._updating = False

    @property
    @log_operation
    def segmentation_data(self) -> Optional[np.ndarray]:
        """Get the segmentation data."""
        with self._lock:
            return self._segmentation_data

    @segmentation_data.setter
    @log_operation
    def segmentation_data(self, value: Union[np.ndarray, Tuple[np.ndarray, int]]):
        """
        Set the segmentation data with enhanced frame preservation.

        Args:
            value: Either full stack (np.ndarray) or tuple of (frame_data, frame_index)
        """
        if self._updating:
            logger.debug("Segmentation data update cancelled - updating in progress")
            return

        with self._lock:
            try:
                self._updating = True

                # Handle single frame updates
                if isinstance(value, tuple) and len(value) == 2:
                    frame_data, frame_index = value
                    self._update_single_frame(frame_data, frame_index)
                    return

                # Handle full stack updates
                if value is not None:
                    if not isinstance(value, np.ndarray):
                        raise ValueError("Segmentation data must be a numpy array")

                    # Initialize if needed
                    if self._segmentation_data is None:
                        if not self._initialized:
                            raise RuntimeError("Must call initialize_stack before first update")
                        shape = (self._num_frames, *value.shape) if value.ndim == 2 else value.shape
                        self._segmentation_data = np.zeros(shape, dtype=value.dtype)

                    # Validate shape consistency
                    if value.shape != self._segmentation_data.shape:
                        raise ValueError(
                            f"Shape mismatch: expected {self._segmentation_data.shape}, "
                            f"got {value.shape}"
                        )

                    # Update the full stack while preserving modified frames
                    self._update_full_stack(value)
                else:
                    self._segmentation_data = None
                    self._frame_states = {}

            except Exception as e:
                logger.error(f"Error updating segmentation data: {e}")
                raise
            finally:
                self._updating = False

    def _update_single_frame(self, frame_data: np.ndarray, frame_index: int) -> None:
        """Update a single frame while preserving others."""
        if not self._initialized:
            raise RuntimeError("Stack not initialized. Call initialize_stack first.")

        if frame_index >= self._num_frames:
            raise ValueError(f"Frame index {frame_index} exceeds stack size {self._num_frames}")

        if self._segmentation_data is None:
            # Initialize with proper shape
            self._segmentation_data = np.zeros((self._num_frames, *frame_data.shape), dtype=frame_data.dtype)

        # Validate frame shape
        if frame_data.shape != self._segmentation_data.shape[1:]:
            raise ValueError(
                f"Frame shape mismatch: expected {self._segmentation_data.shape[1:]}, "
                f"got {frame_data.shape}"
            )

        # Update the frame
        self._segmentation_data[frame_index] = frame_data
        self._frame_states[frame_index] = {
            'modified': True,
            'last_update': np.datetime64('now')
        }

        logger.debug(f"Updated frame {frame_index}")

    def _update_full_stack(self, new_stack: np.ndarray) -> None:
        """Update the full stack while preserving modified frames."""
        if not self._initialized:
            raise RuntimeError("Stack not initialized. Call initialize_stack first.")

        # Create a mask of modified frames
        modified_frames = np.array([
            self._frame_states.get(i, {}).get('modified', False)
            for i in range(self._num_frames)
        ])

        # Update only unmodified frames
        for i in range(self._num_frames):
            if not modified_frames[i]:
                self._segmentation_data[i] = new_stack[i]
                self._frame_states[i] = {
                    'modified': False,
                    'last_update': np.datetime64('now')
                }

        logger.debug(f"Updated full stack, preserved {modified_frames.sum()} modified frames")

    @property
    def tracked_data(self) -> Optional[np.ndarray]:
        """Get the tracked data."""
        with self._lock:
            return self._tracked_data

    @tracked_data.setter
    @log_operation
    def tracked_data(self, data: Optional[np.ndarray]):
        """Set the tracked data with validation."""
        if self._updating:
            return

        with self._lock:
            try:
                self._updating = True
                if data is not None and not isinstance(data, np.ndarray):
                    raise ValueError("Tracked data must be a numpy array")
                self._tracked_data = data
            finally:
                self._updating = False

    def validate_stack_consistency(self) -> bool:
        """Validate that all components have consistent data."""
        with self._lock:
            if self._segmentation_data is None:
                return True

            # Check if we have the expected number of frames
            if self._num_frames is not None:
                if self._segmentation_data.shape[0] != self._num_frames:
                    logger.error(
                        f"Frame count mismatch: expected {self._num_frames}, "
                        f"got {self._segmentation_data.shape[0]}"
                    )
                    return False

            # Validate frame states
            for frame_idx in range(self._num_frames):
                if frame_idx not in self._frame_states:
                    logger.error(f"Missing frame state for frame {frame_idx}")
                    return False

            return True

    def get_frame_state(self, frame_index: int) -> dict:
        """Get the state information for a specific frame."""
        with self._lock:
            if not 0 <= frame_index < self._num_frames:
                raise ValueError(f"Invalid frame index: {frame_index}")
            return self._frame_states.get(frame_index, {}).copy()

    def reset_frame_modifications(self) -> None:
        """Reset all frame modification flags."""
        with self._lock:
            for frame_idx in self._frame_states:
                self._frame_states[frame_idx]['modified'] = False

    def save_tracking_results(self, path: Path) -> None:
        """Save tracking results to a TIFF file."""
        try:
            if self._tracked_data is None:
                raise ValueError("No tracking data to save")

            logger.info(f"Saving tracking results to {path}")
            tifffile.imwrite(str(path), self._tracked_data)
            logger.info("Results saved successfully")

        except Exception as e:
            logger.error(f"Error saving tracking results: {e}")
            raise

    def load_tracking_results(self, path: Path) -> None:
        """Load tracking results from a TIFF file."""
        try:
            logger.info(f"Loading tracking results from {path}")
            self._tracked_data = tifffile.imread(str(path))
            logger.info(f"Loaded tracking data with shape {self._tracked_data.shape}")

        except Exception as e:
            logger.error(f"Error loading tracking results: {e}")
            raise