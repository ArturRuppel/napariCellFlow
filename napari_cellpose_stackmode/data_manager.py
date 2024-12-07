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
    def preprocessed_data(self) -> Optional[np.ndarray]:
        """Get the preprocessed data."""
        with self._lock:
            return self._preprocessed_data

    @preprocessed_data.setter
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
    def segmentation_data(self) -> Optional[np.ndarray]:
        """Get the segmentation data."""
        with self._lock:
            return self._segmentation_data

    @segmentation_data.setter
    def segmentation_data(self, value: Union[np.ndarray, Tuple[np.ndarray, int]]):
        """Set segmentation data with robust frame preservation."""
        if self._updating:
            logger.debug("DataManager: Update cancelled - already updating")
            return

        with self._lock:
            try:
                self._updating = True
                logger.debug("DataManager: Starting segmentation data update")
                logger.debug(f"DataManager: Input type: {type(value)}")
                if isinstance(value, np.ndarray):
                    logger.debug(f"DataManager: Input shape: {value.shape}")
                    logger.debug(f"DataManager: Input unique values: {np.unique(value)}")

                # Handle single frame update
                if isinstance(value, tuple) and len(value) == 2:
                    frame_data, frame_index = value
                    logger.debug(f"DataManager: Updating single frame {frame_index}")

                    if self._segmentation_data is None:
                        shape = (self._num_frames, *frame_data.shape)
                        self._segmentation_data = np.zeros(shape, dtype=frame_data.dtype)
                        self._full_stack = self._segmentation_data.copy()

                    # Update frame
                    if frame_index < self._num_frames:
                        self._segmentation_data[frame_index] = frame_data.copy()
                        self._full_stack[frame_index] = frame_data.copy()
                        self._frame_states[frame_index] = {
                            'modified': True,
                            'last_update': np.datetime64('now')
                        }
                        logger.debug("DataManager: Frame update complete")
                    else:
                        logger.error(f"DataManager: Frame index {frame_index} out of bounds")
                        raise ValueError(f"Frame index {frame_index} out of bounds")

                # Handle full stack update
                else:
                    logger.debug("DataManager: Updating full stack")
                    if value is not None:
                        if value.ndim == 2:
                            value = value[np.newaxis, ...]
                        self._segmentation_data = value.copy()
                        self._full_stack = self._segmentation_data.copy()
                    else:
                        self._segmentation_data = None
                        self._full_stack = None
                        self._frame_states.clear()

            except Exception as e:
                logger.error(f"DataManager: Error updating segmentation data: {e}", exc_info=True)
                raise
            finally:
                self._updating = False
                logger.debug("DataManager: Update complete")

    def _update_single_frame(self, frame_data: np.ndarray, frame_index: int) -> None:
        """Update a single frame while preserving others."""
        if not self._initialized:
            raise RuntimeError("Stack not initialized. Call initialize_stack first.")

        if frame_index >= self._num_frames:
            raise ValueError(f"Frame index {frame_index} exceeds stack size {self._num_frames}")

        with self._lock:
            try:
                # Initialize if needed
                if self._segmentation_data is None:
                    self._segmentation_data = np.zeros((self._num_frames, *frame_data.shape), dtype=frame_data.dtype)

                # Validate frame shape
                if frame_data.shape != self._segmentation_data.shape[1:]:
                    raise ValueError(
                        f"Frame shape mismatch: expected {self._segmentation_data.shape[1:]}, "
                        f"got {frame_data.shape}"
                    )

                # Update the frame
                self._segmentation_data[frame_index] = frame_data.copy()  # Make explicit copy
                self._frame_states[frame_index] = {
                    'modified': True,
                    'last_update': np.datetime64('now')
                }

                logger.debug(f"Updated frame {frame_index}")
                logger.debug(f"Updated frame unique values: {np.unique(frame_data)}")
                logger.debug(f"Full stack unique values: {np.unique(self._segmentation_data)}")

            except Exception as e:
                logger.error(f"Error updating frame: {e}")
                raise

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