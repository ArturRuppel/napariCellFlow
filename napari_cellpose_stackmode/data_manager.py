# data_manager.py
import logging
from pathlib import Path
from typing import Optional
from .debug_logging import log_state_changes, log_array_info, logger, log_data_manager_ops

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data across different components of the application."""

    def __init__(self):
        self._updating = False
        self._preprocessed_data = None
        self._segmentation_data = None
        self._tracked_data = None
        self._num_frames = None  # Store number of frames

    def initialize_stack(self, num_frames: int) -> None:
        """Initialize the number of frames for the data stack"""
        self._num_frames = num_frames



    @property
    def preprocessed_data(self) -> Optional[np.ndarray]:
        """Get the preprocessed data"""
        return self._preprocessed_data

    @preprocessed_data.setter
    def preprocessed_data(self, data: Optional[np.ndarray]):
        """Set the preprocessed data"""
        if self._updating:
            return

        try:
            self._updating = True
            if data is not None and not isinstance(data, np.ndarray):
                raise ValueError("Preprocessed data must be a numpy array")
            self._preprocessed_data = data
        finally:
            self._updating = False

    @property
    @log_data_manager_ops
    def segmentation_data(self) -> Optional[np.ndarray]:
        """Get the segmentation data"""
        return self._segmentation_data

    @segmentation_data.setter
    @log_data_manager_ops
    def segmentation_data(self, value):
        """Set the segmentation data"""
        if self._updating:
            logger.debug("Segmentation data update cancelled - updating in progress")
            return

        try:
            self._updating = True

            # Handle both single frame updates and full stack updates
            if isinstance(value, tuple) and len(value) == 2:
                data, current_frame = value
                if not isinstance(data, np.ndarray):
                    raise ValueError("Segmentation data must be a numpy array")

                # Initialize stack if needed
                if self._segmentation_data is None:
                    if self._num_frames is None:
                        raise ValueError("Must call initialize_stack before first update")
                    self._segmentation_data = np.zeros((self._num_frames, *data.shape), dtype=data.dtype)

                # Update only the specified frame while preserving others
                self._segmentation_data[current_frame] = data
            else:
                # Full stack update - validate and assign
                if value is not None:
                    if not isinstance(value, np.ndarray):
                        raise ValueError("Segmentation data must be a numpy array")
                    if self._segmentation_data is not None and value.shape != self._segmentation_data.shape:
                        raise ValueError(f"Shape mismatch: expected {self._segmentation_data.shape}, got {value.shape}")
                self._segmentation_data = value

            # Validate after update
            if not self.validate_stack_consistency():
                raise ValueError("Stack consistency check failed after update")

        finally:
            self._updating = False
    @property
    @log_data_manager_ops
    def tracked_data(self) -> Optional[np.ndarray]:
        """Get the tracked data"""
        return self._tracked_data

    @tracked_data.setter
    @log_data_manager_ops
    def tracked_data(self, data: Optional[np.ndarray]):
        """Set the tracked data"""
        if self._updating:
            logger.debug("Tracked data update cancelled - updating in progress")
            return

        try:
            self._updating = True
            if data is not None and not isinstance(data, np.ndarray):
                raise ValueError("Tracked data must be a numpy array")
            self._tracked_data = data
        finally:
            self._updating = False

    def save_tracking_results(self, path: Path) -> None:
        """Save tracking results to a TIFF file."""
        try:
            if self.tracked_data is None:
                raise ValueError("No tracking data to save")

            logger.info(f"Saving tracking results to {path}")
            tifffile.imwrite(str(path), self.tracked_data)
            self.last_directory = path.parent
            logger.info("Results saved successfully")

        except Exception as e:
            logger.error(f"Error saving tracking results: {e}")
            raise

    def validate_stack_consistency(self):
        """Validate that all components have consistent data"""
        if self._segmentation_data is None:
            return True

        # Check if we have the expected number of frames
        if self._num_frames is not None:
            if self._segmentation_data.shape[0] != self._num_frames:
                logger.error(f"Frame count mismatch: expected {self._num_frames}, got {self._segmentation_data.shape[0]}")
                return False

        return True

    def load_tracking_results(self, path: Path) -> None:
        """Load tracking results from a TIFF file."""
        try:
            logger.info(f"Loading tracking results from {path}")
            self.tracked_data = tifffile.imread(str(path))
            self.last_directory = path.parent
            logger.info(f"Loaded tracking data with shape {self.tracked_data.shape}")

        except Exception as e:
            logger.error(f"Error loading tracking results: {e}")
            raise

    def store_batch_results(self, path: Path) -> None:
        """Store results during batch processing."""
        if not self.batch_mode:
            return

        save_path = path.parent / f"{path.stem}_tracked.tif"
        self.save_tracking_results(save_path)