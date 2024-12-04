# data_manager.py
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data across different components of the application."""

    def __init__(self):
        self._updating = False
        self._preprocessed_data = None
        self._segmentation_data = None  # Add this
        self._tracked_data = None

    @property
    def segmentation_data(self) -> Optional[np.ndarray]:
        """Get the segmentation data"""
        return self._segmentation_data

    @segmentation_data.setter
    def segmentation_data(self, data: Optional[np.ndarray]):
        """Set the segmentation data"""
        if self._updating:
            return

        try:
            self._updating = True
            if data is not None and not isinstance(data, np.ndarray):
                raise ValueError("Segmentation data must be a numpy array")
            self._segmentation_data = data
        finally:
            self._updating = False

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
    def tracked_data(self) -> Optional[np.ndarray]:
        """Get the tracked data"""
        return self._tracked_data

    @tracked_data.setter
    def tracked_data(self, data: Optional[np.ndarray]):
        """Set the tracked data"""
        if self._updating:
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