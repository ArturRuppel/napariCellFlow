# data_manager.py
import logging
from pathlib import Path
import numpy as np
import tifffile

logger = logging.getLogger(__name__)


class DataManager:
    """Minimal data manager for handling cell tracking data."""

    def __init__(self):
        self.tracked_data = None
        self.last_directory = None
        self.batch_mode = False

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