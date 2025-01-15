"""
Qt-integrated wrapper for Cellpose 3.0 cell segmentation.

This module provides a Qt-friendly interface to Cellpose, with progress signals
and parameter validation. Primarily designed for integration into Qt applications
requiring cell segmentation capabilities.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from cellpose import models
from cellpose.core import use_gpu
from qtpy.QtCore import QObject, Signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SegmentationParameters:
    """Parameters for Cellpose segmentation with validation.

    This dataclass provides a validated parameter set for Cellpose 3.0,
    ensuring all parameters are within valid ranges and logically consistent.
    Implements validation logic that Cellpose doesn't provide natively.
    """
    # Model parameters
    model_type: str = "cyto3"  # "cyto3", "nuclei", "custom"
    custom_model_path: Optional[str] = None

    # Cellpose 3.0 parameters
    diameter: float = 95.0
    flow_threshold: float = 0.6
    cellprob_threshold: float = 0.3
    min_size: int = 25
    normalize: bool = True

    # Channel parameters
    chan_segment: int = 0  # channel to segment
    chan_2: Optional[int] = None  # optional second channel

    # Advanced parameters
    batch_size: int = 8
    gpu: bool = True
    compute_diameter: bool = True  # Auto-compute diameter
    stitch_threshold: float = 0.0  # For handling tiles
    resample: bool = False  # Whether to resample pixels
    anisotropy: Optional[float] = None  # Pixel scaling in z vs xy
    augment: bool = False  # Use augmentation for inference

    def validate(self):
        """Validate parameter values before passing to Cellpose.

        Raises:
            ValueError: If any parameter is invalid or inconsistent:
                - Non-positive diameter with auto-compute disabled
                - Flow/cell probability thresholds outside [0,1]
                - Non-positive minimum size or batch size
        """
        if self.diameter <= 0 and not self.compute_diameter:
            raise ValueError("Cell diameter must be positive or compute_diameter must be True")
        if not 0 <= self.flow_threshold <= 1:
            raise ValueError("Flow threshold must be between 0 and 1")
        if not 0 <= self.cellprob_threshold <= 1:
            raise ValueError("Cell probability threshold must be between 0 and 1")
        if self.min_size <= 0:
            raise ValueError("Minimum size must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")



class CellposeSignals(QObject):
    """Qt signals for Cellpose segmentation events.

    Provides Qt signals for tracking segmentation progress, completion,
    and failures. Separated from main handler to avoid Qt inheritance issues.

    Signals:
        segmentation_completed: Emitted with (masks, metadata) on success
        segmentation_failed: Emitted with error message on failure
        progress_updated: Emitted with (percentage, message) during processing
    """
    segmentation_completed = Signal(object, object)  # masks, metadata
    segmentation_failed = Signal(str)  # error message
    progress_updated = Signal(int, str)  # progress percentage, message


class SegmentationHandler:
    """Qt-integrated handler for Cellpose cell segmentation.

    Wraps Cellpose functionality with Qt signals for progress tracking
    and parameter validation. Designed for integration into Qt applications
    needing cell segmentation capabilities.

    Key Features:
        - Qt signal integration for progress/status updates
        - Parameter validation
        - Result metadata tracking
        - GPU support management
    """
    # Define signals
    segmentation_completed = Signal(np.ndarray, dict)  # masks, metadata
    segmentation_failed = Signal(str)  # error message

    def __init__(self):
        self.params = SegmentationParameters()
        self.model = None
        self.last_results = {}
        # Create signals object
        self.signals = CellposeSignals()

    def initialize_model(self, params: SegmentationParameters):
        """Initialize Cellpose model with validated parameters.

        Args:
            params: Validated parameters for model initialization

        Raises:
            ValueError: If custom model selected without path
            RuntimeError: If model initialization fails

        Signals:
            progress_updated: At start (0%) and completion (100%)
            segmentation_failed: If initialization fails
        """
        try:
            logger.info(f"Initializing Cellpose model: {params.model_type}")
            self.signals.progress_updated.emit(0, "Initializing model...")

            if params.model_type == "custom" and not params.custom_model_path:
                raise ValueError("Custom model path required for custom model type")

            # Check GPU availability
            if params.gpu:
                use_gpu()

            # Initialize model with Cellpose 3.0 parameters
            if params.model_type == "custom":
                self.model = models.CellposeModel(
                    pretrained_model=params.custom_model_path,
                    gpu=params.gpu
                )
            else:
                self.model = models.CellposeModel(
                    model_type=params.model_type,
                    gpu=params.gpu,
                )

            self.params = params
            logger.info("Model initialized successfully")
            self.signals.progress_updated.emit(100, "Model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.signals.segmentation_failed.emit(str(e))
            raise

    def segment_frame(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Segment cells in a single image frame with progress tracking.

        Args:
            image: Input image array (2D or 3D)

        Returns:
            Tuple of:
                - Binary mask array (same shape as input)
                - Results dict containing:
                    - masks: Segmentation masks
                    - flows: Cellpose flow components
                    - styles: Cell style vectors
                    - diameter: Used/computed diameter
                    - parameters: Complete parameter set

        Raises:
            RuntimeError: If model not initialized
            ValueError: If image format invalid

        Signals:
            progress_updated: During key processing steps
            segmentation_completed: When processing succeeds
            segmentation_failed: If processing fails
        """
        if self.model is None:
            error_msg = "Model not initialized. Call initialize_model first."
            self.signals.segmentation_failed.emit(error_msg)
            raise RuntimeError(error_msg)

        try:
            self.signals.progress_updated.emit(10, "Preparing segmentation...")

            # Prepare channels
            channels = [self.params.chan_segment]
            if self.params.chan_2 is not None:
                channels.append(self.params.chan_2)
            else:
                channels.append(0)

            logger.info(f"Running segmentation with channels: {channels}")
            self.signals.progress_updated.emit(30, "Running Cellpose segmentation...")

            # Run Cellpose with 3.0 parameters
            masks, flows, styles = self.model.eval(
                image,
                diameter=None if self.params.compute_diameter else self.params.diameter,
                channels=channels,
                flow_threshold=self.params.flow_threshold,
                cellprob_threshold=self.params.cellprob_threshold,
                min_size=self.params.min_size,
                batch_size=self.params.batch_size,
                normalize=self.params.normalize,
                stitch_threshold=self.params.stitch_threshold,
                resample=self.params.resample,
                anisotropy=self.params.anisotropy,
                augment=self.params.augment
            )

            # Store results
            results = {
                'masks': masks,
                'flows': flows,
                'styles': styles,
                'diameter': self.model.diam_labels if self.params.compute_diameter else self.params.diameter,
                'parameters': self.params.__dict__
            }
            self.last_results = results

            num_cells = len(np.unique(masks)) - 1  # subtract 1 for background
            logger.info(f"Segmentation complete. Found {num_cells} cells")

            self.signals.progress_updated.emit(100, f"Segmentation complete. Found {num_cells} cells")
            self.signals.segmentation_completed.emit(masks, results)

            return masks, results

        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            logger.error(error_msg)
            self.signals.segmentation_failed.emit(error_msg)
            raise
