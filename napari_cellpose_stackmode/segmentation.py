import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from cellpose import models
from dataclasses import dataclass
from qtpy.QtCore import Signal, QObject

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SegmentationParameters:
    """Parameters for Cellpose segmentation"""
    # Model parameters
    model_type: str = "cyto"  # or "nuclei" or "custom"
    custom_model_path: Optional[str] = None

    # Cellpose parameters
    diameter: float = 95.0
    flow_threshold: float = 0.6
    cellprob_threshold: float = 0.3
    min_size: int = 25
    gpu: bool = False

    # Additional parameters
    batch_size: int = 8
    normalize: bool = True

    def validate(self):
        """Validate parameter values"""
        if self.diameter <= 0:
            raise ValueError("Cell diameter must be positive")
        if not 0 <= self.flow_threshold <= 1:
            raise ValueError("Flow threshold must be between 0 and 1")
        if not 0 <= self.cellprob_threshold <= 1:
            raise ValueError("Cell probability threshold must be between 0 and 1")
        if self.min_size <= 0:
            raise ValueError("Minimum size must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


class SegmentationHandler(QObject):
    """Handles cell segmentation using Cellpose with Qt signals"""

    # Define signals
    segmentation_completed = Signal(np.ndarray, dict)  # masks, metadata
    segmentation_failed = Signal(str)  # error message

    def __init__(self):
        """Initialize the segmentation handler"""
        super().__init__()
        self.params = SegmentationParameters()
        self.model: Optional[models.CellposeModel] = None
        self.last_results: Dict[str, Any] = {}

    def initialize_model(self, params: SegmentationParameters) -> None:
        """Initialize or update the Cellpose model"""
        try:
            self.params = params
            self.params.validate()

            if params.model_type == "custom" and params.custom_model_path:
                logger.info(f"Loading custom model from: {params.custom_model_path}")
                self.model = models.CellposeModel(
                    gpu=params.gpu,
                    pretrained_model=params.custom_model_path
                )
            else:
                logger.info(f"Loading built-in {params.model_type} model")
                self.model = models.CellposeModel(
                    gpu=params.gpu,
                    model_type=params.model_type
                )

            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self.segmentation_failed.emit(str(e))
            raise

    def segment_frame(self, image: np.ndarray) -> None:
        """
        Segment a single frame using Cellpose

        Args:
            image: 2D numpy array of the image to segment
        """
        if self.model is None:
            error_msg = "Model not initialized. Call initialize_model first."
            self.segmentation_failed.emit(error_msg)
            raise RuntimeError(error_msg)

        try:
            # Run Cellpose
            masks, flows, styles = self.model.eval(
                image,
                diameter=self.params.diameter,
                flow_threshold=self.params.flow_threshold,
                cellprob_threshold=self.params.cellprob_threshold,
                min_size=self.params.min_size,
                batch_size=self.params.batch_size,
                channels=[0, 0],
                normalize=self.params.normalize
            )

            # Store results
            self.last_results = {
                'flows': flows,
                'styles': styles,
                'diameter': self.params.diameter,
                'parameters': self.params.__dict__
            }

            # Emit completion signal
            self.segmentation_completed.emit(masks, self.last_results)

        except Exception as e:
            logger.error(f"Error during segmentation: {str(e)}")
            self.segmentation_failed.emit(str(e))
            raise

    def save_results(self, output_dir: Path) -> None:
        """
        Save the last segmentation results in Cellpose-compatible format

        Args:
            output_dir: Directory to save results
        """
        if not self.last_results:
            error_msg = "No results to save. Run segment_frame first."
            self.segmentation_failed.emit(error_msg)
            raise RuntimeError(error_msg)

        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save in Cellpose .npy format for GUI compatibility
            output_path = output_dir / "segmentation_results.npy"
            np.save(output_path, self.last_results)

            logger.info(f"Saved results to {output_path}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            self.segmentation_failed.emit(str(e))
            raise