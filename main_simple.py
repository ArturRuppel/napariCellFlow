import json
import logging
import os.path
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import cv2
import numpy as np
from cellpose.models import CellposeModel
from tifffile import tifffile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingParameters:
    """Parameters for image processing and segmentation"""
    # Preprocessing parameters
    median_filter_size: int = 7
    clahe_clip_limit: float = 4
    clahe_grid_size: int = 16

    # Intensity clipping parameters
    initial_lower_percentile: float = 20
    final_lower_percentile: float = 45
    final_upper_percentile: float = 99.0

    # Cell detection parameters
    black_region_threshold: int = 5  # Intensity threshold for excluding dark regions

    # Cellpose parameters
    diameter = 80
    flow_threshold: float = 0.4
    cellprob_threshold: float = -0.1
    min_size: int = 25


class CellSegmentationPipeline:
    def __init__(self, output_dir: Optional[Path] = None, model_path: str = None):
        self.params = ProcessingParameters()
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.preprocessed_dir = self.output_dir / "preprocessed_frames"
        self.segmentation_dir = self.output_dir.parent / "segmentation_frames"

        if model_path is None:
            raise ValueError("model_path must be provided")

        self.initialize_model(model_path=model_path)

    def initialize_model(self, model_path: str) -> None:
        """Initialize CellposeModel with a specific model path"""
        logger.info(f"Loading model from: {model_path}")
        try:
            self.model = CellposeModel(
                gpu=False,
                pretrained_model=model_path,
            )
            logger.info("Successfully loaded model")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise

    def _save_segmentation_frame(self, frame_idx: int, preprocessed_image: np.ndarray,
                                 masks: np.ndarray, flows: list) -> None:
        """Save segmentation results for a single frame"""
        # Save preprocessed image used for segmentation
        image_path = self.segmentation_dir / f"image_{frame_idx:04d}.tif"
        tifffile.imwrite(image_path, preprocessed_image)

        # Create outlines
        from cellpose.utils import masks_to_outlines
        outlines = masks_to_outlines(masks)

        # Create random colors for cells
        ncells = len(np.unique(masks)[1:])
        colors = ((np.random.rand(ncells, 3) * 0.8 + 0.1) * 255).astype(np.uint8)

        # Save in same format as GUI
        model_outputs = {
            'outlines': outlines.astype(np.uint16),
            'colors': colors,
            'masks': masks.astype(np.uint16),
            'chan_choose': [0, 0],
            'filename': str(image_path),
            'flows': flows,
            'ismanual': np.zeros(ncells, dtype=bool),
            'manual_changes': [],
            'model_path': str(self.model.pretrained_model),
            'flow_threshold': self.params.flow_threshold,
            'cellprob_threshold': self.params.cellprob_threshold,
            'diameter': float(self.params.diameter)
        }

        np.save(
            self.segmentation_dir / f"image_{frame_idx:04d}_seg.npy",
            model_outputs
        )

    def _save_preprocessing_steps(self, frame_idx: int, original: np.ndarray,
                                  scaled: np.ndarray, pre_clipped: np.ndarray,
                                  median_filtered: np.ndarray, enhanced: np.ndarray,
                                  final_enhanced: np.ndarray, exclude_mask: np.ndarray):
        """Save each preprocessing step for debugging"""
        steps_dir = self.preprocessed_dir / f"frame_{frame_idx:04d}_steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

        # Save each step
        tifffile.imwrite(steps_dir / "01_original.tif", original)
        tifffile.imwrite(steps_dir / "02_scaled_8bit.tif", scaled)
        tifffile.imwrite(steps_dir / "03_pre_clipped.tif", pre_clipped)
        tifffile.imwrite(steps_dir / "04_median_filtered.tif", median_filtered)
        tifffile.imwrite(steps_dir / "05_clahe_enhanced.tif", enhanced)
        tifffile.imwrite(steps_dir / "06_final_enhanced.tif", final_enhanced)
        tifffile.imwrite(steps_dir / "07_exclude_mask.tif", exclude_mask)

        # Save histograms
        for name, img in [("original", original), ("scaled", scaled),
                          ("pre_clipped", pre_clipped), ("median_filtered", median_filtered),
                          ("enhanced", enhanced), ("final_enhanced", final_enhanced)]:
            hist = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256])
            np.save(steps_dir / f"{name}_histogram.npy", hist)

    def _create_exclude_mask(self, image: np.ndarray, threshold: int = 5) -> np.ndarray:
        """Create mask for dark regions to exclude from detection"""
        return (image <= threshold).astype(np.uint8)

    def _apply_exclude_mask(self, masks: np.ndarray, exclude_mask: np.ndarray) -> np.ndarray:
        """Remove cell masks from excluded regions"""
        updated_masks = masks.copy()

        for cell_id in range(1, masks.max() + 1):
            cell_mask = masks == cell_id
            if np.sum(cell_mask & exclude_mask) / np.sum(cell_mask) > 0.5:
                updated_masks[cell_mask] = 0

        return updated_masks

    def preprocess_frame(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        """Apply preprocessing steps to a single frame with proper intensity scaling"""
        # Store original for debugging
        original = image.copy()

        # Proper scaling to 8-bit
        scaled = self._scale_to_8bit(image)  # This already returns uint8

        # Initial percentile-based clipping
        lower_percentile = np.percentile(scaled, self.params.initial_lower_percentile)
        pre_clipped = np.clip(scaled, lower_percentile, 255).astype(np.uint8)

        # Apply median filter for noise reduction
        median_filtered = cv2.medianBlur(pre_clipped, self.params.median_filter_size)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.params.clahe_clip_limit,
            tileGridSize=(self.params.clahe_grid_size, self.params.clahe_grid_size)
        )
        enhanced = clahe.apply(median_filtered)

        # Final percentile-based clipping
        final_lower_percentile = np.percentile(enhanced, self.params.final_lower_percentile)
        final_upper_percentile = np.percentile(enhanced, self.params.final_upper_percentile)
        final_enhanced = np.clip(enhanced, final_lower_percentile, final_upper_percentile)

        # Rescale to full range
        if final_upper_percentile > final_lower_percentile:
            final_enhanced = ((final_enhanced - final_lower_percentile) *
                              (255.0 / (final_upper_percentile - final_lower_percentile))).clip(0, 255)

        final_enhanced = final_enhanced.astype(np.uint8)

        # Create exclude mask for very dark regions
        exclude_mask = self._create_exclude_mask(final_enhanced)

        # Convert exclude mask to visible format (0 -> black, 1 -> white)
        exclude_mask_vis = exclude_mask * 255

        return final_enhanced, (original, scaled, pre_clipped, median_filtered,
                                enhanced, final_enhanced, exclude_mask_vis)

    def _scale_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """Scale image to 8-bit with proper normalization"""
        # Get the actual min and max of the data
        img_min = np.percentile(image, 1)  # Using 1st percentile to remove outliers
        img_max = np.percentile(image, 99)  # Using 99th percentile to remove outliers

        # Clip the image to these values
        image_clipped = np.clip(image, img_min, img_max)

        # Scale to 0-255
        image_scaled = ((image_clipped - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        return image_scaled

    def process_stack(self, image_stack: np.ndarray, save_intermediate: bool = True) -> Dict:
        """Process an image stack frame by frame"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        self.segmentation_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'masks': [],
            'flows': [],
            'metadata': {
                'parameters': self.params.__dict__,
                'stack_shape': image_stack.shape,
                'processed_frames': 0,
                'failed_frames': [],
                'output_directory': str(self.output_dir),
                'preprocessed_directory': str(self.preprocessed_dir),
                'segmentation_directory': str(self.segmentation_dir)
            }
        }

        # Process all frames independently
        total_frames = image_stack.shape[0]
        for frame_idx in range(total_frames):
            logger.info(f"Processing frame {frame_idx + 1}/{total_frames}")

            try:
                # Preprocess frame
                frame = image_stack[frame_idx]
                preprocessed, preprocessing_steps = self.preprocess_frame(frame)

                # Segment frame
                masks, flows, styles = self.model.eval(
                    preprocessed,
                    diameter=self.params.diameter,
                    batch_size=8,
                    channels=[0, 0],
                    flow_threshold=self.params.flow_threshold,
                    cellprob_threshold=self.params.cellprob_threshold,
                    min_size=self.params.min_size,
                    normalize=True,
                    do_3D=False
                )

                # Apply exclude mask
                exclude_mask = self._create_exclude_mask(preprocessed)
                masks = self._apply_exclude_mask(masks, exclude_mask)

                results['masks'].append(masks)
                results['flows'].append(flows)
                results['metadata']['processed_frames'] += 1

                # Save intermediate results if requested
                if save_intermediate:
                    self._save_preprocessing_steps(frame_idx, *preprocessing_steps)
                    self._save_segmentation_frame(frame_idx, preprocessed, masks, flows)

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                results['metadata']['failed_frames'].append(frame_idx)
                continue

        # Save final stack
        masks_stack = np.stack(results['masks'])
        output_file = self.output_dir.parent / 'segmentation.tif'
        tifffile.imwrite(output_file, masks_stack.astype(np.uint16))

        logger.info(f"Processing complete. Final segmentation saved to: {output_file}")
        return results

    def load_image_stack(self, path: Union[str, Path]) -> np.ndarray:
        """Load image stack from file"""
        path = Path(path)
        logger.info(f"Loading image stack from: {path}")

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in ['.tif', '.tiff']:
            raise ValueError(f"Expected .tif file, got: {path.suffix}")

        stack = tifffile.imread(path)

        if stack.ndim != 3:
            raise ValueError(f"Expected 3 dimensions (t,x,y), got {stack.ndim} dimensions")

        logger.info(f"Loaded stack with shape: {stack.shape}")
        return stack


def main(path, filename, model_path):
    """Example usage of the pipeline"""
    pipeline = CellSegmentationPipeline(
        output_dir=os.path.join(path, "segmentation_output"),
        model_path=model_path
    )

    # Load image stack
    image_stack = pipeline.load_image_stack(os.path.join(path, filename))

    # Process the stack
    results = pipeline.process_stack(image_stack)


if __name__ == "__main__":
    path = "D:/2024-11-27/position5/"
    filename = "registered_membrane_slice_downsized.tif"
    model_path = "D:/cellpose_models/cyto3_xenopus1"

    main(path, filename, model_path)