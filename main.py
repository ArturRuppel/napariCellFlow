import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

import cv2
import numpy as np
from cellpose import models
from cellpose.transforms import normalize99
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
    initial_lower_percentile: float = 1.0
    final_lower_percentile: float = 50
    final_upper_percentile: float = 99.0

    # Cell detection parameters
    black_region_threshold: int = 5  # Intensity threshold for excluding dark regions

    # Cellpose parameters
    diameter = 100
    initial_flow_threshold: float = 0.6
    refined_flow_threshold: float = 0.4
    initial_cellprob_threshold: float = 0.3
    refined_cellprob_threshold: float = -0.1
    initial_min_size: int = 25
    refined_min_size: int = 25


class CellSegmentationPipeline:
    def __init__(self, params: Optional[ProcessingParameters] = None, output_dir: Optional[Path] = None):
        """Initialize the pipeline with processing parameters and output directory"""
        self.params = params or ProcessingParameters()
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.temp_dir = self.output_dir / "temp_frames"
        self.preprocessed_dir = self.output_dir / "preprocessed_frames"
        # Move segmentation_dir to parent directory
        self.segmentation_dir = self.output_dir.parent / "segmentation_frames"
        self.model = models.Cellpose(gpu=False, model_type="cyto3")
        self.results_cache = {}

    def _setup_directories(self):
        """Create necessary directories for output"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        self.segmentation_dir.mkdir(parents=True, exist_ok=True)

    def _save_segmentation_frame(self, frame_idx: int, preprocessed_image: np.ndarray,
                                 masks: np.ndarray, flows: List[np.ndarray],
                                 styles: np.ndarray, diams: np.ndarray):
        """Save segmentation frame and model outputs in Cellpose GUI format"""
        # Save preprocessed image used for segmentation
        image_path = self.segmentation_dir / f"image_{frame_idx:04d}.tif"
        tifffile.imwrite(image_path, preprocessed_image)

        # # Save masks
        # tifffile.imwrite(
        #     self.segmentation_dir / f"masks_{frame_idx:04d}.tif",
        #     masks.astype(np.uint16)
        # )

        # Create outlines
        from cellpose.utils import masks_to_outlines
        outlines = masks_to_outlines(masks)

        # Create random colors for cells
        ncells = len(np.unique(masks)[1:])  # number of cells
        colors = ((np.random.rand(ncells, 3) * 0.8 + 0.1) * 255).astype(np.uint8)

        # Get model path - for Cellpose class it's in the models directory
        import os
        model_path = os.path.join(os.path.expanduser('~'), '.cellpose', 'models', 'cyto3')

        # Save in same format as GUI
        model_outputs = {
            'outlines': outlines.astype(np.uint16),
            'colors': colors,
            'masks': masks.astype(np.uint16),
            'chan_choose': [0, 0],  # assuming default channels
            'filename': str(image_path),
            'flows': flows,
            'ismanual': np.zeros(ncells, dtype=bool),
            'manual_changes': [],
            'model_path': model_path,
            'flow_threshold': 0.4,  # default value
            'cellprob_threshold': 0.0,  # default value
            'normalize_params': {
                'lowhigh': None,
                'percentile': [1.0, 99.0],
                'normalize': True,
                'norm3D': True,
                'sharpen_radius': 0,
                'smooth_radius': 0,
                'tile_norm_blocksize': 0,
                'tile_norm_smooth3D': 1,
                'invert': False
            },
            'restore': None,
            'ratio': 1.0,
            'diameter': float(self.params.diameter)
        }

        np.save(
            self.segmentation_dir / f"image_{frame_idx:04d}_seg.npy",
            model_outputs
        )

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

    def _apply_exclude_mask(self, masks: np.ndarray, exclude_mask: np.ndarray) -> np.ndarray:
        """
        Remove cell masks from excluded (dark) regions

        Parameters:
        -----------
        masks : np.ndarray
            Cellpose output masks
        exclude_mask : np.ndarray
            Binary mask where 1 indicates areas to exclude

        Returns:
        --------
        np.ndarray
            Updated masks with cells in dark regions removed
        """
        # Create a copy of the masks
        updated_masks = masks.copy()

        # Remove masks that overlap with excluded regions
        for cell_id in range(1, masks.max() + 1):
            cell_mask = masks == cell_id
            # If the cell overlaps significantly with excluded region, remove it
            if np.sum(cell_mask & exclude_mask) / np.sum(cell_mask) > 0.5:  # if >50% of cell is in dark region
                updated_masks[cell_mask] = 0

        return updated_masks

    def _save_frame(self, frame_idx: int, mask: np.ndarray):
        """Save individual frame as TIFF file"""
        filename = self.temp_dir / f"frame_{frame_idx:04d}.tif"
        tifffile.imwrite(filename, mask.astype(np.uint16))

    def _save_preprocessed_frame(self, frame_idx: int, frame: np.ndarray):
        """Save preprocessed frame as TIFF file"""
        filename = self.preprocessed_dir / f"preprocessed_{frame_idx:04d}.tif"
        tifffile.imwrite(filename, frame.astype(np.uint8))

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
        """
        Create a mask of very dark regions that should be excluded from cell detection.

        Parameters:
        -----------
        image : np.ndarray
            Preprocessed image
        threshold : int
            Pixel intensity below which areas should be excluded (0-255)

        Returns:
        --------
        np.ndarray
            Binary mask where 1 indicates areas to exclude from cell detection
        """
        return (image <= threshold).astype(np.uint8)

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

    def process_stack(self, image_stack: np.ndarray,
                      save_intermediate: bool = True) -> Dict:
        """Process an entire image stack with progressive saving"""
        self._setup_directories()

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

        total_frames = image_stack.shape[0]

        for frame_idx in range(total_frames):
            logger.info(f"Processing frame {frame_idx + 1}/{total_frames}")

            try:
                # Preprocess frame
                frame = image_stack[frame_idx]
                preprocessed, preprocessing_steps = self.preprocess_frame(frame)

                # Save preprocessing steps
                if save_intermediate:
                    self._save_preprocessing_steps(
                        frame_idx,
                        *preprocessing_steps  # Unpack all preprocessing steps
                    )
                    self._save_preprocessed_frame(frame_idx, preprocessed)

                # First pass segmentation with full model outputs
                masks, flows, styles, diams = self.model.eval(
                    preprocessed,
                    diameter=self.params.diameter,
                    batch_size=8,
                    channels=[0, 0],
                    normalize=True,
                    do_3D=False
                )

                # Save segmentation frame and model outputs
                self._save_segmentation_frame(frame_idx, preprocessed, masks, flows, styles, diams)

                # Apply exclude mask
                exclude_mask = self._create_exclude_mask(preprocessed)
                masks = self._apply_exclude_mask(masks, exclude_mask)

                # Save individual frame result
                if save_intermediate:
                    self._save_frame(frame_idx, masks)
                    self._save_intermediate_results(frame_idx, masks, flows)

                # Prepare temporal context
                context = {
                    'prev_masks': results['masks'][-1] if results['masks'] else None,
                    'frame_index': frame_idx
                }

                # Second pass segmentation
                refined_masks = self.refined_segmentation(preprocessed, context)

                results['masks'].append(refined_masks)
                results['flows'].append(flows)
                results['metadata']['processed_frames'] += 1

                if (frame_idx + 1) % 10 == 0:
                    logger.info(f"Completed {frame_idx + 1}/{total_frames} frames")

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                logger.error(f"Frame shape: {frame.shape if 'frame' in locals() else 'Not created'}")
                logger.error(f"Preprocessed shape: {preprocessed.shape if 'preprocessed' in locals() else 'Not created'}")
                results['metadata']['failed_frames'].append(frame_idx)
                continue

        # Convert masks list to stack and save final result
        masks_stack = np.stack(results['masks'], axis=0)
        output_file = Path(self.output_dir).parent / 'segmentation.tif'
        tifffile.imwrite(output_file, masks_stack.astype(np.uint16))

        # Clean up all temporary directories including segmentation_output
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        if self.preprocessed_dir.exists():
            shutil.rmtree(self.preprocessed_dir)
        # Clean up the entire output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        logger.info(f"Processing complete. Final segmentation saved to: {output_file}")
        logger.info(f"Segmentation frames and model outputs saved in: {self.segmentation_dir}")
        return results

    def _combine_frames_to_stack(self):
        """Combine all temporary frames into a single stack"""
        # Get list of all frames in order
        frame_files = sorted(self.temp_dir.glob("frame_*.tif"))

        if not frame_files:
            raise ValueError("No frames found to combine")

        # Read all frames
        frames = [tifffile.imread(f) for f in frame_files]

        # Combine into stack
        stack = np.stack(frames, axis=0)

        # Save stack
        stack_path = self.output_dir / "masks_stack.tif"
        tifffile.imwrite(stack_path, stack)

        # Clean up temporary files
        shutil.rmtree(self.temp_dir)

        return stack_path

    def initialize_model(self, model_type: str = "cyto3", gpu: bool = True) -> None:
        """Initialize the Cellpose model"""
        logger.info(f"Initializing Cellpose model: {model_type}")
        self.model = models.Cellpose(gpu=gpu, model_type=model_type)

    def load_image_stack(self, path: Union[str, Path]) -> np.ndarray:
        path = Path(path)
        logger.info(f"Loading image stack from: {path}")

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in ['.tif', '.tiff']:
            raise ValueError(f"Expected .tif file, got: {path.suffix}")

        # Load the stack
        stack = tifffile.imread(path)

        # Verify dimensions
        if stack.ndim != 3:
            raise ValueError(f"Expected 3 dimensions (t,x,y), got {stack.ndim} dimensions")

        logger.info(f"Loaded stack with shape: {stack.shape}")
        return stack

    def conservative_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """First pass: Conservative cell segmentation"""
        if self.model is None:
            self.initialize_model()

        # Run Cellpose with documented parameters
        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=self.params.diameter,
            batch_size=8,
            channels=[0, 0],
            normalize=True,
            do_3D=False
        )

        # Create and apply exclude mask post-segmentation
        exclude_mask = self._create_exclude_mask(image)
        masks = self._apply_exclude_mask(masks, exclude_mask)

        return masks, flows

    def refined_segmentation(self, image: np.ndarray, context: Dict) -> np.ndarray:
        """Second pass: Refined segmentation using temporal context"""
        if self.model is None:
            self.initialize_model()

        prev_masks = context.get('prev_masks', None)

        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=self.params.diameter,
            batch_size=8,
            channels=[0, 0],
            normalize=True,
            do_3D=False
        )

        # Create and apply exclude mask post-segmentation
        exclude_mask = self._create_exclude_mask(image)
        masks = self._apply_exclude_mask(masks, exclude_mask)

        if prev_masks is not None:
            masks = self._apply_temporal_consistency(masks, prev_masks)

        return masks

    def _apply_temporal_consistency(self, current_masks: np.ndarray,
                                    reference_masks: np.ndarray) -> np.ndarray:
        """Apply temporal consistency constraints"""
        # Placeholder for temporal consistency logic
        return current_masks

    def _save_intermediate_results(self, frame_idx: int,
                                   masks: np.ndarray,
                                   flows: np.ndarray) -> None:
        """Save intermediate results to cache"""
        self.results_cache[frame_idx] = {
            'masks': masks,
            'flows': flows,
            'timestamp': np.datetime64('now')
        }

    def save_results(self, results: Dict, output_path: Path) -> None:
        """Save processing results"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save masks and flows as numpy arrays
        np.save(output_path / 'masks.npy', np.array(results['masks']))
        np.save(output_path / 'flows.npy', np.array(results['flows']))

        # Save metadata
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(results['metadata'], f, indent=2)

    @classmethod
    def load_results(cls, input_path: Path) -> Dict:
        """Load previously saved results"""
        input_path = Path(input_path)

        results = {
            'masks': np.load(input_path / 'masks.npy'),
            'flows': np.load(input_path / 'flows.npy'),
        }

        with open(input_path / 'metadata.json', 'r') as f:
            results['metadata'] = json.load(f)

        return results

def main(path):
    """Example usage of the pipeline"""
    # Create pipeline with default parameters and specify output directory
    pipeline = CellSegmentationPipeline(output_dir="D:/2024-11-27/position1/segmentation_output")

    custom_model_path = "D:/cellpose_models/cyto3_xenopus"  # full path to model
    pipeline.model = models.Cellpose(
        gpu=False,
        model_type=custom_model_path  # Changed this line
    )

    # Load image stack
    image_stack = pipeline.load_image_stack(path)

    # Process the stack
    results = pipeline.process_stack(image_stack)

if __name__ == "__main__":
    path = "D:/2024-11-27/position1/registered_membrane_slice_downsized.tif"
    main(path)