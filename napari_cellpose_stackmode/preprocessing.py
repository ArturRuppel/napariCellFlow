import logging
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingParameters:
    """Parameters for image preprocessing"""
    # Median filter parameters
    median_filter_size: int = 7

    # CLAHE parameters
    clahe_clip_limit: float = 16.0
    clahe_grid_size: int = 16

    # Intensity clipping parameters
    initial_lower_percentile: float = 1.0
    final_lower_percentile: float = 30.0
    final_upper_percentile: float = 99.0

    # Dark region threshold
    black_region_threshold: int = 3

    def validate(self):
        """Validate parameter values"""
        if self.median_filter_size % 2 != 1:
            raise ValueError("Median filter size must be odd")
        if self.median_filter_size < 1:
            raise ValueError("Median filter size must be positive")
        if self.clahe_clip_limit <= 0:
            raise ValueError("CLAHE clip limit must be positive")
        if self.clahe_grid_size < 1:
            raise ValueError("CLAHE grid size must be positive")
        if not 0 <= self.initial_lower_percentile < 100:
            raise ValueError("Initial lower percentile must be between 0 and 100")
        if not 0 <= self.final_lower_percentile < self.final_upper_percentile <= 100:
            raise ValueError("Final percentiles must be between 0 and 100 and in order")
        if self.black_region_threshold < 0 or self.black_region_threshold > 255:
            raise ValueError("Black region threshold must be between 0 and 255")


class ImagePreprocessor:
    """Handles image preprocessing operations for cell segmentation"""

    def __init__(self, parameters: Optional[PreprocessingParameters] = None):
        self.params = parameters or PreprocessingParameters()

    def preprocess_stack(self, image_stack: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Preprocess an entire image stack.

        Args:
            image_stack: 3D numpy array (t, y, x)

        Returns:
            Tuple of (preprocessed stack, list of preprocessing info per frame)
        """
        if image_stack.ndim != 3:
            raise ValueError(f"Expected 3D stack, got shape {image_stack.shape}")

        processed_stack = np.zeros_like(image_stack, dtype=np.uint8)
        preprocessing_info = []

        for t in range(len(image_stack)):
            processed, info = self.preprocess_frame(image_stack[t])
            processed_stack[t] = processed
            preprocessing_info.append(info)

        return processed_stack, preprocessing_info

    def preprocess_frame(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Preprocess a single image frame.

        Args:
            image: 2D numpy array

        Returns:
            Tuple of (preprocessed image, preprocessing info dictionary)
        """
        # Store original statistics
        info = {
            'original_dtype': image.dtype,
            'original_range': (float(image.min()), float(image.max())),
            'original_mean': float(image.mean()),
            'original_std': float(image.std())
        }

        # Scale to 8-bit
        scaled = self._scale_to_8bit(image)
        info['post_scaling_mean'] = float(scaled.mean())

        # Initial percentile-based clipping
        lower = np.percentile(scaled, self.params.initial_lower_percentile)
        pre_clipped = np.clip(scaled, lower, 255).astype(np.uint8)
        info['initial_clip_threshold'] = float(lower)

        # Median filtering
        median_filtered = cv2.medianBlur(pre_clipped, self.params.median_filter_size)

        # CLAHE enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.params.clahe_clip_limit,
            tileGridSize=(self.params.clahe_grid_size, self.params.clahe_grid_size)
        )
        enhanced = clahe.apply(median_filtered)

        # Final percentile-based clipping
        final_lower = np.percentile(enhanced, self.params.final_lower_percentile)
        final_upper = np.percentile(enhanced, self.params.final_upper_percentile)
        final_enhanced = np.clip(enhanced, final_lower, final_upper)

        # Rescale to full range
        if final_upper > final_lower:
            final_enhanced = ((final_enhanced - final_lower) *
                              (255.0 / (final_upper - final_lower))).clip(0, 255)

        final_enhanced = final_enhanced.astype(np.uint8)

        # Create exclude mask
        exclude_mask = self._create_exclude_mask(final_enhanced)

        # Store final statistics
        info.update({
            'final_lower_threshold': float(final_lower),
            'final_upper_threshold': float(final_upper),
            'final_mean': float(final_enhanced.mean()),
            'final_std': float(final_enhanced.std()),
            'excluded_pixels': int(np.sum(exclude_mask))
        })

        return final_enhanced, info

    def _scale_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """Scale image to 8-bit with proper normalization"""
        img_min = np.percentile(image, 1)
        img_max = np.percentile(image, 99)

        image_clipped = np.clip(image, img_min, img_max)
        image_scaled = ((image_clipped - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        return image_scaled

    def _create_exclude_mask(self, image: np.ndarray) -> np.ndarray:
        """Create mask of dark regions to exclude from analysis"""
        return (image <= self.params.black_region_threshold).astype(np.uint8)

    def update_parameters(self, new_params: PreprocessingParameters) -> None:
        """Update preprocessing parameters"""
        new_params.validate()
        self.params = new_params
        logger.debug(f"Updated preprocessing parameters: {new_params}")