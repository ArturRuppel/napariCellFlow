from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2
import logging
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingParameters:
    """Parameters for image preprocessing with integer intensity values"""
    # Contrast enhancement (absolute intensity values, integers for 8-bit)
    min_intensity: int = 0
    max_intensity: int = 255  # Fixed for 8-bit

    # Optional filters
    enable_median_filter: bool = False
    median_filter_size: int = 3

    enable_gaussian_filter: bool = False
    gaussian_sigma: float = 1.0  # This remains float as it's a kernel parameter

    # CLAHE parameters
    enable_clahe: bool = False
    clahe_clip_limit: float = 16.0
    clahe_grid_size: int = 16

    def validate(self):
        """Validate parameter values"""
        if not 0 <= self.min_intensity < self.max_intensity <= 255:
            raise ValueError("Intensity values must be between 0 and 255")
        if self.median_filter_size % 2 != 1:
            raise ValueError("Median filter size must be odd")
        if self.median_filter_size < 1:
            raise ValueError("Median filter size must be positive")
        if self.gaussian_sigma <= 0:
            raise ValueError("Gaussian sigma must be positive")
        if self.clahe_clip_limit <= 0:
            raise ValueError("CLAHE clip limit must be positive")
        if self.clahe_grid_size < 1:
            raise ValueError("CLAHE grid size must be positive")


class ImagePreprocessor:
    """Handles image preprocessing with standardized 8-bit output.

    This class provides a pipeline for image preprocessing that includes:
    - Bit depth conversion with outlier handling
    - Intensity scaling
    - Optional Gaussian and median filtering
    - Optional CLAHE enhancement

    All operations are performed in a specific order to ensure optimal results
    and maintain image quality throughout the pipeline.
    """

    def __init__(self, parameters: Optional[PreprocessingParameters] = None):
        self.params = parameters or PreprocessingParameters()

    def convert_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """Convert any input image to 8-bit using percentile-based scaling.

        Uses 1st and 99th percentiles to robustly handle outliers while
        preserving meaningful intensity variations. Values outside this
        range are clipped.

        Args:
            image: Input image array of any bit depth
                  Must be a 2D numpy array with numeric dtype

        Returns:
            8-bit image array scaled to use full dynamic range [0, 255]

        Raises:
            ValueError: If input is not a 2D array or contains invalid values

        Note:
            This operation creates a new array and may temporarily double
            memory usage for large images.
        """
        # Use percentiles to ignore outliers
        img_min = np.percentile(image, 1)
        img_max = np.percentile(image, 99)

        # Clip to the percentile range
        image_clipped = np.clip(image, img_min, img_max)

        # Scale to full 8-bit range
        image_scaled = ((image_clipped - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        return image_scaled

    def preprocess_frame(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Preprocess a single image frame through the complete pipeline.

        Processing steps (in order):
        1. Convert to 8-bit if needed (preserves relative intensities)
        2. Apply intensity scaling based on min/max parameters
        3. Apply Gaussian filter if enabled (reduces noise)
        4. Apply median filter if enabled (removes speckle noise)
        5. Apply CLAHE if enabled (enhances local contrast)

        Args:
            image: 2D numpy array of any bit depth
                  Must contain valid numeric values

        Returns:
            Tuple containing:
            - preprocessed image (np.ndarray): 8-bit processed image
            - info (dict): Processing metadata including:
                - original_dtype: Input image dtype
                - original_range: (min, max) of input
                - original_mean: Mean of input
                - original_std: Std dev of input
                - final_mean: Mean after processing
                - final_std: Std dev after processing
                - intensity_range: (min, max) intensity parameters used

        Raises:
            ValueError: If image is invalid or processing fails
        """
        # Store original statistics
        info = {
            'original_dtype': image.dtype,
            'original_range': (int(image.min()), int(image.max())),
            'original_mean': float(image.mean()),
            'original_std': float(image.std())
        }

        # Step 1: Convert to 8-bit if not already
        if image.dtype != np.uint8:
            processed = self.convert_to_8bit(image)
        else:
            processed = image.copy()

        # Step 2: Apply intensity scaling based on parameters
        lut = np.zeros(256, dtype=np.uint8)
        # Create lookup table that maps [min_intensity, max_intensity] to [0, 255]
        # Add +1 to the size calculation to include max_intensity in the range
        lut[self.params.min_intensity:self.params.max_intensity + 1] = np.linspace(
            0, 255, self.params.max_intensity - self.params.min_intensity + 1,
            dtype=np.uint8
        )
        # Ensure all values >= max_intensity map to 255
        lut[self.params.max_intensity:] = 255
        processed = lut[processed]

        # Step 3: Apply gaussian filter if enabled
        if self.params.enable_gaussian_filter:
            processed = cv2.GaussianBlur(
                processed,
                (0, 0),  # Auto kernel size
                self.params.gaussian_sigma
            )

        # Step 4: Apply median filter if enabled
        if self.params.enable_median_filter:
            processed = cv2.medianBlur(
                processed,
                self.params.median_filter_size
            )

        # Step 5: Apply CLAHE if enabled
        if self.params.enable_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.params.clahe_clip_limit,
                tileGridSize=(self.params.clahe_grid_size, self.params.clahe_grid_size)
            )
            processed = clahe.apply(processed)

        # Store final statistics
        info.update({
            'final_mean': float(processed.mean()),
            'final_std': float(processed.std()),
            'intensity_range': (self.params.min_intensity, self.params.max_intensity)
        })

        return processed, info


    def update_parameters(self, new_params: PreprocessingParameters) -> None:
        """Update preprocessing parameters"""
        new_params.validate()
        self.params = new_params
        logger.debug(f"Updated preprocessing parameters: {new_params}")