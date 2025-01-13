Image Preprocessing API Documentation
Overview
The preprocessing.py module provides a framework for preprocessing images with support for intensity scaling, filtering, and histogram equalization. It supports flexible configurations through the PreprocessingParameters dataclass and is designed for use with 8-bit images.

Classes
PreprocessingParameters
Encapsulates parameters for preprocessing images.

Attributes:
min_intensity (int): Minimum intensity for scaling. Defaults to 0.
max_intensity (int): Maximum intensity for scaling. Defaults to 255.
enable_median_filter (bool): Whether to apply a median filter. Defaults to False.
median_filter_size (int): Kernel size for the median filter. Must be an odd positive integer. Defaults to 3.
enable_gaussian_filter (bool): Whether to apply a Gaussian filter. Defaults to False.
gaussian_sigma (float): Sigma value for the Gaussian kernel. Must be positive. Defaults to 1.0.
enable_clahe (bool): Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization). Defaults to False.
clahe_clip_limit (float): Contrast limiting threshold for CLAHE. Must be positive. Defaults to 16.0.
clahe_grid_size (int): Grid size for CLAHE. Must be positive. Defaults to 16.
Methods:
validate(): Validates the parameter values, raising ValueError if any parameter is invalid.
ImagePreprocessor
Handles image preprocessing tasks.

Constructor:
__init__(parameters: Optional[PreprocessingParameters] = None): Initializes the preprocessor with the given parameters. If none are provided, default parameters are used.
Methods:
convert_to_8bit(image: np.ndarray) -> np.ndarray

Description: Converts any input image to an 8-bit image using percentile-based scaling.
Args:
image (np.ndarray): Input image of any bit depth.
Returns: An 8-bit scaled image.
preprocess_frame(image: np.ndarray) -> Tuple[np.ndarray, dict]

Description: Preprocesses a single image frame, applying intensity scaling, filters, and CLAHE if enabled.
Args:
image (np.ndarray): 2D input image of any bit depth.
Returns: A tuple containing:
Preprocessed image (np.ndarray).
Dictionary with preprocessing details (dict).
update_parameters(new_params: PreprocessingParameters) -> None

Description: Updates the preprocessing parameters.
Args:
new_params (PreprocessingParameters): The new parameters to apply.
Returns: None.
Preprocessing Workflow
Convert to 8-bit: Any image not already 8-bit is scaled using percentile-based intensity clipping.
Apply Intensity Scaling: Map intensity values between min_intensity and max_intensity to [0, 255].
Apply Gaussian Filtering (optional): Blurs the image to reduce noise.
Apply Median Filtering (optional): Smoothens the image using a median filter.
Apply CLAHE (optional): Enhances contrast using localized histogram equalization.
Example Usage
python
Copy code
import numpy as np
from preprocessing import ImagePreprocessor, PreprocessingParameters

# Load image (example: 16-bit grayscale)
image = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)

# Define preprocessing parameters
params = PreprocessingParameters(
    min_intensity=100,
    max_intensity=200,
    enable_median_filter=True,
    median_filter_size=5,
    enable_gaussian_filter=True,
    gaussian_sigma=2.0,
    enable_clahe=True,
    clahe_clip_limit=10.0,
    clahe_grid_size=8
)

# Create preprocessor
preprocessor = ImagePreprocessor(parameters=params)

# Preprocess the image
processed_image, info = preprocessor.preprocess_frame(image)

# Print preprocessing info
print("Preprocessing Info:", info)

# Update parameters
new_params = PreprocessingParameters(min_intensity=50, max_intensity=150)
preprocessor.update_parameters(new_params)
Logging
To enable debugging and monitor the preprocessing workflow:

python
Copy code
import logging
logging.basicConfig(level=logging.DEBUG)
For further customization, edit the PreprocessingParameters class or extend the ImagePreprocessor class to include additional methods.