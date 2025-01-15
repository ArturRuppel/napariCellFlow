# Image Preprocessing API Documentation

## Overview
The `preprocessing.py` module provides a robust framework for image preprocessing, specializing in standardizing images to 8-bit depth while preserving meaningful intensity variations. It offers configurable intensity scaling, noise reduction through filtering, and contrast enhancement via CLAHE (Contrast Limited Adaptive Histogram Equalization).

## Key Features
- Robust conversion to 8-bit depth using percentile-based scaling
- Configurable intensity range mapping
- Optional noise reduction via Gaussian and median filtering
- Optional contrast enhancement using CLAHE
- Comprehensive preprocessing metadata tracking

## Classes

### PreprocessingParameters

A configuration class that encapsulates all parameters needed for image preprocessing.

#### Attributes
- `min_intensity` (int, default=0): 
  - Minimum intensity value for scaling
  - Must be in range [0, 254]
  - Values below this are mapped to 0 in output

- `max_intensity` (int, default=255): 
  - Maximum intensity value for scaling
  - Fixed at 255 for 8-bit output
  - Values above this are mapped to 255

- `enable_median_filter` (bool, default=False):
  - Toggle for median filtering
  - Effective for removing speckle noise

- `median_filter_size` (int, default=3):
  - Kernel size for median filter
  - Must be positive odd integer
  - Larger values provide stronger smoothing but may blur features

- `enable_gaussian_filter` (bool, default=False):
  - Toggle for Gaussian filtering
  - Useful for general noise reduction

- `gaussian_sigma` (float, default=1.0):
  - Standard deviation for Gaussian kernel
  - Must be positive
  - Larger values increase blur radius

- `enable_clahe` (bool, default=False):
  - Toggle for CLAHE enhancement
  - Improves local contrast while preventing noise amplification

- `clahe_clip_limit` (float, default=16.0):
  - Contrast limit for CLAHE
  - Must be positive
  - Higher values allow more contrast enhancement

- `clahe_grid_size` (int, default=16):
  - Size of grid for CLAHE
  - Must be positive
  - Affects the size of local regions for contrast enhancement

#### Methods
- `validate()`: 
  - Validates all parameters
  - Raises `ValueError` with descriptive message if any parameter is invalid

### ImagePreprocessor

Main class that implements the preprocessing pipeline.

#### Constructor
```python
ImagePreprocessor(parameters: Optional[PreprocessingParameters] = None)
```
- Creates preprocessor with given parameters
- Uses default parameters if none provided

#### Methods

##### convert_to_8bit
```python
def convert_to_8bit(image: np.ndarray) -> np.ndarray
```
Converts images to 8-bit depth using robust percentile-based scaling.

**Arguments:**
- `image`: Input image array of any bit depth (2D numpy array)

**Returns:**
- 8-bit numpy array with values scaled to [0, 255]

**Details:**
- Uses 1st and 99th percentiles to handle outliers
- Preserves meaningful intensity variations
- Creates new array (input remains unchanged)

##### preprocess_frame
```python
def preprocess_frame(image: np.ndarray) -> Tuple[np.ndarray, dict]
```
Applies complete preprocessing pipeline to an image.

**Arguments:**
- `image`: Input image array (2D numpy array of any bit depth)

**Returns:**
- Tuple containing:
  1. Processed image (8-bit numpy array)
  2. Info dictionary with:
     - `original_dtype`: Input image data type
     - `original_range`: (min, max) of input image
     - `original_mean`: Mean of input image
     - `original_std`: Standard deviation of input
     - `final_mean`: Mean after processing
     - `final_std`: Standard deviation after processing
     - `intensity_range`: (min, max) intensity used for scaling

**Processing Steps:**
1. Convert to 8-bit if needed
2. Apply intensity scaling
3. Apply Gaussian filter (if enabled)
4. Apply median filter (if enabled)
5. Apply CLAHE (if enabled)

##### update_parameters
```python
def update_parameters(new_params: PreprocessingParameters) -> None
```
Updates preprocessing parameters.

**Arguments:**
- `new_params`: New parameters to use

**Behavior:**
- Validates parameters before updating
- Raises `ValueError` if parameters are invalid
- Logs parameter update at debug level

## Usage Examples

### Basic Usage
```python
import numpy as np
from preprocessing import ImagePreprocessor, PreprocessingParameters

# Create preprocessor with default parameters
preprocessor = ImagePreprocessor()

# Load and process an image
image = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
processed, info = preprocessor.preprocess_frame(image)
```

### Custom Configuration
```python
# Configure custom parameters
params = PreprocessingParameters(
    min_intensity=50,
    max_intensity=200,
    enable_gaussian_filter=True,
    gaussian_sigma=1.5,
    enable_clahe=True,
    clahe_clip_limit=10.0
)

# Create preprocessor with custom parameters
preprocessor = ImagePreprocessor(parameters=params)

# Process image
processed, info = preprocessor.preprocess_frame(image)
print(f"Original range: {info['original_range']}")
print(f"Final mean: {info['final_mean']:.2f}")
```

### Updating Parameters
```python
# Update parameters after creation
new_params = PreprocessingParameters(
    min_intensity=100,
    enable_median_filter=True,
    median_filter_size=5
)
preprocessor.update_parameters(new_params)
```

## Performance Considerations

- Memory Usage:
  - Conversion to 8-bit temporarily requires double the image memory
  - Each filter operation creates a new array
  - Consider processing in chunks for very large images

- Processing Time:
  - CLAHE is the most computationally intensive operation
  - Median filtering time increases significantly with kernel size
  - Gaussian filtering is generally fast even with large sigma

## Error Handling

Common exceptions to handle:
- `ValueError`: Invalid parameters or input images
- `TypeError`: Non-numeric or invalid shape input arrays
- `MemoryError`: Insufficient memory for processing

## Logging

The module uses Python's logging framework:
```python
import logging
logging.basicConfig(level=logging.DEBUG)  # For detailed operation logging
```

Log messages include:
- Parameter updates
- Processing stage completion
- Error conditions