# Cell Segmentation API Documentation

## Overview
The `segmentation.py` module provides a robust interface to Cellpose 3.0 for cell segmentation in microscopy images. It offers configurable segmentation parameters, GPU support, progress tracking, and comprehensive error handling through a Qt-based signal system.

## Key Features
- Integration with Cellpose 3.0 models (cyto3, nuclei, custom)
- Automatic or manual cell diameter computation
- GPU acceleration support
- Progress tracking via Qt signals
- Comprehensive error handling and logging
- Support for single and dual-channel segmentation

## Classes

### SegmentationParameters

A configuration class that encapsulates all parameters for Cellpose segmentation.

#### Attributes

##### Model Configuration
- `model_type` (str, default="cyto3"): 
  - Type of Cellpose model to use
  - Options: "cyto3", "nuclei", "custom"
  
- `custom_model_path` (Optional[str], default=None):
  - Path to custom model file
  - Required when model_type is "custom"

##### Core Segmentation Parameters
- `diameter` (float, default=95.0):
  - Expected cell diameter in pixels
  - Ignored if compute_diameter is True

- `flow_threshold` (float, default=0.6):
  - Flow error threshold for cell masks
  - Range: [0, 1]
  - Lower values are more permissive

- `cellprob_threshold` (float, default=0.3):
  - Cell probability threshold
  - Range: [0, 1]
  - Lower values detect more cells

- `min_size` (int, default=25):
  - Minimum cell size in pixels
  - Cells smaller than this are discarded

- `normalize` (bool, default=True):
  - Whether to normalize image intensity

##### Channel Configuration
- `chan_segment` (int, default=0):
  - Primary channel for segmentation

- `chan_2` (Optional[int], default=None):
  - Secondary channel (optional)
  - Used for nuclear or membrane channels

##### Advanced Parameters
- `batch_size` (int, default=8):
  - Batch size for processing
  - Affects memory usage

- `gpu` (bool, default=True):
  - Enable GPU acceleration
  - Requires CUDA-compatible GPU

- `compute_diameter` (bool, default=True):
  - Auto-compute cell diameter
  - Overrides manual diameter setting

- `stitch_threshold` (float, default=0.0):
  - Threshold for stitching tiles
  - Used for large image processing

- `resample` (bool, default=False):
  - Enable pixel resampling
  - Can improve results for anisotropic images

- `anisotropy` (Optional[float], default=None):
  - Pixel scaling ratio (z vs xy)
  - For 3D image processing

- `augment` (bool, default=False):
  - Use test-time augmentation
  - Can improve accuracy but slower

#### Methods
- `validate()`:
  - Validates all parameters
  - Raises `ValueError` for invalid configurations

### CellposeSignals

Qt signal handler for segmentation events.

#### Signals
- `segmentation_completed(object, object)`:
  - Emitted when segmentation succeeds
  - Arguments: masks array, metadata dictionary

- `segmentation_failed(str)`:
  - Emitted when segmentation fails
  - Argument: error message

- `progress_updated(int, str)`:
  - Emitted during processing
  - Arguments: progress percentage, status message

### SegmentationHandler

Main class that manages the Cellpose segmentation pipeline.

#### Constructor
```python
SegmentationHandler()
```
- Creates handler with default parameters
- Initializes signal handling system

#### Methods

##### initialize_model
```python
def initialize_model(params: SegmentationParameters) -> None
```
Initializes Cellpose model with specified parameters.

**Arguments:**
- `params`: SegmentationParameters instance

**Raises:**
- `ValueError`: Invalid parameters
- `RuntimeError`: Model initialization failure

**Behavior:**
- Validates parameters
- Sets up GPU if enabled
- Loads appropriate model type
- Emits progress signals

##### segment_frame
```python
def segment_frame(image: np.ndarray) -> Tuple[np.ndarray, dict]
```
Segments cells in a single image frame.

**Arguments:**
- `image`: Input image array (2D or 3D numpy array)

**Returns:**
- Tuple containing:
  1. Masks array (same shape as input)
  2. Results dictionary with:
     - `masks`: Segmentation masks
     - `flows`: Cellpose flow components
     - `styles`: Cell style vectors
     - `diameter`: Used/computed cell diameter
     - `parameters`: Complete parameter set

**Raises:**
- `RuntimeError`: Model not initialized
- `ValueError`: Invalid input
- Other exceptions from Cellpose

## Usage Examples

### Basic Usage
```python
import numpy as np
from segmentation import SegmentationHandler, SegmentationParameters

# Create handler
handler = SegmentationHandler()

# Initialize with default parameters
params = SegmentationParameters()
handler.initialize_model(params)

# Process an image
image = np.random.rand(512, 512)  # Your image here
masks, results = handler.segment_frame(image)
```

### Custom Configuration
```python
# Configure custom parameters
params = SegmentationParameters(
    model_type="cyto3",
    diameter=50.0,
    flow_threshold=0.5,
    cellprob_threshold=0.4,
    min_size=30,
    gpu=True
)

# Initialize handler with custom parameters
handler = SegmentationHandler()
handler.initialize_model(params)
```

### Progress Tracking
```python
def update_progress(percentage: int, message: str):
    print(f"Progress {percentage}%: {message}")

handler = SegmentationHandler()
handler.signals.progress_updated.connect(update_progress)
```

## Performance Considerations

- Memory Usage:
  - Batch size affects GPU memory usage
  - Large images may require tiling
  - GPU mode requires CUDA-compatible GPU

- Processing Speed:
  - GPU typically 5-10x faster than CPU
  - Augmentation significantly increases processing time
  - Larger batch sizes generally improve GPU utilization

## Error Handling

Common exceptions:
- `ValueError`: Invalid parameters or input
- `RuntimeError`: Model initialization or processing failures
- `MemoryError`: Insufficient memory
- `CUDAOutOfMemoryError`: Insufficient GPU memory

## Logging

The module uses Python's logging framework:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

Log messages include:
- Model initialization
- Processing progress
- Error conditions
- Cell count results