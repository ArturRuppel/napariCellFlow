# Cell Tracking API Documentation

## Overview
The `cell_tracking.py` module provides a robust framework for tracking cells across time-series microscopy images using segmentation data. It implements an overlap-based tracking algorithm with support for gap closing, allowing reliable cell tracking even when cells temporarily disappear from view.

## Key Features
- Automated cell tracking across time-series data
- Gap closing to handle temporary cell disappearance
- Configurable tracking parameters
- Memory-efficient region property caching
- Progress tracking with callback support
- Comprehensive overlap-based tracking algorithm

## Classes

### RegionCache

A dataclass that caches essential region properties to avoid recomputation.

#### Attributes
- `label` (int): Unique identifier for the region
- `centroid` (np.ndarray): Center coordinates of the region
- `area` (float): Area of the region in pixels
- `coords` (np.ndarray): Array of pixel coordinates belonging to the region

### CellTracker

Main class that implements the cell tracking algorithm.

#### Constructor
```python
CellTracker(config: AnalysisConfig)
```
- Creates tracker with given configuration
- Initializes default tracking parameters
- Sets up region caching system

#### Main Methods

##### track_cells
```python
def track_cells(self, segmentation_stack: np.ndarray) -> np.ndarray
```
Tracks cells across frames in a segmentation stack.

**Arguments:**
- `segmentation_stack`: 3D numpy array (t, y, x) containing segmentation labels

**Returns:**
- 3D numpy array with consistent cell labels across frames

**Processing Steps:**
1. Validates input dimensions
2. Filters small cells if minimum size is set
3. Assigns initial IDs in first frame
4. Tracks cells frame by frame using overlap detection
5. Performs gap closing if enabled
6. Updates progress through callback if set

##### update_parameters
```python
def update_parameters(self, new_params: TrackingParameters) -> None
```
Updates tracking parameters and clears region cache.

**Arguments:**
- `new_params`: New tracking parameters to use

##### set_progress_callback
```python
def set_progress_callback(self, callback) -> None
```
Sets callback function for tracking progress updates.

**Arguments:**
- `callback`: Function taking progress (float) and message (str) arguments

#### Internal Methods

##### _handle_gap_closing
```python
def _handle_gap_closing(self, tracked_segmentation: np.ndarray, current_frame: int,
    overlap_matrix: Dict, assigned_ids: Set[int],
    segmentation_stack: np.ndarray) -> Dict
```
Handles tracking of temporarily disappeared cells.

**Key Features:**
- Looks ahead up to max_frame_gap frames
- Adjusts matching criteria based on gap length
- Returns future frame assignments
- Preserves cell IDs across gaps

##### _calculate_overlap_matrix
```python
def _calculate_overlap_matrix(self, current_frame: np.ndarray, next_frame: np.ndarray,
    current_idx: int, next_idx: int, max_displacement: float,
    min_overlap_ratio: float) -> Dict
```
Calculates cell overlap scores between frames.

**Algorithm:**
1. Caches region properties for efficiency
2. Uses vectorized operations for distance calculations
3. Combines overlap ratio and displacement for scoring
4. Returns dictionary of candidate matches sorted by score

##### _cache_regions
```python
def _cache_regions(self, frame: np.ndarray, frame_idx: int) -> List[RegionCache]
```
Caches region properties for efficient reuse.

**Features:**
- Uses frame content hash as cache key
- Stores essential properties only
- Avoids redundant calculations

##### _process_frame_regions
```python
def _process_frame_regions(self, current_frame: np.ndarray, output_frame: np.ndarray,
    overlap_matrix: Dict, assigned_ids: Set[int], all_used_ids: Set[int]) -> None
```
Processes and assigns cell IDs in current frame.

**Features:**
- Memory-efficient unique label finding
- Handles matched and unmatched cells
- Maintains consistent cell IDs

## Usage Examples

### Basic Usage
```python
import numpy as np
from napariCellFlow.cell_tracking import CellTracker
from napariCellFlow.structure import AnalysisConfig, TrackingParameters

# Create configuration
config = AnalysisConfig()

# Create tracker
tracker = CellTracker(config)

# Load segmentation data (t, y, x)
segmentation_stack = np.load('segmentation.npy')

# Track cells
tracked_stack = tracker.track_cells(segmentation_stack)
```

### Custom Parameters
```python
# Configure tracking parameters
params = TrackingParameters(
    max_displacement=50,
    min_overlap_ratio=0.5,
    min_cell_size=100,
    max_frame_gap=3,
    enable_gap_closing=True
)

# Update tracker parameters
tracker.update_parameters(params)
```

### Progress Tracking
```python
def progress_callback(progress: float, message: str):
    print(f"Progress: {progress:.1f}% - {message}")

# Set callback
tracker.set_progress_callback(progress_callback)
```

## Performance Considerations

### Memory Optimization
- Region caching reduces redundant calculations
- Chunk processing for large arrays
- Vectorized operations for distance calculations
- Efficient unique label finding

### Processing Speed
- Gap closing is most computationally intensive
- Performance scales with:
  - Number of cells per frame
  - Maximum displacement parameter
  - Maximum frame gap
  - Image dimensions

### Memory Usage
- Main memory requirements:
  - Input segmentation stack
  - Output tracking stack
  - Region property cache
  - Overlap matrices

## Error Handling

Common exceptions to handle:
- `ValueError`: Invalid input dimensions or parameters
- `MemoryError`: Insufficient memory for processing
- `TypeError`: Invalid input types

## Logging

The module uses Python's logging framework:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Log messages include:
- Gap closing operations
- Region processing status
- Error conditions
- Performance metrics

## Best Practices

### Parameter Tuning
1. Start with default parameters
2. Adjust max_displacement based on cell movement speed
3. Tune min_overlap_ratio based on cell shape changes
4. Enable gap closing for intermittent cell disappearance
5. Set min_cell_size to filter noise if needed

### Memory Management
1. Process large datasets in chunks if possible
2. Clear region cache when changing parameters
3. Monitor memory usage with large frame counts
4. Consider downsampling for initial parameter tuning

### Performance Optimization
1. Use appropriate max_displacement values
2. Limit max_frame_gap for gap closing
3. Filter small objects before tracking
4. Use progress callback for monitoring