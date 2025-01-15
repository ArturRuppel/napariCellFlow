# Edge Analysis Module Documentation

## Overview
The `edge_analysis.py` module provides advanced analysis of cell boundaries and topology changes in segmented microscopy data. It specializes in tracking edge dynamics and detecting T1 transitions (intercalations) between cells over time.

## Key Features
- Robust edge detection between adjacent cells
- Tracking of edge trajectories across frames
- Detection and validation of T1 transitions
- Edge length and boundary calculations
- Group tracking of related edges through topology changes

## Classes

### EdgeGroup
A container class that tracks groups of related edges through intercalation events.

#### Attributes
- `group_id` (int): Unique identifier for the edge group
- `edge_ids` (Set[int]): Set of edge IDs belonging to this group
- `cell_pairs` (Set[FrozenSet[int]]): Set of cell pairs involved in these edges
- `frames` (List[int]): List of frames where group members appear
- `active` (bool): Flag indicating if group is still active

### EdgeAnalyzer
Main class implementing edge detection and topology analysis.

#### Constructor
```python
EdgeAnalyzer(params: Optional[EdgeAnalysisParams] = None)
```
- Creates analyzer with given parameters
- Uses default parameters if none provided

#### Key Methods

##### analyze_sequence
```python
def analyze_sequence(tracked_data: np.ndarray, progress_callback=None) -> EdgeAnalysisResults
```
Process a complete sequence of segmented frames.

**Arguments:**
- `tracked_data`: 3D numpy array (frames, height, width) of segmented cells
- `progress_callback`: Optional callback function(progress: int, message: str)

**Returns:**
- EdgeAnalysisResults object containing:
  - Edge trajectories
  - Intercalation events
  - Analysis metadata

**Processing Steps:**
1. Reset internal state
2. Process frames sequentially
3. Detect edges in each frame
4. Track topology changes between frames
5. Build edge trajectories
6. Compile final results

##### _detect_edges
```python
def _detect_edges(self, frame_data: np.ndarray) -> List[CellBoundary]
```
Detect cell boundaries in a single frame.

**Arguments:**
- `frame_data`: 2D numpy array of segmented cells

**Returns:**
- List of CellBoundary objects

**Details:**
- Identifies shared boundaries between adjacent cells
- Filters isolated edges if enabled
- Applies minimum length and overlap criteria

##### _detect_topology_changes
```python
def _detect_topology_changes(self, frame: int, next_frame: int) -> List[IntercalationEvent]
```
Detect T1 transitions between consecutive frames.

**Arguments:**
- `frame`: Index of first frame
- `next_frame`: Index of second frame

**Returns:**
- List of IntercalationEvent objects

**Details:**
- Compares edge graphs between frames
- Validates potential T1 transitions
- Tracks edge group relationships
- Records event coordinates

##### _find_shared_boundary
```python
def _find_shared_boundary(self, frame: np.ndarray, cell1_id: int, cell2_id: int) -> Optional[CellBoundary]
```
Identify and measure the boundary between two cells.

**Arguments:**
- `frame`: 2D array of segmented cells
- `cell1_id`, `cell2_id`: IDs of cells to analyze

**Returns:**
- CellBoundary object or None if no valid boundary

**Processing Steps:**
1. Create cell masks
2. Dilate masks to find overlap
3. Skeletonize boundary
4. Order boundary pixels
5. Calculate length
6. Apply filters

##### _create_edge_trajectories
```python
def _create_edge_trajectories(self, boundaries_by_frame: Dict[int, List[CellBoundary]], intercalations: List[IntercalationEvent]) -> Dict[int, EdgeData]
```
Create edge trajectories with forward-time merging logic.

**Arguments:**
- `boundaries_by_frame`: Dictionary mapping frames to detected boundaries
- `intercalations`: List of detected intercalation events

**Returns:**
- Dictionary mapping edge IDs to EdgeData objects

**Processing Steps:**
1. Sort events chronologically
2. Process intercalations
3. Create/merge trajectories
4. Process boundaries frame by frame
5. Update trajectory data

## Performance Considerations

- Memory Usage:
  - Stores graph representations of each frame
  - Maintains edge history and group mappings
  - Consider batch processing for very long sequences

- Processing Time:
  - Edge detection scales with number of cell pairs
  - T1 detection requires graph comparison
  - Trajectory building involves multiple passes

## Error Handling

Common exceptions to handle:
- `ValueError`: Invalid parameters or input data
- `TypeError`: Incorrect input types
- `IndexError`: Invalid cell IDs or frame indices

## Logging

The module uses Python's logging framework:
```python
import logging
logging.basicConfig(level=logging.DEBUG)  # For detailed operation logging
```

Log messages include:
- Analysis progress
- Event detection
- Edge group operations
- Processing statistics

## Usage Examples

### Basic Usage
```python
import numpy as np
from napariCellFlow.edge_analysis import EdgeAnalyzer, EdgeAnalysisParams

# Create analyzer with default parameters
analyzer = EdgeAnalyzer()

# Load and analyze sequence
data = np.random.randint(1, 10, (10, 100, 100))  # Example data
results = analyzer.analyze_sequence(data)
```

### Custom Configuration
```python
# Configure custom parameters
params = EdgeAnalysisParams(
    dilation_radius=2,
    min_edge_length=10,
    min_overlap_pixels=5,
    filter_isolated=True
)

# Create analyzer with custom parameters
analyzer = EdgeAnalyzer(params)

# Analyze with progress tracking
def progress_callback(progress, message):
    print(f"{progress}%: {message}")

results = analyzer.analyze_sequence(data, progress_callback)
```

### Accessing Results
```python
# Get edge trajectories
for edge_id, edge_data in results.edges.items():
    print(f"Edge {edge_id}:")
    print(f"  Frames: {edge_data.frames}")
    print(f"  Length changes: {edge_data.lengths}")
    print(f"  Intercalations: {len(edge_data.intercalations)}")
```