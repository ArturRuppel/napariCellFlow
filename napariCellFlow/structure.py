import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization output generation"""
    # Feature toggles
    tracking_plots_enabled: bool = True
    edge_detection_overlay: bool = True
    intercalation_events: bool = True
    edge_length_evolution: bool = True
    create_example_gifs: bool = True
    cell_tracking_overlay: bool = True
    debug_mode: bool = False

    # Output settings
    max_example_gifs: int = 3
    output_dir: Optional[Path] = None

    # Figure settings
    figure_size: Tuple[float, float] = (10, 8)
    dpi: int = 300
    animation_interval: int = 200
    color_scheme: str = 'viridis'
    line_width: float = 1.5
    font_size: int = 10
    alpha: float = 0.7
    histogram_bins: int = 50

    def validate(self):
        """Validate configuration parameters"""
        if self.max_example_gifs < 1:
            raise ValueError("max_example_gifs must be at least 1")
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72")
        if self.animation_interval < 50:
            raise ValueError("animation_interval must be at least 50ms")
        if not all(x > 0 for x in self.figure_size):
            raise ValueError("figure_size dimensions must be positive")
        if self.line_width <= 0:
            raise ValueError("line_width must be positive")
        if self.font_size <= 0:
            raise ValueError("font_size must be positive")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be positive")


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters"""
    dilation_radius: int = 1
    min_overlap_pixels: int = 5
    data_dir: Path = Path("data")

    def validate(self):
        if self.dilation_radius < 1:
            raise ValueError("Dilation radius must be at least 1")
        if self.min_overlap_pixels < 1:
            raise ValueError("Minimum overlap pixels must be at least 1")


@dataclass
class TrackingParameters:
    """Parameters for cell tracking"""
    min_overlap_ratio: float = 0.5
    max_displacement: float = 50.0
    min_cell_size: int = 100
    enable_gap_closing: bool = True
    max_frame_gap: int = 3

    def validate(self):
        if not 0 < self.min_overlap_ratio <= 1:
            raise ValueError("Minimum overlap ratio must be between 0 and 1")
        if self.max_displacement <= 0:
            raise ValueError("Maximum displacement must be positive")
        if self.min_cell_size < 0:
            raise ValueError("Minimum cell size cannot be negative")
        if self.max_frame_gap < 1:
            raise ValueError("Maximum frame gap must be at least 1")


@dataclass
class EdgeAnalysisParams:
    """Combined parameters for edge analysis pipeline"""
    # Edge detection params
    dilation_radius: int = 1
    min_overlap_pixels: int = 5
    min_edge_length: float = 0.0
    filter_isolated: bool = True

    # Intercalation params
    temporal_window: int = 1
    min_contact_frames: int = 1

    def validate(self) -> None:
        """Validate all parameters"""
        if self.dilation_radius < 1 or self.dilation_radius > 10:
            raise ValueError("Dilation radius must be between 1 and 10")
        if self.min_overlap_pixels < 1 or self.min_overlap_pixels > 100:
            raise ValueError("Minimum overlap pixels must be between 1 and 100")
        if self.min_edge_length < 0:
            raise ValueError("Minimum edge length cannot be negative")
        if self.temporal_window < 1:
            raise ValueError("Temporal window must be positive")
        if self.min_contact_frames < 1:
            raise ValueError("Minimum contact frames must be positive")
        if self.min_contact_frames > self.temporal_window:
            raise ValueError("Minimum contact frames cannot exceed temporal window")



@dataclass
class CellBoundary:
    """
    Represents a boundary between two cells.

    Attributes:
        cell_ids (Tuple[int, int]): IDs of the two cells sharing the boundary
        coordinates (np.ndarray): Ordered (y,x) coordinates along boundary
        endpoint1 (np.ndarray): First endpoint (y,x)
        endpoint2 (np.ndarray): Second endpoint (y,x)
        length (float): Length of boundary in pixels
    """
    cell_ids: Tuple[int, int]
    coordinates: np.ndarray  # Ordered (y,x) coordinates along boundary
    endpoint1: np.ndarray  # First endpoint (y,x)
    endpoint2: np.ndarray  # Second endpoint (y,x)
    length: float  # Length of boundary in pixels



@dataclass
class IntercalationEvent:
    """
    Represents a T1 transition (neighbor exchange) event between cells.

    Attributes:
        frame (int): Frame where the event occurs
        losing_cells (Tuple[int, int]): IDs of cells that lose contact
        gaining_cells (Tuple[int, int]): IDs of cells that gain contact
        coordinates (np.ndarray): Central coordinates where event occurs
    """
    frame: int
    losing_cells: Tuple[int, int]
    gaining_cells: Tuple[int, int]
    coordinates: np.ndarray


@dataclass
class EdgeData:
    """Container for edge tracking data through time with proper units."""
    edge_id: int
    frames: List[int]
    cell_pairs: List[Tuple[int, int]]
    lengths: List[float]  # Now in micrometers after conversion
    coordinates: List[np.ndarray]
    time_points: List[float] = field(default_factory=list)
    intercalations: List['IntercalationEvent'] = field(default_factory=list)

    def add_frame(self, frame: int, cells: Tuple[int, int], length: float, coords: np.ndarray) -> None:
        """Add data for a new frame"""
        self.frames.append(frame)
        self.cell_pairs.append(cells)
        self.lengths.append(length)
        self.coordinates.append(coords)

    def add_intercalation(self, event: IntercalationEvent) -> None:
        """Record an intercalation event"""
        self.intercalations.append(event)
        self.intercalations.sort(key=lambda x: x.frame)


@dataclass
class EdgeAnalysisResults:
    """Container for complete analysis results"""

    def __init__(self, params: EdgeAnalysisParams):
        self.edges: Dict[int, EdgeData] = {}
        self.segmentation_data: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {
            'total_frames': 0,
            'frame_ids': [],
            'timestamp': datetime.now().isoformat(),
            'parameters': params
        }

    def add_edge(self, edge: EdgeData) -> None:
        """Add or update edge data"""
        self.edges[edge.edge_id] = edge

    def get_edge(self, edge_id: int) -> Optional[EdgeData]:
        """Retrieve edge data by ID"""
        return self.edges.get(edge_id)

    def set_segmentation_data(self, data: np.ndarray) -> None:
        """Store the original segmentation data"""
        self.segmentation_data = data

    def get_segmentation_data(self) -> Optional[np.ndarray]:
        """Retrieve the stored segmentation data"""
        return self.segmentation_data

    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata field"""
        self.metadata[key] = value

    def validate(self) -> None:
        """Validate results integrity"""
        if not self.edges:
            raise ValueError("No edge data present")
        if not self.metadata['frame_ids']:
            raise ValueError("No frames recorded")
        if not isinstance(self.metadata['parameters'], EdgeAnalysisParams):
            raise ValueError("Invalid or missing analysis parameters")
        if self.segmentation_data is None:
            raise ValueError("No segmentation data present")



