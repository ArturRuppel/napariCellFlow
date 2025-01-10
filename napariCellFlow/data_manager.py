import logging
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List
from threading import Lock
import numpy as np
import tifffile
from .debug_logging import log_operation

logger = logging.getLogger(__name__)
from napariCellFlow.structure import EdgeAnalysisResults, EdgeAnalysisParams


class DataManager:
    """Manages data across different components of the application with robust frame handling."""

    def __init__(self):
        self._updating = False
        self._lock = Lock()
        self._preprocessed_data = None
        self._segmentation_data = None
        self._tracked_data = None
        self._analysis_results = None
        self._num_frames = None
        self._frame_states = {}  # Track state of individual frames
        self._initialized = False

    @property
    def analysis_results(self) -> Optional['EdgeAnalysisResults']:
        """Get the edge analysis results."""
        with self._lock:
            return self._analysis_results

    @analysis_results.setter
    def analysis_results(self, results: Optional['EdgeAnalysisResults']):
        """Set the edge analysis results with validation."""
        if self._updating:
            return

        with self._lock:
            try:
                self._updating = True
                self._analysis_results = results
            finally:
                self._updating = False

    def set_analysis_results(self, boundaries_by_frame: Dict[int, List['CellBoundary']],
                             edge_data: Dict[int, 'EdgeData'],
                             events: List['IntercalationEvent']) -> None:
        """
        Set analysis results from component data.

        Args:
            boundaries_by_frame: Dictionary mapping frame numbers to lists of cell boundaries
            edge_data: Dictionary mapping edge IDs to their tracking data
            events: List of detected intercalation events
        """

        # Create EdgeAnalysisResults object with default parameters
        results = EdgeAnalysisResults(EdgeAnalysisParams())

        # Add edge data
        for edge_id, data in edge_data.items():
            results.add_edge(data)

        # Update metadata
        if boundaries_by_frame:
            results.update_metadata('total_frames', max(boundaries_by_frame.keys()) + 1)
            results.update_metadata('frame_ids', sorted(boundaries_by_frame.keys()))

        self.analysis_results = results

    def initialize_stack(self, num_frames: int) -> None:
        """Initialize the stack with proper dimensionality and frame tracking."""
        with self._lock:
            try:
                logger.info(f"Initializing stack with {num_frames} frames")
                self._num_frames = num_frames
                self._frame_states = {i: {'modified': False, 'last_update': None} for i in range(num_frames)}
                self._initialized = True
                logger.debug("Stack initialization complete")
                logger.debug(f"Stack initialized with shape ({num_frames}, None, None)")

            except Exception as e:
                logger.error(f"Failed to initialize stack: {e}")
                raise ValueError(f"Stack initialization failed: {str(e)}")

    @property
    def preprocessed_data(self) -> Optional[np.ndarray]:
        """Get the preprocessed data."""
        with self._lock:
            return self._preprocessed_data

    @preprocessed_data.setter
    def preprocessed_data(self, data: Optional[np.ndarray]):
        """Set the preprocessed data with validation."""
        if self._updating:
            return

        with self._lock:
            try:
                self._updating = True
                if data is not None:
                    if not isinstance(data, np.ndarray):
                        raise ValueError("Preprocessed data must be a numpy array")
                    # Update number of frames if not already set
                    if self._num_frames is None:
                        self._num_frames = data.shape[0] if data.ndim == 3 else 1
                self._preprocessed_data = data
            finally:
                self._updating = False

    @property
    def segmentation_data(self) -> Optional[np.ndarray]:
        """Get the segmentation data."""
        with self._lock:
            return self._segmentation_data

    @segmentation_data.setter
    def segmentation_data(self, value: Union[np.ndarray, Tuple[np.ndarray, int]]):
        """Set segmentation data with robust frame preservation."""
        if self._updating:
            logger.debug("DataManager: Update cancelled - already updating")
            return

        with self._lock:
            try:
                self._updating = True

                # Handle single frame vs full stack update
                if isinstance(value, tuple):
                    self._update_single_frame(*value)
                else:
                    self._update_full_stack(value)

            except Exception as e:
                logger.error("DataManager: Error updating segmentation data", exc_info=True)
                raise
            finally:
                self._updating = False

    def _update_single_frame(self, frame_data: np.ndarray, frame_index: int):
        """Update a single frame in the segmentation data."""
        if frame_index >= self._num_frames:
            raise ValueError(f"Frame index {frame_index} out of bounds")

        if self._segmentation_data is None:
            # Initialize full stack with frame
            shape = (self._num_frames, *frame_data.shape)
            self._segmentation_data = np.zeros(shape, dtype=frame_data.dtype)

        elif frame_data.shape != self._segmentation_data.shape[1:]:
            raise ValueError(f"Frame shape mismatch: expected {self._segmentation_data.shape[1:]}, got {frame_data.shape}")

        # Update frame and its state
        self._segmentation_data[frame_index] = frame_data.copy()
        self._frame_states[frame_index] = {
            'modified': True,
            'last_update': np.datetime64('now')
        }

    def _update_full_stack(self, stack_data: Optional[np.ndarray]):
        """Update the full segmentation stack."""
        if stack_data is None:
            self._segmentation_data = None
            self._frame_states.clear()
            return

        # Handle 2D data
        if stack_data.ndim == 2:
            stack_data = stack_data[np.newaxis, ...]

        # Validate dimensions
        if stack_data.ndim != 3:
            raise ValueError(f"Invalid data dimensions: {stack_data.shape}")

        # Update stack and frame states
        if stack_data.shape[0] < self._num_frames:
            new_data = np.zeros((self._num_frames, *stack_data.shape[1:]), dtype=stack_data.dtype)
            new_data[:stack_data.shape[0]] = stack_data
            self._segmentation_data = new_data
        else:
            self._segmentation_data = stack_data.copy()

        self._frame_states = {
            i: {'modified': False, 'last_update': np.datetime64('now')}
            for i in range(self._segmentation_data.shape[0])
        }

    @property
    def tracked_data(self) -> Optional[np.ndarray]:
        """Get the tracked data."""
        with self._lock:
            return self._tracked_data

    @tracked_data.setter
    def tracked_data(self, data: Optional[np.ndarray]):
        """Set the tracked data with validation."""
        if self._updating:
            return

        with self._lock:
            try:
                self._updating = True
                if data is not None and not isinstance(data, np.ndarray):
                    raise ValueError("Tracked data must be a numpy array")
                self._tracked_data = data
            finally:
                self._updating = False
