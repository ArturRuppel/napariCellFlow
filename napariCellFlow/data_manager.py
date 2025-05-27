import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List
from threading import Lock
import numpy as np
import tifffile
from .debug_logging import log_operation

logger = logging.getLogger(__name__)
from napariCellFlow.structure import EdgeAnalysisResults, EdgeAnalysisParams, IntercalationEvent, EdgeData

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any


class DataSerializer:
    """Simple serialization using JSON for maximum compatibility"""

    @staticmethod
    def _convert_to_serializable(obj):
        """Convert an object to a JSON-serializable format"""
        # Handle numpy numeric types
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (datetime, np.datetime64)):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # Handle class instances by converting to dict
            return {k: DataSerializer._convert_to_serializable(v)
                    for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: DataSerializer._convert_to_serializable(v)
                    for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DataSerializer._convert_to_serializable(x) for x in obj]
        return obj

    @staticmethod
    def serialize_results(results: 'EdgeAnalysisResults', file_path: Path) -> None:
        """Convert analysis results to simple JSON-compatible dictionary"""
        if not str(file_path).endswith('.json'):
            file_path = file_path.with_suffix('.json')

        # Convert metadata with special handling for EdgeAnalysisParams
        metadata = {}
        for key, value in results.metadata.items():
            if key == 'parameters':
                # Convert EdgeAnalysisParams to dictionary
                metadata[key] = {
                    'dilation_radius': int(value.dilation_radius),  # Ensure integer
                    'min_overlap_pixels': int(value.min_overlap_pixels),  # Ensure integer
                    'min_edge_length': float(value.min_edge_length),  # Ensure float
                    'filter_isolated': bool(value.filter_isolated),  # Ensure boolean
                    'temporal_window': int(value.temporal_window),  # Ensure integer
                    'min_contact_frames': int(value.min_contact_frames)  # Ensure integer
                }
            else:
                metadata[key] = DataSerializer._convert_to_serializable(value)

        # Prepare the main data structure
        data = {
            'metadata': metadata,
            'edges': {}
        }

        # Convert each edge to simple dict
        for edge_id, edge in results.edges.items():
            data['edges'][str(edge_id)] = {
                'frames': [int(f) for f in edge.frames],  # Ensure integers
                'cell_pairs': [[int(c) for c in pair] for pair in edge.cell_pairs],  # Ensure integers
                'lengths': [float(l) for l in edge.lengths],  # Ensure floats
                'coordinates': [coords.tolist() for coords in edge.coordinates],
                'intercalations': [
                    {
                        'frame': int(event.frame),  # Ensure integer
                        'losing_cells': [int(c) for c in event.losing_cells],  # Ensure integers
                        'gaining_cells': [int(c) for c in event.gaining_cells],  # Ensure integers
                        'coordinates': event.coordinates.tolist()
                    }
                    for event in edge.intercalations
                ]
            }

        # Save segmentation data separately as numpy array if present
        if results.segmentation_data is not None:
            np.save(
                file_path.with_suffix('.seg.npy'),
                results.segmentation_data,
                allow_pickle=False
            )

        # Save main data as JSON
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def deserialize_results(file_path: Path) -> Dict[str, Any]:
        """Load data as simple dictionary - no special classes needed"""
        file_path = Path(file_path)

        # Load main JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Load segmentation data if it exists
        seg_file = file_path.with_suffix('.seg.npy')
        if seg_file.exists():
            data['segmentation_data'] = np.load(seg_file, allow_pickle=False)

        # Convert lists back to numpy arrays where needed
        for edge_id, edge in data['edges'].items():
            edge['coordinates'] = [np.array(coords) for coords in edge['coordinates']]
            for event in edge['intercalations']:
                event['coordinates'] = np.array(event['coordinates'])

        return data
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
        self.last_directory = None
        self.data_serializer = DataSerializer()

    @property
    def last_directory(self) -> Optional[Path]:
        """Get the last used directory."""
        return self._last_directory

    @last_directory.setter
    def last_directory(self, path: Optional[Union[str, Path]]):
        """Set the last used directory."""
        if path is not None:
            self._last_directory = Path(path)
        else:
            self._last_directory = None

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

    # In data_manager.py

    def save_analysis_results(self, file_path: Union[str, Path]) -> None:
        """
        Save the current analysis results to JSON format.

        Args:
            file_path: Path to save the results to

        Raises:
            ValueError: If no results exist or path is invalid
            IOError: If saving fails
        """
        if self._analysis_results is None:
            raise ValueError("No analysis results to save")

        file_path = Path(file_path)
        if not str(file_path).endswith('.json'):
            file_path = file_path.with_suffix('.json')

        try:
            self.data_serializer.serialize_results(self._analysis_results, file_path)
            logger.info(f"Analysis results saved to {file_path}")
            self.last_directory = Path(file_path).parent

        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
            raise IOError(f"Failed to save results: {str(e)}")

    def load_analysis_results(self, file_path: Union[str, Path]) -> None:
        """
        Load analysis results from JSON format.

        Args:
            file_path: Path to load the results from

        Raises:
            ValueError: If path is invalid or file format is incorrect
            IOError: If loading fails
        """
        file_path = Path(file_path)
        self.last_directory = Path(file_path).parent

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        try:
            data = self.data_serializer.deserialize_results(file_path)

            # Create EdgeAnalysisParams from loaded data
            params = EdgeAnalysisParams(
                dilation_radius=data['metadata']['parameters']['dilation_radius'],
                min_overlap_pixels=data['metadata']['parameters']['min_overlap_pixels'],
                min_edge_length=data['metadata']['parameters']['min_edge_length'],
                filter_isolated=data['metadata']['parameters']['filter_isolated'],
                temporal_window=data['metadata']['parameters']['temporal_window'],
                min_contact_frames=data['metadata']['parameters']['min_contact_frames']
            )

            # Create new EdgeAnalysisResults
            results = EdgeAnalysisResults(params)

            # Restore metadata
            for key, value in data['metadata'].items():
                if key != 'parameters':  # Skip parameters as we handle them separately
                    results.metadata[key] = value
            results.metadata['parameters'] = params  # Add back the params object

            # Restore edges
            for edge_id_str, edge_data in data['edges'].items():
                edge_id = int(edge_id_str)

                # Create intercalation events
                intercalations = []
                for event_data in edge_data['intercalations']:
                    event = IntercalationEvent(
                        frame=event_data['frame'],
                        losing_cells=tuple(event_data['losing_cells']),
                        gaining_cells=tuple(event_data['gaining_cells']),
                        coordinates=np.array(event_data['coordinates'])
                    )
                    intercalations.append(event)

                # Create edge data
                edge = EdgeData(
                    edge_id=edge_id,
                    frames=edge_data['frames'],
                    cell_pairs=[tuple(pair) for pair in edge_data['cell_pairs']],
                    lengths=edge_data['lengths'],
                    coordinates=[np.array(coords) for coords in edge_data['coordinates']],
                    intercalations=intercalations
                )

                results.add_edge(edge)

            # Restore segmentation data if present
            if 'segmentation_data' in data:
                results.set_segmentation_data(data['segmentation_data'])

            self._analysis_results = results
            logger.info(f"Analysis results loaded from {file_path}")

        except Exception as e:
            logger.error(f"Failed to load analysis results: {e}")
            raise IOError(f"Failed to load results: {str(e)}")

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
