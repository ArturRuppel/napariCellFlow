import colorsys
import logging
from threading import Lock
from typing import Optional, Dict, List
from typing import Tuple, Union

import napari
import numpy as np
from napari.layers import Layer, Labels, Points
from napari.utils.transforms import Affine

from napariCellFlow.structure import EdgeAnalysisResults
from .error_handling import ErrorSeverity, ErrorHandlingMixin, ApplicationError
from .structure import CellBoundary, IntercalationEvent

logger = logging.getLogger(__name__)




class VisualizationManager(ErrorHandlingMixin):
    """Manages visualization of cell tracking results in napari."""

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager"):
        self.viewer = viewer
        self.data_manager = data_manager
        self.tracking_layer = None
        self._updating = False
        self._layer_lock = Lock()

        # Connect to layer removal event with error handling
        try:
            self.viewer.layers.events.removed.connect(self._handle_layer_removal)
        except Exception as e:
            self.handle_error(self.create_error(
                message="Failed to connect layer events",
                details=str(e),
                severity=ErrorSeverity.WARNING,
                recovery_hint="Layer removal events may not be handled properly"
            ))

    def update_tracking_visualization(self, data: Union[np.ndarray, Tuple[np.ndarray, int]]) -> None:
        """Update tracking visualization with layer preservation and error handling."""
        if self._updating:
            logger.debug("VisualizationManager: Update cancelled - already updating")
            return

        try:
            self._updating = True
            with self._layer_lock:
                update_data = self._prepare_update_data(data)

                with self.viewer.events.blocker_all():
                    if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                        self.tracking_layer.data = update_data
                        self.tracking_layer.refresh()
                    else:
                        self.tracking_layer = self.viewer.add_labels(
                            update_data,
                            name='Segmentation',
                            opacity=0.5,
                            visible=True
                        )

        except ValueError as e:
            self.handle_error(self.create_error(
                message="Invalid data format",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check data dimensions and format"
            ))
        except Exception as e:
            self.handle_error(self.create_error(
                message="Failed to update visualization",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                original_error=e
            ))
        finally:
            self._updating = False

    def _prepare_update_data(self, data: Union[np.ndarray, Tuple[np.ndarray, int]]) -> np.ndarray:
        """Prepare data for visualization update with validation."""
        try:
            if isinstance(data, tuple):
                frame_data, frame_index = data
                if not isinstance(frame_data, np.ndarray):
                    raise ValueError("Frame data must be a numpy array")

                num_frames = self.data_manager._num_frames
                if frame_index >= num_frames:
                    raise ValueError(f"Frame index {frame_index} exceeds stack size {num_frames}")

                if self.tracking_layer is None:
                    update_data = np.zeros((num_frames, *frame_data.shape), dtype=frame_data.dtype)
                    update_data[frame_index] = frame_data
                else:
                    update_data = self._update_existing_stack(frame_data, frame_index, num_frames)
            else:
                if not isinstance(data, np.ndarray):
                    raise ValueError("Data must be a numpy array")
                update_data = self._prepare_full_stack(data)

            return update_data

        except ValueError as e:
            raise ApplicationError(
                message="Invalid data format",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check input data format and dimensions"
            )
        except Exception as e:
            raise ApplicationError(
                message="Failed to prepare visualization data",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                original_error=e
            )

    def clear_visualization(self):
        """Remove all visualization layers with error handling."""
        if self._updating:
            return

        try:
            self._updating = True
            self._remove_layers_safely()
            self._current_dims = None
            logger.debug("Cleared all visualization layers")

        except Exception as e:
            self.handle_error(self.create_error(
                message="Failed to clear visualization",
                details=str(e),
                severity=ErrorSeverity.WARNING,
                recovery_hint="Some layers may need to be removed manually"
            ))
        finally:
            self._updating = False

    def _remove_layers_safely(self):
        """Safely remove layers with individual error handling."""
        for layer_name in ['tracking_layer', 'overlay_layer']:
            layer = getattr(self, layer_name, None)
            if layer is not None:
                try:
                    if layer in self.viewer.layers:
                        self.viewer.layers.remove(layer)
                    setattr(self, layer_name, None)
                except Exception as e:
                    self.handle_error(self.create_error(
                        message=f"Failed to remove {layer_name}",
                        details=str(e),
                        severity=ErrorSeverity.WARNING
                    ))

    def _update_existing_stack(self, frame_data: np.ndarray, frame_index: int, num_frames: int) -> np.ndarray:
        """Update existing stack with new frame data."""
        if self.tracking_layer.data.shape[0] < num_frames:
            new_data = np.zeros((num_frames, *frame_data.shape), dtype=frame_data.dtype)
            new_data[:self.tracking_layer.data.shape[0]] = self.tracking_layer.data
            update_data = new_data
        else:
            update_data = self.tracking_layer.data.copy()
        update_data[frame_index] = frame_data
        return update_data

    def _prepare_full_stack(self, data: np.ndarray) -> np.ndarray:
        """Prepare full stack data for update."""
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.shape[0] < self.data_manager._num_frames:
            new_data = np.zeros((self.data_manager._num_frames, *data.shape[1:]), dtype=data.dtype)
            new_data[:data.shape[0]] = data
            return new_data
        return data.copy()

    def _handle_layer_removal(self, event):
        """Handle layer removal events safely."""
        layer = event.value
        if layer == self.tracking_layer:
            logger.debug("VisualizationManager: Tracking layer was removed")
            with self._layer_lock:
                if layer not in self.viewer.layers:
                    self.tracking_layer = None
                    self._current_dims = None

    def _create_tracking_layer(self, data: np.ndarray) -> "napari.layers.Labels":
        """Create or update tracking layer safely."""
        try:
            with self.viewer.events.blocker_all():
                with self._layer_lock:
                    # Always try to update existing layer first
                    if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                        logger.debug("VisualizationManager: Updating existing tracking layer")
                        self.tracking_layer.data = data
                        self.tracking_layer.refresh()
                        return self.tracking_layer

                    # Create new layer only if necessary
                    logger.debug("VisualizationManager: Creating new tracking layer")
                    layer = self.viewer.add_labels(
                        data,
                        name='Segmentation',
                        opacity=0.5,
                        visible=True
                    )
                    self.tracking_layer = layer
                    return layer
        except Exception as e:
            logger.error(f"Error creating tracking layer: {e}")
            raise

    def get_current_tracking_layer(self) -> Optional["napari.layers.Labels"]:
        """Get current tracking layer with validation."""
        with self._layer_lock:
            if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                return self.tracking_layer
            return None

    def _update_full_stack(self, stack_data: np.ndarray) -> None:
        """Update full stack while preserving layer."""
        logger.debug(f"Updating full stack with shape {stack_data.shape}")

        # Handle 2D data
        if stack_data.ndim == 2:
            stack_data = stack_data[np.newaxis, ...]

        # Validate dimensions
        if stack_data.ndim != 3:
            raise ValueError(f"Stack data must be 3D, got shape {stack_data.shape}")

        logger.debug(f"VisualizationManager: Starting full stack update")
        logger.debug(f"VisualizationManager: Input data shape: {stack_data.shape}")
        logger.debug(f"VisualizationManager: Input data unique values: {np.unique(stack_data)}")

        if self.tracking_layer is not None:
            logger.debug(f"VisualizationManager: Current tracking layer: {self.tracking_layer.name}")
            logger.debug(f"VisualizationManager: Tracking layer in viewer: {self.tracking_layer in self.viewer.layers}")
            if self.tracking_layer in self.viewer.layers:
                logger.debug(f"VisualizationManager: Current tracking data shape: {self.tracking_layer.data.shape}")
                logger.debug(f"VisualizationManager: Current tracking unique values: {np.unique(self.tracking_layer.data)}")

        # Log all current layers
        logger.debug("VisualizationManager: Current viewer layers:")
        for layer in self.viewer.layers:
            logger.debug(f"  - {layer.name} ({type(layer)})")

        try:
            # Update existing layer if possible
            if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                self.tracking_layer.data = stack_data
                self.tracking_layer.refresh()
            else:
                # Create new layer only if necessary
                self.tracking_layer = self.viewer.add_labels(
                    stack_data,
                    name='Segmentation',
                    opacity=0.5,
                    visible=True
                )

            self._current_dims = stack_data.shape

            logger.debug("VisualizationManager: After update:")
            logger.debug(f"VisualizationManager: Tracking layer still exists: {self.tracking_layer is not None}")
            if self.tracking_layer is not None:
                logger.debug(f"VisualizationManager: Tracking layer in viewer: {self.tracking_layer in self.viewer.layers}")
                logger.debug(f"VisualizationManager: Updated data shape: {self.tracking_layer.data.shape}")
                logger.debug(f"VisualizationManager: Updated unique values: {np.unique(self.tracking_layer.data)}")


        except Exception as e:
            logger.error(f"Error updating full stack: {e}")
            raise

    def _update_single_frame(self, frame_data: np.ndarray, frame_index: int) -> None:
        """Update a single frame in the visualization."""
        if frame_data.ndim != 2:
            logger.debug(f"Invalid frame data shape: {frame_data.shape}")
            raise ValueError(f"Frame data must be 2D, got shape {frame_data.shape}")

        logger.debug(f"Updating frame {frame_index}")

        if self.tracking_layer is None:
            # Initialize with proper dimensions
            if self._current_dims is None:
                num_frames = int(self.viewer.dims.range[0][1] + 1)
                empty_stack = np.zeros((num_frames, *frame_data.shape), dtype=frame_data.dtype)
                empty_stack[frame_index] = frame_data
                self.tracking_layer = self._create_tracking_layer(empty_stack)
                self._current_dims = empty_stack.shape
            else:
                # Use existing dimensions
                empty_stack = np.zeros(self._current_dims, dtype=frame_data.dtype)
                empty_stack[frame_index] = frame_data
                self.tracking_layer = self._create_tracking_layer(empty_stack)
        else:
            # Update existing layer
            current_data = self.tracking_layer.data.copy()
            if frame_data.shape != current_data.shape[1:]:
                raise ValueError(
                    f"Frame shape {frame_data.shape} doesn't match existing data shape "
                    f"{current_data.shape[1:]}"
                )

            current_data[frame_index] = frame_data
            self.tracking_layer.data = current_data

    def set_data_manager(self, data_manager: "DataManager"):
        """Allow setting the data manager after initialization."""
        self.data_manager = data_manager

    def cleanup(self) -> None:
        """Clean up resources when closing."""
        self.clear_visualization()
        self.viewer = None

    def _generate_distinct_color(self) -> np.ndarray:
        """Generate a distinct color using golden ratio"""
        golden_ratio = 0.618033988749895
        hue = self._color_cycle.random()
        hue += golden_ratio
        hue %= 1
        # Convert to RGB
        hsv = np.array([hue, 0.8, 0.95])
        rgb = np.array(colorsys.hsv_to_rgb(*hsv))
        return np.append(rgb, 1.0)  # Add alpha channel

    def update_edge_analysis_visualization(self, results: EdgeAnalysisResults) -> None:
        """Update final edge analysis visualization with unique colors per edge"""
        try:
            if self._analysis_layer is not None and self._analysis_layer in self.viewer.layers:
                self.viewer.layers.remove(self._analysis_layer)

            edge_coords = []
            edge_colors = []
            edge_properties = {
                'edge_id': [],
                'frame': [],
                'length': [],
                'has_intercalation': []
            }

            # Generate unique color for each edge
            edge_colors_map = {edge_id: self._generate_distinct_color()
                               for edge_id in results.edges.keys()}

            for edge_id, edge in results.edges.items():
                edge_color = edge_colors_map[edge_id]

                for frame_idx, frame in enumerate(edge.frames):
                    coords = edge.coordinates[frame_idx]
                    time_coords = np.column_stack((
                        np.full(len(coords), frame),
                        coords
                    ))
                    edge_coords.append(time_coords)
                    edge_colors.extend([edge_color] * len(coords))

                    edge_properties['edge_id'].extend([edge_id] * len(coords))
                    edge_properties['frame'].extend([frame] * len(coords))
                    edge_properties['length'].extend([edge.lengths[frame_idx]] * len(coords))
                    edge_properties['has_intercalation'].extend([bool(edge.intercalations)] * len(coords))

            if edge_coords:
                coords_array = np.vstack(edge_coords)

                self._analysis_layer = self.viewer.add_points(
                    coords_array,
                    name="Edge Analysis",
                    size=5,  # Increased thickness
                    face_color=edge_colors,
                    opacity=0.8,
                    properties=edge_properties,
                    ndim=3
                )

            logger.info("Updated edge analysis visualization")

        except Exception as e:
            logger.error(f"Error updating edge analysis visualization: {e}")
            raise

    def update_intercalation_visualization(self, results: EdgeAnalysisResults) -> None:
        """Visualize edges involved in intercalations"""
        try:
            if self._intercalation_layer is not None and self._intercalation_layer in self.viewer.layers:
                self.viewer.layers.remove(self._intercalation_layer)

            edge_coords = []
            edge_properties = {
                'frame': [],
                'cell_pair': [],
                'edge_id': []
            }

            # Use a single color for all intercalation edges
            edge_color = np.array([1, 1, 0, 1])  # Yellow

            # Process each edge that has intercalations
            for edge_id, edge in results.edges.items():
                if not edge.intercalations:
                    continue

                for event in edge.intercalations:
                    # Get frames right before and after intercalation
                    event_frame = event.frame
                    frames_to_show = [event_frame, event_frame + 1]

                    for frame in frames_to_show:
                        if frame in edge.frames:
                            idx = edge.frames.index(frame)
                            coords = edge.coordinates[idx]
                            time_coords = np.column_stack((
                                np.full(len(coords), frame),
                                coords
                            ))
                            edge_coords.append(time_coords)

                            # Add properties
                            edge_properties['frame'].extend([frame] * len(coords))
                            edge_properties['edge_id'].extend([edge_id] * len(coords))
                            cell_pair = f"{edge.cell_pairs[idx][0]}-{edge.cell_pairs[idx][1]}"
                            edge_properties['cell_pair'].extend([cell_pair] * len(coords))

            if edge_coords:
                coords_array = np.vstack(edge_coords)

                self._intercalation_layer = self.viewer.add_points(
                    coords_array,
                    name="Intercalation Events",
                    size=6,  # Thick lines
                    face_color=edge_color,
                    opacity=1.0,
                    properties=edge_properties,
                    ndim=3
                )

                logger.info("Updated intercalation visualization")
            else:
                logger.warning("No intercalation events to visualize")

        except Exception as e:
            logger.error(f"Error updating intercalation visualization: {e}")
            raise

    def update_edge_visualization(self, boundaries_by_frame: Dict[int, List[CellBoundary]]) -> None:
        """Update initial edge detection visualization"""
        try:
            if self._edge_layer is not None and self._edge_layer in self.viewer.layers:
                self.viewer.layers.remove(self._edge_layer)

            edge_coords = []
            edge_properties = {
                'frame': [],
                'cell_pair': []
            }

            for frame, boundaries in sorted(boundaries_by_frame.items()):
                for boundary in boundaries:
                    coords = boundary.coordinates
                    time_coords = np.column_stack((
                        np.full(len(coords), frame),
                        coords
                    ))
                    edge_coords.append(time_coords)

                    cell_pair = f"{boundary.cell_ids[0]}-{boundary.cell_ids[1]}"
                    edge_properties['frame'].extend([frame] * len(coords))
                    edge_properties['cell_pair'].extend([cell_pair] * len(coords))

            if edge_coords:
                coords_array = np.vstack(edge_coords)

                self._edge_layer = self.viewer.add_points(
                    coords_array,
                    name="Cell Edges",
                    size=4,  # Increased thickness
                    face_color='cyan',
                    opacity=0.8,
                    properties=edge_properties,
                    ndim=3
                )

            logger.info("Updated edge visualization")

        except Exception as e:
            logger.error(f"Error updating edge visualization: {e}")
            raise

    def clear_edge_layers(self) -> None:
        """Remove all edge-related layers"""
        layer_names = ['Edge Analysis', 'Intercalation Events', 'Cell Edges']
        for name in layer_names:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

    def clear_all_layers(self) -> None:
        """Remove all managed layers from the viewer"""
        if self._tracked_layer is not None and self._tracked_layer in self.viewer.layers:
            self.viewer.layers.remove(self._tracked_layer)
        self._tracked_layer = None

        self.clear_edge_layers()
        self._edge_layer = None
        self._intercalation_layer = None
        self._analysis_layer = None

    @property
    def tracked_layer(self) -> Optional[Labels]:
        """The current tracked cells layer"""
        return self._tracked_layer

    @property
    def edge_layer(self) -> Optional[Points]:
        """The current edge detection layer"""
        return self._edge_layer

    @property
    def analysis_layer(self) -> Optional[Points]:
        """The current edge analysis layer"""
        return self._analysis_layer
