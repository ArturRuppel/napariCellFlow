import logging
from threading import Lock
from typing import Optional, Tuple, Union
import numpy as np
import napari
from napari.utils.transforms import Affine

from napari_cellpose_stackmode import data_manager
from napari_cellpose_stackmode.debug_logging import log_operation

logger = logging.getLogger(__name__)


class VisualizationManager:
    """Manages visualization of cell tracking results in napari."""

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager"):
        self.viewer = viewer
        self.data_manager = data_manager
        self.tracking_layer = None
        self.overlay_layer = None
        self._updating = False
        self._current_dims = None
        self._layer_lock = Lock()  # Add thread safety

        # Connect to layer removal event
        self.viewer.layers.events.removed.connect(self._handle_layer_removal)

    def update_tracking_visualization(self, data: Union[np.ndarray, Tuple[np.ndarray, int]]) -> None:
        """Update tracking visualization with layer preservation."""
        if self._updating:
            logger.debug("VisualizationManager: Update cancelled - already updating")
            return

        try:
            self._updating = True
            with self._layer_lock:
                # Handle single frame or full stack update
                update_data = self._prepare_update_data(data)

                # Update or create layer
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

        except Exception as e:
            logger.error(f"VisualizationManager: Error updating visualization: {e}")
            raise
        finally:
            self._updating = False

    def _prepare_update_data(self, data: Union[np.ndarray, Tuple[np.ndarray, int]]) -> np.ndarray:
        """Prepare data for visualization update."""
        if isinstance(data, tuple):
            frame_data, frame_index = data
            num_frames = self.data_manager._num_frames

            if self.tracking_layer is None:
                # Create new full-sized stack
                update_data = np.zeros((num_frames, *frame_data.shape), dtype=frame_data.dtype)
                update_data[frame_index] = frame_data
            else:
                # Update existing stack
                update_data = self._update_existing_stack(frame_data, frame_index, num_frames)
        else:
            # Handle full stack update
            update_data = self._prepare_full_stack(data)

        return update_data

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

    def validate_stack_consistency(self):
        """Validate that visualization state is consistent"""
        if self.tracking_layer is None:
            return True

        if self._current_dims is not None:
            if self.tracking_layer.data.shape != self._current_dims:
                logger.error(f"Visualization shape mismatch: expected {self._current_dims}, got {self.tracking_layer.data.shape}")
                return False

        # Check against data manager if available
        if self.data_manager is not None and self.data_manager.segmentation_data is not None:
            stack_shape = self.data_manager.segmentation_data.shape
            visualization_shape = self.tracking_layer.data.shape

            if stack_shape != visualization_shape:
                logger.error(f"Stack shape mismatch: DataManager={stack_shape}, Visualization={visualization_shape}")
                return False

        return True

    def clear_visualization(self):
        """Remove all visualization layers."""
        if self._updating:
            return

        try:
            self._updating = True

            if self.tracking_layer is not None:
                if self.tracking_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.tracking_layer)
                self.tracking_layer = None

            if self.overlay_layer is not None:
                if self.overlay_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.overlay_layer)
                self.overlay_layer = None

            self._current_dims = None
            logger.debug("Cleared all visualization layers")

        except Exception as e:
            logger.error(f"Error clearing visualization: {e}")
            raise
        finally:
            self._updating = False

    def set_data_manager(self, data_manager: "DataManager"):
        """Allow setting the data manager after initialization."""
        self.data_manager = data_manager

    def _setup_layer_transforms(self, layer: "napari.layers.Layer", data_shape: Tuple[int, ...]) -> None:
        """Set up proper transforms for the layer based on data dimensions."""
        ndim = len(data_shape)
        scale = np.ones(ndim)
        translate = np.zeros(ndim)

        affine_matrix = np.eye(ndim + 1)
        affine_matrix[:-1, :-1] = np.diag(scale)
        affine_matrix[:-1, -1] = translate

        transform = Affine(affine_matrix=affine_matrix)
        layer.affine = transform

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the data for the current frame."""
        if self.tracking_layer is None:
            return None

        current_step = int(self.viewer.dims.point[0])
        return self.tracking_layer.data[current_step]

    def set_layer_visibility(self, visible: bool) -> None:
        """Set the visibility of the tracking layer."""
        if self.tracking_layer is not None:
            self.tracking_layer.visible = visible

    def cleanup(self) -> None:
        """Clean up resources when closing."""
        self.clear_visualization()
        self.viewer = None


