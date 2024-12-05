import logging
from typing import Optional, Tuple, Union
import numpy as np
import napari
from napari.utils.transforms import Affine

from napari_cellpose_stackmode import data_manager
from napari_cellpose_stackmode.debug_logging import log_visualization_ops

logger = logging.getLogger(__name__)


class VisualizationManager:
    """Manages visualization of cell tracking results in napari."""

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager"):
        self.viewer = viewer
        self.data_manager = data_manager
        self.tracking_layer = None
        self.overlay_layer = None
        self._updating = False
        self._current_dims = None  # Track current data dimensions

    @log_visualization_ops
    def update_tracking_visualization(self, data: Union[np.ndarray, Tuple[np.ndarray, int]]) -> None:
        """Update the tracking visualization with new data."""
        if self._updating:
            logger.debug("Visualization update cancelled - updating in progress")
            return

        try:
            self._updating = True
            logger.debug(f"Updating visualization with data type: {type(data)}")

            # Handle input data
            if isinstance(data, tuple):
                frame_data, frame_index = data
                self._update_single_frame(frame_data, frame_index)
            else:
                self._update_full_stack(data)

            # Validate after update
            if not self.validate_stack_consistency():
                raise ValueError("Stack consistency validation failed after visualization update")

        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            raise
        finally:
            self._updating = False
            if self.tracking_layer is not None:
                self.tracking_layer.refresh()

    @log_visualization_ops
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
            current_data = self.tracking_layer.data.copy()  # Create a copy to prevent reference issues
            if frame_data.shape != current_data.shape[1:]:
                raise ValueError(
                    f"Frame shape {frame_data.shape} doesn't match existing data shape "
                    f"{current_data.shape[1:]}"
                )

            # Update only the specified frame
            current_data[frame_index] = frame_data
            self.tracking_layer.data = current_data

    @log_visualization_ops
    def _update_full_stack(self, stack_data: np.ndarray) -> None:
        """Update the visualization with a full stack of data."""
        logger.debug(f"Updating full stack with shape {stack_data.shape}")
        # Handle 2D data
        if stack_data.ndim == 2:
            stack_data = stack_data[np.newaxis, ...]

        # Validate dimensions
        if stack_data.ndim != 3:
            raise ValueError(f"Stack data must be 3D, got shape {stack_data.shape}")

        # Create or update tracking layer
        if self.tracking_layer is None:
            self.tracking_layer = self._create_tracking_layer(stack_data)
            self._current_dims = stack_data.shape
        else:
            # Check if dimensions changed
            if stack_data.shape != self.tracking_layer.data.shape:
                logger.warning(
                    f"Data dimensions changed from {self.tracking_layer.data.shape} "
                    f"to {stack_data.shape}"
                )
                # Remove existing layer and create new one
                self.clear_visualization()
                self.tracking_layer = self._create_tracking_layer(stack_data)
                self._current_dims = stack_data.shape
            else:
                # Update existing layer
                self.tracking_layer.data = stack_data

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

    def _create_tracking_layer(self, data: np.ndarray) -> "napari.layers.Labels":
        """Create a new tracking layer with proper settings."""
        layer = self.viewer.add_labels(
            data,
            name='Cell Tracking',
            opacity=0.5,
            visible=True
        )
        return layer

    def set_data_manager(self, data_manager: "DataManager"):
        """Allow setting the data manager after initialization."""
        self.data_manager = data_manager
    @log_visualization_ops
    def update_tracking_visualization(self, data: Union[np.ndarray, Tuple[np.ndarray, int]]) -> None:
        """Update the tracking visualization with new data."""
        if self._updating:
            logger.debug("Visualization update cancelled - updating in progress")
            return

        try:
            self._updating = True
            logger.debug(f"Updating visualization with data type: {type(data)}")

            # Handle input data
            if isinstance(data, tuple):
                frame_data, frame_index = data
                self._update_single_frame(frame_data, frame_index)
            else:
                self._update_full_stack(data)

            # Validate after update
            if not self.validate_stack_consistency():
                raise ValueError("Stack consistency validation failed after visualization update")

        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            raise
        finally:
            self._updating = False
            if self.tracking_layer is not None:
                self.tracking_layer.refresh()

    @log_visualization_ops
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
            current_data = self.tracking_layer.data.copy()  # Create a copy to prevent reference issues
            if frame_data.shape != current_data.shape[1:]:
                raise ValueError(
                    f"Frame shape {frame_data.shape} doesn't match existing data shape "
                    f"{current_data.shape[1:]}"
                )

            # Update only the specified frame
            current_data[frame_index] = frame_data
            self.tracking_layer.data = current_data

    @log_visualization_ops
    def _update_full_stack(self, stack_data: np.ndarray) -> None:
        """Update the visualization with a full stack of data."""
        logger.debug(f"Updating full stack with shape {stack_data.shape}")
        # Handle 2D data
        if stack_data.ndim == 2:
            stack_data = stack_data[np.newaxis, ...]

        # Validate dimensions
        if stack_data.ndim != 3:
            raise ValueError(f"Stack data must be 3D, got shape {stack_data.shape}")

        # Create or update tracking layer
        if self.tracking_layer is None:
            self.tracking_layer = self._create_tracking_layer(stack_data)
            self._current_dims = stack_data.shape
        else:
            # Check if dimensions changed
            if stack_data.shape != self.tracking_layer.data.shape:
                logger.warning(
                    f"Data dimensions changed from {self.tracking_layer.data.shape} "
                    f"to {stack_data.shape}"
                )
                # Remove existing layer and create new one
                self.clear_visualization()
                self.tracking_layer = self._create_tracking_layer(stack_data)
                self._current_dims = stack_data.shape
            else:
                # Update existing layer
                self.tracking_layer.data = stack_data

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
