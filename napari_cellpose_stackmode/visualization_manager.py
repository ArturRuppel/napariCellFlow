import logging
from typing import Optional, Tuple, Union
import numpy as np
import napari
from napari.utils.transforms import Affine

from napari_cellpose_stackmode.debug_logging import log_visualization_ops

logger = logging.getLogger(__name__)


class VisualizationManager:
    """Manages visualization of cell tracking results in napari."""

    def __init__(self, viewer: "napari.Viewer"):
        self.viewer = viewer
        self.tracking_layer = None
        self.overlay_layer = None
        self._updating = False
        self._current_dims = None  # Track current data dimensions

    @log_visualization_ops
    def update_tracking_visualization(self, data: Union[np.ndarray, Tuple[np.ndarray, int]]) -> None:
        """
        Update the tracking visualization with new data.

        Args:
            data: Either a full 3D stack (np.ndarray) or a tuple of (2D frame, frame_index)
        """
        if self._updating:
            logger.debug("Visualization update cancelled - updating in progress")
            return

        try:
            self._updating = True
            logger.debug(f"Updating visualization with data type: {type(data)}")

            # Log shape and unique values of input data
            if isinstance(data, np.ndarray):
                logger.debug(f"Input data shape: {data.shape}")
                logger.debug(f"Unique values in input data: {np.unique(data)}")
            elif isinstance(data, tuple):
                logger.debug(f"Input frame data shape: {data[0].shape}")
                logger.debug(f"Unique values in input frame data: {np.unique(data[0])}")
                logger.debug(f"Input frame index: {data[1]}")

            # Handle input data
            if isinstance(data, tuple):
                frame_data, frame_index = data
                self._update_single_frame(frame_data, frame_index)
            else:
                self._update_full_stack(data)

            # Log final state of tracking layer data
            if self.tracking_layer is not None:
                logger.debug(f"Final tracking layer data shape: {self.tracking_layer.data.shape}")
                logger.debug(f"Unique values in final tracking layer data: {np.unique(self.tracking_layer.data)}")


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

        # Initialize tracking layer if needed
        if self.tracking_layer is None:
            num_frames = int(self.viewer.dims.range[0][1] + 1)
            empty_stack = np.zeros((num_frames, *frame_data.shape), dtype=frame_data.dtype)
            empty_stack[frame_index] = frame_data
            self.tracking_layer = self._create_tracking_layer(empty_stack)
            self._current_dims = empty_stack.shape
        else:
            # Ensure dimensions match
            if frame_data.shape != self.tracking_layer.data.shape[1:]:
                raise ValueError(
                    f"Frame shape {frame_data.shape} doesn't match existing data shape "
                    f"{self.tracking_layer.data.shape[1:]}"
                )

            # Update single frame while preserving others
            current_data = self.tracking_layer.data.copy()
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

    def _create_tracking_layer(self, data: np.ndarray) -> "napari.layers.Labels":
        """Create a new tracking layer with proper settings."""
        layer = self.viewer.add_labels(
            data,
            name='Cell Tracking',
            opacity=0.5,
            visible=True
        )

        # Set up proper transforms
        self._setup_layer_transforms(layer, data.shape)

        return layer

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

    def clear_visualization(self) -> None:
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
