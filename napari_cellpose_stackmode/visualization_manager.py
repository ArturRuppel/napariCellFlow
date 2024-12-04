import logging
from typing import Optional, Tuple, Union
import numpy as np
import napari
from napari.utils.transforms import Affine

logger = logging.getLogger(__name__)


class VisualizationManager:
    """Manages visualization of cell tracking results in napari."""

    def __init__(self, viewer: "napari.Viewer"):
        self.viewer = viewer
        self.tracking_layer = None
        self.overlay_layer = None
        self._updating = False

    def update_tracking_visualization(self, tracked_data: Union[np.ndarray, Tuple[np.ndarray, int]], full_update: bool = False) -> None:
        """Update the tracking visualization in napari viewer.

        Args:
            tracked_data: Either full stack array or tuple of (single frame data, frame index)
            full_update: Force update of entire stack
        """
        if self._updating:
            return

        try:
            self._updating = True

            # Handle different input types
            if isinstance(tracked_data, tuple):
                frame_data, frame_idx = tracked_data
                if self.tracking_layer is None:
                    raise ValueError("Layer must exist for single frame updates")

                # Update single frame
                with self.viewer.events.blocker_all():
                    current_data = self.tracking_layer.data
                    current_data[frame_idx] = frame_data
                    # Use direct data assignment to avoid full refresh
                    self.tracking_layer._data[frame_idx] = frame_data

                logger.debug(f"Updated single frame {frame_idx}")

            else:
                # Full stack update
                tracked_data = np.asarray(tracked_data)
                if tracked_data.ndim == 2:
                    tracked_data = tracked_data[np.newaxis, ...]

                with self.viewer.events.blocker_all():
                    if self.tracking_layer is None:
                        self.tracking_layer = self.viewer.add_labels(
                            tracked_data,
                            name='Cell Tracking',
                            opacity=0.7,
                            visible=True
                        )
                    else:
                        self.tracking_layer.data = tracked_data

                logger.debug(f"Updated full stack with shape {tracked_data.shape}")

            # Only refresh the current view
            if self.tracking_layer is not None:
                current_step = int(self.viewer.dims.point[0])
                self.tracking_layer.refresh()

        except Exception as e:
            logger.error(f"Error updating tracking visualization: {e}")
            raise
        finally:
            self._updating = False

    def clear_visualization(self) -> None:
        """Remove all visualization layers."""
        if self._updating:
            return

        try:
            self._updating = True
            if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                self.viewer.layers.remove(self.tracking_layer)
                self.tracking_layer = None

            if self.overlay_layer is not None and self.overlay_layer in self.viewer.layers:
                self.viewer.layers.remove(self.overlay_layer)
                self.overlay_layer = None

            logger.debug("Cleared all visualization layers")

        except Exception as e:
            logger.error(f"Error clearing visualization: {e}")
            raise
        finally:
            self._updating = False

    def _setup_layer_transforms(self, layer: "napari.layers.Layer", data_shape: Tuple[int, ...]) -> None:
        """Set up proper transforms for the layer based on data dimensions."""
        # Create identity transform for each dimension
        ndim = len(data_shape)
        scale = np.ones(ndim)
        translate = np.zeros(ndim)

        # Create affine transform
        affine_matrix = np.eye(ndim + 1)
        affine_matrix[:-1, :-1] = np.diag(scale)
        affine_matrix[:-1, -1] = translate

        transform = Affine(affine_matrix=affine_matrix)
        layer.affine = transform

    def set_layer_visibility(self, layer_name: str, visible: bool) -> None:
        """Set the visibility of a specific layer."""
        if self._updating:
            return

        try:
            self._updating = True
            for layer in self.viewer.layers:
                if layer.name == layer_name:
                    layer.visible = visible
                    logger.debug(f"Set {layer_name} visibility to {visible}")
                    break
        except Exception as e:
            logger.error(f"Error setting layer visibility: {e}")
            raise
        finally:
            self._updating = False
