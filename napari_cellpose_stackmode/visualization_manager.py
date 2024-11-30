import logging
from typing import Optional
import numpy as np
import napari

logger = logging.getLogger(__name__)


class VisualizationManager:
    """Manages visualization of cell tracking results in napari."""

    def __init__(self, viewer: "napari.Viewer"):
        """Initialize visualization manager.

        Args:
            viewer: The napari viewer instance
        """
        self.viewer = viewer
        self.tracking_layer = None
        self.overlay_layer = None

    def update_tracking_visualization(self, tracked_data: Optional[np.ndarray]) -> None:
        """Update the tracking visualization in napari viewer.

        Args:
            tracked_data: 3D numpy array of tracked cell labels (t, y, x)
        """
        try:
            if tracked_data is None:
                logger.warning("No tracking data to visualize")
                return

            # Remove existing tracking layer if it exists
            if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                self.viewer.layers.remove(self.tracking_layer)

            # Add new tracking layer
            self.tracking_layer = self.viewer.add_labels(
                tracked_data,
                name='Cell Tracking',
                opacity=0.7,
                visible=True
            )

            logger.info(f"Updated tracking visualization with data shape {tracked_data.shape}")

        except Exception as e:
            logger.error(f"Error updating tracking visualization: {e}")
            raise

    def clear_visualization(self) -> None:
        """Remove all visualization layers."""
        try:
            if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                self.viewer.layers.remove(self.tracking_layer)
                self.tracking_layer = None

            if self.overlay_layer is not None and self.overlay_layer in self.viewer.layers:
                self.viewer.layers.remove(self.overlay_layer)
                self.overlay_layer = None

            logger.info("Cleared all visualization layers")

        except Exception as e:
            logger.error(f"Error clearing visualization: {e}")
            raise

    def set_layer_visibility(self, layer_name: str, visible: bool) -> None:
        """Set the visibility of a specific layer.

        Args:
            layer_name: Name of the layer to modify
            visible: Whether to show or hide the layer
        """
        try:
            for layer in self.viewer.layers:
                if layer.name == layer_name:
                    layer.visible = visible
                    logger.debug(f"Set {layer_name} visibility to {visible}")
                    break
        except Exception as e:
            logger.error(f"Error setting layer visibility: {e}")
            raise