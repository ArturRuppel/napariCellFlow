import logging
from functools import wraps
import numpy as np
from typing import Optional, Any, Callable
import inspect

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('cell_tracking_debug')
fh = logging.FileHandler('cell_tracking_debug.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def log_array_info(arr: Optional[np.ndarray], name: str) -> None:
    """Log detailed information about a numpy array"""
    if arr is None:
        logger.debug(f"{name} is None")
        return

    logger.debug(f"{name} info:")
    logger.debug(f"  Shape: {arr.shape}")
    logger.debug(f"  Dtype: {arr.dtype}")
    logger.debug(f"  Unique values: {np.unique(arr)}")
    if arr.size > 0:
        logger.debug(f"  Min: {arr.min()}, Max: {arr.max()}")


def log_state_changes(func: Callable) -> Callable:
    """Decorator to log state changes in key methods"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Log entry
        logger.debug(f"Entering {func.__name__}")
        logger.debug(f"  Args: {args}")
        logger.debug(f"  Kwargs: {kwargs}")

        # Log initial state
        if hasattr(self, 'masks_layer') and self.masks_layer is not None:
            log_array_info(self.masks_layer.data, "Initial masks_layer.data")
        if hasattr(self, '_full_masks'):
            log_array_info(self._full_masks, "Initial _full_masks")

        # Get current frame if available
        current_frame = None
        if hasattr(self, 'viewer') and hasattr(self.viewer, 'dims'):
            current_frame = int(self.viewer.dims.point[0])
            logger.debug(f"Current frame: {current_frame}")

        # Execute function
        result = func(self, *args, **kwargs)

        # Log final state
        if hasattr(self, 'masks_layer') and self.masks_layer is not None:
            log_array_info(self.masks_layer.data, "Final masks_layer.data")
        if hasattr(self, '_full_masks'):
            log_array_info(self._full_masks, "Final _full_masks")

        # Log exit
        logger.debug(f"Exiting {func.__name__}")
        return result

    return wrapper


def log_data_manager_ops(func):
    """Decorator for logging DataManager operations"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Don't log property access, only property setting
        if len(args) == 0 and not kwargs:
            return func(self)

        logger.debug(f"DataManager: Entering {func.__name__}")

        # Only log data info for setter operations
        if len(args) > 0 or kwargs:
            if hasattr(self, '_segmentation_data'):
                log_array_info(self._segmentation_data, "Initial segmentation_data")

        result = func(self, *args, **kwargs)

        # Only log final state for setter operations
        if len(args) > 0 or kwargs:
            if hasattr(self, '_segmentation_data'):
                log_array_info(self._segmentation_data, "Final segmentation_data")
            logger.debug(f"DataManager: Exiting {func.__name__}")

        return result

    return wrapper

def log_visualization_ops(func: Callable) -> Callable:
    """Decorator for logging VisualizationManager operations"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logger.debug(f"VisualizationManager: Entering {func.__name__}")

        # Log visualization state
        if self.tracking_layer is not None:
            log_array_info(self.tracking_layer.data, "Initial tracking_layer.data")

        result = func(self, *args, **kwargs)

        # Log final state
        if self.tracking_layer is not None:
            log_array_info(self.tracking_layer.data, "Final tracking_layer.data")

        logger.debug(f"VisualizationManager: Exiting {func.__name__}")
        return result

    return wrapper