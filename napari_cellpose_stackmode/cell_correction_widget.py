import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QApplication,
    QHBoxLayout, QShortcut
)
from qtpy.QtGui import QKeySequence
import napari
from napari.layers import Labels
import logging
import cv2
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union, Dict
from .debug_logging import logger, log_operation

logger = logging.getLogger(__name__)


@dataclass
class UndoAction:
    """Represents a single undoable action"""
    frame: Optional[int]  # Frame number for 3D stacks, None for 2D
    previous_state: np.ndarray  # Previous state of the masks
    description: str  # Description of the action


class CellCorrectionWidget(QWidget):
    """Widget for interactive cell correction in napari."""

    correction_made = Signal(np.ndarray)  # Emitted when masks are modified
    LINE_THICKNESS = 2
    START_POINT_RADIUS = 5
    LINE_COLOR = 3
    MAX_UNDO_STEPS = 20

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager",
                 visualization_manager: "VisualizationManager", parent: QWidget = None):
        super().__init__(parent)
        self.viewer = viewer
        self.data_manager = data_manager
        self.vis_manager = visualization_manager

        # Add flags for state management
        self._updating = False
        self._loading_external = False

        # Initialize masks layer
        self.masks_layer = None
        if self.vis_manager.tracking_layer is not None:
            self.masks_layer = self.vis_manager.tracking_layer
        elif hasattr(self.data_manager, 'segmentation_data') and self.data_manager.segmentation_data is not None:
            self.set_masks_layer(self.data_manager.segmentation_data)

        # Connect to viewer layer events
        self.viewer.layers.events.inserted.connect(self._handle_layer_added)
        self.viewer.layers.events.removed.connect(self._handle_layer_removal)

        self.drawing_layer = None
        self.current_frame = 0
        self.is_drawing = False
        self.drawing_points = []
        self.next_cell_id = 1
        self.selected_cell = None
        self.drawing_started = False
        self.start_point = None
        self.CLOSURE_THRESHOLD = 10
        self.MIN_DRAWING_POINTS = 20
        self.ctrl_pressed = False
        self.toggle_state = False
        self._updating = False

        # Undo history
        self.undo_stack = deque(maxlen=self.MAX_UNDO_STEPS)

        self._setup_ui()
        self._connect_events()
        self._setup_shortcuts()
        QApplication.instance().installEventFilter(self)

    # In cell_correction_widget.py

    def eventFilter(self, watched_object, event):
        """Global event filter to catch key events regardless of focus"""
        try:
            if event.type() == event.KeyPress:
                if event.key() == Qt.Key_Control:
                    logger.debug("CellCorrection: Ctrl key pressed")
                    self.ctrl_pressed = True
                    self._update_drawing_state()
                    return False  # Don't consume the event
            elif event.type() == event.KeyRelease:
                if event.key() == Qt.Key_Control:
                    logger.debug("CellCorrection: Ctrl key released")
                    logger.debug(f"CellCorrection: Current drawing state - is_drawing: {self.is_drawing}, drawing_started: {self.drawing_started}")

                    # Only handle Ctrl release if we're not in the middle of drawing
                    if not self.drawing_started:
                        self.ctrl_pressed = False
                        self._update_drawing_state()
                    else:
                        logger.debug("CellCorrection: Ignoring Ctrl release while drawing active")

                    return False  # Don't consume the event
            return super().eventFilter(watched_object, event)
        except Exception as e:
            logger.error(f"CellCorrection: Error in event filter: {e}", exc_info=True)
            return False

    def _update_drawing_state(self):
        """Update drawing state with proper synchronization."""
        try:
            logger.debug("CellCorrection: Updating drawing state")
            logger.debug(f"CellCorrection: Current state - toggle_state: {self.toggle_state}, ctrl_pressed: {self.ctrl_pressed}")

            # Calculate new drawing state
            new_drawing_state = self.toggle_state or self.ctrl_pressed

            # Only update if state actually changes
            if new_drawing_state != self.is_drawing:
                logger.debug(f"CellCorrection: Drawing state changing from {self.is_drawing} to {new_drawing_state}")
                self.is_drawing = new_drawing_state

                # Only update UI if we're not actively drawing
                if not self.drawing_started:
                    with self.viewer.events.blocker_all():
                        self._update_ui_state()
                        if not new_drawing_state:  # Only clear when exiting drawing mode
                            self._clear_drawing()

            # Single refresh at the end if needed
            if self.masks_layer is not None and self.masks_layer in self.viewer.layers:
                logger.debug("CellCorrection: Refreshing masks layer")
                self.masks_layer.refresh()

        except Exception as e:
            logger.error(f"CellCorrection: Error updating drawing state: {e}", exc_info=True)

    def _clear_drawing(self):
        try:
            logger.debug("CellCorrection: Starting drawing clear")

            # Only remove drawing preview layer
            if self.drawing_layer is not None and self.drawing_layer in self.viewer.layers:
                with self.viewer.events.blocker_all():
                    self.viewer.layers.remove(self.drawing_layer)
                self.drawing_layer = None

            # Ensure we maintain reference to the correct segmentation layer
            if self.vis_manager.tracking_layer is not None:
                self.masks_layer = self.vis_manager.tracking_layer

            if self.masks_layer is not None and self.masks_layer in self.viewer.layers:
                self.masks_layer.visible = True

        except Exception as e:
            logger.error(f"CellCorrection: Error clearing drawing: {e}")

    def _update_ui_state(self):
        """Update UI elements based on current state"""
        try:
            # Update status label based on mode
            if self.is_drawing:
                self.status_label.setText("Drawing Mode: Right-click and drag to draw cell contour")
            else:
                self.status_label.setText("Selection Mode: Right-click to select cells")

            # Update button state
            self.toggle_draw_btn.blockSignals(True)
            self.toggle_draw_btn.setChecked(self.is_drawing)
            self.toggle_draw_btn.blockSignals(False)

        except Exception as e:
            logger.error(f"CellCorrection: Error updating UI state: {e}")


    def _on_mouse_drag(self, viewer, event):
        """Handle mouse drag events"""
        try:
            if self.masks_layer is None and self.vis_manager.tracking_layer is not None:
                logger.debug("CellCorrection: Recovering masks layer from visualization manager")
                self.masks_layer = self.vis_manager.tracking_layer

            if self.masks_layer is None:
                logger.debug("CellCorrection: No masks layer available")
                return

            pos = viewer.cursor.position
            coords = np.round(pos).astype(int)[-2:]

            if event.button == Qt.RightButton:
                if self.is_drawing:
                    if not self.drawing_started:
                        logger.debug("CellCorrection: Starting new drawing")
                        self.drawing_started = True
                        self.start_point = coords
                        self.drawing_points = [coords]
                        self._update_drawing_preview()
                    else:
                        logger.debug("CellCorrection: Adding point to drawing")
                        self.drawing_points.append(coords)
                        self._update_drawing_preview()
                else:
                    self._handle_selection(coords)
            elif event.button == Qt.LeftButton and self.is_drawing:
                self._delete_cell_at_position(coords)

        except Exception as e:
            logger.error(f"CellCorrection: Error in mouse drag: {e}", exc_info=True)

    def _initialize_or_recover_layer(self):
        """Initialize or recover the masks layer."""
        try:
            if self.masks_layer is None or self.masks_layer not in self.viewer.layers:
                logger.debug("CellCorrection: Attempting to recover masks layer")
                if self.vis_manager.tracking_layer is not None:
                    self.masks_layer = self.vis_manager.tracking_layer
                    logger.debug("CellCorrection: Recovered masks layer from visualization manager")
                else:
                    logger.debug("CellCorrection: No tracking layer available")
        except Exception as e:
            logger.error(f"CellCorrection: Error initializing/recovering layer: {e}")

    def _update_drawing_preview(self):
        """Update the preview of the cell being drawn."""
        if not self.drawing_points or len(self.drawing_points) < 2:
            return

        try:
            # Ensure we have a valid masks layer
            if self.masks_layer is None or self.masks_layer not in self.viewer.layers:
                if self.vis_manager.tracking_layer is not None:
                    self.masks_layer = self.vis_manager.tracking_layer
                else:
                    return

            # Get current frame and shape
            current_frame = int(self.viewer.dims.point[0])
            shape = (self.masks_layer.data.shape[1:]
                     if len(self.masks_layer.data.shape) == 3
                     else self.masks_layer.data.shape)

            # Create or update drawing layer
            if self.drawing_layer is None or self.drawing_layer not in self.viewer.layers:
                preview_shape = (self.masks_layer.data.shape
                                 if len(self.masks_layer.data.shape) == 3
                                 else shape)
                preview_data = np.zeros(preview_shape, dtype=np.uint8)

                self.drawing_layer = self.viewer.add_labels(
                    preview_data,
                    name='Drawing Preview',
                    opacity=0.8,
                    visible=True
                )

            # Keep both layers visible
            self.drawing_layer.visible = True
            self.masks_layer.visible = True

            # Create mask for current frame
            if len(self.masks_layer.data.shape) == 3:
                preview_data = self.drawing_layer.data.copy()
                mask = np.zeros(shape, dtype=np.uint8)
            else:
                preview_data = np.zeros_like(self.masks_layer.data)
                mask = preview_data

            # Draw the preview
            points = np.array(self.drawing_points)
            for i in range(len(points) - 1):
                p1 = (int(points[i][1]), int(points[i][0]))
                p2 = (int(points[i + 1][1]), int(points[i + 1][0]))
                cv2.line(mask, p1, p2, self.LINE_COLOR, self.LINE_THICKNESS)

            # Draw start point circle
            if self.start_point is not None:
                center = (int(self.start_point[1]), int(self.start_point[0]))
                cv2.circle(mask, center, self.START_POINT_RADIUS, self.LINE_COLOR, -1)

            # Update the preview layer data
            if len(self.masks_layer.data.shape) == 3:
                preview_data[current_frame] = mask
                self.drawing_layer.data = preview_data
            else:
                self.drawing_layer.data = mask

        except Exception as e:
            logger.error(f"Error updating drawing preview: {e}")
            self._clear_drawing()

    def _finish_drawing(self):
        if not self.drawing_points or len(self.drawing_points) < 3:
            logger.debug("CellCorrection: Not enough points to complete drawing")
            self._clear_drawing()
            return

        try:
            if self._updating:
                return

            self._updating = True
            current_frame = int(self.viewer.dims.point[0])
            logger.debug(f"CellCorrection: Finishing drawing on frame {current_frame}")

            # Get existing data from visualization manager first
            if self.vis_manager.tracking_layer is None:
                raise ValueError("No tracking layer available")

            # Get a deep copy of the full stack
            full_stack = self.vis_manager.tracking_layer.data.copy()
            logger.debug(f"Full stack unique values before: {np.unique(full_stack)}")

            # Calculate next cell ID based on the maximum value in the entire stack
            self.next_cell_id = int(full_stack.max()) + 1
            logger.debug(f"Using cell ID: {self.next_cell_id}")

            # Create new cell mask
            frame_shape = full_stack.shape[1:]
            new_cell_mask = np.zeros(frame_shape, dtype=np.uint8)
            points = np.array(self.drawing_points + [self.start_point])
            points = np.clip(points, 0, np.array(frame_shape) - 1).astype(np.int32)
            cv2.fillPoly(new_cell_mask, [points[:, ::-1]], 1)

            # Get the current frame data
            current_frame_data = full_stack[current_frame].copy()

            # Only update empty areas
            empty_mask = (current_frame_data == 0)
            add_mask = np.logical_and(new_cell_mask > 0, empty_mask)

            # Update only the new cell area in the current frame
            current_frame_data[add_mask] = self.next_cell_id

            # Put the updated frame back into the stack
            full_stack[current_frame] = current_frame_data

            logger.debug(f"Full stack unique values after: {np.unique(full_stack)}")

            # Update through visualization manager
            self.vis_manager.update_tracking_visualization(full_stack)

            # Update data manager
            self.data_manager.segmentation_data = full_stack

            # Update UI state
            self.status_label.setText(f"Added new cell {self.next_cell_id}")

            # Increment next_cell_id for the next drawing
            self.next_cell_id += 1

            # Verify data consistency
            if self.vis_manager.tracking_layer is not None:
                final_data = self.vis_manager.tracking_layer.data
                logger.debug(f"Final layer unique values: {np.unique(final_data)}")

        except Exception as e:
            logger.error(f"CellCorrection: Error finishing drawing: {e}", exc_info=True)
            raise
        finally:
            self._updating = False
            self.drawing_started = False
            self.start_point = None
            self._clear_drawing()
    @log_operation
    def _handle_layer_added(self, event):
        """Handle when a new layer is added to the viewer"""
        if self._updating:
            logger.debug("Skipping layer addition - updating in progress")
            return

        layer = event.value
        logger.debug(f"New layer added: {layer.name}, type: {type(layer)}")

        if isinstance(layer, napari.layers.Labels):
            try:
                self._updating = True
                logger.debug("Processing new Labels layer")

                # Initialize with the new layer
                self.masks_layer = layer
                if layer.name != 'Drawing Preview':
                    layer.name = 'Segmentation'
                    logger.debug("Renamed layer to Segmentation")

                # Store full data
                data = layer.data
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                logger.debug(f"Layer data shape: {data.shape}")
                logger.debug(f"Unique values in data: {np.unique(data)}")

                # Update data manager
                if not self.data_manager._initialized:
                    logger.debug("Initializing data manager")
                    self.data_manager.initialize_stack(data.shape[0])
                self.data_manager.segmentation_data = data

                # Update visualization manager
                if layer.name != 'Drawing Preview':
                    logger.debug("Setting tracking layer in visualization manager")
                    self.vis_manager.tracking_layer = layer

                # Update next cell ID
                self.next_cell_id = int(data.max()) + 1
                logger.debug(f"Set next cell ID to: {self.next_cell_id}")

            except Exception as e:
                logger.error(f"Failed to initialize with external layer: {e}", exc_info=True)
            finally:
                self._updating = False
                logger.debug("Layer initialization complete")

    def _on_mouse_wheel(self, viewer, event):
        """Handle mouse wheel events with proper state synchronization."""
        if self._updating:
            return

        try:
            self._updating = True
            current_frame = int(self.viewer.dims.point[0])

            # Clear any active drawing
            self._clear_drawing()

            # Update masks layer if needed
            if self.masks_layer is not None and self.masks_layer in self.viewer.layers:
                # Ensure proper data shape
                if len(self.masks_layer.data.shape) == 3:
                    self.masks_layer.refresh()
                    # Update visualization manager
                    if self.vis_manager.tracking_layer is not None:
                        self.vis_manager.tracking_layer.refresh()

        except Exception as e:
            logger.error(f"Error during mouse wheel: {e}")
        finally:
            self._updating = False

    def _validate_coords(self, coords):
        """Validate coordinates are within image bounds."""
        if self.masks_layer is None:
            return False

        try:
            # Get current frame shape
            if len(self.masks_layer.data.shape) == 3:
                shape = self.masks_layer.data.shape[1:]
            else:
                shape = self.masks_layer.data.shape

            # Validate coordinates
            return (0 <= coords[0] < shape[0] and
                    0 <= coords[1] < shape[1])

        except Exception as e:
            logger.error(f"Error validating coordinates: {e}")
            return False

    def _handle_layer_removal(self, event):
        """Handle layer removal events"""
        if event.value == self.masks_layer:
            self.masks_layer = None
            self._full_masks = None

    def _validate_loaded_segmentation(self, data: np.ndarray) -> bool:
        """Validate loaded segmentation data"""
        try:
            # Check dimensions
            if data.ndim not in [2, 3]:
                raise ValueError(f"Invalid segmentation dimensions: {data.ndim}")

            # Check data type
            if not np.issubdtype(data.dtype, np.integer):
                raise ValueError(f"Segmentation must contain integer labels, got {data.dtype}")

            # Check value range
            if data.min() < 0:
                raise ValueError("Segmentation contains negative values")

            return True

        except Exception as e:
            logger.error(f"Segmentation validation failed: {e}")
            return False

    def _init_with_external_data(self, data: np.ndarray):
        """Initialize widget with externally loaded data"""
        if self._updating:
            return

        try:
            self._updating = True

            # Ensure proper dimensionality
            if data.ndim == 2:
                data = data[np.newaxis, ...]

            # Initialize data manager
            if not self.data_manager._initialized:
                self.data_manager.initialize_stack(data.shape[0])
            self.data_manager.segmentation_data = data

            # Update visualization
            self.vis_manager.update_tracking_visualization(data)

            # Update internal references
            self.masks_layer = self.vis_manager.tracking_layer
            self._full_masks = data.copy()
            self.next_cell_id = int(data.max()) + 1

        except Exception as e:
            logger.error(f"Failed to initialize with external data: {e}")
            raise
        finally:
            self._updating = False

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Control buttons in a horizontal layout
        buttons_layout = QHBoxLayout()

        # Drawing mode toggle
        self.toggle_draw_btn = QPushButton("Toggle Drawing Mode (or hold Ctrl)")
        self.toggle_draw_btn.setCheckable(True)
        buttons_layout.addWidget(self.toggle_draw_btn)

        # Undo button
        self.undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.undo_btn.setEnabled(False)
        buttons_layout.addWidget(self.undo_btn)

        layout.addLayout(buttons_layout)

        # Updated instructions
        instructions = QLabel(
            "Drawing Mode:\n"
            "- Hold Ctrl + Right click and drag to draw cell boundary\n"
            "- Return to start point (marked with circle) to complete the cell\n"
            "- Left click to delete cells\n"
            "- Ctrl+Z to undo last action\n\n"
            "Selection Mode:\n"
            "- Right-click: Select cell\n"
            "- Delete: Remove selected cell"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

    def _store_undo_state(self, description: str):
        """Store current state for undo"""
        if self._updating:
            return

        try:
            if self.masks_layer is None:
                return

            if hasattr(self, '_full_masks') and self._full_masks is not None:
                current_frame = int(self.viewer.dims.point[0])
                # Store only the current frame's state for 3D data
                self.undo_stack.append(UndoAction(
                    frame=current_frame,
                    previous_state=self._full_masks[current_frame].copy(),
                    description=description
                ))
            else:
                # Store the entire 2D mask
                self.undo_stack.append(UndoAction(
                    frame=None,
                    previous_state=self.masks_layer.data.copy(),
                    description=description
                ))

            self.undo_btn.setEnabled(True)
            logger.debug(f"Stored undo state: {description}")

        except Exception as e:
            logger.error(f"Error storing undo state: {e}")

    def _connect_events(self):
        """Connect all event handlers"""
        # Remove any existing callbacks first
        if self._on_mouse_drag in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.remove(self._on_mouse_drag)
        if self._on_mouse_move in self.viewer.mouse_move_callbacks:
            self.viewer.mouse_move_callbacks.remove(self._on_mouse_move)
        if self._on_mouse_wheel in self.viewer.mouse_wheel_callbacks:
            self.viewer.mouse_wheel_callbacks.remove(self._on_mouse_wheel)

        # Add mouse callbacks explicitly to viewer
        self.viewer.mouse_drag_callbacks.append(self._on_mouse_drag)
        self.viewer.mouse_move_callbacks.append(self._on_mouse_move)
        self.viewer.mouse_wheel_callbacks.append(self._on_mouse_wheel)

        print("Mouse callbacks connected")  # Debug

        # Bind keyboard events
        self.viewer.bind_key('Delete', self._on_delete_pressed)

        # Add toggle button connection
        self.toggle_draw_btn.toggled.connect(self._handle_toggle_button)

        # Add undo button connection
        self.undo_btn.clicked.connect(self.undo_last_action)

    def _handle_toggle_button(self, checked):
        """Handle toggle button state changes"""
        self.toggle_state = checked
        self._update_drawing_state()

    def _ensure_layer_sync(self):
        """Ensure synchronization between layers and managers."""
        try:
            if self.masks_layer is None or self.masks_layer not in self.viewer.layers:
                # Try to get from visualization manager
                if self.vis_manager.tracking_layer is not None:
                    self.masks_layer = self.vis_manager.tracking_layer
                    return True

            # Update visualization manager reference
            if self.masks_layer is not None:
                self.vis_manager.tracking_layer = self.masks_layer

                # Ensure data manager is synchronized
                if self.data_manager.segmentation_data is None:
                    self.data_manager.segmentation_data = self.masks_layer.data.copy()
                elif not np.array_equal(self.data_manager.segmentation_data, self.masks_layer.data):
                    current_frame = int(self.viewer.dims.point[0])
                    self.data_manager.segmentation_data = (self.masks_layer.data[current_frame], current_frame)

            return self.masks_layer is not None and self.masks_layer in self.viewer.layers

        except Exception as e:
            logger.error(f"Error ensuring layer sync: {e}")
            return False

    def undo_last_action(self):
        """Undo the last action"""
        if not self.undo_stack:
            return

        try:
            self._updating = True
            action = self.undo_stack.pop()

            if hasattr(self, '_full_masks') and self._full_masks is not None:
                if action.frame is not None:
                    self._full_masks[action.frame] = action.previous_state
                    self.masks_layer.data = action.previous_state
                    self.correction_made.emit(self._full_masks)
            else:
                self.masks_layer.data = action.previous_state
                self.correction_made.emit(action.previous_state)

            self.status_label.setText(f"Undid: {action.description}")
            logger.info(f"Undid action: {action.description}")

            # Update undo button state
            self.undo_btn.setEnabled(len(self.undo_stack) > 0)

        except Exception as e:
            logger.error(f"Error during undo: {e}")
            raise
        finally:
            self._updating = False
            self._clear_drawing()

    def _delete_cell_at_position(self, coords):
        """Delete cell at the given coordinates."""
        if self._updating or not self._validate_coords(coords):
            logger.debug("Deletion skipped - updating flag or invalid coords")
            return

        current_mask = self._get_current_frame_mask()
        if current_mask is None:
            logger.debug("Deletion skipped - no current mask")
            return

        cell_id = current_mask[coords[0], coords[1]]
        if cell_id > 0:
            try:
                self._store_undo_state(f"Delete cell {cell_id}")
                self._updating = True

                # Create a copy of just the current frame
                new_frame = current_mask.copy()
                new_frame[new_frame == cell_id] = 0

                # Get current frame index
                current_slice = int(self.viewer.dims.point[0])

                # Update only the current frame in data manager
                self.data_manager.segmentation_data = (new_frame, current_slice)

                # Update visualization for single frame
                self.vis_manager.update_tracking_visualization(
                    (new_frame, current_slice)
                )

            finally:
                self._updating = False

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts"""
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_last_action)

    def _on_correction_made(self, updated_masks: np.ndarray):
        """Handle corrections without triggering cascading updates."""
        if self._updating:
            return

        try:
            self._updating = True

            logger.debug(f"Updated masks shape: {updated_masks.shape}")
            logger.debug(f"Unique values in updated masks: {np.unique(updated_masks)}")

            # Block napari events during update
            with self.viewer.events.blocker_all():
                # Update data manager
                self.data_manager.segmentation_data = updated_masks

                # Update visualization for current frame only
                if hasattr(self, '_full_masks') and self._full_masks is not None:
                    current_slice = int(self.viewer.dims.point[0])
                    self.masks_layer.data = updated_masks[current_slice]
                else:
                    self.masks_layer.data = updated_masks

                # Update status
                num_cells = len(np.unique(updated_masks)) - 1
                self.status_label.setText(f"Correction applied. Current cell count: {num_cells}")

        except Exception as e:
            logger.error(f"Error applying correction: {str(e)}")
        finally:
            self._updating = False

    def set_masks_layer(self, masks: np.ndarray):
        """Set or update the masks layer."""
        if self._updating:
            return

        try:
            self._updating = True
            logger.debug(f"Setting masks layer with shape {masks.shape}")

            # Ensure proper dimensionality
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]

            # Update through visualization manager first
            if self.vis_manager is not None:
                self.vis_manager.update_tracking_visualization(masks)
                self.masks_layer = self.vis_manager.tracking_layer
            else:
                # Create new layer if needed
                if self.masks_layer is None or self.masks_layer not in self.viewer.layers:
                    self.masks_layer = self.viewer.add_labels(
                        masks,
                        name='Segmentation',
                        opacity=0.5
                    )
                else:
                    self.masks_layer.data = masks

            # Update visualization manager reference
            if self.vis_manager is not None:
                self.vis_manager.tracking_layer = self.masks_layer

            # Set next cell ID
            self.next_cell_id = int(masks.max()) + 1

        except Exception as e:
            logger.error(f"Error setting masks layer: {str(e)}")
            raise
        finally:
            self._updating = False

    def _get_next_cell_id(self):
        """Get the next available unique cell ID."""
        if self.masks_layer is None:
            return 1

        max_id = int(self.masks_layer.data.max())
        return max_id + 1

    def _get_current_frame_mask(self):
        """Get mask for the current frame."""
        if self.masks_layer is None:
            return None

        if len(self.masks_layer.data.shape) == 3:
            # For 3D data, get current frame
            current_frame = int(self.viewer.dims.point[0])
            return self.masks_layer.data[current_frame]
        else:
            # For 2D data, return as is
            return self.masks_layer.data

    def _create_empty_mask(self):
        """Create an empty mask matching the current frame dimensions."""
        if self.masks_layer is None:
            return None

        if len(self.masks_layer.data.shape) == 3:
            # For 3D data, create empty frame
            shape = self.masks_layer.data.shape[1:]  # Get y,x dimensions
            return np.zeros(shape, dtype=self.masks_layer.data.dtype)
        else:
            # For 2D data
            return np.zeros_like(self.masks_layer.data)

    def _handle_selection(self, coords):
        """Handle cell selection."""
        if not self._validate_coords(coords):
            return

        current_mask = self._get_current_frame_mask()
        if current_mask is None:
            return

        cell_id = current_mask[coords[0], coords[1]]
        if cell_id > 0:
            self.selected_cell = cell_id
            self.status_label.setText(f"Selected cell: {cell_id}")
            self._highlight_selected_cell()

    def _highlight_selected_cell(self):
        """Highlight the selected cell."""
        if self.selected_cell is None:
            return

        self._clear_drawing()

        # Create highlight mask
        if len(self.masks_layer.data.shape) == 3:
            current_frame = int(self.viewer.dims.point[0])
            highlight = np.zeros_like(self.masks_layer.data)
            highlight[current_frame][self.masks_layer.data[current_frame] == self.selected_cell] = 1
        else:
            highlight = np.zeros_like(self.masks_layer.data)
            highlight[self.masks_layer.data == self.selected_cell] = 1

        self.drawing_layer = self.viewer.add_labels(
            highlight,
            name='Selected Cell',
            opacity=0.6
        )

    def _on_delete_pressed(self, viewer):
        """Handle delete key press to remove selected cell."""
        if self.selected_cell is None or self.masks_layer is None:
            return

        try:
            if hasattr(self, '_full_masks') and self._full_masks is not None:
                current_slice = int(self.viewer.dims.point[0])
                new_mask = self._full_masks[current_slice].copy()
                new_mask[new_mask == self.selected_cell] = 0

                # Update both the full stack and current view
                self._full_masks[current_slice] = new_mask
                self.masks_layer.data = new_mask
                self.correction_made.emit(self._full_masks)
            else:
                new_mask = self.masks_layer.data.copy()
                new_mask[new_mask == self.selected_cell] = 0
                self.masks_layer.data = new_mask
                self.correction_made.emit(new_mask)

            self._clear_drawing()
            self.selected_cell = None
            self.status_label.setText(f"Deleted cell {self.selected_cell}")

        except Exception as e:
            logger.error(f"Error deleting cell: {e}")
            raise

    def cleanup(self):
        """Clean up resources and disconnect events"""
        try:
            # Remove application-wide event filter
            QApplication.instance().removeEventFilter(self)

            # Disconnect layer events
            self.viewer.layers.events.inserted.disconnect(self._handle_layer_added)
            self.viewer.layers.events.removed.disconnect(self._handle_layer_removal)

            # Remove mouse callbacks
            if hasattr(self.viewer, 'mouse_drag_callbacks'):
                if self._on_mouse_drag in self.viewer.mouse_drag_callbacks:
                    self.viewer.mouse_drag_callbacks.remove(self._on_mouse_drag)
            if hasattr(self.viewer, 'mouse_move_callbacks'):
                if self._on_mouse_move in self.viewer.mouse_move_callbacks:
                    self.viewer.mouse_move_callbacks.remove(self._on_mouse_move)
            if hasattr(self.viewer, 'mouse_wheel_callbacks'):
                if self._on_mouse_wheel in self.viewer.mouse_wheel_callbacks:
                    self.viewer.mouse_wheel_callbacks.remove(self._on_mouse_wheel)

            # Clear drawing state
            self._clear_drawing()

            # Clear references
            self.viewer = None
            self.masks_layer = None
            self._full_masks = None

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _on_mouse_move(self, viewer, event):
        """Handle mouse movement for drawing preview."""
        if not self.is_drawing or not self.drawing_started:
            return

        pos = self.viewer.cursor.position
        coords = np.round(pos).astype(int)[-2:]  # Take last 2 dimensions for y,x

        if self._validate_coords(coords):
            # Add point if it's different from the last point
            if not self.drawing_points or not np.array_equal(coords, self.drawing_points[-1]):
                self.drawing_points.append(coords)

                # Only check for closure if we have enough points
                if len(self.drawing_points) > self.MIN_DRAWING_POINTS:
                    dist_to_start = np.linalg.norm(coords - self.start_point)
                    if dist_to_start < self.CLOSURE_THRESHOLD:
                        self._finish_drawing()
                        return

                self._update_drawing_preview()

    def _handle_key_press(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Control:
            self.ctrl_pressed = True
            self._update_drawing_state()
        # Make sure to call the parent class's keyPressEvent
        self.viewer.window.qt_viewer.__class__.keyPressEvent(self.viewer.window.qt_viewer, event)

    def _handle_key_release(self, event):
        """Handle key release events."""
        if event.key() == Qt.Key_Control:
            self.ctrl_pressed = False
            self._update_drawing_state()
        # Make sure to call the parent class's keyReleaseEvent
        self.viewer.window.qt_viewer.__class__.keyReleaseEvent(self.viewer.window.qt_viewer, event)

    def _on_ctrl_pressed(self):
        """Handle Ctrl key press."""
        self.ctrl_pressed = True
        if not self.toggle_draw_btn.isChecked():
            self.is_drawing = True
            self.status_label.setText("Drawing Mode: Right-click and drag to draw cell contour")

    def _on_ctrl_released(self):
        """Handle Ctrl key release."""
        self.ctrl_pressed = False
        if not self.toggle_draw_btn.isChecked():
            self.is_drawing = False
            if not self.drawing_started:
                self.status_label.setText("Selection Mode: Right-click to select cells")

    def _toggle_drawing_mode(self, enabled: bool):
        """Toggle between drawing and selection modes."""
        self.is_drawing = enabled
        self.drawing_started = False
        self.start_point = None
        self._clear_drawing()

        if enabled:
            self.status_label.setText("Drawing Mode: Right-click and drag to draw cell contour")
        else:
            self.status_label.setText("Selection Mode: Right-click to select cells")


    def _handle_layer_removed(self, event):
        """Handle layer removal events."""
        try:
            layer = event.value
            logger.debug(f"CellCorrection: Layer removed: {layer.name}")

            if layer == self.masks_layer:
                logger.debug("CellCorrection: Masks layer was removed")
                # Only clear reference if it's actually gone from viewer
                if layer not in self.viewer.layers:
                    self.masks_layer = None
                    # Try to recover from visualization manager
                    if self.vis_manager.tracking_layer is not None:
                        logger.debug("CellCorrection: Recovering masks layer from visualization manager")
                        self.masks_layer = self.vis_manager.tracking_layer

            logger.debug(f"CellCorrection: After removal - masks_layer exists: {self.masks_layer is not None}")
            if self.masks_layer is not None:
                logger.debug(f"CellCorrection: After removal - masks_layer in viewer: {self.masks_layer in self.viewer.layers}")

        except Exception as e:
            logger.error(f"CellCorrection: Error handling layer removal: {e}")