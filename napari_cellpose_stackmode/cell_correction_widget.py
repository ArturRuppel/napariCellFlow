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
from .debug_logging import log_state_changes, log_array_info, logger

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

        # Initialize masks layer from visualization manager if available
        self.masks_layer = None
        if self.vis_manager.tracking_layer is not None:
            self.masks_layer = self.vis_manager.tracking_layer
            print(f"Got masks layer from visualization manager")
        elif hasattr(self.data_manager, 'segmentation_data') and self.data_manager.segmentation_data is not None:
            self.set_masks_layer(self.data_manager.segmentation_data)
            print(f"Initialized masks layer from data manager")

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

    def _on_mouse_drag(self, viewer, event):
        """Handle mouse drag events"""
        print(f"Mouse drag event: button={event.button}, is_drawing={self.is_drawing}")  # Debug

        # Try to get masks layer if not set
        if self.masks_layer is None and self.vis_manager.tracking_layer is not None:
            print("Retrieving masks layer from visualization manager")  # Debug
            self.masks_layer = self.vis_manager.tracking_layer

        # Try to initialize from data manager if still no layer
        if self.masks_layer is None and self.data_manager.segmentation_data is not None:
            print("Initializing masks layer from data manager")  # Debug
            self.set_masks_layer(self.data_manager.segmentation_data)

        if self.masks_layer is None:
            print("No masks layer available - cannot process mouse events")  # Debug
            return

        pos = viewer.cursor.position
        coords = np.round(pos).astype(int)[-2:]  # Take last 2 dimensions for y,x
        print(f"Mouse position: {coords}")  # Debug

        if event.button == Qt.RightButton:
            print(f"Right button, drawing_mode={self.is_drawing}")  # Debug
            if self.is_drawing:
                if not self.drawing_started:
                    print("Starting new drawing")  # Debug
                    self.drawing_started = True
                    self.start_point = coords
                    self.drawing_points = [coords]
                    self._update_drawing_preview()
                else:
                    print(f"Adding point: {coords}")  # Debug
                    self.drawing_points.append(coords)
                    self._update_drawing_preview()
            else:
                self._handle_selection(coords)
        elif event.button == Qt.LeftButton and self.is_drawing:
            print("Left button delete attempt")  # Debug
            self._delete_cell_at_position(coords)
    def _update_drawing_preview(self):
        """Update the preview of the cell being drawn."""
        print(f"Updating preview with {len(self.drawing_points)} points")  # Debug

        if not self.drawing_points or len(self.drawing_points) < 2:
            return

        if self.drawing_layer is None:
            print("Creating new drawing layer")  # Debug
            empty_data = np.zeros_like(self.masks_layer.data)
            self.drawing_layer = self.viewer.add_labels(
                empty_data,
                name='Drawing Preview',
                opacity=0.8
            )

        mask = self._create_empty_mask()
        if mask is None:
            print("Failed to create empty mask")  # Debug
            return

        # Draw the path with thicker lines
        points = np.array(self.drawing_points)
        for i in range(len(points) - 1):
            cv2.line(mask,
                     (points[i][1], points[i][0]),
                     (points[i + 1][1], points[i + 1][0]),
                     self.LINE_COLOR,
                     self.LINE_THICKNESS)

        # Draw start point circle
        if self.start_point is not None:
            cv2.circle(mask,
                       (self.start_point[1], self.start_point[0]),
                       self.START_POINT_RADIUS,
                       self.LINE_COLOR,
                       -1)  # Filled circle

        print(f"Updating drawing layer with shape {mask.shape}")  # Debug

        if len(self.masks_layer.data.shape) == 3:
            current_frame = int(self.viewer.dims.point[0])
            new_data = self.drawing_layer.data.copy()
            new_data[current_frame] = mask
            self.drawing_layer.data = new_data
        else:
            self.drawing_layer.data = mask

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

    def _update_drawing_state(self):
        """Update the drawing state based on toggle button and ctrl key"""
        # Determine if we should be in drawing mode
        new_drawing_state = self.toggle_state or self.ctrl_pressed
        print(f"Drawing state changed: {new_drawing_state}")  # Debug print

        # Only update if the state is actually changing
        if new_drawing_state != self.is_drawing:
            self.is_drawing = new_drawing_state
            self._update_ui_state()

            # Clear drawing if we're switching modes and not mid-drawing
            if not self.drawing_started:
                self._clear_drawing()

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
    def _on_mouse_wheel(self, viewer, event):
        """Handle mouse wheel events for slice navigation."""
        if hasattr(self, '_full_masks') and self._full_masks is not None:
            try:
                self._updating = True
                current_slice = int(self.viewer.dims.point[0])
                logger.debug(f"Mouse wheel event, current slice: {current_slice}")

                if self.masks_layer is not None:
                    logger.debug(f"Before update, masks layer data shape: {self.masks_layer.data.shape}")
                    logger.debug(f"Unique values in masks layer data before update: {np.unique(self.masks_layer.data)}")

                    # Update just the current slice data while maintaining dimensionality
                    with self.viewer.events.blocker_all():
                        if self._full_masks.ndim == 3:
                            current_data = self.masks_layer.data
                            current_data[current_slice] = self._full_masks[current_slice].copy()
                            self.masks_layer.refresh()
                        else:
                            self.masks_layer.data = self._full_masks

                    logger.debug(f"After update, masks layer data shape: {self.masks_layer.data.shape}")
                    logger.debug(f"Unique values in masks layer data after update: {np.unique(self.masks_layer.data)}")

                    self._clear_drawing()

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

    @log_state_changes
    def set_masks_layer(self, masks: np.ndarray):
        """Set or update the masks layer."""
        if self._updating:
            logger.debug("Set masks cancelled - updating in progress")
            return

        try:
            self._updating = True
            logger.debug(f"Setting masks layer with shape {masks.shape}")

            # First store the full masks
            self._full_masks = masks.copy()

            # If layer doesn't exist, initialize it
            if self.vis_manager.tracking_layer is None:
                if masks.ndim == 3:
                    self.masks_layer = self.viewer.add_labels(
                        masks,
                        name='Cell Tracking',
                        opacity=0.5
                    )
                else:
                    self.masks_layer = self.viewer.add_labels(
                        masks[np.newaxis, ...] if masks.ndim == 2 else masks,
                        name='Cell Tracking',
                        opacity=0.5
                    )
                self.vis_manager.tracking_layer = self.masks_layer
            else:
                # Use existing layer
                self.masks_layer = self.vis_manager.tracking_layer

                # Update the data in a way that preserves dimensionality
                if masks.ndim == 2:
                    masks = masks[np.newaxis, ...]

                # Ensure the layer's data maintains proper dimensionality
                current_data = self.masks_layer.data
                if current_data.ndim != masks.ndim:
                    if current_data.ndim > masks.ndim:
                        masks = masks[np.newaxis, ...]
                    else:
                        current_data = current_data[np.newaxis, ...]

                # Update the layer data
                self.masks_layer.data = masks

            self.next_cell_id = masks.max() + 1

        except Exception as e:
            logger.error(f"Error setting masks layer: {str(e)}")
            raise
        finally:
            self._updating = False

    @log_state_changes
    def _finish_drawing(self):
        """Complete the cell drawing process."""
        if not self.drawing_points or len(self.drawing_points) < 3:
            logger.debug("Drawing cancelled - insufficient points")
            self._clear_drawing()
            return

        try:
            if self._updating:
                logger.debug("Drawing cancelled - updating in progress")
                return

            self._updating = True
            current_frame = int(self.viewer.dims.point[0])
            logger.debug(f"Finishing drawing in frame {current_frame}")

            # Store state before adding new cell
            self._store_undo_state(f"Add cell {self.next_cell_id}")

            # Close the contour
            self.drawing_points.append(self.start_point)
            points = np.array(self.drawing_points)

            # Create mask for the new cell
            current_mask = self._get_current_frame_mask()
            if current_mask is None:
                return

            new_cell_mask = np.zeros_like(current_mask, dtype=np.uint8)
            cv2.fillPoly(new_cell_mask, [points[:, ::-1]], 1)

            # Create a mask of empty areas
            empty_mask = (current_mask == 0)

            # Assign a new unique ID to the new cell
            new_cell_id = self._get_next_cell_id()

            # Apply the new cell mask only to empty areas
            current_mask[np.logical_and(new_cell_mask > 0, empty_mask)] = new_cell_id

            # Get current frame index
            current_frame = int(self.viewer.dims.point[0])

            # Update the appropriate slice
            if len(self.masks_layer.data.shape) == 3:  # 3D data
                new_mask = self.masks_layer.data.copy()
                new_mask[current_frame] = current_mask
                self.masks_layer.data = new_mask
                if hasattr(self, '_full_masks'):
                    self._full_masks[current_frame] = current_mask
            else:  # 2D data
                self.masks_layer.data = current_mask

            # Update data manager
            self.data_manager.segmentation_data = self.masks_layer.data

            # Ensure visualization is updated
            if len(self.masks_layer.data.shape) == 3:
                self.vis_manager.update_tracking_visualization(
                    (self.masks_layer.data[current_frame], current_frame)
                )
            else:
                self.vis_manager.update_tracking_visualization(self.masks_layer.data)

            # Update status
            self.status_label.setText(f"Added new cell {new_cell_id}")

        except Exception as e:
            logger.error(f"Error finishing drawing: {e}")
            raise
        finally:
            self._updating = False
            self.drawing_started = False
            self.start_point = None
            self._clear_drawing()

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

    def eventFilter(self, watched_object, event):
        """Global event filter to catch key events regardless of focus"""
        if event.type() == event.KeyPress:
            if event.key() == Qt.Key_Control:
                self.ctrl_pressed = True
                print(f"Ctrl pressed, ctrl_pressed={self.ctrl_pressed}")  # Debug
                self._update_drawing_state()
                return False  # Don't consume the event
        elif event.type() == event.KeyRelease:
            if event.key() == Qt.Key_Control:
                self.ctrl_pressed = False
                print(f"Ctrl released, ctrl_pressed={self.ctrl_pressed}")  # Debug
                self._update_drawing_state()
                return False  # Don't consume the event
        return super().eventFilter(watched_object, event)

    def cleanup(self):
        """Clean up resources and disconnect events"""
        # Remove application-wide event filter
        QApplication.instance().removeEventFilter(self)

        # Remove mouse callbacks
        if self._on_mouse_drag in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.remove(self._on_mouse_drag)
        if self._on_mouse_move in self.viewer.mouse_move_callbacks:
            self.viewer.mouse_move_callbacks.remove(self._on_mouse_move)
        if self._on_mouse_wheel in self.viewer.mouse_wheel_callbacks:
            self.viewer.mouse_wheel_callbacks.remove(self._on_mouse_wheel)

        self._clear_drawing()

    def _update_ui_state(self):
        """Update UI elements based on current state"""
        # Update status label
        if self.is_drawing:
            self.status_label.setText("Drawing Mode: Right-click and drag to draw cell contour")
        else:
            self.status_label.setText("Selection Mode: Right-click to select cells")

        # Update button state without triggering the toggle signal
        self.toggle_draw_btn.blockSignals(True)
        self.toggle_draw_btn.setChecked(self.is_drawing)
        self.toggle_draw_btn.blockSignals(False)

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


    def _validate_coords(self, coords):
        """Validate coordinates are within image bounds."""
        if self.masks_layer is None:
            return False
        shape = self.masks_layer.data.shape[-2:]  # Get y,x dimensions
        return (0 <= coords[0] < shape[0] and
                0 <= coords[1] < shape[1])

    def _clear_drawing(self):
        """Clear current drawing state."""
        self.drawing_points = []
        if self.drawing_layer is not None:
            try:
                self.viewer.layers.remove(self.drawing_layer)
            except ValueError:
                pass  # Layer already removed
            self.drawing_layer = None

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

