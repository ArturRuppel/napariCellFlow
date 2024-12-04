import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QApplication
import napari
from napari.layers import Labels
from napari.utils.events import Event
import cv2


class CellCorrectionWidget(QWidget):
    """Widget for interactive cell correction in napari."""

    correction_made = Signal(np.ndarray)  # Emitted when masks are modified

    def __init__(self, viewer: "napari.Viewer", parent: QWidget = None):
        super().__init__(parent)
        self.viewer = viewer
        self.masks_layer = None
        self.drawing_layer = None
        self.current_frame = 0
        self.is_drawing = False
        self.drawing_points = []
        self.next_cell_id = 1
        self.selected_cell = None
        self.drawing_started = False
        self.start_point = None
        self.CLOSURE_THRESHOLD = 10  # pixels
        self.MIN_DRAWING_POINTS = 5  # Minimum points needed before checking for closure
        self.ctrl_pressed = False
        self.toggle_state = False

        self._setup_ui()
        self._connect_events()

        # Install event filter on the application instance to catch all key events
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, watched_object, event):
        """Global event filter to catch key events regardless of focus"""
        if event.type() == event.KeyPress:
            if event.key() == Qt.Key_Control:
                self.ctrl_pressed = True
                self._update_drawing_state()
                return False  # Don't consume the event
        elif event.type() == event.KeyRelease:
            if event.key() == Qt.Key_Control:
                self.ctrl_pressed = False
                self._update_drawing_state()
                return False  # Don't consume the event
        return super().eventFilter(watched_object, event)

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Control buttons
        self.toggle_draw_btn = QPushButton("Toggle Drawing Mode (or hold Ctrl)")
        self.toggle_draw_btn.setCheckable(True)
        layout.addWidget(self.toggle_draw_btn)

        # Instructions
        instructions = QLabel(
            "Drawing Mode:\n"
            "- Hold Ctrl + Right click and drag to draw cell boundary\n"
            "- Return to start point to complete the cell\n\n"
            "Selection Mode:\n"
            "- Right-click: Select cell\n"
            "- Delete: Remove selected cell"
        )
        layout.addWidget(instructions)

        # Ensure the widget accepts focus
        self.setFocusPolicy(Qt.StrongFocus)

    def _connect_events(self):
        # Mouse events
        self.viewer.mouse_drag_callbacks.append(self._on_mouse_drag)
        self.viewer.mouse_move_callbacks.append(self._on_mouse_move)
        self.viewer.mouse_wheel_callbacks.append(self._on_mouse_wheel)

        # Key bindings
        self.viewer.bind_key('Delete', self._on_delete_pressed)

        # Button events
        self.toggle_draw_btn.toggled.connect(self._handle_toggle_button)

    def _handle_toggle_button(self, checked):
        """Handle toggle button state changes"""
        self.toggle_state = checked
        self._update_drawing_state()

    def _update_drawing_state(self):
        """Update the drawing state based on toggle button and ctrl key"""
        # Determine if we should be in drawing mode
        new_drawing_state = self.toggle_state or self.ctrl_pressed

        # Only update if the state is actually changing
        if new_drawing_state != self.is_drawing:
            self.is_drawing = new_drawing_state
            self._update_ui_state()

            # Clear drawing if we're switching modes and not mid-drawing
            if not self.drawing_started:
                self._clear_drawing()

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

    def _on_mouse_drag(self, viewer, event):
        if self.masks_layer is None:
            return

        pos = self.viewer.cursor.position
        coords = np.round(pos).astype(int)[-2:]  # Take last 2 dimensions for y,x

        if event.button == Qt.RightButton:
            if self.is_drawing:
                if not self.drawing_started:
                    # Start drawing
                    self.drawing_started = True
                    self.start_point = coords
                    self.drawing_points = [coords]
                    self._update_drawing_preview()
            else:
                # Selection mode
                self._handle_selection(coords)

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

    def _update_drawing_preview(self):
        """Update the preview of the cell being drawn."""
        if not self.drawing_points or len(self.drawing_points) < 2:
            return

        if self.drawing_layer is None:
            self.drawing_layer = self.viewer.add_labels(
                np.zeros_like(self.masks_layer.data),
                name='Drawing Preview',
                opacity=0.6
            )

        # Create mask from drawn points
        points = np.array(self.drawing_points)
        mask = np.zeros_like(self.masks_layer.data)

        # Draw the current path
        for i in range(len(points) - 1):
            cv2.line(mask,
                     (points[i][1], points[i][0]),  # OpenCV uses (x,y) order
                     (points[i + 1][1], points[i + 1][0]),
                     1,
                     1)

        self.drawing_layer.data = mask

    def _finish_drawing(self):
        """Complete the cell drawing process."""
        if not self.drawing_points or len(self.drawing_points) < 3:
            self._clear_drawing()
            return

        # Close the contour by adding the start point
        self.drawing_points.append(self.start_point)

        # Create mask from closed contour
        points = np.array(self.drawing_points)
        mask = np.zeros_like(self.masks_layer.data)

        # Fill the polygon
        cv2.fillPoly(mask, [points[:, ::-1]], 1)  # Reverse point order for OpenCV

        # Add new cell to masks, but only in empty spaces
        new_mask = self.masks_layer.data.copy()
        # Only fill areas where there are no existing cells
        empty_space = new_mask == 0
        new_mask[np.logical_and(mask > 0, empty_space)] = self.next_cell_id
        self.next_cell_id += 1

        # Update masks
        self.masks_layer.data = new_mask
        self.correction_made.emit(new_mask)

        # Reset drawing state
        self.drawing_started = False
        self.start_point = None
        self._clear_drawing()

    def _on_mouse_wheel(self, viewer, event):
        """Handle mouse wheel events to update current frame."""
        if hasattr(self.masks_layer, 'data') and len(self.masks_layer.data.shape) > 2:
            self.current_frame = self.viewer.dims.point[0]

    def _handle_selection(self, coords):
        """Handle cell selection."""
        if not self._validate_coords(coords):
            return

        cell_id = self.masks_layer.data[coords[0], coords[1]]
        if cell_id > 0:
            self.selected_cell = cell_id
            self.status_label.setText(f"Selected cell: {cell_id}")
            self._highlight_selected_cell()

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

    def _highlight_selected_cell(self):
        """Highlight the selected cell."""
        if self.selected_cell is None:
            return

        # Clear any existing drawing layer
        self._clear_drawing()

        # Create highlight mask
        highlight = np.zeros_like(self.masks_layer.data)
        highlight[self.masks_layer.data == self.selected_cell] = 1

        # Create new highlight layer
        self.drawing_layer = self.viewer.add_labels(
            highlight,
            name='Selected Cell',
            opacity=0.6
        )

    def _on_delete_pressed(self, viewer):
        """Handle delete key press to remove selected cell."""
        if self.selected_cell is None or self.masks_layer is None:
            return

        # Remove the selected cell
        new_mask = self.masks_layer.data.copy()
        new_mask[new_mask == self.selected_cell] = 0

        # Update masks
        self.masks_layer.data = new_mask
        self.correction_made.emit(new_mask)

        self._clear_drawing()
        self.selected_cell = None
        self.status_label.setText(f"Deleted cell {self.selected_cell}")

    def set_masks_layer(self, masks: np.ndarray):
        """Set or update the masks layer."""
        if self.masks_layer is None:
            self.masks_layer = self.viewer.add_labels(
                masks,
                name='Cell Masks',
                opacity=0.5
            )
        else:
            self.masks_layer.data = masks

        self.next_cell_id = masks.max() + 1

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

