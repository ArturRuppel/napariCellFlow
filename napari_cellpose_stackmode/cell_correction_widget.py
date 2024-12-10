import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import napari
import numpy as np
from napari.layers import Labels
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QApplication,
    QHBoxLayout, QShortcut
)

from .debug_logging import logger

logger = logging.getLogger(__name__)
#############################
# TODO
# currently Ctrl+Z only undoes deletion events but no drawing events. In fact, if I first remove a cell, then draw a bunch and then do ctrl+z, then the drawn cells disappear and the removed cell reappears
#############################

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import cv2
import napari
import numpy as np
from napari.layers import Labels
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QApplication,
    QHBoxLayout, QShortcut
)

from .debug_logging import logger

logger = logging.getLogger(__name__)


class ActionType(Enum):
    DRAW = "draw"
    DELETE = "delete"


@dataclass
class UndoAction:
    """Represents a single undoable action with full stack context"""
    action_type: ActionType
    frame: Optional[int]  # Frame number for 3D stacks, None for 2D
    previous_state: np.ndarray  # Previous state of the entire stack
    description: str  # Description of the action
    affected_cell_ids: set  # Set of cell IDs affected by this action


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
        self._undo_in_progress = False

        # Initialize masks layer and full stack
        self.masks_layer = None
        self._full_stack = None

        # Initialize from available data sources
        if self.vis_manager.tracking_layer is not None:
            self.masks_layer = self.vis_manager.tracking_layer
            self._initialize_full_stack(self.masks_layer.data)
        elif hasattr(self.data_manager, 'segmentation_data') and self.data_manager.segmentation_data is not None:
            self.set_masks_layer(self.data_manager.segmentation_data)
            self._initialize_full_stack(self.data_manager.segmentation_data)

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

        # Enhanced undo history
        self.undo_stack = deque(maxlen=self.MAX_UNDO_STEPS)

        self._setup_ui()
        self._connect_events()
        self._setup_shortcuts()
        QApplication.instance().installEventFilter(self)


    def _store_undo_state(self, action_type: ActionType, description: str, affected_cells: set = None):
        """Store current state for undo with full stack context"""
        if self._updating or self._undo_in_progress or self._full_stack is None:
            return

        try:
            current_frame = int(self.viewer.dims.point[0])

            # Store the entire stack state
            self.undo_stack.append(UndoAction(
                action_type=action_type,
                frame=current_frame,
                previous_state=self._full_stack.copy(),
                description=description,
                affected_cell_ids=affected_cells or set()
            ))

            self.undo_btn.setEnabled(True)
            logger.debug(f"Stored undo state: {description} ({action_type})")

        except Exception as e:
            logger.error(f"Error storing undo state: {e}")

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts using napari's key bindings"""
        # Remove any existing shortcuts to avoid duplicates
        if hasattr(self, 'undo_shortcut'):
            self.undo_shortcut.setEnabled(False)
            self.undo_shortcut.deleteLater()
        if hasattr(self, 'undo_shortcut_alt'):
            self.undo_shortcut_alt.setEnabled(False)
            self.undo_shortcut_alt.deleteLater()

        # Add the key binding directly to napari's viewer
        @self.viewer.bind_key('Control-Z')
        def undo_callback(viewer):
            # Check if we're not already processing an undo
            if not self._undo_in_progress:
                self.undo_last_action()

        # Connect undo button only once
        self.undo_btn.clicked.disconnect()  # Disconnect any existing connections
        self.undo_btn.clicked.connect(self.undo_last_action)

        logger.debug("Shortcuts setup completed with napari key bindings")

    def undo_last_action(self):
        """Undo the last action with proper state restoration"""
        if not self.undo_stack or self._undo_in_progress:
            return

        try:
            self._undo_in_progress = True
            self._updating = True

            action = self.undo_stack.pop()
            logger.debug(f"Undoing action: {action.description}")

            # Restore the entire stack state
            self._full_stack = action.previous_state.copy()

            # Update visualization
            self.masks_layer.data = self._full_stack
            self.vis_manager.update_tracking_visualization(self._full_stack)

            # Update data manager
            self.data_manager.segmentation_data = self._full_stack

            # Emit correction signal
            self.correction_made.emit(self._full_stack)

            self.status_label.setText(f"Undid: {action.description}")
            logger.info(f"Undid action: {action.description} ({action.action_type})")

            # Update undo button state
            self.undo_btn.setEnabled(len(self.undo_stack) > 0)

            # Force a refresh of the layer
            if self.masks_layer and self.masks_layer in self.viewer.layers:
                self.masks_layer.refresh()

        except Exception as e:
            logger.error(f"Error during undo: {e}")
        finally:
            self._undo_in_progress = False
            self._updating = False
            self._clear_drawing()

    def cleanup(self):
        """Clean up resources and disconnect events"""
        try:
            # Remove application-wide event filter
            QApplication.instance().removeEventFilter(self)

            # Remove the key binding
            self.viewer.keymap.pop('Control-Z', None)

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
            self._full_stack = None

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _finish_drawing(self):
        """Finish drawing with proper undo state management"""
        if not self.drawing_points or len(self.drawing_points) < 3 or self._updating:
            self._clear_drawing()
            return

        try:
            current_frame = int(self.viewer.dims.point[0])

            if self._full_stack is None:
                raise ValueError("No tracking data available")

            # Create new cell mask
            frame_shape = self._full_stack.shape[1:]
            new_cell_mask = self._create_cell_mask(frame_shape)

            # Store undo state before modification
            self._store_undo_state(
                ActionType.DRAW,
                f"Draw new cell {self.next_cell_id}",
                affected_cells={self.next_cell_id}
            )

            self._updating = True

            # Update current frame data
            current_frame_data = self._full_stack[current_frame].copy()
            empty_mask = (current_frame_data == 0)
            add_mask = np.logical_and(new_cell_mask > 0, empty_mask)
            current_frame_data[add_mask] = self.next_cell_id

            # Update full stack
            self._full_stack[current_frame] = current_frame_data

            # Update visualization and data manager
            self.vis_manager.update_tracking_visualization(self._full_stack)
            self.data_manager.segmentation_data = self._full_stack

            self.status_label.setText(f"Added new cell {self.next_cell_id}")
            self.next_cell_id += 1

        except Exception as e:
            logger.error(f"Error finishing drawing: {e}")
            raise
        finally:
            self._updating = False
            self.drawing_started = False
            self.start_point = None
            self._clear_drawing()

    def _delete_cell_at_position(self, coords):
        """Delete cell at the given coordinates with proper undo state management"""
        if self._updating or not self._validate_coords(coords) or self._full_stack is None:
            return

        current_frame = int(self.viewer.dims.point[0])
        current_mask = self._full_stack[current_frame]
        cell_id = current_mask[coords[0], coords[1]]

        if cell_id > 0:
            try:
                # Store undo state before modification
                self._store_undo_state(
                    ActionType.DELETE,
                    f"Delete cell {cell_id}",
                    affected_cells={cell_id}
                )

                self._updating = True

                # Create a copy of the current frame and update
                new_frame = current_mask.copy()
                new_frame[new_frame == cell_id] = 0

                # Update full stack
                self._full_stack[current_frame] = new_frame

                # Update data manager and visualization
                self.data_manager.segmentation_data = (new_frame, current_frame)
                self.vis_manager.update_tracking_visualization((new_frame, current_frame))

                # Force a refresh of the layer
                if self.masks_layer and self.masks_layer in self.viewer.layers:
                    self.masks_layer.refresh()

            finally:
                self._updating = False


    def _initialize_full_stack(self, data: np.ndarray):
        """Initialize the full stack with proper dimensionality"""
        if data is None:
            self._full_stack = None
            return

        # Ensure proper dimensionality
        if data.ndim == 2:
            self._full_stack = data[np.newaxis, ...].copy()
        else:
            self._full_stack = data.copy()

        # Set initial next cell ID
        self.next_cell_id = int(self._full_stack.max()) + 1
        logger.debug(f"Initialized full stack with shape {self._full_stack.shape}")

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
                    # Initialize full stack from the layer data
                    self._initialize_full_stack(layer.data)

                # Update data manager if needed
                if not self.data_manager._initialized:
                    logger.debug("Initializing data manager")
                    if self._full_stack is not None:
                        self.data_manager.initialize_stack(self._full_stack.shape[0])
                        self.data_manager.segmentation_data = self._full_stack

                # Update visualization manager
                if layer.name != 'Drawing Preview':
                    logger.debug("Setting tracking layer in visualization manager")
                    self.vis_manager.tracking_layer = layer

            except Exception as e:
                logger.error(f"Failed to initialize with external layer: {e}", exc_info=True)
            finally:
                self._updating = False
                logger.debug("Layer initialization complete")

    def set_masks_layer(self, masks: np.ndarray):
        """Set or update the masks layer."""
        if self._updating:
            return

        try:
            self._updating = True
            logger.debug(f"Setting masks layer with shape {masks.shape}")

            # Initialize full stack first
            self._initialize_full_stack(masks)

            # Update through visualization manager first
            if self.vis_manager is not None:
                self.vis_manager.update_tracking_visualization(self._full_stack)
                self.masks_layer = self.vis_manager.tracking_layer
            else:
                # Create new layer if needed
                if self.masks_layer is None or self.masks_layer not in self.viewer.layers:
                    self.masks_layer = self.viewer.add_labels(
                        self._full_stack,
                        name='Segmentation',
                        opacity=0.5
                    )
                else:
                    self.masks_layer.data = self._full_stack

            # Update visualization manager reference
            if self.vis_manager is not None:
                self.vis_manager.tracking_layer = self.masks_layer

        except Exception as e:
            logger.error(f"Error setting masks layer: {str(e)}")
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
            "- Hold Ctrl + Left click to delete cells\n"
            "- Hold Ctrl + Right click and drag to draw cell boundary\n"
            "- Return to start point (marked with circle) to complete the cell\n"
            "- Ctrl+Z to undo last action"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

    def eventFilter(self, watched_object, event):
        """Global event filter to catch key events regardless of focus"""
        if event.type() not in (event.KeyPress, event.KeyRelease):
            return super().eventFilter(watched_object, event)

        if event.key() != Qt.Key_Control:
            return False

        try:
            is_press = event.type() == event.KeyPress
            logger.debug(f"CellCorrection: Ctrl key {'pressed' if is_press else 'released'}")

            # Only update state if we're not actively drawing
            if not self.drawing_started:
                self.ctrl_pressed = is_press
                self._update_drawing_state()

            return False  # Don't consume the event

        except Exception as e:
            logger.error(f"CellCorrection: Error handling Ctrl key: {e}")
            return False

    def _update_drawing_state(self):
        """Update drawing state with proper synchronization."""
        try:
            new_drawing_state = self.toggle_state or self.ctrl_pressed

            # Only update if state actually changes
            if new_drawing_state == self.is_drawing:
                return

            logger.debug(f"CellCorrection: Drawing state changing from {self.is_drawing} to {new_drawing_state}")
            self.is_drawing = new_drawing_state

            # Only update UI if we're not actively drawing
            if not self.drawing_started:
                with self.viewer.events.blocker_all():
                    self._update_ui_state()
                    if not new_drawing_state:
                        self._clear_drawing()

            # Single refresh at the end if needed
            if self.masks_layer and self.masks_layer in self.viewer.layers:
                self.masks_layer.refresh()

        except Exception as e:
            logger.error(f"CellCorrection: Error updating drawing state: {e}")

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

    def _create_cell_mask(self, frame_shape):
        """Create a mask for the new cell"""
        new_cell_mask = np.zeros(frame_shape, dtype=np.uint8)
        points = np.array(self.drawing_points + [self.start_point])
        points = np.clip(points, 0, np.array(frame_shape) - 1).astype(np.int32)
        cv2.fillPoly(new_cell_mask, [points[:, ::-1]], 1)
        return new_cell_mask

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

        # Add toggle button connection
        self.toggle_draw_btn.toggled.connect(self._handle_toggle_button)

        # Add undo button connection
        self.undo_btn.clicked.connect(self.undo_last_action)

    def _handle_toggle_button(self, checked):
        """Handle toggle button state changes"""
        self.toggle_state = checked
        self._update_drawing_state()

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

