import logging
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, List

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

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Set, Dict
import numpy as np


class ActionType(Enum):
    DRAW = "draw"
    DELETE = "delete"
    SEGMENTATION = "segmentation"


@dataclass
class UndoAction:
    """Enhanced undo action with better state management"""
    action_type: ActionType
    frame: int  # Frame where action occurred
    previous_state: np.ndarray  # Previous state of affected frame
    affected_frames: Set[int]  # All frames affected by this action
    affected_cells: Set[int]  # Cell IDs affected
    description: str
    metadata: Dict = None  # Additional action-specific data


class UndoManager:
    """Manages undo operations with frame awareness"""

    def __init__(self, max_steps: int = 20):
        self.stack = []
        self.max_steps = max_steps
        self._locked = False

    def push(self, action: UndoAction):
        """Push a new action onto the stack"""
        if self._locked:
            return

        self.stack.append(action)
        if len(self.stack) > self.max_steps:
            self.stack.pop(0)

    def pop(self) -> Optional[UndoAction]:
        """Pop the most recent action"""
        if not self.stack or self._locked:
            return None
        return self.stack.pop()

    def get_frame_actions(self, frame: int) -> List[UndoAction]:
        """Get all actions affecting a specific frame"""
        return [action for action in self.stack
                if frame in action.affected_frames]

    @contextmanager
    def lock(self):
        """Context manager to temporarily lock the undo stack"""
        self._locked = True
        try:
            yield
        finally:
            self._locked = False


class CellCorrectionWidget(QWidget):
    """Widget for interactive cell correction in napari."""

    correction_made = Signal(np.ndarray)  # Emitted when masks are modified
    LINE_THICKNESS = 2
    START_POINT_RADIUS = 5
    LINE_COLOR = 3
    MAX_UNDO_STEPS = 20
    CLOSURE_THRESHOLD = 10
    MIN_DRAWING_POINTS = 20

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager",
                 visualization_manager: "VisualizationManager", parent: QWidget = None):
        super().__init__(parent)

        # Core components
        self.viewer = viewer
        self.data_manager = data_manager
        self.vis_manager = visualization_manager

        # State flags - simplified
        self._updating = False
        self._drawing_in_progress = False

        # Layer management
        self.masks_layer = None
        self._full_stack = None
        self._frame_states = {}  # Cache of frame states

        # Drawing state
        self.drawing_layer = None
        self.is_drawing = False
        self.drawing_points = []
        self.next_cell_id = 1
        self.drawing_started = False
        self.start_point = None
        self.ctrl_pressed = False
        self.toggle_state = False

        # Single undo manager
        self.undo_manager = UndoManager()

        # Initialize from available data
        if self.vis_manager.tracking_layer is not None:
            self.masks_layer = self.vis_manager.tracking_layer
            self._initialize_full_stack(self.masks_layer.data)
        elif hasattr(self.data_manager, 'segmentation_data') and self.data_manager.segmentation_data is not None:
            self.set_masks_layer(self.data_manager.segmentation_data)

        # Setup UI and events
        self._setup_ui()
        self._connect_events()
        self._setup_shortcuts()

        # Install event filter for key handling
        QApplication.instance().installEventFilter(self)

    def _update_undo_button(self):
        """Update undo button state based on undo stack"""
        if hasattr(self, 'undo_btn'):
            self.undo_btn.setEnabled(len(self.undo_manager.stack) > 0)

    def _update_drawing_preview(self):
        """Update the preview of the cell being drawn with proper state checks"""
        if not self.drawing_points or len(self.drawing_points) < 2 or self._updating:
            return

        try:
            self._updating = True

            # Ensure valid masks layer
            if self.masks_layer is None or self.masks_layer not in self.viewer.layers:
                if self.vis_manager.tracking_layer is not None:
                    self.masks_layer = self.vis_manager.tracking_layer
                else:
                    return

            # Get shape information
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

                with self.viewer.events.blocker_all():
                    self.drawing_layer = self.viewer.add_labels(
                        preview_data,
                        name='Drawing Preview',
                        opacity=0.8,
                        visible=True
                    )

            # Create preview mask
            preview_data = self.drawing_layer.data.copy()
            if len(self.masks_layer.data.shape) == 3:
                mask = np.zeros(shape, dtype=np.uint8)
            else:
                preview_data = np.zeros_like(self.masks_layer.data)
                mask = preview_data

            # Draw preview
            points = np.array(self.drawing_points)
            for i in range(len(points) - 1):
                p1 = (int(points[i][1]), int(points[i][0]))
                p2 = (int(points[i + 1][1]), int(points[i + 1][0]))
                cv2.line(mask, p1, p2, self.LINE_COLOR, self.LINE_THICKNESS)

            # Draw start point indicator
            if self.start_point is not None:
                center = (int(self.start_point[1]), int(self.start_point[0]))
                cv2.circle(mask, center, self.START_POINT_RADIUS, self.LINE_COLOR, -1)

            # Update layer data
            with self.viewer.events.blocker_all():
                if len(self.masks_layer.data.shape) == 3:
                    preview_data[current_frame] = mask
                    self.drawing_layer.data = preview_data
                else:
                    self.drawing_layer.data = mask

                # Ensure visibility
                self.drawing_layer.visible = True
                self.masks_layer.visible = True

        except Exception as e:
            logger.error(f"Error updating drawing preview: {e}")
            self._clear_drawing()
        finally:
            self._updating = False

    def _clear_drawing(self):
        """Clear drawing state with proper cleanup"""
        try:
            with self.viewer.events.blocker_all():
                # Remove drawing preview layer
                if self.drawing_layer is not None:
                    if self.drawing_layer in self.viewer.layers:
                        self.viewer.layers.remove(self.drawing_layer)
                    self.drawing_layer = None

                # Reset drawing state
                self.drawing_points = []
                self.start_point = None
                self.drawing_started = False

                # Ensure correct segmentation layer
                if self.vis_manager.tracking_layer is not None:
                    self.masks_layer = self.vis_manager.tracking_layer

                if self.masks_layer is not None and self.masks_layer in self.viewer.layers:
                    self.masks_layer.visible = True

        except Exception as e:
            logger.error(f"Error clearing drawing: {e}")

    def _connect_events(self):
        """Connect all event handlers with proper cleanup"""
        try:
            # Remove existing callbacks
            self._remove_existing_callbacks()

            # Add mouse callbacks
            self.viewer.mouse_drag_callbacks.append(self._on_mouse_drag)
            self.viewer.mouse_move_callbacks.append(self._on_mouse_move)
            self.viewer.mouse_wheel_callbacks.append(self._on_mouse_wheel)

            # Connect UI controls
            self.toggle_draw_btn.toggled.connect(self._handle_toggle_button)
            self.undo_btn.clicked.connect(self.undo_last_action)

            logger.debug("Event handlers connected successfully")

        except Exception as e:
            logger.error(f"Error connecting events: {e}")
            raise

    def _remove_existing_callbacks(self):
        """Remove existing callbacks safely"""
        try:
            callbacks_to_remove = [
                (self.viewer.mouse_drag_callbacks, self._on_mouse_drag),
                (self.viewer.mouse_move_callbacks, self._on_mouse_move),
                (self.viewer.mouse_wheel_callbacks, self._on_mouse_wheel)
            ]

            for callback_list, callback in callbacks_to_remove:
                if callback in callback_list:
                    callback_list.remove(callback)

        except Exception as e:
            logger.error(f"Error removing callbacks: {e}")

    def cleanup(self):
        """Clean up resources with proper error handling"""
        try:
            # Remove event filter
            QApplication.instance().removeEventFilter(self)

            # Remove shortcuts
            self.viewer.keymap.pop('Control-Z', None)

            # Remove callbacks
            self._remove_existing_callbacks()

            # Clear drawing state
            self._clear_drawing()

            # Clear references
            self.viewer = None
            self.masks_layer = None
            self._full_stack = None
            self.drawing_layer = None

            logger.debug("Widget cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _validate_coords(self, coords):
        """Validate coordinates with comprehensive checks"""
        try:
            if self.masks_layer is None:
                return False

            # Get current shape
            if len(self.masks_layer.data.shape) == 3:
                shape = self.masks_layer.data.shape[1:]
            else:
                shape = self.masks_layer.data.shape

            # Basic bounds check
            if not (isinstance(coords, (tuple, list, np.ndarray)) and len(coords) == 2):
                return False

            # Validate each coordinate
            return (0 <= coords[0] < shape[0] and
                    0 <= coords[1] < shape[1])

        except Exception as e:
            logger.error(f"Error validating coordinates: {e}")
            return False
    def _delete_cell_at_position(self, coords):
        """Delete cell at the given coordinates with proper state management"""
        if self._updating or not self._validate_coords(coords) or self._full_stack is None:
            return

        current_frame = int(self.viewer.dims.point[0])
        current_mask = self._full_stack[current_frame]
        cell_id = current_mask[coords[0], coords[1]]

        if cell_id > 0:
            try:
                # Create undo action
                action = self._create_undo_action(
                    frame=current_frame,
                    action_type=ActionType.DELETE,
                    cell_ids={cell_id},
                    description=f"Delete cell {cell_id}"
                )

                if action is None:
                    logger.error("Failed to create undo action for cell deletion")
                    return

                self._updating = True

                # Create a copy of the current frame and update
                new_frame = current_mask.copy()
                new_frame[new_frame == cell_id] = 0
                self._full_stack[current_frame] = new_frame

                # Store undo action and update UI
                self.undo_manager.push(action)
                self._update_undo_button()
                self.status_label.setText(f"Deleted cell {cell_id}")

                # Update visualization
                self.vis_manager.update_tracking_visualization(self._full_stack)
                self.correction_made.emit(self._full_stack)

            finally:
                self._updating = False
                self._clear_drawing()
    def _restore_frame_state(self, frame: int, state: np.ndarray) -> bool:
        """Restore a frame to a previous state"""
        if self._full_stack is None:
            return False

        try:
            self._full_stack[frame] = state.copy()
            self.vis_manager.update_tracking_visualization(self._full_stack)
            self.data_manager.segmentation_data = self._full_stack
            return True
        except Exception as e:
            logger.error(f"Failed to restore frame state: {e}")
            return False

    def _store_frame_state(self, frame: int) -> bool:
        """Store the current state of a specific frame. Returns True if successful."""
        if self._full_stack is None:
            logger.warning("Cannot store frame state: No stack available")
            return False

        if frame < 0 or frame >= self._full_stack.shape[0]:
            logger.error(f"Invalid frame index: {frame}")
            return False

        try:
            self._frame_states[frame] = self._full_stack[frame].copy()
            return True
        except Exception as e:
            logger.error(f"Failed to store frame state: {e}")
            return False

    def _create_undo_action(self, frame: int, action_type: ActionType, cell_ids: Set[int], description: str) -> Optional[UndoAction]:
        """Create an undo action with proper state management"""
        if not self._store_frame_state(frame):
            return None

        return UndoAction(
            action_type=action_type,
            frame=frame,
            previous_state=self._frame_states[frame],
            affected_frames={frame},
            affected_cells=cell_ids,
            description=description
        )

    def undo_last_action(self):
        """Undo last action with proper frame handling"""
        if self._updating:
            return

        action = self.undo_manager.pop()
        if not action:
            return

        try:
            self._updating = True
            self._undo_in_progress = True

            # Restore frame state
            self._full_stack[action.frame] = action.previous_state.copy()

            # Update visualization
            self.vis_manager.update_tracking_visualization(self._full_stack)
            self.data_manager.segmentation_data = self._full_stack

            # Update UI
            self.status_label.setText(f"Undid: {action.description}")
            self.undo_btn.setEnabled(len(self.undo_manager.stack) > 0)

        finally:
            self._updating = False
            self._undo_in_progress = False
            self._clear_drawing()

    def _finish_drawing(self):
        """Finish drawing with proper state management"""
        if not self.drawing_points or len(self.drawing_points) < self.MIN_DRAWING_POINTS:
            self._clear_drawing()
            return

        try:
            current_frame = int(self.viewer.dims.point[0])
            frame_shape = self._full_stack[current_frame].shape

            # Create new cell mask
            new_cell_mask = self._create_cell_mask(frame_shape)

            # Create undo action
            action = self._create_undo_action(
                frame=current_frame,
                action_type=ActionType.DRAW,
                cell_ids={self.next_cell_id},
                description=f"Draw cell {self.next_cell_id}"
            )

            if action is None:
                raise RuntimeError("Failed to create undo action")

            # Apply the change
            frame_data = self._full_stack[current_frame].copy()
            empty_mask = (frame_data == 0)
            add_mask = np.logical_and(new_cell_mask > 0, empty_mask)
            frame_data[add_mask] = self.next_cell_id

            # Update stack and store undo action
            self._full_stack[current_frame] = frame_data
            self.undo_manager.push(action)
            self._update_undo_button()

            # Update visualization
            self.vis_manager.update_tracking_visualization(self._full_stack)
            self.correction_made.emit(self._full_stack)

            # Update UI
            self.status_label.setText(f"Added cell {self.next_cell_id}")
            self.next_cell_id += 1

        except Exception as e:
            logger.error(f"Error finishing drawing: {e}")
            raise
        finally:
            self._clear_drawing()
            self.drawing_started = False
            self.start_point = None

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts with proper cleanup"""
        try:
            # Clear existing shortcuts
            if hasattr(self, '_undo_callback'):
                self.viewer.keymap.pop('Control-Z', None)

            # Create new undo callback
            def undo_callback(viewer):
                if not self._updating:
                    self.undo_last_action()

            # Store reference to callback
            self._undo_callback = undo_callback

            # Add the key binding
            self.viewer.bind_key('Control-Z', undo_callback)

            # Connect undo button - ensure single connection
            self.undo_btn.clicked.disconnect() if self.undo_btn.receivers(self.undo_btn.clicked) > 0 else None
            self.undo_btn.clicked.connect(self.undo_last_action)

            logger.debug("Shortcuts setup completed")

        except Exception as e:
            logger.error(f"Error setting up shortcuts: {e}")
            raise

    def _on_mouse_drag(self, viewer, event):
        """Handle mouse drag events with proper state management"""
        try:
            if self._updating:
                return

            # Ensure we have valid layer
            if self.masks_layer is None:
                if self.vis_manager.tracking_layer is not None:
                    self.masks_layer = self.vis_manager.tracking_layer
                else:
                    return

            # Get coordinates
            pos = viewer.cursor.position
            if pos is None:
                return

            coords = np.round(pos).astype(int)[-2:]
            if not self._validate_coords(coords):
                return

            # Handle right button drawing
            if event.button == Qt.RightButton and self.is_drawing:
                if not self.drawing_started:
                    self.drawing_started = True
                    self.start_point = coords
                    self.drawing_points = [coords]
                else:
                    self.drawing_points.append(coords)
                self._update_drawing_preview()

            # Handle left button deletion
            elif event.button == Qt.LeftButton and self.is_drawing:
                self._delete_cell_at_position(coords)

        except Exception as e:
            logger.error(f"Error in mouse drag handling: {e}")
            self._clear_drawing()

    def _on_mouse_move(self, viewer, event):
        """Handle mouse movement for drawing preview with stability checks"""
        if not self.is_drawing or not self.drawing_started or self._updating:
            return

        try:
            pos = viewer.cursor.position
            if pos is None:
                return

            coords = np.round(pos).astype(int)[-2:]
            if not self._validate_coords(coords):
                return

            # Only add point if different from last point
            if not self.drawing_points or not np.array_equal(coords, self.drawing_points[-1]):
                self.drawing_points.append(coords)

                # Check for closure
                if len(self.drawing_points) > self.MIN_DRAWING_POINTS:
                    dist_to_start = np.linalg.norm(coords - self.start_point)
                    if dist_to_start < self.CLOSURE_THRESHOLD:
                        self._finish_drawing()
                        return

                self._update_drawing_preview()

        except Exception as e:
            logger.error(f"Error in mouse move handling: {e}")
            self._clear_drawing()

    def _on_mouse_wheel(self, viewer, event):
        """Handle mouse wheel events with proper state cleanup"""
        if self._updating:
            return

        try:
            self._updating = True

            # Clear any active drawing
            self._clear_drawing()

            # Refresh visualization
            if self.masks_layer is not None and self.masks_layer in self.viewer.layers:
                self.masks_layer.refresh()
                if self.vis_manager.tracking_layer is not None:
                    self.vis_manager.tracking_layer.refresh()

        except Exception as e:
            logger.error(f"Error in mouse wheel handling: {e}")
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

    def _create_cell_mask(self, frame_shape):
        """Create a mask for the new cell"""
        new_cell_mask = np.zeros(frame_shape, dtype=np.uint8)
        points = np.array(self.drawing_points + [self.start_point])
        points = np.clip(points, 0, np.array(frame_shape) - 1).astype(np.int32)
        cv2.fillPoly(new_cell_mask, [points[:, ::-1]], 1)
        return new_cell_mask

    def _handle_layer_removal(self, event):
        """Handle layer removal events"""
        if event.value == self.masks_layer:
            self.masks_layer = None
            self._full_masks = None

    def _handle_toggle_button(self, checked):
        """Handle toggle button state changes"""
        self.toggle_state = checked
        self._update_drawing_state()
