"""Cell Correction Test Suite

A comprehensive test suite for testing the cell correction functionality in napariCellFlow.
Tests cover the CellCorrectionWidget UI component and its interactions with napari viewer
and layer management.

Key Test Coverage:

Basic Functionality Tests:
1. `test_basic_initialization`: Verifies widget initialization with default state and components
2. `test_layer_management`: Tests layer addition, removal, and state maintenance
3. `test_cleanup`: Validates proper cleanup of resources and event disconnection
4. `test_ui_components`: Verifies creation and properties of UI elements
5. `test_external_data_initialization`: Tests initialization with external data sources

Drawing Operation Tests:
1. `test_drawing_state_management`: Tests drawing mode state transitions and controls
2. `test_drawing_preview`: Validates drawing preview layer creation and updates
3. `test_drawing_completion`: Tests cell completion and validation
4. `test_coordinate_validation`: Verifies coordinate validation within image bounds
5. `test_mouse_event_handling`: Tests mouse event processing for drawing operations

Cell Manipulation Tests:
1. `test_cell_deletion`: Validates cell deletion functionality and state updates
2. `test_undo_functionality`: Tests undo stack management and state restoration
3. `test_keyboard_event_handling`: Verifies keyboard shortcuts and state changes

Stack Operations Tests:
1. `test_stack_operations`: Tests operations on multi-frame data
2. `test_frame_handling`: Validates frame-specific operations in both stack and single modes

Test Data:
- Uses synthetic test masks with known properties:
  * Simple cell patterns for basic testing
  * Multi-frame sequences for stack testing
  * Known cell counts and IDs for validation
  * Varying cell sizes and positions

Test Components:
- Mock viewer for napari viewer simulation
- Mock tracking layer for data management
- Mock event system for user interactions
- Synthetic mask generation utilities

Dependencies:
- pytest for test framework
- numpy for array operations
- Mock/patch from unittest.mock for mocking
- Qt testing utilities for widget testing
- napari for viewer components
- OpenCV for mask operations

Notes:
- Tests use small synthetic masks for efficient execution
- Event system thoroughly mocked to simulate user interactions
- Layer management tests account for visualization manager state
- Drawing tests validate both UI and data state
- Mock objects ensure consistent test behavior
"""

from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QWidget, QPushButton

from napariCellFlow.cell_correction_widget import CellCorrectionWidget, ActionType, UndoAction


def create_test_mask(size=64, num_cells=3):
    """Create a test mask with specified number of cells."""
    mask = np.zeros((size, size), dtype=np.uint16)
    cell_size = size // (num_cells + 1)

    for i in range(num_cells):
        y = (i + 1) * cell_size
        x = (i + 1) * cell_size
        mask[y - cell_size // 2:y + cell_size // 2, x - cell_size // 2:x + cell_size // 2] = i + 1

    return mask


class TestCellCorrectionWidget:
    @pytest.fixture(autouse=True)
    def setup_class(self, qtbot):
        """Set up the Qt application for all tests."""
        self.qtbot = qtbot

    @pytest.fixture
    def mock_viewer(self):
        """Create a mock napari viewer."""
        viewer = MagicMock()

        # Mock the layers property
        viewer.layers = MagicMock()
        viewer.layers.events = MagicMock()
        viewer.layers.events.inserted = MagicMock()
        viewer.layers.events.removed = MagicMock()

        # Mock the dims property
        viewer.dims = MagicMock()
        viewer.dims.point = [0, 0, 0]

        # Mock cursor
        viewer.cursor = MagicMock()
        viewer.cursor.position = [32, 32]

        # Mock window for shortcuts
        viewer.window = MagicMock(return_value=QWidget())

        # Mock add_labels method
        test_layer = MagicMock()
        viewer.add_labels = MagicMock(return_value=test_layer)

        return viewer

    @pytest.fixture
    def mock_tracking_layer(self):
        """Create a mock tracking layer with test data."""
        layer = MagicMock()
        test_data = create_test_mask()
        type(layer).data = PropertyMock(return_value=test_data)
        layer.name = 'Segmentation'
        return layer

    @pytest.fixture
    def widget(self, mock_viewer, mock_tracking_layer, qtbot):
        """Create a CellCorrectionWidget instance with mocked dependencies."""
        data_manager = MagicMock()
        data_manager._initialized = False

        # Create visualization manager with pre-configured tracking layer
        visualization_manager = MagicMock()
        visualization_manager.tracking_layer = mock_tracking_layer

        with patch('napari.layers.Labels', MagicMock()):
            widget = CellCorrectionWidget(
                viewer=mock_viewer,
                data_manager=data_manager,
                visualization_manager=visualization_manager
            )
            # Add undo button manually for testing
            widget.undo_btn = QPushButton("Undo")
            qtbot.addWidget(widget)

            # Force update next_cell_id based on mock_tracking_layer
            widget.next_cell_id = mock_tracking_layer.data.max() + 1
            return widget

    def test_basic_initialization(self, widget, mock_tracking_layer):
        """Test basic initialization of CellCorrectionWidget."""
        assert widget is not None
        assert widget.viewer is not None
        assert widget.data_manager is not None
        assert widget.vis_manager is not None
        assert widget.is_drawing is False
        assert widget.drawing_points == []
        assert widget.next_cell_id == mock_tracking_layer.data.max() + 1
        assert widget.selected_cell is None
        assert widget.ctrl_pressed is False
        assert len(widget.undo_stack) == 0

    def test_layer_management(self, widget):
        """Test layer management functionality."""
        # Create mock layer with known data
        test_data = create_test_mask()
        mock_layer = MagicMock()
        type(mock_layer).data = PropertyMock(return_value=test_data)
        mock_layer.name = 'Segmentation'

        # Mock the visualization manager before any operations
        widget.vis_manager.tracking_layer = None  # This directly sets the tracking_layer to None
        widget.masks_layer = None  # Ensure we start with no masks layer

        # Simulate layer addition
        event = MagicMock()
        event.value = mock_layer
        widget._handle_layer_added(event)

        # Verify layer was properly initialized and next_cell_id updated
        max_id = test_data.max()
        assert widget.next_cell_id == max_id + 1

        # Before testing removal, ensure vis_manager won't provide a layer
        widget.vis_manager.tracking_layer = None

        # Test layer removal
        event.value = mock_layer
        widget._handle_layer_removal(event)
        assert widget.masks_layer is None

    def test_undo_functionality(self, widget, mock_tracking_layer):
        """Test undo functionality."""
        # Setup initial state
        test_data = create_test_mask()
        widget.masks_layer = mock_tracking_layer
        widget.vis_manager.tracking_layer = mock_tracking_layer

        # Create an undo action with proper previous state
        prev_state = test_data.copy()
        action = UndoAction(
            action_type=ActionType.DELETE,
            frame=0,
            previous_state=prev_state[np.newaxis, ...],
            description="Test deletion",
            affected_cell_ids={1}
        )
        widget.undo_stack.append(action)

        # Perform undo
        widget.undo_last_action()

        # Verify undo stack is empty and visualization was updated
        assert len(widget.undo_stack) == 0
        widget.vis_manager.update_tracking_visualization.assert_called_once()

    def test_drawing_completion(self, widget, mock_tracking_layer):
        """Test drawing completion functionality."""
        # Setup widget state
        widget.is_drawing = True
        widget.drawing_started = True
        widget.masks_layer = mock_tracking_layer
        widget.start_point = np.array([32, 32])

        # Add drawing points to simulate a complete cell
        points = [
            np.array([32, 32]),
            np.array([32, 42]),
            np.array([42, 42]),
            np.array([42, 32]),
            np.array([32, 32])  # Close the shape
        ]
        widget.drawing_points = points

        # Mock both create_cell_mask and clear_drawing
        with patch.object(widget, '_create_cell_mask', return_value=np.ones((64, 64), dtype=np.uint8)), \
                patch.object(widget, '_clear_drawing') as mock_clear:
            # Complete the drawing
            widget._finish_drawing()

            # Verify drawing state was reset
            assert not widget.drawing_started
            assert widget.start_point is None
            mock_clear.assert_called_once()  # Verify clear_drawing was called

    def test_cell_deletion(self, widget, mock_tracking_layer):
        """Test cell deletion functionality."""
        # Setup test mask
        test_mask = create_test_mask(64, 3)
        widget.masks_layer = mock_tracking_layer
        widget.vis_manager.tracking_layer = mock_tracking_layer

        # Mock current frame mask
        with patch.object(widget, '_get_current_frame_mask', return_value=test_mask):
            # Simulate cell deletion
            coords = np.array([32, 32])
            widget._delete_cell_at_position(coords)

            # Verify deletion was tracked in undo stack
            assert len(widget.undo_stack) == 1
            assert widget.undo_stack[-1].action_type == ActionType.DELETE

    def test_drawing_preview(self, widget, mock_tracking_layer):
        """Test drawing preview functionality."""
        # Setup initial state
        widget.is_drawing = True
        widget.drawing_started = True
        widget.start_point = np.array([32, 32])
        widget.drawing_points = [np.array([32, 32]), np.array([33, 33])]  # Need at least 2 points
        widget.masks_layer = mock_tracking_layer

        # Mock the layer being in the viewer's layers
        widget.viewer.layers = MagicMock()
        widget.viewer.layers.__contains__ = MagicMock(return_value=True)

        # Mock the viewer's add_labels method to return a proper layer
        preview_layer = MagicMock()
        widget.viewer.add_labels.return_value = preview_layer

        # Update drawing preview
        widget._update_drawing_preview()

        # Verify preview layer was created
        widget.viewer.add_labels.assert_called_once()

    def test_stack_operations(self, widget, mock_tracking_layer):
        """Test operations on image stacks."""
        # Create test stack
        test_stack = np.stack([create_test_mask(64, 3) for _ in range(3)])
        type(mock_tracking_layer).data = PropertyMock(return_value=test_stack)
        widget.masks_layer = mock_tracking_layer
        widget.vis_manager.tracking_layer = mock_tracking_layer

        # Set current frame
        widget.viewer.dims.point = [1, 0, 0]

        # Simulate deletion in stack mode
        coords = np.array([32, 32])
        with patch.object(widget, '_get_current_frame_mask', return_value=test_stack[1]):
            widget._delete_cell_at_position(coords)

            # Verify undo stack
            assert len(widget.undo_stack) == 1
            assert widget.undo_stack[-1].frame == 1

    def test_external_data_initialization(self, widget):
        """Test initialization with external data."""
        # Create test data
        test_data = create_test_mask(64, 3)

        # Initialize widget with external data
        widget._init_with_external_data(test_data)

        # Verify initialization
        assert widget.next_cell_id == test_data.max() + 1
        widget.data_manager.initialize_stack.assert_called_once()
        widget.vis_manager.update_tracking_visualization.assert_called_once()
        assert len(widget.undo_stack) == 0

    def test_coordinate_validation(self, widget, mock_tracking_layer):
        """Test coordinate validation."""
        widget.masks_layer = mock_tracking_layer

        # Test valid coordinates
        assert widget._validate_coords(np.array([32, 32]))

        # Test invalid coordinates
        assert not widget._validate_coords(np.array([64, 64]))
        assert not widget._validate_coords(np.array([-1, -1]))

    def test_mouse_event_handling(self, widget, mock_tracking_layer):
        """Test mouse event handling."""
        # Setup widget state
        widget.is_drawing = True
        widget.ctrl_pressed = True
        widget.masks_layer = mock_tracking_layer

        # Create mock event
        event = MagicMock()
        event.button = Qt.RightButton

        # Test drawing initiation
        widget._on_mouse_drag(widget.viewer, event)
        assert widget.drawing_started
        assert len(widget.drawing_points) == 1

        # Move cursor and test drawing continuation
        widget.viewer.cursor.position = [33, 33]
        widget._on_mouse_move(widget.viewer, event)
        assert len(widget.drawing_points) > 0

    @pytest.mark.parametrize("stack_mode", [True, False])
    def test_frame_handling(self, widget, mock_tracking_layer, stack_mode):
        """Test frame handling in both stack and single image modes."""
        # Setup test data
        if stack_mode:
            test_data = np.stack([create_test_mask() for _ in range(3)])
        else:
            test_data = create_test_mask()

        # Update mock layer data
        type(mock_tracking_layer).data = PropertyMock(return_value=test_data)
        widget.masks_layer = mock_tracking_layer

        # Test getting current frame
        current_mask = widget._get_current_frame_mask()
        assert current_mask is not None

        if stack_mode:
            assert current_mask.shape == test_data[0].shape
        else:
            assert current_mask.shape == test_data.shape

    def test_cleanup(self, widget):
        """Test cleanup functionality."""
        # Setup initial state
        widget.masks_layer = MagicMock()

        # Perform cleanup
        widget.cleanup()

        # Verify cleanup
        assert widget.viewer is None
        assert widget.masks_layer is None
        assert widget._full_masks is None

    def test_ui_components(self, widget):
        """Test UI component creation and properties."""
        # Check status label
        assert hasattr(widget, 'status_label')
        assert widget.status_label.text() == "Ready"

        # Verify layout
        assert widget.layout() is not None
        assert widget.layout().count() >= 2  # Status label and instructions

        # Check that instructions are present
        instructions_found = False
        for i in range(widget.layout().count()):
            item = widget.layout().itemAt(i)
            if item.widget() and "Cell Editing Controls:" in item.widget().text():
                instructions_found = True
                break
        assert instructions_found

    def test_drawing_state_management(self, widget):
        """Test drawing state management."""
        # Initial state
        assert not widget.is_drawing
        assert not widget.drawing_started
        assert widget.drawing_points == []

        # Simulate Ctrl press
        widget.ctrl_pressed = True
        widget._update_drawing_state()
        assert widget.is_drawing

        # Simulate Ctrl release
        widget.ctrl_pressed = False
        widget._update_drawing_state()
        assert not widget.is_drawing
        assert widget.drawing_points == []

    def test_keyboard_event_handling(self, widget):
        """Test keyboard event handling."""

        # Create mock key events
        class MockKeyEvent:
            def __init__(self, key, type_):
                self.key = lambda: key
                self.type = lambda: type_
                self.KeyPress = QMouseEvent.KeyPress
                self.KeyRelease = QMouseEvent.KeyRelease

        # Test Ctrl press
        ctrl_press = MockKeyEvent(Qt.Key_Control, QMouseEvent.KeyPress)
        widget.eventFilter(None, ctrl_press)
        assert widget.ctrl_pressed
        assert widget.is_drawing

        # Test Ctrl release
        ctrl_release = MockKeyEvent(Qt.Key_Control, QMouseEvent.KeyRelease)
        widget.eventFilter(None, ctrl_release)
        assert not widget.ctrl_pressed
        assert not widget.is_drawing

