"""Cell Tracking Test Suite

A comprehensive test suite for the cell tracking functionality in napariCellFlow. Tests cover both
the CellTracker class implementation and the CellTrackingWidget UI component.

Key Test Coverage:

CellTracker Tests (`TestCellTracker` class):
1. `test_initialization`: Verifies tracker initialization with config and empty region cache
2. `test_invalid_input_dimensions`: Ensures proper error handling for 2D instead of 3D input
3. `test_progress_callback`: Validates progress callback functionality during tracking
4. `test_small_cell_filtering`: Tests filtering of cells below minimum size threshold
5. `test_cell_tracking_basic`: Verifies basic cell tracking across frames:
   - Checks shape preservation
   - Validates cell ID consistency
   - Confirms ID preservation between frames
6. `test_gap_closing`: Tests tracking continuity when cells temporarily disappear

CellTrackingWidget Tests (`TestCellTrackingWidget` class):
1. `test_initialization`: Verifies widget creation with mock viewer and managers
2. `test_parameter_controls`: Tests UI control updates for:
   - Overlap ratio
   - Maximum displacement
3. `test_gap_closing_controls`: Validates gap closing checkbox behavior and spin box enabling
4. `test_run_analysis`: Tests analysis execution with mock layer data
5. `test_reset_parameters`: Verifies parameter reset functionality to defaults

Test Data:
Uses a standard test stack (3x20x20) containing:
- Frame 1: Two separate 3x3 square cells
- Frame 2: Same cells moved by 1 pixel
- Frame 3: One cell unchanged, one cell split into two

Dependencies:
- pytest for test framework
- numpy for array operations
- Mock/patch from unittest.mock for mocking
- Qt testing utilities for widget interaction
- napariCellFlow package components

Notes:
The test coverage for the CellTracker focuses on core functionality and error cases,
while the widget tests cover both UI interaction and integration with the tracking system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from napariCellFlow.cell_tracking import CellTracker
from napariCellFlow.cell_tracking_widget import CellTrackingWidget
from napariCellFlow.structure import AnalysisConfig, TrackingParameters


def create_test_stack(shape=(3, 20, 20)):
    """Create a simple test segmentation stack with cells that satisfy tracking parameters"""
    stack = np.zeros(shape, dtype=np.int32)

    # First frame: two cells
    stack[0, 5:8, 5:8] = 1  # Cell 1: 3x3 square
    stack[0, 12:15, 12:15] = 2  # Cell 2: 3x3 square

    # Second frame: same cells moved slightly
    stack[1, 6:9, 6:9] = 1  # Cell 1 moved by 1 pixel
    stack[1, 13:16, 13:16] = 2  # Cell 2 moved by 1 pixel

    # Third frame: one cell split
    stack[2, 7:10, 7:10] = 1  # Cell 1 moved by 1 pixel
    stack[2, 14:17, 14:17] = 2  # Original Cell 2
    stack[2, 14:17, 11:14] = 3  # Split from Cell 2

    return stack


class TestCellTracker:
    @pytest.fixture
    def tracker(self):
        config = AnalysisConfig()
        tracker = CellTracker(config)
        tracker.params = TrackingParameters(
            min_overlap_ratio=0.3,
            max_displacement=5,
            min_cell_size=4,
            enable_gap_closing=True,
            max_frame_gap=2
        )
        return tracker

    def test_initialization(self, tracker):
        assert tracker is not None
        assert isinstance(tracker.config, AnalysisConfig)
        assert len(tracker._region_cache) == 0

    def test_invalid_input_dimensions(self, tracker):
        invalid_stack = np.zeros((10, 10))
        with pytest.raises(ValueError):
            tracker.track_cells(invalid_stack)

    def test_progress_callback(self, tracker):
        mock_callback = Mock()
        tracker.set_progress_callback(mock_callback)
        test_stack = create_test_stack()

        tracker.track_cells(test_stack)
        mock_callback.assert_called()

    def test_small_cell_filtering(self, tracker):
        test_stack = create_test_stack()
        tracker.params.min_cell_size = 20

        tracked = tracker.track_cells(test_stack)
        assert np.sum(tracked > 0) == 0

    def test_cell_tracking_basic(self, tracker):
        test_stack = create_test_stack()
        tracked = tracker.track_cells(test_stack)

        assert tracked.shape == test_stack.shape
        assert np.any(tracked > 0), "No cells were tracked"

        cell1_count = len(np.unique(tracked[tracked == 1]))
        assert cell1_count == 1, "Cell 1 ID not consistent across frames"

        frame0_ids = set(np.unique(tracked[0]))
        frame1_ids = set(np.unique(tracked[1]))
        frame0_ids.discard(0)
        frame1_ids.discard(0)
        assert len(frame0_ids.intersection(frame1_ids)) > 0

    def test_gap_closing(self, tracker):
        """Test gap closing when a cell temporarily disappears"""
        params = TrackingParameters(
            min_overlap_ratio=0.1,  # Lower this from default
            max_displacement=5,
            min_cell_size=4,
            enable_gap_closing=True,
            max_frame_gap=2
        )
        tracker.update_parameters(params)

        test_stack = create_test_stack()

        # Add debug prints
        print("Frame 0 cell position:", np.where(test_stack[0] == 1))
        print("Frame 2 cell position:", np.where(test_stack[2] == 1))

        # Remove cell from frame 1
        mask = test_stack[1] == 1
        test_stack[1][mask] = 0

        tracked = tracker.track_cells(test_stack)

        # Add more debug info
        print("Frame 0 tracked IDs:", np.unique(tracked[0][tracked[0] > 0]))
        print("Frame 1 tracked IDs:", np.unique(tracked[1][tracked[1] > 0]))
        print("Frame 2 tracked IDs:", np.unique(tracked[2][tracked[2] > 0]))

        cell1_frame0 = tracked[0, 5:8, 5:8]
        cell1_frame2 = tracked[2, 7:10, 7:10]

        id_frame0 = np.unique(cell1_frame0[cell1_frame0 > 0])[0]
        id_frame2 = np.unique(cell1_frame2[cell1_frame2 > 0])[0]

        assert id_frame0 == id_frame2
class TestCellTrackingWidget:
    @pytest.fixture
    def widget(self, make_napari_viewer):
        """Create widget with mock viewer"""
        viewer = make_napari_viewer()
        data_manager = Mock()
        visualization_manager = Mock()
        widget = CellTrackingWidget(
            viewer,
            data_manager,
            visualization_manager
        )
        return widget

    def test_initialization(self, widget):
        """Test widget initialization"""
        assert widget is not None
        assert widget.tracker is not None
        assert widget.tracking_params is not None

    def test_parameter_controls(self, qtbot, widget):
        """Test parameter control initialization and updates"""
        assert widget.overlap_spin.value() == widget.tracking_params.min_overlap_ratio
        assert widget.displacement_spin.value() == widget.tracking_params.max_displacement

        # Test parameter updates using setValue directly
        widget.overlap_spin.setValue(0.8)
        qtbot.wait(100)  # Wait for event processing
        assert abs(widget.tracking_params.min_overlap_ratio - 0.8) < 0.001

        widget.displacement_spin.setValue(40)
        qtbot.wait(100)
        assert widget.tracking_params.max_displacement == 40

    def test_gap_closing_controls(self, qtbot, widget):
        """Test gap closing control behavior"""
        assert widget.gap_frames_spin.isEnabled() == widget.gap_closing_check.isChecked()

        # Toggle gap closing
        widget.gap_closing_check.setChecked(not widget.gap_closing_check.isChecked())
        qtbot.wait(100)
        assert widget.gap_frames_spin.isEnabled() == widget.gap_closing_check.isChecked()

    @patch('napariCellFlow.cell_tracking_widget.CellTrackingWidget._get_active_labels_layer')
    def test_run_analysis(self, mock_get_layer, qtbot, widget):
        """Test analysis execution"""
        mock_layer = Mock()
        mock_layer.data = np.zeros((3, 10, 10), dtype=np.int32)
        mock_get_layer.return_value = mock_layer

        widget.track_btn.click()
        qtbot.wait(100)
        assert widget.data_manager.tracked_data is not None
        widget.visualization_manager.update_tracking_visualization.assert_called_once()

    def test_reset_parameters(self, qtbot, widget):
        """Test parameter reset functionality"""
        # Change all parameters to non-default values
        widget.overlap_spin.setValue(0.8)  # default is usually ~0.3
        widget.displacement_spin.setValue(50)  # default is usually ~20
        widget.cell_size_spin.setValue(100)  # default is usually ~4
        widget.gap_closing_check.setChecked(not widget.gap_closing_check.isChecked())  # toggle from default
        widget.gap_frames_spin.setValue(10)  # default is usually 2
        qtbot.wait(100)

        # Reset parameters
        widget.reset_btn.click()
        qtbot.wait(100)

        # Verify all parameters are reset to defaults
        default_params = TrackingParameters()
        assert abs(widget.overlap_spin.value() - default_params.min_overlap_ratio) < 0.001
        assert widget.displacement_spin.value() == default_params.max_displacement
        assert widget.cell_size_spin.value() == default_params.min_cell_size
        assert widget.gap_closing_check.isChecked() == default_params.enable_gap_closing
        assert widget.gap_frames_spin.value() == default_params.max_frame_gap