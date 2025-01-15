"""Edge Analysis Test Suite

A comprehensive test suite for the edge analysis functionality in napariCellFlow. Tests cover both
the EdgeAnalyzer class implementation and the EdgeAnalysisWidget UI component.

Key Test Coverage:

EdgeAnalyzer Tests (`TestEdgeAnalyzer` class):
1. `test_initialization`: Verifies analyzer initialization with config and empty caches
2. `test_parameter_updates`: Tests parameter update functionality
3. `test_edge_detection`: Tests basic edge detection between cells
4. `test_boundary_ordering`: Validates correct boundary pixel ordering
5. `test_intercalation_detection`: Tests T1 transition detection
6. `test_edge_tracking`: Verifies edge tracking across frames
7. `test_edge_length_calculation`: Tests edge length measurements
8. `test_isolated_edge_filtering`: Tests filtering of isolated edges

EdgeAnalysisWidget Tests (`TestEdgeAnalysisWidget` class):
1. `test_initialization`: Verifies widget creation with mock viewer and managers
2. `test_parameter_controls`: Tests UI control updates
3. `test_visualization_controls`: Validates visualization parameter updates
4. `test_analysis_execution`: Tests running analysis with mock data
5. `test_results_saving_loading`: Tests save/load functionality
6. `test_visualization_generation`: Tests visualization creation

Dependencies:
- pytest for test framework
- numpy for array operations
- Mock/patch from unittest.mock for mocking
- Qt testing utilities for widget interaction
- napariCellFlow package components
"""
import inspect
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import napari
import numpy as np
import pytest
from psygnal import Signal
from qtpy.QtWidgets import (
    QFileDialog
)

from napariCellFlow.data_manager import DataManager
from napariCellFlow.edge_analysis import EdgeAnalyzer
from napariCellFlow.edge_analysis_widget import EdgeAnalysisWidget
from napariCellFlow.structure import (
    EdgeAnalysisParams, EdgeAnalysisResults, CellBoundary,
    VisualizationConfig
)
from napariCellFlow.visualization_manager import VisualizationManager


def create_test_segmentation(shape=(20, 20)):
    """Create a simple test segmentation with four cells forming a cross pattern"""
    seg = np.zeros(shape, dtype=np.int32)

    # Create four cells that share boundaries
    seg[5:10, 8:12] = 1  # Top cell
    seg[10:14, 8:12] = 2  # Bottom cell
    seg[8:12, 5:10] = 3  # Left cell
    seg[8:12, 10:14] = 4  # Right cell

    return seg


def create_test_sequence():
    """Create a sequence showing alternating T1 transitions between frames

    Creates a realistic intercalation sequence where:
    - Even frames: Cells 2&4 share an edge, while cells 1&3 are separated
    - Odd frames: Cells 1&4 share an edge, while cells 2&3 are separated

    The sequence alternates between these two states to test detection of
    repeated T1 transitions.
    """
    sequence = np.zeros((4, 20, 20), dtype=np.int32)

    # Configuration 1: Cells 2&4 share edge, 1&3 separated
    config1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 1, 1, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 1, 1, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 2, 2, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 2, 2, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 2, 2, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 2, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

    # Configuration 2: Cells 1&4 share edge, 2&3 separated
    config2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 1, 1, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 2, 2, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 2, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

    # Create alternating sequence
    sequence[0] = config1
    sequence[1] = config2
    sequence[2] = config1
    sequence[3] = config2

    return sequence

    return sequence

@pytest.fixture
def mock_viewer():
    viewer = Mock()
    viewer.layers = Mock()
    viewer.layers.selection = Mock()
    viewer.layers.selection.active = None
    # Mock the events
    viewer.layers.events = Mock()
    viewer.layers.events.removed = Mock()
    viewer.layers.events.inserted = Mock()
    viewer.layers.events.removed.connect = Mock()
    viewer.layers.events.inserted.connect = Mock()
    viewer.layers.selection.events = Mock()
    viewer.layers.selection.events.changed = Mock()
    viewer.layers.selection.events.changed.connect = Mock()
    return viewer

@pytest.fixture
def mock_vis_manager():
    vis_manager = Mock(spec=VisualizationManager)
    vis_manager.update_edge_visualization = Mock()
    vis_manager.update_edge_analysis_visualization = Mock()
    vis_manager.update_intercalation_visualization = Mock()
    vis_manager.clear_edge_layers = Mock()
    return vis_manager

@pytest.fixture
def mock_data_manager():
    return Mock(spec=DataManager)

@pytest.fixture
def widget(mock_viewer, mock_vis_manager, mock_data_manager, qtbot):
    widget = EdgeAnalysisWidget(
        viewer=mock_viewer,
        data_manager=mock_data_manager,
        visualization_manager=mock_vis_manager
    )
    qtbot.addWidget(widget)
    return widget
class TestEdgeAnalyzer:
    @pytest.fixture
    def analyzer(self):
        params = EdgeAnalysisParams(
            dilation_radius=1,
            min_overlap_pixels=3,
            min_edge_length=2.0,
            filter_isolated=True
        )
        return EdgeAnalyzer(params)

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert isinstance(analyzer.params, EdgeAnalysisParams)
        assert len(analyzer._edge_history) == 0
        assert len(analyzer._edge_groups) == 0

    def test_parameter_updates(self, analyzer):
        """Test parameter update functionality"""
        new_params = EdgeAnalysisParams(
            dilation_radius=2,
            min_overlap_pixels=5,
            min_edge_length=3.0,
            filter_isolated=False
        )

        analyzer.update_parameters(new_params)
        assert analyzer.params.dilation_radius == 2
        assert analyzer.params.min_overlap_pixels == 5
        assert analyzer.params.min_edge_length == 3.0
        assert analyzer.params.filter_isolated is False

    def test_edge_detection(self, analyzer):
        """Test basic edge detection between cells"""
        seg = create_test_segmentation()
        boundaries = analyzer._detect_edges(seg)

        assert len(boundaries) > 0
        for boundary in boundaries:
            assert isinstance(boundary, CellBoundary)
            assert len(boundary.coordinates) >= 2
            assert boundary.length >= analyzer.params.min_edge_length

    def test_boundary_ordering(self, analyzer):
        """Test boundary pixel ordering"""
        seg = create_test_segmentation()
        boundaries = analyzer._detect_edges(seg)

        for boundary in boundaries:
            coords = boundary.coordinates
            # Check that consecutive points are adjacent
            for i in range(len(coords) - 1):
                diff = np.abs(coords[i + 1] - coords[i])
                assert np.all(diff <= 1)

    def test_intercalation_detection(self, analyzer):
        """Test T1 transition detection"""
        print("\n=== Starting intercalation detection test ===")

        # Debug sequence creation
        sequence = create_test_sequence()

        print(f"\nTest sequence shape: {sequence.shape}")
        print("Unique cell IDs in each frame:")
        for i, frame in enumerate(sequence):
            print(f"Frame {i}: {np.unique(frame[frame > 0])}")

        # Add debug prints to key methods in EdgeAnalyzer
        original_detect_edges = analyzer._detect_edges

        def debug_detect_edges(frame_data):
            print(f"\nDetecting edges in frame shape: {frame_data.shape}")
            boundaries = original_detect_edges(frame_data)
            print(f"Found {len(boundaries)} boundaries")
            for b in boundaries:
                print(f"Boundary between cells {b.cell_ids} with length {b.length:.2f}")
            return boundaries

        analyzer._detect_edges = debug_detect_edges

        original_detect_topology = analyzer._detect_topology_changes

        def debug_topology_changes(frame, next_frame):
            print(f"\nAnalyzing topology changes between frames {frame} and {next_frame}")
            G1 = analyzer._frame_graphs[frame]
            G2 = analyzer._frame_graphs[next_frame]
            print(f"Frame {frame} edges: {list(G1.edges())}")
            print(f"Frame {next_frame} edges: {list(G2.edges())}")
            events = original_detect_topology(frame, next_frame)
            print(f"Detected {len(events)} topology changes")
            for event in events:
                print(f"Event: {event.losing_cells} → {event.gaining_cells}")
            return events

        analyzer._detect_topology_changes = debug_topology_changes

        original_validate_t1 = analyzer._validate_t1_transition

        def debug_validate_t1(lost_edge, gained_edge, G1, G2):
            print(f"\nValidating T1 transition:")
            print(f"Lost edge: {lost_edge}")
            print(f"Gained edge: {gained_edge}")
            valid = original_validate_t1(lost_edge, gained_edge, G1, G2)
            print(f"Valid T1? {valid}")
            if not valid:
                all_cells = lost_edge | gained_edge
                print(f"All cells involved: {all_cells}")
                print(f"Number of unique cells: {len(all_cells)}")
                lost_cells = tuple(sorted(lost_edge))
                gained_cells = tuple(sorted(gained_edge))
                edges_to_check = [
                    tuple(sorted((lost_cells[0], gained_cells[0]))),
                    tuple(sorted((lost_cells[0], gained_cells[1]))),
                    tuple(sorted((lost_cells[1], gained_cells[0]))),
                    tuple(sorted((lost_cells[1], gained_cells[1])))
                ]
                print("Required connecting edges:", edges_to_check)
                edges_t = set(tuple(sorted(e)) for e in G1.edges())
                edges_t_plus_1 = set(tuple(sorted(e)) for e in G2.edges())
                print(f"Frame t edges: {edges_t}")
                print(f"Frame t+1 edges: {edges_t_plus_1}")
            return valid

        analyzer._validate_t1_transition = debug_validate_t1

        # Run analysis
        print("\nRunning sequence analysis...")
        results = analyzer.analyze_sequence(sequence)

        # Debug results
        print("\nAnalysis Results:")
        print(f"Total edges found: {len(results.edges)}")
        for edge_id, edge_data in results.edges.items():
            print(f"\nEdge {edge_id}:")
            print(f"Frames: {edge_data.frames}")
            print(f"Cell pairs: {edge_data.cell_pairs}")
            print(f"Number of intercalations: {len(edge_data.intercalations)}")
            for event in edge_data.intercalations:
                print(f"  Intercalation at frame {event.frame}: {event.losing_cells} → {event.gaining_cells}")

        # Original assertions
        assert len(results.edges) > 0
        intercalation_count = sum(
            len(edge.intercalations)
            for edge in results.edges.values()
            if edge.intercalations
        )
        print(f"\nTotal intercalation count: {intercalation_count}")
        assert intercalation_count > 0

    def test_edge_tracking(self, analyzer):
        """Test edge tracking across frames"""
        sequence = create_test_sequence()
        results = analyzer.analyze_sequence(sequence)

        # Check edge continuity
        for edge_id, edge_data in results.edges.items():
            assert len(edge_data.frames) > 0
            assert len(edge_data.cell_pairs) == len(edge_data.frames)
            assert len(edge_data.coordinates) == len(edge_data.frames)

    def test_edge_length_calculation(self, analyzer):
        """Test edge length measurements"""
        seg = create_test_segmentation()
        boundaries = analyzer._detect_edges(seg)

        for boundary in boundaries:
            # Length should be greater than or equal to the straight-line distance
            endpoints_dist = np.sqrt(np.sum((boundary.endpoint2 - boundary.endpoint1) ** 2))
            assert boundary.length >= endpoints_dist

    def test_isolated_edge_filtering(self, analyzer):
        """Test filtering of isolated edges"""
        # Create segmentation with an isolated edge
        seg = create_test_segmentation()
        seg[15:18, 12:15] = 5
        seg[15:18, 15:18] = 6  # Add isolated cell pair
        # import matplotlib.pyplot as plt
        # plt.imshow(seg)
        # plt.show()
        # Test with filtering enabled
        analyzer.params.filter_isolated = True
        boundaries = analyzer._detect_edges(seg)
        cell_ids = {cell_id for boundary in boundaries for cell_id in boundary.cell_ids}
        assert 5 not in cell_ids

        # Test with filtering disabled
        analyzer.params.filter_isolated = False
        boundaries = analyzer._detect_edges(seg)
        cell_ids = {cell_id for boundary in boundaries for cell_id in boundary.cell_ids}
        assert 5 in cell_ids


class TestEdgeAnalysisWidget:
    @pytest.fixture
    def widget(self, make_napari_viewer):
        """Create widget with mock viewer"""
        viewer = make_napari_viewer()
        data_manager = Mock()
        visualization_manager = Mock()
        widget = EdgeAnalysisWidget(
            viewer,
            data_manager,
            visualization_manager
        )
        return widget

    def test_initialization(self, widget):
        """Test widget initialization"""
        assert widget is not None
        assert isinstance(widget.analysis_params, EdgeAnalysisParams)
        assert isinstance(widget.visualization_config, VisualizationConfig)
        assert widget.analyzer is not None

    def test_parameter_controls(self, qtbot, widget):
        """Test parameter control updates"""
        # First check the initial value
        assert widget.analyzer.params.dilation_radius == 1

        # Create a signal spy to verify the signal is emitted
        with qtbot.waitSignal(widget.parameters_updated, timeout=1000):
            widget.dilation_spin.setValue(3)

        # After signal emission, verify both UI and parameters are updated
        assert widget.dilation_spin.value() == 3
        assert widget.analyzer.params.dilation_radius == 3

        # Test another parameter to ensure all updates work
        with qtbot.waitSignal(widget.parameters_updated, timeout=1000):
            widget.overlap_spin.setValue(7)

        assert widget.overlap_spin.value() == 7
        assert widget.analyzer.params.min_overlap_pixels == 7
    def test_visualization_controls(self, qtbot, widget):
        """Test visualization parameter updates"""
        # Toggle visualization options
        widget.tracking_checkbox.setChecked(False)
        qtbot.wait(100)
        assert widget.visualization_config.tracking_plots_enabled is False

        widget.edge_checkbox.setChecked(False)
        qtbot.wait(100)
        assert widget.visualization_config.edge_detection_overlay is False

    @patch('napariCellFlow.edge_analysis_widget.EdgeAnalysisWidget._get_active_labels_layer')
    def test_analysis_execution(self, mock_get_layer, qtbot, widget):
        """Test running analysis

        Note: This test should be updated to properly handle async worker completion
        """
        # Create mock layer with test data
        mock_layer = Mock()
        mock_layer.__class__ = napari.layers.Labels
        test_data = create_test_sequence()
        mock_layer.data = test_data
        mock_get_layer.return_value = mock_layer

        # Create actual QObject with real signals for our mock worker
        from qtpy.QtCore import QObject, Signal
        class MockWorker(QObject):
            progress = Signal(int, str)
            finished = Signal(object)
            error = Signal(Exception)

        mock_worker = MockWorker()

        # Add debug prints to the original handler
        original_handler = widget._handle_analysis_complete

        def debug_handler(results):
            print("\nStarting handler execution")
            try:
                print("Setting current results")
                widget._current_results = results

                print("Setting segmentation data")
                results.set_segmentation_data(widget.segmentation_data)

                print("Setting data manager results")
                widget.data_manager.analysis_results = results

                print("Extracting boundaries")
                boundaries_by_frame = widget._extract_boundaries(results)
                widget._current_boundaries = boundaries_by_frame

                print("Updating visualizations")
                widget.vis_manager.update_edge_visualization(boundaries_by_frame)
                widget.vis_manager.update_intercalation_visualization(results)
                widget.vis_manager.update_edge_analysis_visualization(results)

                print("About to emit processing_completed")
                widget.processing_completed.emit(results)
                print("Emitted processing_completed")

            except Exception as e:
                print(f"Error in handler: {str(e)}")
                raise e

        widget._handle_analysis_complete = debug_handler

        with patch('napariCellFlow.edge_analysis_widget.AnalysisWorker') as mock_worker_class:
            mock_worker_class.return_value = mock_worker

            # Run analysis
            widget.analyze_btn.click()
            qtbot.wait(100)

            # Connect our mock worker's finished signal to the widget's handler
            mock_worker.finished.connect(widget._handle_analysis_complete)

            # Wait for processing_completed signal
            with qtbot.waitSignal(widget.processing_completed, timeout=1000):
                # Create proper EdgeAnalysisResults with all required data
                mock_results = EdgeAnalysisResults(EdgeAnalysisParams())
                mock_edge = Mock()
                mock_edge.intercalations = []
                mock_edge.coordinates = [np.array([[0, 0], [1, 1]])]  # Add required attributes
                mock_edge.frames = [0]
                mock_edge.cell_pairs = [(1, 2)]
                mock_edge.lengths = [1.0]

                mock_results.edges = {"edge1": mock_edge}
                mock_results.set_segmentation_data(test_data)
                widget.segmentation_data = test_data

                # Now emit the signal
                mock_worker.finished.emit(mock_results)

            # Verify results
            assert widget._current_results is not None

    def test_results_saving_loading(self, monkeypatch, qtbot, widget):
        """Test save/load functionality"""
        print("\n=== Starting test_results_saving_loading ===")

        # Track signal emissions
        signals_received = {'processing': False, 'edges': False}

        def on_processing_completed(results):
            print("\nProcessing completed signal received!")
            print(f"Results edges: {len(results.edges) if results else 'None'}")
            signals_received['processing'] = True

        def on_edges_detected(boundaries):
            print("\nEdges detected signal received!")
            print(f"Boundaries: {len(boundaries) if boundaries else 'None'}")
            signals_received['edges'] = True

        # Connect debug handlers to signals
        widget.processing_completed.connect(on_processing_completed)
        widget.edges_detected.connect(on_edges_detected)

        print("\nSetting up mock dialogs...")
        mock_save_dialog = MagicMock()
        mock_save_dialog.exec_.return_value = True
        mock_save_dialog.selectedFiles.return_value = [str(Path('/tmp/save_test.pkl'))]

        mock_load_dialog = MagicMock()
        mock_load_dialog.exec_.return_value = True
        mock_load_dialog.selectedFiles.return_value = [str(Path('/tmp/load_test.pkl'))]

        current_mode = {'mode': 'save'}

        def create_mock_dialog(*args, **kwargs):
            print(f"\nCreating mock dialog with args: {args}")
            print(f"Dialog kwargs: {kwargs}")

            stack = inspect.stack()
            calling_method = stack[1].function if len(stack) > 1 else None
            print(f"Calling method: {calling_method}")

            if calling_method == '_get_save_path' or kwargs.get('acceptMode') == QFileDialog.AcceptSave:
                current_mode['mode'] = 'save'
                print("Creating SAVE dialog")
                return mock_save_dialog
            else:
                current_mode['mode'] = 'load'
                print("Creating LOAD dialog")
                return mock_load_dialog

        print("Setting up mock dialog class...")
        mock_dialog_class = MagicMock(side_effect=create_mock_dialog)
        monkeypatch.setattr('napariCellFlow.edge_analysis_widget.QFileDialog', mock_dialog_class)

        print("\nSetting up mock results...")
        mock_results = EdgeAnalysisResults(widget.analysis_params)
        mock_edge = Mock()
        mock_edge.frames = [0]
        mock_edge.cell_pairs = [(1, 2)]
        mock_edge.coordinates = [np.array([[0, 0], [1, 1]])]
        mock_edge.lengths = [1.0]
        mock_edge.intercalations = []
        mock_results.edges = {'edge1': mock_edge}
        print(f"Mock results created with {len(mock_results.edges)} edges")

        test_data = create_test_sequence()
        mock_results.set_segmentation_data(test_data)

        print("\nSetting up widget state...")
        widget._current_results = mock_results
        widget.segmentation_data = test_data

        print("\nConfiguring data manager...")
        widget.data_manager = MagicMock()
        widget.data_manager.last_directory = Path('/tmp')
        widget.data_manager.analysis_results = mock_results

        def mock_load_results(path):
            print(f"\nMock load_analysis_results called with path: {path}")
            loaded_results = EdgeAnalysisResults(widget.analysis_params)
            loaded_edge = Mock()
            loaded_edge.frames = [0, 1]
            loaded_edge.cell_pairs = [(1, 2), (2, 3)]
            loaded_edge.coordinates = [np.array([[0, 0], [1, 1]]), np.array([[1, 1], [2, 2]])]
            loaded_edge.lengths = [1.0, 1.414]
            loaded_edge.intercalations = []
            loaded_results.edges = {'loaded_edge': loaded_edge}
            loaded_results.set_segmentation_data(test_data)
            widget.data_manager.analysis_results = loaded_results
            print("Mock results loaded successfully")
            return loaded_results

        widget.data_manager.load_analysis_results = MagicMock(side_effect=mock_load_results)
        widget.data_manager.save_analysis_results = MagicMock()

        print("\nSetting up UI state...")
        widget.save_btn.setEnabled(True)
        widget._update_ui_state()  # Important: This updates ALL button states
        print(f"Save button enabled state: {widget.save_btn.isEnabled()}")

        print("\nInitial widget state:")
        print(f"Has current results: {widget._current_results is not None}")
        print(f"Has segmentation data: {widget.segmentation_data is not None}")
        print(f"Save button enabled: {widget.save_btn.isEnabled()}")
        print(f"Save button connected: {widget.save_btn.receivers(widget.save_btn.clicked)}")

        print("\nTesting save functionality...")
        with patch('pathlib.Path.exists', return_value=True):
            print("About to click save button...")
            original_get_save_path = widget._get_save_path
            widget._get_save_path = MagicMock(return_value=Path('/tmp/save_test.pkl'))

            with qtbot.waitSignal(widget.save_btn.clicked, timeout=1000):
                widget.save_btn.click()
            qtbot.wait(100)

            # Restore original method
            widget._get_save_path = original_get_save_path

            print("\nPost-save state:")
            print(f"Save method called: {widget.data_manager.save_analysis_results.called}")
            print(f"Dialog mode: {current_mode['mode']}")
            print(f"Dialog class called: {mock_dialog_class.called}")

            assert widget.data_manager.save_analysis_results.called, "Save operation was not triggered"

            print("\nClearing widget state...")
            widget._current_results = None
            widget.segmentation_data = None
            widget._current_boundaries = None
            widget._update_ui_state()

            print("\nTesting load functionality...")
            widget.load_btn.click()

            # Wait for signals and process events
            for _ in range(10):  # Try a few times
                qtbot.wait(100)
                if signals_received['edges'] and signals_received['processing']:
                    break

            print("\nVerifying loaded state...")
            assert widget._current_results is not None, "Results were not loaded"
            assert widget.segmentation_data is not None, "Segmentation data was not loaded"
            assert widget._current_boundaries is not None, "Boundaries were not computed"
            assert len(widget._current_results.edges) > 0, "No edges were loaded"

            print("\nVerifying visualizations...")
            assert widget.vis_manager.update_edge_visualization.called
            assert widget.vis_manager.update_intercalation_visualization.called
            assert widget.vis_manager.update_edge_analysis_visualization.called

            print("\nVerifying final UI state...")
            assert widget.save_btn.isEnabled()
            assert widget.generate_vis_btn.isEnabled()

            print("\nTest completed successfully")


    @pytest.mark.parametrize("visualization_config", [
        {"tracking": True, "edge": False, "intercalation": False, "length": False, "gifs": False}
    ])
    @patch('napariCellFlow.edge_analysis_widget.QThread')
    @patch('napariCellFlow.edge_analysis_visualization.Visualizer')
    @patch('napariCellFlow.edge_analysis_widget.VisualizationWorker')
    @patch('napariCellFlow.edge_analysis_widget.EdgeAnalysisWidget._get_output_directory')
    def test_visualization_generation(self, mock_get_dir, mock_worker_class, mock_visualizer_class, mock_qthread, visualization_config, qtbot, widget, tmp_path):
        """Test that visualization generation process works correctly"""
        print("\n=== Starting test_visualization_generation ===")

        # Create mock visualizer instance
        mock_visualizer_instance = Mock()
        mock_visualizer_class.return_value = mock_visualizer_instance

        # Replace widget's visualizer with our mock
        widget.visualizer = mock_visualizer_instance

        # Setup test data and mock results
        print("\nSetting up test data...")
        test_data = create_test_sequence()
        widget._current_results = EdgeAnalysisResults(widget.analysis_params)
        mock_edge = Mock()
        mock_edge.frames = [0, 1]
        mock_edge.cell_pairs = [(1, 2), (2, 3)]
        mock_edge.coordinates = [np.array([[0, 0], [1, 1]]), np.array([[1, 1], [2, 2]])]
        mock_edge.lengths = [1.0, 1.414]
        mock_edge.intercalations = [Mock(frame=0, losing_cells=(1, 2), gaining_cells=(2, 3))]
        widget._current_results.edges = {'edge1': mock_edge}
        widget._current_results.set_segmentation_data(test_data)
        widget.segmentation_data = test_data

        # Set output directory
        output_dir = tmp_path / "visualization_test"
        mock_get_dir.return_value = output_dir

        # Create mock worker that will emit signals
        print("\nSetting up mock worker...")
        mock_worker = Mock()
        mock_worker.progress = Signal(int, str)
        mock_worker.finished = Signal()
        mock_worker.error = Signal(Exception)
        mock_worker_class.return_value = mock_worker

        # Configure visualization options
        widget.tracking_checkbox.setChecked(visualization_config["tracking"])
        widget.edge_checkbox.setChecked(visualization_config["edge"])
        widget.intercalation_checkbox.setChecked(visualization_config["intercalation"])
        widget.edge_length_checkbox.setChecked(visualization_config["length"])
        widget.example_gifs_checkbox.setChecked(visualization_config["gifs"])
        qtbot.wait(100)

        # Generate visualizations
        print("\nGenerating visualizations...")
        widget._generate_visualizations()
        qtbot.wait(100)

        # Verify worker was created with correct parameters
        print("\nChecking worker creation...")
        mock_worker_class.assert_called_once()

        args, _ = mock_worker_class.call_args
        print(f"\nWorker creation args: {args}")

        # Verify the arguments
        visualizer, results, output_directory = args
        assert visualizer == mock_visualizer_instance
        assert isinstance(results, EdgeAnalysisResults)
        assert results == widget._current_results
        assert output_directory == output_dir

        # Verify visualization thread setup
        assert widget._visualization_thread is not None
        assert widget._visualization_worker is not None
        assert widget._visualization_worker == mock_worker

        # Clean up
        if widget._visualization_thread and widget._visualization_thread.isRunning():
            widget._visualization_thread.quit()
            widget._visualization_thread.wait()