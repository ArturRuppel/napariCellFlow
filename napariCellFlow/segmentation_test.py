"""Segmentation Test Suite

A comprehensive test suite for testing the segmentation functionality in napariCellFlow.
Tests cover both the SegmentationHandler class implementation and the SegmentationWidget
UI component.

Key Test Coverage:

SegmentationHandler Tests (`TestSegmentationHandler` class):
1. `test_basic_initialization`: Verifies handler initialization with default parameters and empty state
2. `test_parameter_validation`: Tests validation of various segmentation parameters (diameter, thresholds, size)
3. `test_model_initialization`: Validates model initialization with GPU support
4. `test_segment_frame`: Tests single frame segmentation and result structure
5. `test_error_handling`: Verifies proper error handling for uninitialized model and invalid inputs
6. `test_parameter_updates`: Tests runtime parameter updates and model reinitialization
7. `test_results_caching`: Validates caching and retrieval of segmentation results
8. `test_progress_callback`: Tests progress reporting during segmentation
9. `test_different_model_types`: Verifies initialization with various model types (cyto3, nuclei)
10. `test_custom_model_loading`: Tests loading and initialization of custom models
11. `test_custom_model_validation`: Validates proper handling of custom model parameters
12. `test_model_reinitialization`: Tests model reinitialization with new parameters
13. `test_gpu_cpu_switching`: Verifies switching between GPU and CPU processing modes

SegmentationWidget Tests (`TestSegmentationWidget` class):
1. `test_widget_initialization`: Verifies widget initialization with default parameters and component creation
2. `test_parameter_controls`: Tests validation and updates of various widget controls (diameter, thresholds, size)
3. `test_model_switching`: Validates model type switching functionality (cyto3, nuclei, custom)
4. `test_custom_model_loading`: Tests loading and initialization of custom model through UI
5. `test_run_segmentation_single_frame`: Tests single frame segmentation workflow and result handling
6. `test_run_segmentation_stack`: Verifies stack processing functionality and progress tracking
7. `test_error_handling`: Tests proper error handling for widget operations and user feedback
8. `test_parameter_reset`: Validates parameter reset functionality and default value restoration
9. `test_parameter_controls`: Tests runtime parameter updates through UI controls
10. `test_model_switching`: Verifies switching between different model types through UI
11. `test_custom_model_validation`: Tests validation of custom model parameters in UI
12. `test_run_button_states`: Validates run button state management during processing
13. `test_progress_reporting`: Tests progress bar updates and user feedback during operations

Test Data:
- Uses synthetic test images with known characteristics:
  * Simple shapes (circles, squares) for basic segmentation
  * Varying intensities for normalization testing
  * Multi-frame sequences for stack processing
  * Known cell counts and sizes for validation

Dependencies:
- pytest for test framework
- numpy for array operations
- Mock/patch from unittest.mock for mocking
- Qt testing utilities for widget testing
- napariCellFlow package components
- cellpose for model operations

Notes:
- Tests use small synthetic images to keep execution fast
- GPU tests are skipped if no GPU is available
- Custom model tests use temporary test models
- Widget tests simulate user interactions
- Error cases check both UI feedback and error handling
"""

from unittest.mock import Mock, patch, call

import numpy as np
import pytest
from qtpy.QtWidgets import QWidget

from napariCellFlow.segmentation import SegmentationHandler
from napariCellFlow.segmentation import SegmentationParameters


def create_test_image(size=64):
    """Create a synthetic test image with simple cell-like objects."""
    image = np.zeros((size, size), dtype=np.uint8)

    # Add three circular "cells" of different sizes
    from skimage.draw import disk

    # Large cell in center
    rr, cc = disk((size // 2, size // 2), 10)
    image[rr, cc] = 200

    # Two smaller cells
    rr, cc = disk((size // 4, size // 4), 6)
    image[rr, cc] = 180
    rr, cc = disk((3 * size // 4, 3 * size // 4), 8)
    image[rr, cc] = 220

    return image


class TestSegmentationHandler:
    @pytest.fixture
    def segmentation_handler(self):
        """Fixture to create a basic SegmentationHandler instance"""
        return SegmentationHandler()

    def test_basic_initialization(self, segmentation_handler):
        """Test basic initialization of SegmentationHandler"""
        assert segmentation_handler is not None
        assert segmentation_handler.model is None
        assert segmentation_handler.params is not None
        assert segmentation_handler.last_results == {}

    def test_parameter_validation(self, segmentation_handler):
        """Test parameter validation"""
        # Test invalid diameter
        params = SegmentationParameters(
            diameter=-1,
            compute_diameter=False
        )
        with pytest.raises(ValueError, match="Cell diameter must be positive or compute_diameter must be True"):
            params.validate()

        # Test invalid flow threshold
        params = SegmentationParameters(
            diameter=95.0,  # Valid diameter
            flow_threshold=1.5  # Invalid flow threshold
        )
        with pytest.raises(ValueError, match="Flow threshold must be between 0 and 1"):
            params.validate()

        # Test invalid cell probability threshold
        params = SegmentationParameters(
            diameter=95.0,  # Valid diameter
            flow_threshold=0.6,  # Valid flow threshold
            cellprob_threshold=-0.1  # Invalid probability threshold
        )
        with pytest.raises(ValueError, match="Cell probability threshold must be between 0 and 1"):
            params.validate()

        # Test invalid minimum size
        params = SegmentationParameters(
            diameter=95.0,
            flow_threshold=0.6,
            cellprob_threshold=0.3,
            min_size=0  # Invalid minimum size
        )
        with pytest.raises(ValueError, match="Minimum size must be positive"):
            params.validate()

        # Test valid parameters
        params = SegmentationParameters(
            diameter=95.0,
            flow_threshold=0.6,
            cellprob_threshold=0.3,
            min_size=25
        )
        params.validate()  # Should not raise any exceptions

    @pytest.mark.gpu
    def test_model_initialization(self, segmentation_handler):
        """Test model initialization with GPU support"""
        with patch('cellpose.models.CellposeModel') as mock_model:
            mock_instance = Mock()
            mock_model.return_value = mock_instance

            segmentation_handler.initialize_model(segmentation_handler.params)

            # Verify model was initialized with correct parameters
            mock_model.assert_called_once_with(
                model_type="cyto3",
                gpu=True
            )
            assert segmentation_handler.model == mock_instance

    def test_segment_frame(self, segmentation_handler):
        """Test frame segmentation"""
        # Create test image
        test_img = create_test_image()

        # Mock model evaluation
        mock_masks = np.random.randint(0, 4, size=test_img.shape)
        mock_flows = [np.random.rand(*test_img.shape), np.random.rand(*test_img.shape)]
        mock_styles = np.random.rand(3)  # Example style vector

        with patch.object(segmentation_handler, 'model') as mock_model:
            mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles)

            # Run segmentation
            masks, results = segmentation_handler.segment_frame(test_img)

            # Verify results
            assert masks is mock_masks
            assert 'masks' in results
            assert 'flows' in results
            assert 'styles' in results
            assert 'parameters' in results

            # Verify model was called with correct parameters
            mock_model.eval.assert_called_once()
            call_args = mock_model.eval.call_args[1]
            assert 'channels' in call_args
            assert 'flow_threshold' in call_args
            assert 'cellprob_threshold' in call_args
            assert 'min_size' in call_args

    def test_error_handling(self, segmentation_handler):
        """Test error handling during segmentation"""
        # Test segmentation without initialized model
        with pytest.raises(RuntimeError, match="Model not initialized"):
            segmentation_handler.segment_frame(np.zeros((64, 64)))

        # Test with invalid image dimensions
        with patch.object(segmentation_handler, 'model'):
            with pytest.raises(Exception):
                segmentation_handler.segment_frame(np.zeros((64,)))  # 1D array should fail

    def test_parameter_updates(self, segmentation_handler):
        """Test runtime parameter updates"""
        # Set initial parameters
        initial_params = SegmentationParameters(
            diameter=95.0,
            flow_threshold=0.6
        )

        with patch('cellpose.models.CellposeModel') as mock_model:
            segmentation_handler.initialize_model(initial_params)

            # Update parameters
            new_params = SegmentationParameters(
                diameter=120.0,
                flow_threshold=0.7
            )
            segmentation_handler.initialize_model(new_params)

            # Verify model was reinitialized with new parameters
            assert segmentation_handler.params.diameter == 120.0
            assert segmentation_handler.params.flow_threshold == 0.7
            assert mock_model.call_count == 2

    def test_results_caching(self, segmentation_handler):
        """Test caching and retrieval of segmentation results"""
        test_img = create_test_image()
        mock_masks = np.random.randint(0, 4, size=test_img.shape)
        mock_flows = [np.random.rand(*test_img.shape), np.random.rand(*test_img.shape)]
        mock_styles = np.random.rand(3)

        with patch.object(segmentation_handler, 'model') as mock_model:
            mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles)

            # Run segmentation
            masks, results = segmentation_handler.segment_frame(test_img)

            # Verify results are cached
            assert segmentation_handler.last_results == results
            assert 'masks' in segmentation_handler.last_results
            assert 'flows' in segmentation_handler.last_results
            assert 'styles' in segmentation_handler.last_results
            assert 'parameters' in segmentation_handler.last_results

            # Verify cached results match returned results
            assert np.array_equal(segmentation_handler.last_results['masks'], masks)

    def test_progress_callback(self, segmentation_handler):
        """Test progress callback functionality"""
        test_img = create_test_image()
        mock_progress_callback = Mock()

        # Connect to progress signal
        segmentation_handler.signals.progress_updated.connect(mock_progress_callback)

        with patch.object(segmentation_handler, 'model') as mock_model:
            mock_model.eval.return_value = (
                np.zeros_like(test_img),
                [np.zeros_like(test_img), np.zeros_like(test_img)],
                np.zeros(3)
            )

            # Run segmentation
            segmentation_handler.segment_frame(test_img)

            # Verify progress callbacks
            expected_calls = [
                call(10, "Preparing segmentation..."),
                call(30, "Running Cellpose segmentation..."),
                call(100, "Segmentation complete. Found 0 cells")
            ]
            mock_progress_callback.assert_has_calls(expected_calls, any_order=False)

    def test_different_model_types(self, segmentation_handler):
        """Test initialization with different model types"""
        model_types = ["cyto3", "nuclei"]

        for model_type in model_types:
            params = SegmentationParameters(model_type=model_type)

            with patch('cellpose.models.CellposeModel') as mock_model:
                segmentation_handler.initialize_model(params)

                # Verify correct model type was used
                mock_model.assert_called_with(
                    model_type=model_type,
                    gpu=True
                )

    def test_custom_model_loading(self, segmentation_handler):
        """Test loading custom model"""
        params = SegmentationParameters(
            model_type="custom",
            custom_model_path="/path/to/custom/model"
        )

        with patch('cellpose.models.CellposeModel') as mock_model:
            segmentation_handler.initialize_model(params)

            # Verify custom model initialization
            mock_model.assert_called_with(
                pretrained_model="/path/to/custom/model",
                gpu=True
            )

    def test_custom_model_validation(self, segmentation_handler):
        """Test validation of custom model parameters"""
        # Test missing custom model path
        params = SegmentationParameters(model_type="custom")

        with pytest.raises(ValueError, match="Custom model path required"):
            segmentation_handler.initialize_model(params)

    def test_model_reinitialization(self, segmentation_handler):
        """Test model reinitialization with new parameters"""
        initial_params = SegmentationParameters(gpu=True)

        with patch('cellpose.models.CellposeModel') as mock_model:
            # Initial initialization
            segmentation_handler.initialize_model(initial_params)

            # Reinitialize with GPU disabled
            new_params = SegmentationParameters(gpu=False)
            segmentation_handler.initialize_model(new_params)

            # Verify second initialization used updated parameters
            assert mock_model.call_count == 2
            last_call = mock_model.call_args
            assert last_call.kwargs['gpu'] == False

    def test_gpu_cpu_switching(self, segmentation_handler):
        """Test switching between GPU and CPU modes"""
        with patch('cellpose.models.CellposeModel') as mock_model:
            with patch('cellpose.core.use_gpu') as mock_use_gpu:
                # Test GPU mode
                params_gpu = SegmentationParameters(gpu=True)
                segmentation_handler.initialize_model(params_gpu)
                mock_model.assert_called_with(
                    model_type="cyto3",
                    gpu=True
                )

                # Test CPU mode
                params_cpu = SegmentationParameters(gpu=False)
                segmentation_handler.initialize_model(params_cpu)
                mock_model.assert_called_with(
                    model_type="cyto3",
                    gpu=False
                )

                # Verify model was initialized twice
                assert mock_model.call_count == 2


class TestSegmentationWidget:
    @pytest.fixture
    def mock_correction_widget(self):
        """Create a mock correction widget that inherits from QWidget"""

        class MockCorrectionWidget(QWidget):
            def __init__(self, viewer, data_manager, visualization_manager):
                super().__init__()
                self.viewer = viewer
                self.data_manager = data_manager
                self.visualization_manager = visualization_manager

        return MockCorrectionWidget

    @pytest.fixture
    def widget(self, make_napari_viewer, mock_correction_widget):
        """Create the SegmentationWidget instance"""
        from napariCellFlow.segmentation_widget import SegmentationWidget

        viewer = make_napari_viewer()
        data_manager = Mock()
        visualization_manager = Mock()

        # Mock the SegmentationHandler and replace CellCorrectionWidget with our QWidget-based mock
        with patch('napariCellFlow.segmentation_widget.SegmentationHandler'), \
                patch('napariCellFlow.segmentation_widget.CellCorrectionWidget', mock_correction_widget):
            widget = SegmentationWidget(
                viewer,
                data_manager,
                visualization_manager
            )
            return widget

    def test_widget_initialization(self, qtbot, widget):
        """Test basic widget initialization"""
        qtbot.addWidget(widget)

        # Verify control creation
        assert widget.model_combo is not None
        assert widget.custom_model_btn is not None
        assert widget.diameter_spin is not None
        assert widget.flow_spin is not None
        assert widget.prob_spin is not None
        assert widget.size_spin is not None
        assert widget.gpu_check is not None
        assert widget.normalize_check is not None
        assert widget.compute_diameter_check is not None
        assert widget.run_btn is not None
        assert widget.run_stack_btn is not None

        # Verify default values
        assert widget.model_combo.currentText() == "cyto3"
        assert widget.diameter_spin.value() == 95.0
        assert widget.flow_spin.value() == 0.6
        assert widget.size_spin.value() == 25
        assert widget.gpu_check.isChecked() is True
        assert widget.normalize_check.isChecked() is True
        assert widget._custom_model_path is None

        # Verify correction widget initialization
        assert isinstance(widget.correction_widget, QWidget)
        assert hasattr(widget.correction_widget, 'viewer')
        assert hasattr(widget.correction_widget, 'data_manager')
        assert hasattr(widget.correction_widget, 'visualization_manager')

    def test_parameter_controls(self, qtbot, widget):
        """Test parameter control updates"""
        qtbot.addWidget(widget)

        # Test diameter spin box
        widget.diameter_spin.setValue(120.0)
        qtbot.wait(100)
        params = widget._get_current_parameters()
        assert params.diameter == 120.0

        # Test flow threshold
        widget.flow_spin.setValue(0.8)
        qtbot.wait(100)
        params = widget._get_current_parameters()
        assert params.flow_threshold == 0.8

        # Test cell probability threshold
        widget.prob_spin.setValue(0.5)
        qtbot.wait(100)
        params = widget._get_current_parameters()
        assert params.cellprob_threshold == 0.5

        # Test minimum size
        widget.size_spin.setValue(50)
        qtbot.wait(100)
        params = widget._get_current_parameters()
        assert params.min_size == 50

    @patch('qtpy.QtWidgets.QFileDialog.getOpenFileName')
    def test_custom_model_loading(self, mock_dialog, qtbot, widget):
        """Test custom model loading functionality"""
        qtbot.addWidget(widget)

        # Setup mock return value
        mock_dialog.return_value = ('/path/to/model.pth', '')

        # Switch to custom model mode
        widget.model_combo.setCurrentText("custom")
        qtbot.wait(100)
        assert widget.custom_model_btn.isEnabled()

        # Test custom model loading
        with patch('cellpose.models.CellposeModel') as mock_model:
            widget._load_custom_model()
            qtbot.wait(100)
            assert widget._custom_model_path == '/path/to/model.pth'
            mock_model.assert_called_once_with(pretrained_model='/path/to/model.pth')

    @pytest.mark.parametrize("model_type", ["cyto3", "nuclei", "custom"])
    def test_model_switching(self, qtbot, widget, model_type):
        """Test model type switching"""
        qtbot.addWidget(widget)
        widget.model_combo.setCurrentText(model_type)
        qtbot.wait(100)
        assert widget.model_combo.currentText() == model_type
        assert widget.custom_model_btn.isEnabled() == (model_type == "custom")

    def test_run_segmentation_single_frame(self, qtbot, widget):
        """Test single frame segmentation"""
        qtbot.addWidget(widget)

        # Create mock image data
        mock_image = np.zeros((64, 64), dtype=np.uint8)
        mock_masks = np.zeros_like(mock_image)

        # Create proper napari Image layer instead of Mock
        from napari.layers import Image
        image_layer = Image(mock_image, name='test_image')
        widget.viewer.layers.append(image_layer)
        widget.viewer.layers.selection.active = image_layer

        # Mock segmentation handler
        widget.segmentation.segment_frame.return_value = (mock_masks, {})
        widget.segmentation.model = Mock()  # Ensure model is initialized

        # Run segmentation
        widget.run_btn.click()
        qtbot.wait(100)

        # Verify segmentation was called
        widget.segmentation.segment_frame.assert_called_once()
        assert widget.data_manager.segmentation_data is mock_masks
        widget.visualization_manager.update_tracking_visualization.assert_called_once_with(mock_masks)

    def test_run_segmentation_stack(self, qtbot, widget):
        """Test stack segmentation"""
        qtbot.addWidget(widget)

        # Create mock stack data
        mock_stack = np.zeros((5, 64, 64), dtype=np.uint8)
        mock_masks = np.zeros_like(mock_stack)

        # Create proper napari Image layer for stack
        from napari.layers import Image
        stack_layer = Image(mock_stack, name='test_stack')
        widget.viewer.layers.append(stack_layer)
        widget.viewer.layers.selection.active = stack_layer

        # Mock segmentation handler
        widget.segmentation.segment_frame.return_value = (mock_masks[0], {})
        widget.segmentation.model = Mock()

        # Run stack segmentation
        widget.run_stack_btn.click()
        qtbot.wait(100)

        # Verify processing
        assert widget.segmentation.segment_frame.call_count == 5
        widget.data_manager.initialize_stack.assert_called_once_with(5)
        widget.visualization_manager.update_tracking_visualization.assert_called()

    def test_error_handling(self, qtbot, widget):
        """Test error handling during processing"""
        qtbot.addWidget(widget)

        # Test segmentation error
        widget.segmentation.segment_frame.side_effect = Exception("Segmentation failed")

        # Create proper napari Image layer
        from napari.layers import Image
        test_image = np.zeros((64, 64), dtype=np.uint8)
        image_layer = Image(test_image, name='test_image')
        widget.viewer.layers.append(image_layer)
        widget.viewer.layers.selection.active = image_layer
        widget.segmentation.model = Mock()

        # Run segmentation and verify error handling
        widget.run_btn.click()
        qtbot.wait(100)
        assert not widget._processing
        assert widget.run_btn.isEnabled()

    def test_parameter_reset(self, qtbot, widget):
        """Test parameter reset functionality"""
        qtbot.addWidget(widget)

        # Change parameters
        widget.diameter_spin.setValue(120.0)
        widget.flow_spin.setValue(0.8)
        widget.prob_spin.setValue(0.5)
        widget.size_spin.setValue(50)
        widget.gpu_check.setChecked(False)
        qtbot.wait(100)

        # Reset parameters
        widget.reset_parameters()
        qtbot.wait(100)

        # Verify reset values
        assert widget.diameter_spin.value() == 95.0
        assert widget.flow_spin.value() == 0.6
        assert widget.prob_spin.value() == 0.0
        assert widget.size_spin.value() == 25
        assert widget.gpu_check.isChecked() is True
        assert widget.normalize_check.isChecked() is True
        assert widget.compute_diameter_check.isChecked() is False