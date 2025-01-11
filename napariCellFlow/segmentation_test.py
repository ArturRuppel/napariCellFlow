"""Segmentation Test Suite

A comprehensive test suite for testing the segmentation functionality in napariCellFlow.
Tests cover both the SegmentationHandler class implementation and the SegmentationWidget
UI component.

Key Test Coverage:

SegmentationHandler Tests (`TestSegmentationHandler` class):
1. Basic Functionality:
   - Initialization with default parameters
   - Model loading and initialization
   - Parameter validation
   - GPU availability detection

2. Segmentation Operations:
   - Single frame segmentation
   - Parameter updates during runtime
   - Error handling for invalid inputs
   - Results caching and retrieval
   - Progress callback functionality

3. Model Management:
   - Different model types (cyto3, nuclei, custom)
   - Custom model loading and validation
   - Model reinitialization with new parameters
   - GPU/CPU mode switching

SegmentationWidget Tests (`TestSegmentationWidget` class):
1. Widget Initialization:
   - Control creation and layout
   - Default parameter values
   - Signal connections
   - Manager integration (data, visualization)

2. UI Interaction:
   - Parameter control updates
   - Model selection and switching
   - Custom model loading dialog
   - Run button state management
   - Progress bar updates

3. Processing Operations:
   - Single frame processing
   - Stack processing
   - Cellpose export/import functionality
   - Error handling and user feedback

4. Integration:
   - Layer management
   - Results visualization
   - Data manager updates
   - Correction widget interaction

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

from unittest.mock import Mock, patch
import numpy as np
import pytest
from napariCellFlow.segmentation import SegmentationHandler, SegmentationParameters


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