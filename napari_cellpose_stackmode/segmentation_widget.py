import json
from datetime import datetime
from typing import Optional
import logging
from pathlib import Path
from qtpy.QtCore import Qt

import napari
import numpy as np
from napari.layers import Image
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFormLayout, QProgressBar,
    QPushButton, QFileDialog, QMessageBox, QProgressDialog,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
)
from tifffile import tifffile

from .cell_correction_widget import CellCorrectionWidget
from .segmentation import SegmentationHandler, SegmentationParameters
from .data_manager import DataManager
from .visualization_manager import VisualizationManager
from .segmentation_state import SegmentationStateManager
from .debug_logging import logger, log_operation


class SegmentationWidget(QWidget):
    """Widget for controlling cell segmentation in napari with enhanced frame handling"""

    segmentation_completed = Signal(np.ndarray, dict)  # masks, metadata
    segmentation_failed = Signal(str)
    progress_updated = Signal(int, str)  # progress percentage, message

    def __init__(self, viewer: "napari.Viewer", data_manager: DataManager,
                 visualization_manager: VisualizationManager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.vis_manager = visualization_manager

        # Track processing state
        self._current_frame = 0
        self._processing = False
        self._batch_processing = False
        self._last_error = None

        # Initialize handlers
        self.state_manager = SegmentationStateManager()
        self.segmentation = SegmentationHandler()

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

    @log_operation
    def _run_segmentation(self, preserve_existing=False):
        """Run segmentation on current frame with enhanced error handling."""
        if self._processing:
            logger.warning("Segmentation already in progress")
            return

        self._processing = True
        self._update_ui_state()

        try:
            if not self._ensure_model_initialized():
                return

            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ValueError("No image layer selected")

            # Get number of frames and current frame
            num_frames = image_layer.data.shape[0] if image_layer.data.ndim > 2 else 1
            self._current_frame = int(self.viewer.dims.point[0])

            # Check if we're in preview mode
            preview_mode = any(layer.name == 'Preview' for layer in self.viewer.layers)

            # Case 1: Preview mode without initialized stack
            if preview_mode and not self.data_manager._initialized:
                QMessageBox.warning(
                    self,
                    "Initialization Required",
                    "Please initialize segmentation on the full stack first before segmenting preview frames.\n\n"
                    "Tip: Run 'Initialize Stack' or segment the unprocessed stack first."
                )
                return

            # Case 2: Regular segmentation - initialize if needed
            if not preview_mode and not self.data_manager._initialized:
                logger.debug(f"Initializing data manager with {num_frames} frames")
                self.data_manager.initialize_stack(num_frames)

            # Get frame data
            frame_data = (image_layer.data[self._current_frame]
                          if image_layer.data.ndim > 2 else image_layer.data)

            # Run segmentation
            masks, metadata = self.segmentation.segment_frame(frame_data)
            self.data_manager.segmentation_data = (masks, self._current_frame)
            self.vis_manager.update_tracking_visualization((masks, self._current_frame))

            self.progress_bar.setValue(100)
            self.status_label.setText(f"Frame {self._current_frame + 1} completed")
            self.segmentation_completed.emit(masks, metadata)

        except Exception as e:
            self._on_segmentation_failed(str(e))
        finally:
            self._processing = False
            self._last_error = None
            self._update_ui_state()

    @log_operation
    def _run_stack_segmentation(self, preserve_existing=False):
        """Run segmentation on entire stack with enhanced progress tracking and error handling."""
        if self._processing or self._batch_processing:
            logger.warning("Processing already in progress")
            return

        try:
            self._batch_processing = True
            self._update_ui_state()

            # Validate prerequisites
            if not self._ensure_model_initialized():
                return

            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ValueError("No image layer selected")

            if image_layer.data.ndim < 3:
                raise ValueError("Selected layer is not a stack")

            # Add these debug logs
            logger.debug(f"Image layer shape: {image_layer.data.shape}")
            num_frames = image_layer.data.shape[0]
            logger.debug(f"Detected {num_frames} frames from image data")

            # Initialize state with correct number of frames
            num_frames = image_layer.data.shape[0]
            logger.debug(f"Stack segmentation starting with {num_frames} frames")

            # Always reinitialize data manager with correct number of frames
            self.data_manager.initialize_stack(num_frames)
            logger.debug("Data manager initialized")

            # Create progress dialog
            progress = QProgressDialog("Processing image stack...", "Cancel", 0, num_frames, self)
            progress.setWindowModality(Qt.WindowModal)

            try:
                for frame_idx in range(num_frames):
                    logger.debug(f"Processing frame {frame_idx}/{num_frames}")
                    if progress.wasCanceled():
                        raise InterruptedError("Processing canceled by user")

                    # Update progress
                    progress.setValue(frame_idx)
                    progress.setLabelText(f"Processing frame {frame_idx + 1}/{num_frames}")

                    # Process frame
                    self._current_frame = frame_idx
                    frame_data = image_layer.data[frame_idx]
                    logger.debug(f"Frame data shape: {frame_data.shape}")

                    try:
                        # Run segmentation on frame
                        masks, metadata = self.segmentation.segment_frame(frame_data)
                        logger.debug(f"Segmentation complete for frame {frame_idx}")
                        logger.debug(f"Masks shape: {masks.shape}")

                        # Update data manager
                        self.data_manager.segmentation_data = (masks, frame_idx)
                        logger.debug(f"Data manager updated for frame {frame_idx}")

                        # Update visualization
                        self.vis_manager.update_tracking_visualization(
                            (masks, frame_idx)
                        )
                        logger.debug(f"Visualization updated for frame {frame_idx}")

                    except Exception as e:
                        logger.error(f"Error processing frame {frame_idx}: {e}")
                        raise ValueError(f"Failed on frame {frame_idx}: {str(e)}")

            finally:
                progress.close()

            # Verify final state
            if not self.data_manager.validate_stack_consistency():
                raise ValueError("Stack consistency validation failed after processing")

            self.status_label.setText("Stack processing completed")
            self.progress_bar.setValue(100)

        except InterruptedError as e:
            logger.info(f"Stack processing interrupted: {str(e)}")
            QMessageBox.information(self, "Processing Canceled", "Stack processing was canceled")

        except Exception as e:
            self._on_segmentation_failed(str(e))

        finally:
            self._batch_processing = False
            self._last_error = None
            self._update_ui_state()
    def _ensure_model_initialized(self) -> bool:
        """Make sure model is initialized before running segmentation"""
        if self.segmentation.model is None:
            try:
                self._initialize_model()
                return True
            except Exception as e:
                self._on_segmentation_failed(f"Model initialization failed: {str(e)}")
                return False
        return True

    def _on_segmentation_failed(self, error_msg: str):
        """Enhanced error handling with proper state cleanup."""
        logger.error(f"Segmentation failed: {error_msg}")

        self._last_error = error_msg
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Error: {error_msg}")

        QMessageBox.critical(self, "Error", f"Segmentation failed: {error_msg}")
        self.segmentation_failed.emit(error_msg)

        # Reset processing states
        self._processing = False
        self._batch_processing = False
        self._update_ui_state()

    def _update_ui_state(self):
        """Update UI elements based on current processing state."""
        is_busy = self._processing or self._batch_processing
        has_image_layer = self._get_active_image_layer() is not None

        # Update button states
        self.run_btn.setEnabled(not is_busy and has_image_layer)
        self.run_stack_btn.setEnabled(not is_busy and has_image_layer)
        self.export_btn.setEnabled(not is_busy and self.data_manager.segmentation_data is not None)

        # Update model-related controls
        model_controls = [
            self.model_combo,
            self.custom_model_btn,
            self.diameter_spin,
            self.flow_spin,
            self.prob_spin,
            self.size_spin,
            self.gpu_check,
            self.normalize_check,
            self.compute_diameter_check
        ]

        for control in model_controls:
            control.setEnabled(not is_busy)

        # Update progress indicators
        if is_busy:
            self.progress_bar.setEnabled(True)
        else:
            if self._last_error:
                self.status_label.setText(f"Error: {self._last_error}")
            else:
                self.status_label.setText("Ready")
            self.progress_bar.setEnabled(False)
            self.progress_bar.setValue(0)  # Reset progress bar when not busy
    def _validate_frame_update(self, frame_index: int, data: np.ndarray) -> bool:
        """Validate frame data before update."""
        try:
            if not self.data_manager._initialized:
                raise ValueError("Data manager not initialized")

            if frame_index >= self.data_manager._num_frames:
                raise ValueError(f"Frame index {frame_index} out of bounds")

            if self.data_manager.segmentation_data is not None:
                expected_shape = self.data_manager.segmentation_data.shape[1:]
                if data.shape != expected_shape:
                    raise ValueError(f"Frame shape mismatch: expected {expected_shape}, got {data.shape}")

            return True

        except Exception as e:
            logger.error(f"Frame validation failed: {e}")
            return False

    @log_operation
    def _on_segmentation_completed(self, masks: np.ndarray, results: dict):
        """Handle completion of segmentation"""
        try:
            logger.debug("Processing segmentation completion")
            current_frame = int(self.viewer.dims.point[0])

            # Update data manager with current frame only
            self.data_manager.segmentation_data = (masks, current_frame)

            # Validate stack consistency
            if not self.validate_stack_consistency():
                raise ValueError("Stack consistency validation failed after segmentation")

            # Update visualization for current frame only
            self.vis_manager.update_tracking_visualization((masks, current_frame))

            # Enable buttons and update status
            self.export_btn.setEnabled(True)

            # Emit completion signal with full stack
            if self.data_manager.segmentation_data is not None:
                self.segmentation_completed.emit(self.data_manager.segmentation_data, results)
            else:
                self.segmentation_completed.emit(masks, results)

        except Exception as e:
            self._on_segmentation_failed(str(e))

    def validate_stack_consistency(self):
        """Validate that all components have consistent data"""
        if self.data_manager.segmentation_data is None:
            return True

        stack_shape = self.data_manager.segmentation_data.shape
        visualization_shape = (
            self.vis_manager.tracking_layer.data.shape
            if self.vis_manager.tracking_layer is not None else None
        )

        if visualization_shape is not None and stack_shape != visualization_shape:
            logger.error(f"Stack shape mismatch: DataManager={stack_shape}, Visualization={visualization_shape}")
            return False

        return True

    def _update_button_states(self):
        """Update button states based on current state"""
        state = self.state_manager.state

        # Disable buttons during processing
        processing_buttons = [self.run_btn, self.run_stack_btn,  # Removed save_btn
                              self.export_btn, self.import_btn]

        for button in processing_buttons:
            button.setEnabled(not state.is_processing)

        # Enable/disable based on data availability
        has_data = state.full_stack is not None
        self.export_btn.setEnabled(has_data and not state.is_processing)

    def _connect_signals(self):
        """Connect signals between components"""
        # Connect segmentation handler signals
        self.segmentation.signals.segmentation_completed.connect(self._on_segmentation_completed)
        self.segmentation.signals.segmentation_failed.connect(self._on_segmentation_failed)
        self.segmentation.signals.progress_updated.connect(self._update_progress)

        # Connect export/import buttons
        self.export_btn.clicked.connect(self.export_to_cellpose)
        self.import_btn.clicked.connect(self.import_corrections)

        # Model selection signals
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.custom_model_btn.clicked.connect(self._load_custom_model)

        # Action button signals
        self.run_btn.clicked.connect(self._run_segmentation)
        self.run_stack_btn.clicked.connect(self._run_stack_segmentation)

        # Viewer layer events
        self.viewer.layers.events.inserted.connect(self._update_button_states)
        self.viewer.layers.events.removed.connect(self._update_button_states)

    def shutdown(self):
        """Clean up resources"""
        try:
            # Disconnect layer events
            self.viewer.layers.events.inserted.disconnect(self._update_button_states)
            self.viewer.layers.events.removed.disconnect(self._update_button_states)

            # Clear references
            self.viewer = None
            self.data_manager = None
            self.vis_manager = None
            self.segmentation = None

            if hasattr(self, 'correction_widget'):
                self.correction_widget.cleanup()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def _update_progress(self, progress: int, message: str):
        """Update progress bar and status message"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        logger.debug(f"Progress: {progress}%, Message: {message}")

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all controls"""
        controls = [
            self.run_btn,
            self.run_stack_btn,
            self.model_combo,
            self.custom_model_btn,
            self.diameter_spin,
            self.flow_spin,
            self.prob_spin,
            self.size_spin,
            self.gpu_check,
            self.normalize_check
        ]
        for control in controls:
            control.setEnabled(enabled)

    def _on_correction_made(self, updated_masks: np.ndarray):
        """Handle corrections made to the segmentation"""
        try:
            self.data_manager.segmentation_data = updated_masks
            self.vis_manager.update_tracking_visualization(updated_masks)
            num_cells = len(np.unique(updated_masks)) - 1
            self.status_label.setText(f"Correction applied. Current cell count: {num_cells}")
        except Exception as e:
            self._on_segmentation_failed(f"Error applying correction: {str(e)}")

    def _setup_ui(self):
        """Create and arrange the widget UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Cell Segmentation")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Model Selection Section
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()
        model_group.setLayout(model_layout)

        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["cyto3", "nuclei", "custom"])
        model_layout.addWidget(self.model_combo)

        self.custom_model_btn = QPushButton("Load Custom...")
        self.custom_model_btn.setEnabled(False)
        model_layout.addWidget(self.custom_model_btn)

        layout.addWidget(model_group)

        # Parameters Section
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        params_group.setLayout(params_layout)

        # Cell diameter
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(0.1, 1000.0)
        self.diameter_spin.setValue(95.0)
        params_layout.addRow("Cell Diameter:", self.diameter_spin)

        # Flow threshold
        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0.0, 1.0)
        self.flow_spin.setValue(0.6)
        self.flow_spin.setSingleStep(0.1)
        params_layout.addRow("Flow Threshold:", self.flow_spin)

        # Cell probability threshold
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setValue(0.3)
        self.prob_spin.setSingleStep(0.1)
        params_layout.addRow("Cell Probability:", self.prob_spin)

        # Minimum size
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 10000)
        self.size_spin.setValue(25)
        params_layout.addRow("Min Size:", self.size_spin)

        # Additional options
        options_layout = QHBoxLayout()
        self.gpu_check = QCheckBox("Use GPU")
        self.normalize_check = QCheckBox("Normalize")
        self.compute_diameter_check = QCheckBox("Auto-compute diameter")
        self.gpu_check.setChecked(True)
        self.normalize_check.setChecked(True)
        self.compute_diameter_check.setChecked(True)

        options_group = QGroupBox("Advanced Options")
        options_group_layout = QVBoxLayout()
        options_group_layout.addWidget(self.gpu_check)
        options_group_layout.addWidget(self.normalize_check)
        options_group_layout.addWidget(self.compute_diameter_check)
        options_group.setLayout(options_group_layout)

        layout.addWidget(params_group)
        layout.addWidget(options_group)

        # Action Buttons
        button_group = QGroupBox("Actions")
        button_layout = QVBoxLayout()
        button_group.setLayout(button_layout)

        # Progress bar and status
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        # Buttons
        buttons_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Segmentation")
        self.run_stack_btn = QPushButton("Process Stack")

        buttons_layout.addWidget(self.run_btn)
        buttons_layout.addWidget(self.run_stack_btn)

        button_layout.addLayout(buttons_layout)
        button_layout.addWidget(self.progress_bar)
        button_layout.addWidget(self.status_label)

        layout.addWidget(button_group)

        # Add Correction Workflow section
        correction_group = QGroupBox("Manual Correction Tools")
        correction_layout = QVBoxLayout()
        correction_group.setLayout(correction_layout)

        # Add correction widget
        self.correction_widget = CellCorrectionWidget(
            self.viewer,
            self.data_manager,
            self.vis_manager
        )
        correction_layout.addWidget(self.correction_widget)

        # Add Cellpose integration section
        cellpose_group = QGroupBox("Cellpose Integration")
        cellpose_layout = QVBoxLayout()
        cellpose_group.setLayout(cellpose_layout)

        # Add informative label
        info_label = QLabel(
            "You can export the current segmentation to edit in Cellpose, "
            "then import the corrected results back."
        )
        info_label.setWordWrap(True)
        cellpose_layout.addWidget(info_label)

        # Cellpose integration buttons
        cellpose_buttons_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export to Cellpose")
        self.export_btn.setToolTip("Save current segmentation in Cellpose format for external editing")

        self.import_btn = QPushButton("Import from Cellpose")
        self.import_btn.setToolTip("Load segmentation that was edited in Cellpose")

        cellpose_buttons_layout.addWidget(self.export_btn)
        cellpose_buttons_layout.addWidget(self.import_btn)
        cellpose_layout.addLayout(cellpose_buttons_layout)

        # Add both groups to main layout
        layout.addWidget(correction_group)
        layout.addWidget(cellpose_group)

        # Initially disable buttons
        self.export_btn.setEnabled(False)
        self.import_btn.setEnabled(False)
        self._update_button_states()

        # Add stretch at the bottom
        layout.addStretch()

    def _on_model_changed(self, model_type: str):
        """Handle model type change"""
        self.custom_model_btn.setEnabled(model_type == "custom")
        if model_type != "custom":
            self._custom_model_path = None
            # Initialize model when changing to built-in models
            try:
                self._initialize_model()
            except Exception as e:
                logger.error(f"Failed to initialize model on type change: {str(e)}")

    def _load_custom_model(self):
        """Open file dialog to select custom model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom Model",
            str(Path.home()),
            "All Files (*.*);; Model Files (*.pth)"  # Allow all files, but also keep .pth option
        )

        if file_path:
            try:
                # Validate that this is a valid model path before storing it
                from cellpose.models import CellposeModel
                # Try to load the model to verify it's valid
                test_model = CellposeModel(pretrained_model=file_path)

                self._custom_model_path = file_path
                # Initialize the actual model for use
                self._initialize_model()

                logger.info(f"Successfully loaded custom model from: {file_path}")

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load model: {str(e)}\n\n"
                    "Please ensure this is a valid Cellpose model file."
                )
                self._custom_model_path = None

    def _get_current_parameters(self) -> SegmentationParameters:
        """Get current parameter values from UI"""
        return SegmentationParameters(
            model_type=self.model_combo.currentText(),
            custom_model_path=getattr(self, '_custom_model_path', None),
            diameter=self.diameter_spin.value(),
            flow_threshold=self.flow_spin.value(),
            cellprob_threshold=self.prob_spin.value(),
            min_size=self.size_spin.value(),
            gpu=self.gpu_check.isChecked(),
            normalize=self.normalize_check.isChecked()
        )

    def _initialize_model(self):
        """Initialize the segmentation model with current parameters"""
        try:
            params = self._get_current_parameters()
            params.validate()  # Make sure parameters are valid
            self.segmentation.initialize_model(params)
            logger.info("Model initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize model: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _get_active_image_layer(self) -> Optional[Image]:
        """Get the currently active image layer"""
        active_layer = self.viewer.layers.selection.active
        if isinstance(active_layer, Image):
            return active_layer

        # If no image layer is active, look for the first image layer
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                return layer

        return None

    def export_to_cellpose(self):
        """Export current segmentation for Cellpose GUI editing"""
        try:
            # Get save location
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Export Directory",
                str(Path.home())
            )

            if not save_dir:
                return

            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Verify we have segmentation data
            if self.data_manager.segmentation_data is None:
                raise ValueError("No segmentation data available")

            # Get original image data
            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ValueError("No image layer found")

            image_data = image_layer.data

            # Ensure dimensions match
            if image_data.shape != self.data_manager.segmentation_data.shape:
                raise ValueError(
                    f"Image shape {image_data.shape} does not match "
                    f"segmentation shape {self.data_manager.segmentation_data.shape}"
                )

            # Save metadata about the export
            metadata = {
                'timestamp': str(datetime.now()),
                'original_image_shape': image_data.shape,
                'segmentation_parameters': self.segmentation.params.__dict__,
                'total_frames': len(image_data)
            }

            with open(save_dir / 'export_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            # Progress dialog for long exports
            progress = QProgressDialog(
                "Exporting segmentation data...",
                "Cancel",
                0,
                len(image_data),
                self
            )
            progress.setWindowModality(Qt.WindowModal)

            try:
                for i in range(len(image_data)):
                    if progress.wasCanceled():
                        raise InterruptedError("Export cancelled by user")

                    progress.setValue(i)
                    progress.setLabelText(f"Exporting frame {i + 1}/{len(image_data)}")

                    # Get current frame data
                    current_image = image_data[i]
                    current_masks = self.data_manager.segmentation_data[i]

                    # Ensure proper dimensions and type
                    if current_image.ndim > 2:
                        current_image = current_image.squeeze()
                    if current_masks.ndim > 2:
                        current_masks = current_masks.squeeze()

                    # Convert to uint8 for image and uint16 for masks
                    if current_image.dtype != np.uint8:
                        current_image = self._scale_to_8bit(current_image)
                    current_masks = current_masks.astype(np.uint16)

                    # Generate outlines using Cellpose utility
                    from cellpose.utils import masks_to_outlines
                    outlines = masks_to_outlines(current_masks)

                    # Create random colors for cell labels (Cellpose GUI expectation)
                    ncells = len(np.unique(current_masks)[1:])  # exclude 0 background
                    colors = ((np.random.rand(ncells, 3) * 0.8 + 0.1) * 255).astype(np.uint8)

                    # Save frame image
                    tifffile.imwrite(
                        save_dir / f'img_{i:03d}.tif',
                        current_image
                    )

                    # Prepare Cellpose-format data dictionary
                    cellpose_data = {
                        'img': current_image,
                        'masks': current_masks,
                        'outlines': outlines,
                        'colors': colors,
                        'filename': f'img_{i:03d}.tif',
                        'flows': self.segmentation.last_results.get('flows', [None, None]),
                        'chan_choose': [0, 0],  # Default to first channel
                        'ismanual': np.zeros(ncells, dtype=bool),  # Track manual edits
                        'filename': str(save_dir / f'img_{i:03d}.tif'),
                        'diameter': float(self.segmentation.params.diameter)
                    }

                    # Save Cellpose data
                    np.save(
                        save_dir / f'img_{i:03d}_seg.npy',
                        cellpose_data
                    )

                # Final progress update
                progress.setValue(len(image_data))

                # Enable import button and update status
                self.import_btn.setEnabled(True)
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Segmentation exported to {save_dir}\n"
                    f"Total frames exported: {len(image_data)}\n\n"
                    "You can now open and edit these files in the Cellpose GUI"
                )

                # Store export directory for later import
                self._last_export_dir = save_dir

            except InterruptedError:
                QMessageBox.information(
                    self,
                    "Export Cancelled",
                    "Export operation was cancelled"
                )
                return

        except Exception as e:
            error_msg = f"Failed to export segmentation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(
                self,
                "Export Failed",
                error_msg
            )

        finally:
            if 'progress' in locals():
                progress.close()

    def _scale_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """Scale image data to 8-bit range"""
        img_min = np.percentile(image, 1)  # 1st percentile for robustness
        img_max = np.percentile(image, 99)  # 99th percentile for robustness

        scaled = np.clip(image, img_min, img_max)
        scaled = ((scaled - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        return scaled

    def import_corrections(self):
        """Import corrected masks from Cellpose"""
        try:
            # Get directory containing corrected files
            import_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory with Corrected Files",
                str(Path.home())
            )

            if not import_dir:
                return

            import_dir = Path(import_dir)

            # Load corrected masks
            corrected_masks = []
            for i in range(len(self.data_manager.segmentation_data)):
                mask_file = import_dir / f'img_{i:03d}_seg.npy'
                if not mask_file.exists():
                    raise FileNotFoundError(f"Missing mask file: {mask_file}")

                data = np.load(mask_file, allow_pickle=True).item()
                corrected_masks.append(data['masks'])

            # Update segmentation data
            corrected_stack = np.stack(corrected_masks)
            self.data_manager.segmentation_data = corrected_stack

            # Update visualization
            self.vis_manager.update_tracking_visualization(corrected_stack)

            QMessageBox.information(
                self,
                "Import Complete",
                "Successfully imported corrected segmentation"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Import Failed",
                f"Failed to import corrections: {str(e)}"
            )

    def _setup_correction_tools(self):
        self.correction_widget = CellCorrectionWidget(self.viewer)
        # Add to your widget's layout
        self.layout().addWidget(self.correction_widget)
