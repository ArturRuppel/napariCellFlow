from pathlib import Path
from typing import Optional
import napari
import numpy as np
from napari.layers import Image
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFormLayout,
    QPushButton, QFileDialog, QMessageBox, QProgressBar, QProgressDialog,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
)

from .cell_correction_widget import CellCorrectionWidget
from .segmentation import SegmentationHandler, SegmentationParameters
from .data_manager import DataManager
from .visualization_manager import VisualizationManager
from .error_handling import ErrorSignals, ProcessingError

class SegmentationWidget(QWidget):
    """Widget for controlling cell segmentation in napari"""

    def __init__(self, viewer: "napari.Viewer", data_manager: DataManager,
                 visualization_manager: VisualizationManager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.vis_manager = visualization_manager

        # Initialize error handling
        self.error_signals = ErrorSignals()
        self.error_signals.processing_error.connect(self._handle_processing_error)
        self.error_signals.critical_error.connect(self._handle_critical_error)
        self.error_signals.warning.connect(self._handle_warning)

        # Track processing state
        self._processing = False
        self._batch_processing = False

        # Initialize segmentation handler
        self.segmentation = SegmentationHandler()

        self._setup_ui()
        self._connect_signals()
        self._initialize_ui_state()
        self._update_ui_state()

    def _setup_ui(self):
        """Initialize the user interface"""
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

        layout.addWidget(correction_group)
        # Add Cellpose integration section
        cellpose_group = QGroupBox("Cellpose Integration")
        cellpose_layout = QVBoxLayout()
        cellpose_group.setLayout(cellpose_layout)

        info_label = QLabel(
            "You can export the current segmentation to edit in Cellpose, "
            "then import the corrected results back."
        )
        info_label.setWordWrap(True)
        cellpose_layout.addWidget(info_label)

        # Cellpose integration buttons
        cellpose_buttons_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export to Cellpose")
        self.import_btn = QPushButton("Import from Cellpose")

        cellpose_buttons_layout.addWidget(self.export_btn)
        cellpose_buttons_layout.addWidget(self.import_btn)
        cellpose_layout.addLayout(cellpose_buttons_layout)

        layout.addWidget(cellpose_group)
        layout.addStretch()

    def _connect_signals(self):
        """Connect all signal handlers"""
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.custom_model_btn.clicked.connect(self._load_custom_model)
        self.run_btn.clicked.connect(self._run_segmentation)
        self.run_stack_btn.clicked.connect(self._run_stack_segmentation)
        self.export_btn.clicked.connect(self.export_to_cellpose)
        self.import_btn.clicked.connect(self.import_corrections)

        # Add layer change handlers
        self.viewer.layers.events.inserted.connect(self._update_ui_state)
        self.viewer.layers.events.removed.connect(self._update_ui_state)
        self.viewer.layers.selection.events.changed.connect(self._update_ui_state)

    def _handle_processing_error(self, error: ProcessingError):
        """Handle recoverable processing errors"""
        self._update_status(f"Error: {error.message}")
        self.progress_bar.setValue(0)
        QMessageBox.warning(self, "Processing Error",
                            f"{error.message}\n\nDetails: {error.details}")
        self._set_controls_enabled(True)

    def _handle_critical_error(self, error: ProcessingError):
        """Handle non-recoverable errors"""
        self._update_status(f"Critical Error: {error.message}")
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Critical Error",
                             f"{error.message}\n\nDetails: {error.details}")
        self._set_controls_enabled(False)

    def _handle_warning(self, message: str):
        """Handle warnings"""
        self.status_label.setText(f"Warning: {message}")

    def _run_segmentation(self):
        """Run segmentation on current frame"""
        if self._processing:
            self.error_signals.warning.emit("Segmentation already in progress")
            return

        try:
            self._processing = True
            self._update_ui_state()

            if not self._ensure_model_initialized():
                return

            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ProcessingError(message="No image layer selected",
                                      component="SegmentationWidget")

            # Initialize data manager if needed
            if not self.data_manager._initialized:
                num_frames = image_layer.data.shape[0] if image_layer.data.ndim > 2 else 1
                self.data_manager.initialize_stack(num_frames)

            # Get current frame data
            frame_data = self._get_current_frame_data(image_layer)
            masks, metadata = self.segmentation.segment_frame(frame_data)

            current_frame = int(self.viewer.dims.point[0])
            self.data_manager.segmentation_data = (masks, current_frame)
            self.vis_manager.update_tracking_visualization((masks, current_frame))

            self._update_status("Segmentation completed", 100)

        except Exception as e:
            error = ProcessingError(
                message="Segmentation failed",
                details=str(e),
                component="SegmentationWidget"
            )
            self.error_signals.processing_error.emit(error)
        finally:
            self._processing = False
            self._update_ui_state()

    def _run_stack_segmentation(self):
        """Run segmentation on entire stack"""
        if self._processing or self._batch_processing:
            self.error_signals.warning.emit("Processing already in progress")
            return

        try:
            self._batch_processing = True
            self._update_ui_state()

            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ProcessingError(message="No image layer selected")

            if image_layer.data.ndim < 3:
                raise ProcessingError(message="Selected layer is not a stack")

            num_frames = image_layer.data.shape[0]
            self.data_manager.initialize_stack(num_frames)

            progress = self._create_progress_dialog(num_frames)

            for frame_idx in range(num_frames):
                if progress.wasCanceled():
                    raise ProcessingError(message="Processing canceled by user",
                                          recoverable=True)

                progress.setValue(frame_idx)
                progress.setLabelText(f"Processing frame {frame_idx + 1}/{num_frames}")

                frame_data = image_layer.data[frame_idx]
                masks, metadata = self.segmentation.segment_frame(frame_data)
                self.data_manager.segmentation_data = (masks, frame_idx)
                self.vis_manager.update_tracking_visualization((masks, frame_idx))

        except Exception as e:
            error = ProcessingError(
                message="Stack processing failed",
                details=str(e),
                component="SegmentationWidget"
            )
            self.error_signals.processing_error.emit(error)
        finally:
            self._batch_processing = False
            self._update_ui_state()
            if 'progress' in locals():
                progress.close()

    def _create_progress_dialog(self, max_value: int) -> QProgressDialog:
        """Create and configure progress dialog"""
        progress = QProgressDialog("Processing image stack...", "Cancel",
                                   0, max_value, self)
        progress.setWindowModality(Qt.WindowModal)
        return progress

    def _get_current_frame_data(self, layer: Image) -> np.ndarray:
        """Get data for current frame"""
        if layer.data.ndim > 2:
            return layer.data[int(self.viewer.dims.point[0])]
        return layer.data

    def _ensure_model_initialized(self) -> bool:
        """Ensure model is initialized"""
        if self.segmentation.model is None:
            try:
                self._initialize_model()
                return True
            except Exception as e:
                self.error_signals.processing_error.emit(
                    ProcessingError(
                        message="Model initialization failed",
                        details=str(e)
                    )
                )
                return False
        return True

    def _get_active_image_layer(self) -> Optional[Image]:
        """Get currently active image layer"""
        active_layer = self.viewer.layers.selection.active
        if isinstance(active_layer, Image):
            return active_layer

        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                return layer
        return None

    def _update_ui_state(self):
        """Update UI based on current state"""
        is_busy = self._processing or self._batch_processing
        has_image = self._get_active_image_layer() is not None
        model_initialized = self.segmentation.model is not None

        # Enable main processing buttons if we have an image and aren't busy
        self.run_btn.setEnabled(not is_busy and has_image and model_initialized)
        self.run_stack_btn.setEnabled(not is_busy and has_image and model_initialized)

        # Enable/disable parameter controls
        self._set_controls_enabled(not is_busy)

        # Update model-related UI
        self.custom_model_btn.setEnabled(self.model_combo.currentText() == "custom" and not is_busy)

    def _initialize_ui_state(self):
        """Initialize UI state after setup"""
        # Initialize the model for the default selection
        try:
            self._initialize_model()
        except Exception as e:
            self.error_signals.processing_error.emit(
                ProcessingError(
                    message="Failed to initialize default model",
                    details=str(e)
                )
            )

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all controls"""
        controls = [
            self.model_combo, self.custom_model_btn,
            self.diameter_spin, self.flow_spin, self.prob_spin,
            self.size_spin, self.gpu_check, self.normalize_check,
            self.compute_diameter_check
        ]
        for control in controls:
            control.setEnabled(enabled)

    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and progress"""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)

    def shutdown(self):
        """Clean up resources"""
        if hasattr(self, 'correction_widget'):
            self.correction_widget.cleanup()

    def _on_model_changed(self, model_type: str):
        """Handle model type change"""
        self.custom_model_btn.setEnabled(model_type == "custom")
        if model_type != "custom":
            self._custom_model_path = None
            try:
                self._initialize_model()
            except Exception as e:
                self.error_signals.processing_error.emit(
                    ProcessingError(
                        message="Failed to initialize model",
                        details=str(e),
                        component="SegmentationWidget"
                    )
                )

    def _load_custom_model(self):
        """Open file dialog to select custom model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom Model",
            str(Path.home()),
            "All Files (*.*);; Model Files (*.pth)"
        )

        if file_path:
            try:
                from cellpose.models import CellposeModel
                # Verify model is valid
                test_model = CellposeModel(pretrained_model=file_path)

                self._custom_model_path = file_path
                self._initialize_model()

            except Exception as e:
                self.error_signals.processing_error.emit(
                    ProcessingError(
                        message="Failed to load custom model",
                        details=str(e),
                        component="SegmentationWidget"
                    )
                )
                self._custom_model_path = None

    def _initialize_model(self):
        """Initialize the segmentation model with current parameters"""
        try:
            params = self._get_current_parameters()
            params.validate()
            self.segmentation.initialize_model(params)
        except Exception as e:
            raise ProcessingError(
                message="Failed to initialize model",
                details=str(e),
                component="SegmentationWidget"
            )

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

    def export_to_cellpose(self):
        """Export current segmentation for Cellpose GUI editing"""
        try:
            save_dir = self._get_export_directory()
            if not save_dir:
                return

            if self.data_manager.segmentation_data is None:
                raise ProcessingError(message="No segmentation data available")

            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ProcessingError(message="No image layer found")

            # Verify dimensions match
            if image_layer.data.shape != self.data_manager.segmentation_data.shape:
                raise ProcessingError(
                    message="Image and segmentation dimensions do not match",
                    details=f"Image shape {image_layer.data.shape} vs "
                            f"segmentation shape {self.data_manager.segmentation_data.shape}"
                )

            self._save_export_metadata(save_dir, image_layer.data.shape)
            self._export_frames(save_dir, image_layer.data)

            self.import_btn.setEnabled(True)
            QMessageBox.information(
                self,
                "Export Complete",
                f"Segmentation exported to {save_dir}\n"
                f"Total frames exported: {len(image_layer.data)}"
            )
            self._last_export_dir = save_dir

        except Exception as e:
            self.error_signals.processing_error.emit(
                ProcessingError(
                    message="Failed to export segmentation",
                    details=str(e),
                    component="SegmentationWidget"
                )
            )

    def _get_export_directory(self) -> Optional[Path]:
        """Get directory for exporting data"""
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            str(Path.home())
        )
        return Path(save_dir) if save_dir else None

    def _save_export_metadata(self, save_dir: Path, data_shape: tuple):
        """Save metadata about the export"""
        from datetime import datetime
        import json

        metadata = {
            'timestamp': str(datetime.now()),
            'original_image_shape': data_shape,
            'segmentation_parameters': self.segmentation.params.__dict__,
            'total_frames': len(data_shape)
        }

        with open(save_dir / 'export_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def _export_frames(self, save_dir: Path, image_data: np.ndarray):
        """Export individual frames"""
        from cellpose.utils import masks_to_outlines
        import tifffile

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
                    raise ProcessingError(message="Export cancelled by user")

                progress.setValue(i)
                progress.setLabelText(f"Exporting frame {i + 1}/{len(image_data)}")

                current_image = self._prepare_frame_image(image_data[i])
                current_masks = self._prepare_frame_masks(i)

                # Generate Cellpose-format data
                cellpose_data = self._create_cellpose_data(
                    current_image,
                    current_masks,
                    f'img_{i:03d}.tif'
                )

                # Save frame files
                tifffile.imwrite(save_dir / f'img_{i:03d}.tif', current_image)
                np.save(save_dir / f'img_{i:03d}_seg.npy', cellpose_data)

        finally:
            progress.close()

    def _prepare_frame_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image frame for export"""
        if image.ndim > 2:
            image = image.squeeze()
        return self._scale_to_8bit(image) if image.dtype != np.uint8 else image

    def _prepare_frame_masks(self, frame_idx: int) -> np.ndarray:
        """Prepare masks frame for export"""
        masks = self.data_manager.segmentation_data[frame_idx]
        if masks.ndim > 2:
            masks = masks.squeeze()
        return masks.astype(np.uint16)

    def _create_cellpose_data(self, image: np.ndarray, masks: np.ndarray, filename: str) -> dict:
        """Create Cellpose-format data dictionary"""
        from cellpose.utils import masks_to_outlines
        outlines = masks_to_outlines(masks)
        ncells = len(np.unique(masks)[1:])
        colors = ((np.random.rand(ncells, 3) * 0.8 + 0.1) * 255).astype(np.uint8)

        return {
            'img': image,
            'masks': masks,
            'outlines': outlines,
            'colors': colors,
            'filename': filename,
            'flows': self.segmentation.last_results.get('flows', [None, None]),
            'chan_choose': [0, 0],
            'ismanual': np.zeros(ncells, dtype=bool),
            'diameter': float(self.segmentation.params.diameter)
        }

    def _scale_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """Scale image data to 8-bit range"""
        img_min = np.percentile(image, 1)
        img_max = np.percentile(image, 99)
        scaled = np.clip(image, img_min, img_max)
        scaled = ((scaled - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return scaled

    def import_corrections(self):
        """Import corrected masks from Cellpose"""
        try:
            import_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory with Corrected Files",
                str(Path.home())
            )

            if not import_dir:
                return

            import_dir = Path(import_dir)
            corrected_masks = []

            for i in range(len(self.data_manager.segmentation_data)):
                mask_file = import_dir / f'img_{i:03d}_seg.npy'
                if not mask_file.exists():
                    raise ProcessingError(
                        message=f"Missing mask file: {mask_file}",
                        details="Make sure all frames were exported and corrected"
                    )

                data = np.load(mask_file, allow_pickle=True).item()
                corrected_masks.append(data['masks'])

            corrected_stack = np.stack(corrected_masks)
            self.data_manager.segmentation_data = corrected_stack
            self.vis_manager.update_tracking_visualization(corrected_stack)

            QMessageBox.information(
                self,
                "Import Complete",
                "Successfully imported corrected segmentation"
            )

        except Exception as e:
            self.error_signals.processing_error.emit(
                ProcessingError(
                    message="Failed to import corrections",
                    details=str(e),
                    component="SegmentationWidget"
                )
            )