from pathlib import Path
from typing import Optional

import napari
import numpy as np
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout, QLabel, QFormLayout,
    QPushButton, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
)
from qtpy.QtWidgets import (
    QMessageBox, QProgressDialog
)

from .base_widget import BaseAnalysisWidget, ProcessingError
from .cell_correction_widget import CellCorrectionWidget
from .data_manager import DataManager
from .segmentation import SegmentationHandler, SegmentationParameters
from .visualization_manager import VisualizationManager


class SegmentationWidget(BaseAnalysisWidget):
    """Widget for controlling cell segmentation in napari"""

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: "DataManager",
            visualization_manager: "VisualizationManager"
    ):
        super().__init__(
            viewer=viewer,
            data_manager=data_manager,
            visualization_manager=visualization_manager,
            widget_title="Cell Segmentation"
        )

        self._last_export_dir = None
        self._custom_model_path = None

        # Initialize segmentation handler first
        self.segmentation = SegmentationHandler()

        # Setup UI and connect signals BEFORE setting initial values
        self._setup_ui()
        self._connect_signals()

        # Now initialize model and update UI state
        self._initialize_model()
        self._update_ui_state()

    def _connect_signals(self):
        """Connect all signal handlers"""
        # Model signals
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.custom_model_btn.clicked.connect(self._load_custom_model)

        # Parameter change signals - connect BEFORE setting initial values
        self.diameter_spin.valueChanged.connect(self.parameters_updated.emit)
        self.flow_spin.valueChanged.connect(self.parameters_updated.emit)
        self.prob_spin.valueChanged.connect(self.parameters_updated.emit)
        self.size_spin.valueChanged.connect(self.parameters_updated.emit)
        self.gpu_check.stateChanged.connect(self.parameters_updated.emit)
        self.normalize_check.stateChanged.connect(self.parameters_updated.emit)
        self.compute_diameter_check.stateChanged.connect(self.parameters_updated.emit)

        # Action signals
        self.run_btn.clicked.connect(self._run_segmentation)
        self.run_stack_btn.clicked.connect(self._run_stack_segmentation)
        self.export_btn.clicked.connect(self.export_to_cellpose)
        self.import_btn.clicked.connect(self.import_from_cellpose)

        # Layer change handlers
        self.viewer.layers.events.inserted.connect(self._update_ui_state)
        self.viewer.layers.events.removed.connect(self._update_ui_state)
        self.viewer.layers.selection.events.changed.connect(self._update_ui_state)
    def _setup_ui(self):
        """Initialize the user interface"""
        # Model Selection Section
        model_group = self._create_parameter_group("Model Selection")
        model_layout = QHBoxLayout()

        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["cyto3", "nuclei", "custom"])
        model_layout.addWidget(self.model_combo)

        self.custom_model_btn = QPushButton("Load Custom...")
        self.custom_model_btn.setEnabled(False)
        model_layout.addWidget(self.custom_model_btn)

        model_group.layout().addLayout(model_layout)
        self.main_layout.addWidget(model_group)

        # Parameters Section
        params_group = self._create_parameter_group("Parameters")
        form_layout = QFormLayout()

        # Cell diameter
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(0.1, 1000.0)
        self.diameter_spin.setValue(95.0)
        self.diameter_spin.setToolTip("Expected diameter of cells in pixels")
        form_layout.addRow("Cell Diameter:", self.diameter_spin)

        # Flow threshold
        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0.0, 1.0)
        self.flow_spin.setValue(0.6)
        self.flow_spin.setSingleStep(0.1)
        self.flow_spin.setToolTip("Flow threshold for cell detection (0-1)")
        form_layout.addRow("Flow Threshold:", self.flow_spin)

        # Cell probability threshold
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setValue(0.3)
        self.prob_spin.setSingleStep(0.1)
        self.prob_spin.setToolTip("Probability threshold for cell detection (0-1)")
        form_layout.addRow("Cell Probability:", self.prob_spin)

        # Minimum size
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 10000)
        self.size_spin.setValue(25)
        self.size_spin.setToolTip("Minimum size of detected cells in pixels")
        form_layout.addRow("Min Size:", self.size_spin)

        params_group.layout().addLayout(form_layout)
        self.main_layout.addWidget(params_group)

        # Options Section
        options_group = self._create_parameter_group("Advanced Options")

        self.gpu_check = QCheckBox("Use GPU")
        self.normalize_check = QCheckBox("Normalize")
        self.compute_diameter_check = QCheckBox("Auto-compute diameter")

        # Set checkbox defaults
        self.gpu_check.setChecked(True)
        self.normalize_check.setChecked(True)
        self.compute_diameter_check.setChecked(True)

        # Add tooltips after creating checkboxes
        self.gpu_check.setToolTip("Use GPU acceleration if available")
        self.normalize_check.setToolTip("Normalize image data before processing")
        self.compute_diameter_check.setToolTip("Automatically compute cell diameter")

        options_group.layout().addWidget(self.gpu_check)
        options_group.layout().addWidget(self.normalize_check)
        options_group.layout().addWidget(self.compute_diameter_check)

        self.main_layout.addWidget(options_group)

        # Action Buttons
        button_group = self._create_parameter_group("Actions")
        buttons_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Segmentation")
        self.run_stack_btn = QPushButton("Process Stack")

        buttons_layout.addWidget(self.run_btn)
        buttons_layout.addWidget(self.run_stack_btn)

        button_group.layout().addLayout(buttons_layout)
        self.main_layout.addWidget(button_group)

        # Register all controls
        self._controls = [
            self.model_combo, self.custom_model_btn, self.diameter_spin,
            self.flow_spin, self.prob_spin, self.size_spin, self.gpu_check,
            self.normalize_check, self.compute_diameter_check,
            self.run_btn, self.run_stack_btn
        ]

        # Initialize and add correction widget
        self.correction_widget = CellCorrectionWidget(
            self.viewer,
            self.data_manager,
            self.vis_manager
        )
        correction_group = self._create_parameter_group("Manual Correction Tools")
        correction_group.layout().addWidget(self.correction_widget)
        self.main_layout.addWidget(correction_group)

        # Add Cellpose export/import
        cellpose_group = self._create_parameter_group("Cellpose Integration")

        info_label = QLabel(
            "You can export the current segmentation to edit in Cellpose, "
            "then import the corrected results back."
        )
        info_label.setWordWrap(True)
        cellpose_group.layout().addWidget(info_label)

        # Create and configure export/import buttons
        self.export_btn = QPushButton("Export to Cellpose")
        self.import_btn = QPushButton("Import from Cellpose")

        # Add buttons to layout
        button_container = QHBoxLayout()
        button_container.addWidget(self.export_btn)
        button_container.addWidget(self.import_btn)
        cellpose_group.layout().addLayout(button_container)

        # Add group to main layout
        self.main_layout.addWidget(cellpose_group)

        # Register additional controls
        self._controls.extend([self.export_btn, self.import_btn])

    def _update_ui_state(self):
        """Update UI based on current state"""
        is_busy = self._processing
        has_image = self._get_active_image_layer() is not None
        model_initialized = self.segmentation.model is not None

        self._set_controls_enabled(not is_busy)
        self.run_btn.setEnabled(not is_busy and has_image and model_initialized)
        self.run_stack_btn.setEnabled(not is_busy and has_image and model_initialized)
        self.custom_model_btn.setEnabled(
            self.model_combo.currentText() == "custom" and not is_busy
        )

    def _run_segmentation(self):
        """Run segmentation on current frame"""
        if self._processing:
            return

        try:
            self._processing = True
            self._update_ui_state()

            if not self._ensure_model_initialized():
                return

            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ProcessingError(message="No image layer selected")

            # Initialize data manager if needed
            if not self.data_manager._initialized:
                num_frames = (
                    image_layer.data.shape[0]
                    if image_layer.data.ndim > 2
                    else 1
                )
                self.data_manager.initialize_stack(num_frames)

            # Get current frame data
            frame_data = self._get_current_frame_data(image_layer)
            masks, metadata = self.segmentation.segment_frame(frame_data)

            current_frame = int(self.viewer.dims.point[0])
            self.data_manager.segmentation_data = (masks, current_frame)
            self.vis_manager.update_tracking_visualization((masks, current_frame))

            self._update_status("Segmentation completed", 100)
            self.processing_completed.emit(masks)

        except Exception as e:
            self._handle_error(ProcessingError(
                message="Segmentation failed",
                details=str(e),
                component=self.__class__.__name__
            ))
        finally:
            self._processing = False
            self._update_ui_state()

    def _run_stack_segmentation(self):
        """Run segmentation on entire stack"""
        if self._processing:
            return

        try:
            self._processing = True
            self._update_ui_state()

            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ProcessingError(message="No image layer selected")

            if image_layer.data.ndim < 3:
                raise ProcessingError(message="Selected layer is not a stack")

            num_frames = image_layer.data.shape[0]
            self.data_manager.initialize_stack(num_frames)

            progress = self._create_progress_dialog(
                num_frames,
                "Processing image stack..."
            )

            masks_list = []
            for frame_idx in range(num_frames):
                if progress.wasCanceled():
                    raise ProcessingError(
                        message="Processing canceled by user",
                        component=self.__class__.__name__
                    )

                progress.setValue(frame_idx)
                progress.setLabelText(
                    f"Processing frame {frame_idx + 1}/{num_frames}"
                )

                frame_data = image_layer.data[frame_idx]
                masks, metadata = self.segmentation.segment_frame(frame_data)
                masks_list.append(masks)

            masks_stack = np.stack(masks_list)
            self.data_manager.segmentation_data = masks_stack
            self.vis_manager.update_tracking_visualization(masks_stack)

            self.processing_completed.emit(masks_stack)

        except Exception as e:
            self._handle_error(ProcessingError(
                message="Stack processing failed",
                details=str(e),
                component=self.__class__.__name__
            ))
        finally:
            self._processing = False
            self._update_ui_state()
            if 'progress' in locals():
                progress.close()

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
                self._handle_error(ProcessingError(
                    message="Model initialization failed",
                    details=str(e),
                    component=self.__class__.__name__
                ))
                return False
        return True

    def _on_model_changed(self, model_type: str):
        """Handle model type change"""
        self.custom_model_btn.setEnabled(model_type == "custom")
        if model_type != "custom":
            self._custom_model_path = None
            try:
                self._initialize_model()
            except Exception as e:
                self._handle_error(ProcessingError(
                    message="Failed to initialize model",
                    details=str(e),
                    component=self.__class__.__name__
                ))

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
                self._handle_error(ProcessingError(
                    message="Failed to load custom model",
                    details=str(e),
                    component=self.__class__.__name__
                ))
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
                component=self.__class__.__name__
            )

    def _get_current_parameters(self) -> SegmentationParameters:
        """Get current parameter values from UI"""
        return SegmentationParameters(
            model_type=self.model_combo.currentText(),
            custom_model_path=self._custom_model_path,
            diameter=self.diameter_spin.value(),
            flow_threshold=self.flow_spin.value(),
            cellprob_threshold=self.prob_spin.value(),
            min_size=self.size_spin.value(),
            gpu=self.gpu_check.isChecked(),
            normalize=self.normalize_check.isChecked()
        )

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'correction_widget'):
            self.correction_widget.cleanup()
        super().cleanup()

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

    def export_to_cellpose(self):
        """Export current segmentation for Cellpose GUI editing"""
        try:
            save_dir = self._get_export_directory()
            if not save_dir:
                return

            if self.data_manager.segmentation_data is None:
                raise ProcessingError(
                    message="No segmentation data available",
                    component=self.__class__.__name__
                )

            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ProcessingError(
                    message="No image layer found",
                    component=self.__class__.__name__
                )

            # Verify dimensions match
            if image_layer.data.shape != self.data_manager.segmentation_data.shape:
                raise ProcessingError(
                    message="Image and segmentation dimensions do not match",
                    details=f"Image shape {image_layer.data.shape} vs "
                            f"segmentation shape {self.data_manager.segmentation_data.shape}",
                    component=self.__class__.__name__
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
            self._handle_error(ProcessingError(
                message="Failed to export segmentation",
                details=str(e),
                component=self.__class__.__name__
            ))

    def import_from_cellpose(self):
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
                        details="Make sure all frames were exported and corrected",
                        component=self.__class__.__name__
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
            self._handle_error(ProcessingError(
                message="Failed to import corrections",
                details=str(e),
                component=self.__class__.__name__
            ))