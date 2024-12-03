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
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QPushButton, QFileDialog, QMessageBox, QProgressDialog,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
)
from tifffile import tifffile

from .segmentation import SegmentationHandler, SegmentationParameters
from .data_manager import DataManager
from .visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class SegmentationWidget(QWidget):
    """Widget for controlling cell segmentation in napari"""

    # Define signals for communication with main widget
    segmentation_completed = Signal(np.ndarray, dict)  # masks, metadata
    segmentation_failed = Signal(str)

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: DataManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.vis_manager = visualization_manager

        # Initialize segmentation handler
        self.segmentation = SegmentationHandler()

        self._setup_ui()
        self._connect_signals()

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

    def _run_segmentation(self):
        """Run segmentation on current frame"""
        try:
            if not self._ensure_model_initialized():
                return

            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ValueError("No image layer selected")

            # Get current frame data
            data = image_layer.data
            if data.ndim > 2:
                current_step = self.viewer.dims.point[0]
                data = data[int(current_step)]

            # Run segmentation
            self.segmentation.segment_frame(data)

        except Exception as e:
            self._on_segmentation_failed(str(e))

    def _setup_ui(self):
        """Create and arrange the widget UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Model Selection Section
        model_group = QWidget()
        model_layout = QHBoxLayout()
        model_group.setLayout(model_layout)

        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["cyto", "nuclei", "custom"])
        model_layout.addWidget(self.model_combo)

        self.custom_model_btn = QPushButton("Load Custom...")
        self.custom_model_btn.setEnabled(False)
        model_layout.addWidget(self.custom_model_btn)

        layout.addWidget(model_group)

        # Parameters Section
        params_group = QWidget()
        params_layout = QVBoxLayout()
        params_group.setLayout(params_layout)

        # Cell diameter
        diameter_layout = QHBoxLayout()
        diameter_layout.addWidget(QLabel("Cell Diameter:"))
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(0.1, 1000.0)
        self.diameter_spin.setValue(95.0)
        diameter_layout.addWidget(self.diameter_spin)
        params_layout.addLayout(diameter_layout)

        # Flow threshold
        flow_layout = QHBoxLayout()
        flow_layout.addWidget(QLabel("Flow Threshold:"))
        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0.0, 1.0)
        self.flow_spin.setValue(0.6)
        self.flow_spin.setSingleStep(0.1)
        flow_layout.addWidget(self.flow_spin)
        params_layout.addLayout(flow_layout)

        # Cell probability threshold
        prob_layout = QHBoxLayout()
        prob_layout.addWidget(QLabel("Cell Probability:"))
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setValue(0.3)
        self.prob_spin.setSingleStep(0.1)
        prob_layout.addWidget(self.prob_spin)
        params_layout.addLayout(prob_layout)

        # Minimum size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Min Size:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 10000)
        self.size_spin.setValue(25)
        size_layout.addWidget(self.size_spin)
        params_layout.addLayout(size_layout)

        # Additional options
        options_layout = QHBoxLayout()
        self.gpu_check = QCheckBox("Use GPU")
        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.setChecked(True)
        options_layout.addWidget(self.gpu_check)
        options_layout.addWidget(self.normalize_check)
        params_layout.addLayout(options_layout)

        layout.addWidget(params_group)

        # Action Buttons
        button_group = QWidget()
        button_layout = QHBoxLayout()
        button_group.setLayout(button_layout)

        self.run_btn = QPushButton("Run Segmentation")
        self.run_stack_btn = QPushButton("Process Stack")
        self.save_btn = QPushButton("Save Results")

        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.run_stack_btn)
        button_layout.addWidget(self.save_btn)

        layout.addWidget(button_group)

        # Add Correction Workflow section
        correction_group = QGroupBox("Manual Correction Workflow")
        correction_layout = QVBoxLayout()
        correction_group.setLayout(correction_layout)

        # Add informative label
        info_label = QLabel(
            "Export segmentation to edit in Cellpose GUI, then import corrections"
        )
        info_label.setWordWrap(True)
        correction_layout.addWidget(info_label)

        # Correction workflow buttons
        self.export_btn = QPushButton("Export to Cellpose")
        self.export_btn.clicked.connect(self.export_to_cellpose)
        self.export_btn.setEnabled(False)  # Enable only after segmentation

        self.launch_gui_btn = QPushButton("Launch Cellpose GUI")
        self.launch_gui_btn.clicked.connect(self.launch_cellpose_gui)

        self.import_btn = QPushButton("Import Corrections")
        self.import_btn.clicked.connect(self.import_corrections)
        self.import_btn.setEnabled(False)  # Enable after export

        correction_layout.addWidget(self.export_btn)
        correction_layout.addWidget(self.launch_gui_btn)
        correction_layout.addWidget(self.import_btn)

        layout.addWidget(correction_group)

        # Initially disable buttons
        self.save_btn.setEnabled(False)
        self._update_button_states()

        # Add stretch at the bottom
        layout.addStretch()

    def _connect_signals(self):
        """Connect widget signals and slots"""
        # Model selection signals
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.custom_model_btn.clicked.connect(self._load_custom_model)

        # Action button signals
        self.run_btn.clicked.connect(self._run_segmentation)
        self.run_stack_btn.clicked.connect(self._run_stack_segmentation)
        self.save_btn.clicked.connect(self._save_results)

        # Segmentation handler signals
        self.segmentation.segmentation_completed.connect(self._on_segmentation_completed)
        self.segmentation.segmentation_failed.connect(self._on_segmentation_failed)

        # Viewer layer events
        self.viewer.layers.events.inserted.connect(self._update_button_states)
        self.viewer.layers.events.removed.connect(self._update_button_states)

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

    def _run_stack_segmentation(self):
        """Run segmentation on entire stack with progress tracking"""
        try:
            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ValueError("No image layer selected")

            if image_layer.data.ndim < 3:
                raise ValueError("Selected layer is not a stack")

            # Initialize model if needed
            if self.segmentation.model is None:
                self._initialize_model()

            # Process each frame
            masks_list = []
            total_frames = image_layer.data.shape[0]

            # Create progress dialog
            from qtpy.QtWidgets import QProgressDialog
            from qtpy.QtCore import Qt
            progress = QProgressDialog("Processing image stack...", "Cancel", 0, total_frames, self)
            progress.setWindowModality(Qt.WindowModal)

            try:
                for i in range(total_frames):
                    if progress.wasCanceled():
                        raise InterruptedError("Processing canceled by user")

                    # Update progress
                    progress.setValue(i)
                    progress.setLabelText(f"Processing frame {i + 1}/{total_frames}")

                    # Extract single frame and ensure proper dimensions
                    frame = image_layer.data[i]
                    if frame.ndim > 2:
                        frame = frame.squeeze()
                    if frame.ndim != 2:
                        raise ValueError(f"Invalid frame dimensions: {frame.shape}")

                    # Run segmentation on frame
                    try:
                        masks, results = self.segmentation.segment_frame(frame)
                        masks_list.append(masks)
                        logger.debug(f"Processed frame {i + 1}/{total_frames}, found {len(np.unique(masks)) - 1} cells")
                    except Exception as e:
                        raise ValueError(f"Failed to segment frame {i}: {str(e)}")

                # Stack all masks together
                masks_stack = np.stack(masks_list)

                # Store final results with all frames
                final_results = {
                    'masks': masks_stack,
                    'parameters': self.segmentation.params.__dict__,
                    'total_frames': total_frames
                }

                # Update visualization and emit completion signal
                self._on_segmentation_completed(masks_stack, final_results)

            finally:
                # Ensure progress dialog is closed
                progress.close()

        except InterruptedError as e:
            logger.info(f"Stack processing interrupted: {str(e)}")
            QMessageBox.information(self, "Processing Canceled", "Stack processing was canceled")

        except Exception as e:
            error_msg = f"Stack processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
            self._on_segmentation_failed(error_msg)
    def _save_results(self):
        """Save segmentation results"""
        try:
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Save Directory",
                str(Path.home())
            )

            if save_dir:
                self.segmentation.save_results(Path(save_dir))
                QMessageBox.information(
                    self,
                    "Success",
                    "Results saved successfully"
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")

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

    def _update_button_states(self, event=None):
        """Update button states based on current conditions"""
        has_image = self._get_active_image_layer() is not None
        self.run_btn.setEnabled(has_image)
        self.run_stack_btn.setEnabled(
            has_image and
            isinstance(self._get_active_image_layer().data, np.ndarray) and
            self._get_active_image_layer().data.ndim > 2
        )

    def _on_segmentation_completed(self, masks: np.ndarray, results: dict):
        """Handle successful segmentation"""
        # Update data manager
        self.data_manager.segmentation_data = masks

        # Update visualization
        self.vis_manager.update_tracking_visualization(masks)

        # Enable buttons
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)  # Enable export for corrections

        # Emit completion signal
        self.segmentation_completed.emit(masks, results)
    def _on_segmentation_failed(self, error_msg: str):
        """Handle segmentation failure"""
        logger.error(f"Segmentation failed: {error_msg}")
        QMessageBox.critical(self, "Error", f"Segmentation failed: {error_msg}")
        self.segmentation_failed.emit(error_msg)

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

    def shutdown(self):
        """Clean up resources"""
        # Disconnect layer events
        self.viewer.layers.events.inserted.disconnect(self._update_button_states)
        self.viewer.layers.events.removed.disconnect(self._update_button_states)

        # Clear references
        self.viewer = None
        self.data_manager = None
        self.vis_manager = None
        self.segmentation = None

    def launch_cellpose_gui(self):
        """Launch the Cellpose GUI"""
        try:
            import subprocess
            import sys

            # Launch Cellpose GUI in a separate process
            subprocess.Popen([
                sys.executable,
                "-m",
                "cellpose",
                "--gui"
            ])

        except Exception as e:
            QMessageBox.critical(
                self,
                "Launch Failed",
                f"Failed to launch Cellpose GUI: {str(e)}"
            )

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