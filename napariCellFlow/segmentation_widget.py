from pathlib import Path
from typing import Optional

import napari
import numpy as np
from napari.layers import Image
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFormLayout,
    QPushButton, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QSizePolicy, QProgressBar, QProgressDialog
)

from .base_widget import BaseAnalysisWidget, ProcessingError
from .cell_correction_widget import CellCorrectionWidget
from .data_manager import DataManager
from .segmentation import SegmentationHandler, SegmentationParameters
from .visualization_manager import VisualizationManager


class SegmentationWidget(BaseAnalysisWidget):
    """Widget for controlling cell segmentation in napari"""

    segmentation_completed = Signal(np.ndarray)  # Processed masks

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: DataManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(
            viewer=viewer,
            data_manager=data_manager,
            visualization_manager=visualization_manager
        )

        self._last_export_dir = None
        self._custom_model_path = None

        # Initialize segmentation handler first
        self.segmentation = SegmentationHandler()

        # Initialize all controls
        self._initialize_controls()

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

        # Initialize model and update UI state
        self._initialize_model()
        self._update_ui_state()

    def _setup_ui(self):
        """Initialize the user interface"""
        # Create right side container
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        right_container.setFixedWidth(350)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # Model Selection Section
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.custom_model_btn)
        model_group.setLayout(model_layout)
        right_layout.addWidget(model_group)

        # Parameters Section
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        form_layout = QFormLayout()
        form_layout.addRow("Cell Diameter:", self.diameter_spin)
        form_layout.addRow("Flow Threshold:", self.flow_spin)
        form_layout.addRow("Cell Probability:", self.prob_spin)
        form_layout.addRow("Min Size:", self.size_spin)
        params_layout.addLayout(form_layout)
        params_layout.addWidget(self.reset_params_btn)
        params_group.setLayout(params_layout)
        right_layout.addWidget(params_group)

        # Options Section
        options_group = QGroupBox("Advanced Options")
        options_layout = QVBoxLayout()
        options_layout.addWidget(self.gpu_check)
        options_layout.addWidget(self.normalize_check)
        options_layout.addWidget(self.compute_diameter_check)
        options_group.setLayout(options_layout)
        right_layout.addWidget(options_group)

        # Action Buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.run_stack_btn)
        action_layout.addLayout(button_layout)
        action_group.setLayout(action_layout)
        right_layout.addWidget(action_group)

        # Initialize and add correction widget
        self.correction_widget = CellCorrectionWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )
        correction_group = QGroupBox("Manual Correction Tools")
        correction_layout = QVBoxLayout()
        correction_layout.addWidget(self.correction_widget)
        correction_group.setLayout(correction_layout)
        right_layout.addWidget(correction_group)

        # Cellpose Integration
        cellpose_group = QGroupBox("Cellpose Integration")
        cellpose_layout = QVBoxLayout()
        info_label = QLabel(
            "Export segmentation data for model training in Cellpose, "
            "then import the processed results back."
        )
        info_label.setWordWrap(True)
        cellpose_layout.addWidget(info_label)
        cellpose_buttons = QHBoxLayout()
        cellpose_buttons.addWidget(self.export_btn)
        cellpose_buttons.addWidget(self.import_btn)
        cellpose_layout.addLayout(cellpose_buttons)
        cellpose_group.setLayout(cellpose_layout)
        right_layout.addWidget(cellpose_group)

        # Add status section
        status_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        right_layout.addLayout(status_layout)

        right_layout.addStretch()
        right_container.setLayout(right_layout)

        # Add to main layout
        self.main_layout.addWidget(right_container)
        self.main_layout.addStretch(1)

        # Register controls
        self._register_controls()

    def _register_controls(self):
        """Register all controls with base widget"""
        for control in [
            self.model_combo, self.custom_model_btn,
            self.diameter_spin, self.flow_spin, self.prob_spin, self.size_spin,
            self.gpu_check, self.normalize_check, self.compute_diameter_check,
            self.run_btn, self.run_stack_btn,
            self.export_btn, self.import_btn, self.reset_params_btn
        ]:
            self.register_control(control)

    def _initialize_controls(self):
        """Initialize all UI controls"""
        # Model selection controls
        self.model_combo = QComboBox()
        self.model_combo.addItems(["cyto3", "nuclei", "custom"])
        self.custom_model_btn = QPushButton("Load Custom...")
        self.custom_model_btn.setEnabled(False)

        # Parameter controls
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(0.1, 1000.0)
        self.diameter_spin.setValue(95.0)

        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0.0, 1.0)
        self.flow_spin.setValue(0.6)
        self.flow_spin.setSingleStep(0.1)

        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setValue(0.0)
        self.prob_spin.setSingleStep(0.1)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 10000)
        self.size_spin.setValue(25)

        # Option controls
        self.gpu_check = QCheckBox("Use GPU")
        self.normalize_check = QCheckBox("Normalize")
        self.compute_diameter_check = QCheckBox("Auto-compute diameter")

        # Set checkbox defaults
        self.gpu_check.setChecked(True)
        self.normalize_check.setChecked(True)
        self.compute_diameter_check.setChecked(False)

        # Action buttons
        self.run_btn = QPushButton("Segment Frame")
        self.run_stack_btn = QPushButton("Run Segmentation")
        self.export_btn = QPushButton("Export to Cellpose")
        self.import_btn = QPushButton("Import from Cellpose")
        self.reset_params_btn = QPushButton("Reset Parameters")

    def _create_cellpose_group(self) -> QGroupBox:
        """Create Cellpose integration group"""
        group = QGroupBox("Cellpose Integration")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        info_label = QLabel(
            "Export segmentation data for model training in Cellpose, "
            "then import the processed results back."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.reset_params_btn)
        layout.addLayout(button_layout)

        group.setLayout(layout)
        return group

    def _connect_signals(self):
        """Connect all signal handlers"""
        # Model signals
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.custom_model_btn.clicked.connect(self._load_custom_model)

        # Parameter change signals
        for control in [
            self.diameter_spin, self.flow_spin, self.prob_spin, self.size_spin,
            self.gpu_check, self.normalize_check, self.compute_diameter_check
        ]:
            if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.valueChanged.connect(self.parameters_updated.emit)
            else:
                control.stateChanged.connect(self.parameters_updated.emit)

        # Action signals
        self.run_btn.clicked.connect(self._run_segmentation)
        self.run_stack_btn.clicked.connect(self._run_stack_segmentation)
        self.export_btn.clicked.connect(self.export_to_cellpose)
        self.import_btn.clicked.connect(self.import_from_cellpose)
        self.reset_params_btn.clicked.connect(self.reset_parameters)

        # Layer change handlers
        if self.viewer is not None:
            self.viewer.layers.events.inserted.connect(self._update_ui_state)
            self.viewer.layers.events.removed.connect(self._update_ui_state)
            self.viewer.layers.selection.events.changed.connect(self._update_ui_state)

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.diameter_spin.setValue(95.0)
        self.flow_spin.setValue(0.6)
        self.prob_spin.setValue(0.0)
        self.size_spin.setValue(25)
        self.gpu_check.setChecked(True)
        self.normalize_check.setChecked(True)
        self.compute_diameter_check.setChecked(False)
        self.parameters_updated.emit()

    def import_from_cellpose(self):
        """Import processed masks and images from Cellpose"""
        if getattr(self, '_processing', False):
            return

        try:
            # First open the file dialog
            import_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory with Processed Files",
                str(self._last_export_dir) if self._last_export_dir else str(Path.home())
            )

            if not import_dir:
                return  # User cancelled

            self._processing = True
            self._set_controls_enabled(False)
            import_dir = Path(import_dir)

            self._update_status("Starting import...", 0)

            # Look for files to determine number of frames
            import glob
            mask_files = sorted(glob.glob(str(import_dir / 'img_*_seg.npy')))
            if not mask_files:
                raise ProcessingError(
                    "No mask files found",
                    "Selected directory doesn't contain any Cellpose mask files (img_*_seg.npy)"
                )

            num_frames = len(mask_files)
            imported_masks = []
            imported_images = []

            # Load data
            import tifffile
            for i in range(num_frames):
                progress = int(90 * i / num_frames)
                self._update_status(f"Loading frame {i + 1}/{num_frames}", progress)

                # Check for files
                mask_file = import_dir / f'img_{i:03d}_seg.npy'
                image_file = import_dir / f'img_{i:03d}.tif'

                if not mask_file.exists() or not image_file.exists():
                    raise ProcessingError(
                        "Missing files",
                        f"Could not find mask or image file for frame {i}"
                    )

                # Load data
                try:
                    data = np.load(mask_file, allow_pickle=True).item()
                    if 'masks' not in data:
                        raise ProcessingError(
                            "Invalid mask data",
                            f"Frame {i} doesn't contain mask data"
                        )
                    imported_masks.append(data['masks'])

                    image = tifffile.imread(image_file)
                    imported_images.append(image)
                except Exception as e:
                    raise ProcessingError(
                        f"Error loading frame {i}",
                        f"Failed to load data from files: {str(e)}"
                    )

            self._update_status("Processing data...", 95)

            try:
                # Stack the data
                imported_stack = np.stack(imported_masks)
                image_stack = np.stack(imported_images)

                # Validate dimensions
                if imported_stack.ndim < 2 or image_stack.ndim < 2:
                    raise ProcessingError(
                        "Invalid data dimensions",
                        "Imported data has incorrect dimensionality"
                    )

                # Initialize data manager with correct number of frames
                self.data_manager.initialize_stack(num_frames)

                # Update visualizations
                self.data_manager.segmentation_data = imported_stack

                # Update or create image layer
                if 'Cellpose_Imported' in self.viewer.layers:
                    self.viewer.layers['Cellpose_Imported'].data = image_stack
                else:
                    self.viewer.add_image(
                        image_stack,
                        name='Cellpose_Imported',
                        visible=True
                    )

                # Update tracking visualization
                self.visualization_manager.update_tracking_visualization(imported_stack)

                # Store successful import directory
                self._last_export_dir = import_dir

                self._update_status("Import complete", 100)

            except Exception as e:
                raise ProcessingError(
                    "Failed to process imported data",
                    str(e)
                )

        except ProcessingError as pe:
            self._handle_error(pe)
        except Exception as e:
            self._handle_error(ProcessingError(
                "Failed to import data",
                str(e),
                self.__class__.__name__
            ))
        finally:
            self._processing = False
            self._set_controls_enabled(True)
            # Ensure UI state is properly updated regardless of success/failure
            self._update_ui_state()

    def _run_segmentation(self):
        """Run segmentation on current frame"""
        if getattr(self, '_processing', False):
            return

        try:
            self._processing = True
            self._set_controls_enabled(False)
            self._update_status("Starting segmentation...", 0)

            # Store the currently selected image layer
            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ProcessingError("No image layer selected")

            # Update model with current parameters
            current_params = self._get_current_parameters()
            self.segmentation.params = current_params

            if not self._ensure_model_initialized():
                return

            # Get current frame data
            frame_data = self._get_current_frame_data(image_layer)
            masks, metadata = self.segmentation.segment_frame(frame_data)

            # Initialize data manager with proper dimensions
            num_frames = (
                image_layer.data.shape[0]
                if image_layer.data.ndim > 2
                else 1
            )

            # Check if we need to create a new layer due to shape mismatch
            if 'Segmentation' in self.viewer.layers:
                existing_layer = self.viewer.layers['Segmentation']
                if existing_layer.data.shape[1:] != masks.shape:
                    self.viewer.layers.remove('Segmentation')

            # Initialize or reinitialize data manager if needed
            if (not self.data_manager._initialized or
                    self.data_manager.segmentation_data is None or
                    (image_layer.data.ndim > 2 and
                     self.data_manager.segmentation_data.shape != (num_frames,) + masks.shape)):
                self.data_manager.initialize_stack(num_frames)
                self.data_manager.segmentation_data = np.zeros(
                    (num_frames,) + masks.shape,
                    dtype=masks.dtype
                )

            # Store results
            if image_layer.data.ndim > 2:
                current_frame = int(self.viewer.dims.point[0])
                self.data_manager.segmentation_data[current_frame] = masks
                self.visualization_manager.update_tracking_visualization(
                    (masks, current_frame)
                )
            else:
                self.data_manager.segmentation_data = masks
                self.visualization_manager.update_tracking_visualization(masks)

            # Ensure the image layer stays selected
            self.viewer.layers.selection.active = image_layer

            self._update_status("Segmentation completed", 100)
            self.segmentation_completed.emit(masks)

        except Exception as e:
            self._handle_error(ProcessingError(
                message="Segmentation failed",
                details=str(e),
                component=self.__class__.__name__
            ))
        finally:
            self._processing = False
            self._set_controls_enabled(True)

    def _run_stack_segmentation(self):
        """Run segmentation on entire stack"""
        if getattr(self, '_processing', False):
            return

        try:
            self._processing = True
            self._set_controls_enabled(False)
            self._update_status("Starting stack processing...", 0)

            # Store the currently selected image layer
            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ProcessingError("No image layer selected")

            if image_layer.data.ndim < 3:
                raise ProcessingError("Selected layer is not a stack")

            # Update model with current parameters
            current_params = self._get_current_parameters()
            self.segmentation.params = current_params

            if not self._ensure_model_initialized():
                return

            # Process first frame to get shape
            self._update_status("Processing first frame...", 5)
            first_frame = self.segmentation.segment_frame(image_layer.data[0])[0]

            # Check if we need to create a new layer due to shape mismatch
            if 'Segmentation' in self.viewer.layers:
                existing_layer = self.viewer.layers['Segmentation']
                if existing_layer.data.shape[1:] != first_frame.shape:
                    self.viewer.layers.remove('Segmentation')

            # Initialize data manager with proper dimensions
            num_frames = image_layer.data.shape[0]
            if (not self.data_manager._initialized or
                    self.data_manager.segmentation_data is None or
                    self.data_manager.segmentation_data.shape != (num_frames,) + first_frame.shape):
                self.data_manager.initialize_stack(num_frames)
                self.data_manager.segmentation_data = np.zeros(
                    (num_frames,) + first_frame.shape,
                    dtype=first_frame.dtype
                )

            # Store first frame result
            self.data_manager.segmentation_data[0] = first_frame

            # Process remaining frames
            for frame_idx in range(1, num_frames):
                # Calculate progress percentage (5-95% range)
                progress = 5 + int(90 * frame_idx / (num_frames - 1))
                self._update_status(
                    f"Processing frame {frame_idx + 1}/{num_frames}",
                    progress
                )

                frame_data = image_layer.data[frame_idx]
                masks, _ = self.segmentation.segment_frame(frame_data)
                self.data_manager.segmentation_data[frame_idx] = masks

            # Update visualization
            self._update_status("Updating visualization...", 95)
            self.visualization_manager.update_tracking_visualization(
                self.data_manager.segmentation_data
            )

            # Ensure the image layer stays selected
            self.viewer.layers.selection.active = image_layer

            self._update_status("Stack processing complete", 100)
            self.segmentation_completed.emit(self.data_manager.segmentation_data)

        except Exception as e:
            self._handle_error(ProcessingError(
                message="Stack processing failed",
                details=str(e),
                component=self.__class__.__name__
            ))
        finally:
            self._processing = False
            self._set_controls_enabled(True)

    def _on_model_changed(self, model_type: str):
        """Handle model type change"""
        self.custom_model_btn.setEnabled(model_type == "custom")
        if model_type != "custom":
            self._custom_model_path = None
            try:
                self._initialize_model()
            except Exception as e:
                self._handle_error(ProcessingError(
                    "Failed to initialize model",
                    str(e),
                    self.__class__.__name__
                ))

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
            normalize=self.normalize_check.isChecked(),
            compute_diameter=self.compute_diameter_check.isChecked()
        )

    def _ensure_model_initialized(self) -> bool:
        """Ensure model is initialized"""
        if self.segmentation.model is None:
            try:
                self._initialize_model()
                return True
            except Exception as e:
                self._handle_error(ProcessingError(
                    "Model initialization failed",
                    str(e),
                    self.__class__.__name__
                ))
                return False
        return True

    def _initialize_model(self):
        """Initialize the segmentation model with current parameters"""
        try:
            params = self._get_current_parameters()
            params.validate()
            self.segmentation.initialize_model(params)
        except Exception as e:
            raise ProcessingError(
                "Failed to initialize model",
                str(e),
                self.__class__.__name__
            )

    def _update_ui_state(self):
        """Update UI based on current state"""
        # Get current state
        has_image = bool(self._get_active_image_layer() is not None)  # Ensure boolean
        model_initialized = bool(self.segmentation.model is not None)  # Ensure boolean
        has_segmentation_data = bool(  # Ensure boolean
            self.data_manager.segmentation_data is not None and
            np.any(self.data_manager.segmentation_data)
        )

        # Update basic controls
        for control in self._controls:
            if control not in [self.run_btn, self.run_stack_btn, self.export_btn, self.import_btn]:
                control.setEnabled(True)

        # Update model-specific controls
        self.custom_model_btn.setEnabled(bool(self.model_combo.currentText() == "custom"))

        # Update action buttons based on state
        self.run_btn.setEnabled(bool(has_image and model_initialized))
        self.run_stack_btn.setEnabled(bool(has_image and model_initialized))

        # Export button enabled only when we have both image and segmentation data
        self.export_btn.setEnabled(bool(has_image and has_segmentation_data))

        # Import button is always enabled as it's an entry point for data
        self.import_btn.setEnabled(True)

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
                    "Failed to load custom model",
                    str(e),
                    self.__class__.__name__
                ))
                self._custom_model_path = None

    def export_to_cellpose(self):
        """Export current segmentation for Cellpose GUI editing"""
        if getattr(self, '_processing', False):
            return

        try:
            self._processing = True
            self._set_controls_enabled(False)

            if self.data_manager.segmentation_data is None:
                raise ProcessingError("No segmentation data available")

            image_layer = self._get_active_image_layer()
            if image_layer is None:
                raise ProcessingError("No image layer found")

            # Get export directory
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Export Directory",
                str(Path.home())
            )
            if not save_dir:
                return

            save_dir = Path(save_dir)

            # Verify dimensions match
            if image_layer.data.shape != self.data_manager.segmentation_data.shape:
                raise ProcessingError(
                    "Image and segmentation dimensions do not match",
                    f"Image shape {image_layer.data.shape} vs "
                    f"segmentation shape {self.data_manager.segmentation_data.shape}"
                )

            # Export frames
            import tifffile
            total_frames = len(image_layer.data)

            self._update_status("Starting export...", 0)

            try:
                for i in range(total_frames):
                    # Calculate progress percentage (0-95% range for export)
                    progress = int(95 * i / total_frames)
                    self._update_status(f"Exporting frame {i + 1}/{total_frames}", progress)

                    # Get current image and masks
                    current_image = image_layer.data[i]
                    current_masks = self.data_manager.segmentation_data[i]

                    # Scale image to 8-bit if needed
                    if current_image.dtype != np.uint8:
                        img_min = np.percentile(current_image, 1)
                        img_max = np.percentile(current_image, 99)
                        current_image = np.clip(current_image, img_min, img_max)
                        current_image = ((current_image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

                    # Generate Cellpose-format data
                    from cellpose.utils import masks_to_outlines
                    outlines = masks_to_outlines(current_masks)
                    ncells = len(np.unique(current_masks)[1:])
                    colors = ((np.random.rand(ncells, 3) * 0.8 + 0.1) * 255).astype(np.uint8)

                    cellpose_data = {
                        'img': current_image,
                        'masks': current_masks,
                        'outlines': outlines,
                        'colors': colors,
                        'filename': f'img_{i:03d}.tif',
                        'flows': self.segmentation.last_results.get('flows', [None, None]),
                        'chan_choose': [0, 0],
                        'ismanual': np.zeros(ncells, dtype=bool),
                        'diameter': float(self.segmentation.params.diameter)
                    }

                    # Save files
                    tifffile.imwrite(save_dir / f'img_{i:03d}.tif', current_image)
                    np.save(save_dir / f'img_{i:03d}_seg.npy', cellpose_data)

                self._last_export_dir = save_dir
                self.import_btn.setEnabled(True)

                self._update_status("Export completed successfully", 100)

            except Exception as e:
                raise ProcessingError("Export operation failed", str(e))

        except Exception as e:
            self._handle_error(ProcessingError(
                "Failed to export segmentation",
                str(e),
                self.__class__.__name__
            ))
        finally:
            self._processing = False
            self._set_controls_enabled(True)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'correction_widget'):
            self.correction_widget.cleanup()
        super().cleanup()

    def _create_model_group(self) -> QGroupBox:
        """Create model selection controls group"""
        group = QGroupBox("Model Selection")
        layout = QHBoxLayout()
        layout.setSpacing(4)

        layout.addWidget(QLabel("Model:"))
        layout.addWidget(self.model_combo)
        layout.addWidget(self.custom_model_btn)

        group.setLayout(layout)
        return group

    def _create_parameter_group(self) -> QGroupBox:
        """Create segmentation parameters group"""
        group = QGroupBox("Parameters")
        form_layout = QFormLayout()
        form_layout.setSpacing(4)

        # Configure diameter control
        self.diameter_spin.setRange(0.1, 1000.0)
        self.diameter_spin.setValue(95.0)
        self.diameter_spin.setToolTip("Expected diameter of cells in pixels")
        form_layout.addRow("Cell Diameter:", self.diameter_spin)

        # Configure flow threshold control
        self.flow_spin.setRange(0.0, 1.0)
        self.flow_spin.setValue(0.6)
        self.flow_spin.setSingleStep(0.1)
        self.flow_spin.setToolTip("Flow threshold for cell detection (0-1)")
        form_layout.addRow("Flow Threshold:", self.flow_spin)

        # Configure probability threshold control
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setValue(0.3)
        self.prob_spin.setSingleStep(0.1)
        self.prob_spin.setToolTip("Probability threshold for cell detection (0-1)")
        form_layout.addRow("Cell Probability:", self.prob_spin)

        # Configure minimum size control
        self.size_spin.setRange(1, 10000)
        self.size_spin.setValue(25)
        self.size_spin.setToolTip("Minimum size of detected cells in pixels")
        form_layout.addRow("Min Size:", self.size_spin)

        group_widget = QWidget()
        group_widget.setLayout(form_layout)

        group_layout = QVBoxLayout()
        group_layout.addWidget(group_widget)
        group.setLayout(group_layout)

        return group

    def _create_options_group(self) -> QGroupBox:
        """Create advanced options group"""
        group = QGroupBox("Advanced Options")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Configure checkboxes
        self.gpu_check.setChecked(True)
        self.normalize_check.setChecked(True)
        self.compute_diameter_check.setChecked(True)

        # Add tooltips
        self.gpu_check.setToolTip("Use GPU acceleration if available")
        self.normalize_check.setToolTip("Normalize image data before processing")
        self.compute_diameter_check.setToolTip("Automatically compute cell diameter")

        layout.addWidget(self.gpu_check)
        layout.addWidget(self.normalize_check)
        layout.addWidget(self.compute_diameter_check)

        group.setLayout(layout)
        return group

    def _create_action_group(self) -> QGroupBox:
        """Create action buttons group"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.run_stack_btn)
        layout.addLayout(button_layout)

        group.setLayout(layout)
        return group

    def _handle_layer_removal(self, event):
        """Handle layer removal events"""
        removed_layer = event.value
        # Update UI state when layers are removed
        self._update_ui_state()

    def _get_current_frame_data(self, layer: Image) -> np.ndarray:
        """Get data for current frame"""
        if layer.data.ndim > 2:
            return layer.data[int(self.viewer.dims.point[0])]
        return layer.data

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
