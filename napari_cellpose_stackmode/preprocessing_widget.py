from typing import Optional
import numpy as np
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                            QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
                            QProgressBar, QMessageBox, QCheckBox)
from qtpy.QtCore import Signal

import napari
from napari.layers import Image
from napari.utils.events import Event

from .preprocessing import PreprocessingParameters, ImagePreprocessor
from .data_manager import DataManager
from .visualization_manager import VisualizationManager


class PreprocessingWidget(QWidget):
    """Widget for controlling image preprocessing parameters and operations"""

    # Signals
    preprocessing_completed = Signal(np.ndarray, list)  # Processed stack and info
    preprocessing_failed = Signal(str)  # Error message
    parameters_updated = Signal()

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: DataManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.visualization_manager = visualization_manager
        self.preprocessor = ImagePreprocessor()

        # Track preview state
        self.preview_enabled = False
        self.original_layer = None
        self.preview_layer = None

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Image Preprocessing")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Parameters section
        param_group = QFormLayout()

        # Median filter settings
        self.median_size_spin = QSpinBox()
        self.median_size_spin.setRange(1, 31)
        self.median_size_spin.setSingleStep(2)
        self.median_size_spin.setValue(self.preprocessor.params.median_filter_size)
        self.median_size_spin.setToolTip("Size of median filter kernel (must be odd)")
        param_group.addRow("Median Filter Size:", self.median_size_spin)

        # CLAHE parameters
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setRange(0.1, 100.0)
        self.clahe_clip_spin.setSingleStep(0.5)
        self.clahe_clip_spin.setValue(self.preprocessor.params.clahe_clip_limit)
        self.clahe_clip_spin.setToolTip("Clip limit for contrast enhancement")
        param_group.addRow("CLAHE Clip Limit:", self.clahe_clip_spin)

        self.clahe_grid_spin = QSpinBox()
        self.clahe_grid_spin.setRange(1, 64)
        self.clahe_grid_spin.setValue(self.preprocessor.params.clahe_grid_size)
        self.clahe_grid_spin.setToolTip("Size of grid for local contrast enhancement")
        param_group.addRow("CLAHE Grid Size:", self.clahe_grid_spin)

        # Intensity clipping parameters
        self.initial_lower_spin = QDoubleSpinBox()
        self.initial_lower_spin.setRange(0.0, 99.0)
        self.initial_lower_spin.setValue(self.preprocessor.params.initial_lower_percentile)
        self.initial_lower_spin.setToolTip("Initial lower percentile for intensity clipping")
        param_group.addRow("Initial Lower %:", self.initial_lower_spin)

        self.final_lower_spin = QDoubleSpinBox()
        self.final_lower_spin.setRange(0.0, 99.0)
        self.final_lower_spin.setValue(self.preprocessor.params.final_lower_percentile)
        self.final_lower_spin.setToolTip("Final lower percentile for intensity clipping")
        param_group.addRow("Final Lower %:", self.final_lower_spin)

        self.final_upper_spin = QDoubleSpinBox()
        self.final_upper_spin.setRange(1.0, 100.0)
        self.final_upper_spin.setValue(self.preprocessor.params.final_upper_percentile)
        self.final_upper_spin.setToolTip("Final upper percentile for intensity clipping")
        param_group.addRow("Final Upper %:", self.final_upper_spin)

        # Dark region threshold
        self.black_threshold_spin = QSpinBox()
        self.black_threshold_spin.setRange(0, 255)
        self.black_threshold_spin.setValue(self.preprocessor.params.black_region_threshold)
        self.black_threshold_spin.setToolTip("Threshold for excluding dark regions")
        param_group.addRow("Dark Region Threshold:", self.black_threshold_spin)

        layout.addLayout(param_group)

        # Preview checkbox
        self.preview_checkbox = QCheckBox("Show Preview")
        self.preview_checkbox.setToolTip("Toggle preview of preprocessing on current frame")
        layout.addWidget(self.preview_checkbox)

        # Buttons
        button_layout = QHBoxLayout()

        self.preprocess_btn = QPushButton("Preprocess Stack")
        self.preprocess_btn.clicked.connect(self.run_preprocessing)
        button_layout.addWidget(self.preprocess_btn)

        self.reset_btn = QPushButton("Reset Parameters")
        self.reset_btn.clicked.connect(self.reset_parameters)
        button_layout.addWidget(self.reset_btn)

        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def connect_signals(self):
        """Connect widget signals"""
        # Parameter update signals
        for spin in [self.median_size_spin, self.clahe_clip_spin,
                     self.clahe_grid_spin, self.initial_lower_spin,
                     self.final_lower_spin, self.final_upper_spin,
                     self.black_threshold_spin]:
            spin.valueChanged.connect(self.update_parameters)

        # Preview checkbox signal
        self.preview_checkbox.toggled.connect(self.toggle_preview)

        # Connect to viewer events
        self.viewer.dims.events.current_step.connect(self.update_preview_frame)

    def update_parameters(self):
        """Update preprocessing parameters from UI controls"""
        try:
            params = PreprocessingParameters(
                median_filter_size=self.median_size_spin.value(),
                clahe_clip_limit=self.clahe_clip_spin.value(),
                clahe_grid_size=self.clahe_grid_spin.value(),
                initial_lower_percentile=self.initial_lower_spin.value(),
                final_lower_percentile=self.final_lower_spin.value(),
                final_upper_percentile=self.final_upper_spin.value(),
                black_region_threshold=self.black_threshold_spin.value()
            )

            # Validate and update
            params.validate()
            self.preprocessor.update_parameters(params)

            self.preprocess_btn.setEnabled(True)
            self.status_label.setText("Parameters updated")
            self.parameters_updated.emit()

            # Update preview if enabled
            if self.preview_enabled:
                self.update_preview_frame()

        except ValueError as e:
            self.preprocess_btn.setEnabled(False)
            self.preview_checkbox.setEnabled(False)
            self.status_label.setText(f"Invalid parameters: {str(e)}")

    def toggle_preview(self, enabled: bool):
        """Toggle preview mode"""
        self.preview_enabled = enabled

        try:
            if enabled:
                # If we don't have a stored original layer, find it
                if self.original_layer is None:
                    # First try to get the selected layer
                    selected = self.viewer.layers.selection.active
                    if isinstance(selected, Image):
                        self.original_layer = selected
                    else:
                        # If no image is selected, look for the first image layer
                        for layer in self.viewer.layers:
                            if isinstance(layer, Image):
                                self.original_layer = layer
                                break

                    if self.original_layer is None:
                        QMessageBox.warning(self, "Warning", "No image layer found")
                        self.preview_checkbox.setChecked(False)
                        return

                # Create preview layer if needed
                if self.preview_layer is None:
                    self.preview_layer = self.viewer.add_image(
                        np.zeros_like(self.original_layer.data[0] if self.original_layer.data.ndim == 3
                                      else self.original_layer.data),
                        name='Preview',
                        visible=True
                    )

                # Update preview
                self.update_preview_frame()
            else:
                # Remove preview layer
                if self.preview_layer is not None:
                    self.viewer.layers.remove(self.preview_layer)
                    self.preview_layer = None

                # Don't clear original_layer reference - keep it for next toggle

        except Exception as e:
            logger.error(f"Error toggling preview: {e}")
            self.preview_checkbox.setChecked(False)
            self.preview_enabled = False
            if self.preview_layer is not None:
                self.viewer.layers.remove(self.preview_layer)
                self.preview_layer = None
    def update_preview_frame(self, event: Optional[Event] = None):
        """Update the preview for the current frame"""
        if not self.preview_enabled or self.original_layer is None:
            return

        try:
            # Get current frame with consistent data type
            if self.original_layer.data.ndim == 3:
                current_step = self.viewer.dims.current_step[0]
                frame = self.original_layer.data[current_step].copy()
            else:
                frame = self.original_layer.data.copy()

            # Ensure frame is in the correct format
            if frame.ndim != 2:
                raise ValueError(f"Invalid frame dimensions: {frame.shape}")

            # Process frame
            processed_frame, info = self.preprocessor.preprocess_frame(frame)

            # Ensure preview layer has correct data type
            if self.preview_layer is None:
                self.preview_layer = self.viewer.add_image(
                    np.zeros_like(frame, dtype=np.uint8),
                    name='Preview',
                    visible=True
                )

            # Update preview layer data
            self.preview_layer.data = processed_frame

            # Match contrast limits with full processing
            if hasattr(self.preview_layer, 'contrast_limits'):
                self.preview_layer.contrast_limits = (0, 255)

            # Display processing info
            info_text = (
                f"Preview - Original range: {info['original_range']}\n"
                f"Initial clip: {info['initial_clip_threshold']:.1f}\n"
                f"Final range: [{info['final_lower_threshold']:.1f}, "
                f"{info['final_upper_threshold']:.1f}]"
            )
            self.status_label.setText(info_text)

        except Exception as e:
            error_msg = f"Preview failed: {str(e)}"
            self.status_label.setText(error_msg)
            self.preview_checkbox.setChecked(False)
    def reset_parameters(self):
        """Reset all parameters to defaults"""
        if self.preview_enabled:
            self.preview_checkbox.setChecked(False)

        default_params = PreprocessingParameters()

        self.median_size_spin.setValue(default_params.median_filter_size)
        self.clahe_clip_spin.setValue(default_params.clahe_clip_limit)
        self.clahe_grid_spin.setValue(default_params.clahe_grid_size)
        self.initial_lower_spin.setValue(default_params.initial_lower_percentile)
        self.final_lower_spin.setValue(default_params.final_lower_percentile)
        self.final_upper_spin.setValue(default_params.final_upper_percentile)
        self.black_threshold_spin.setValue(default_params.black_region_threshold)

        self.preprocessor.update_parameters(default_params)
        self.status_label.setText("Parameters reset to defaults")
        self.parameters_updated.emit()

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all controls"""
        self.preprocess_btn.setEnabled(enabled)
        self.preview_checkbox.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)
        self.median_size_spin.setEnabled(enabled)
        self.clahe_clip_spin.setEnabled(enabled)
        self.clahe_grid_spin.setEnabled(enabled)
        self.initial_lower_spin.setEnabled(enabled)
        self.final_lower_spin.setEnabled(enabled)
        self.final_upper_spin.setEnabled(enabled)
        self.black_threshold_spin.setEnabled(enabled)

    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and progress bar"""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)

    def run_preprocessing(self):
        """Run preprocessing on the entire stack"""
        if self.preview_enabled:
            self.preview_checkbox.setChecked(False)

        selected = self.viewer.layers.selection.active

        if not isinstance(selected, Image):
            QMessageBox.warning(self, "Warning", "Please select an image layer")
            return

        try:
            # Disable controls during processing
            self._set_controls_enabled(False)
            self._update_status("Starting preprocessing...", 0)

            # Get the image data
            stack = self._ensure_stack_format(selected.data)
            total_frames = len(stack)

            # Process frames
            processed_frames = []
            preprocessing_info = []

            for frame_idx in range(total_frames):
                progress = int(5 + (90 * frame_idx / total_frames))
                self._update_status(f"Processing frame {frame_idx + 1}/{total_frames}", progress)

                # Get frame with consistent data type
                frame = stack[frame_idx].copy()

                # Process frame using same method as preview
                try:
                    processed_frame, frame_info = self.preprocessor.preprocess_frame(frame)
                    processed_frames.append(processed_frame)
                    preprocessing_info.append(frame_info)
                except Exception as e:
                    raise ValueError(f"Error processing frame {frame_idx}: {str(e)}")

            # Combine processed frames into stack
            processed_stack = np.stack(processed_frames, axis=0)

            # Update visualization
            self._update_status("Updating visualization...", 95)

            # Remove existing preprocessed layer if it exists
            for layer in self.viewer.layers[:]:
                if layer.name == 'Preprocessed':
                    self.viewer.layers.remove(layer)

            # Add new preprocessed layer with consistent settings
            preprocessed_layer = self.viewer.add_image(
                processed_stack,
                name='Preprocessed',
                visible=True,
                metadata={'preprocessing_info': preprocessing_info}
            )

            # Set consistent contrast limits
            if hasattr(preprocessed_layer, 'contrast_limits'):
                preprocessed_layer.contrast_limits = (0, 255)

            # Store results in data manager
            self.data_manager.preprocessed_data = processed_stack
            self.data_manager.preprocessing_info = preprocessing_info

            # Signal completion and update status
            self._update_status("Preprocessing complete", 100)
            self.preprocessing_completed.emit(processed_stack, preprocessing_info)

            # Display summary statistics
            total_pixels = np.prod(processed_stack.shape)
            excluded_pixels = sum(info['excluded_pixels'] for info in preprocessing_info)
            excluded_percent = (excluded_pixels / total_pixels) * 100

            summary = (
                f"Preprocessing complete:\n"
                f"Processed {total_frames} frames\n"
                f"Excluded regions: {excluded_percent:.1f}% of total pixels\n"
                f"Mean intensity: {processed_stack.mean():.1f}"
            )
            self.status_label.setText(summary)

        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            self._update_status(error_msg, 0)
            self.preprocessing_failed.emit(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
        finally:
            self._set_controls_enabled(True)
    def _ensure_stack_format(self, data: np.ndarray) -> np.ndarray:
        """Ensure data is in [t, y, x] format"""
        if data.ndim == 2:
            return data[np.newaxis, :, :]
        elif data.ndim == 3:
            if data.shape[-1] < data.shape[0] and data.shape[-1] < data.shape[1]:
                return np.moveaxis(data, -1, 0)
            return data
        else:
            raise ValueError(f"Unexpected data dimensions: {data.shape}")