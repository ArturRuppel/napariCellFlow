from typing import Optional

import napari
import numpy as np
from napari.layers import Image
from napari.utils.events import Event
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QSlider,
    QSpinBox, QDoubleSpinBox, QProgressBar, QCheckBox,
    QFormLayout
)
from qtrangeslider import QRangeSlider

from .data_manager import DataManager
from .preprocessing import PreprocessingParameters, ImagePreprocessor
from .visualization_manager import VisualizationManager
import logging

logger = logging.getLogger(__name__)


class PreprocessingWidget(QWidget):
    """Widget for controlling image preprocessing parameters"""

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

        # Track intensity range (always 0-255 after initial conversion)
        self.current_min_intensity = 0
        self.current_max_intensity = 255

        self.setup_ui()
        self.connect_signals()

        # Add layer removal event handler
        self.viewer.layers.events.removed.connect(self._handle_layer_removal)

    def _handle_layer_removal(self, event):
        """Handle layer removal events"""
        removed_layer = event.value

        # Check if the removed layer was our preview layer
        if removed_layer == self.preview_layer:
            logger.debug("Preview layer was removed")
            self.preview_layer = None
            self.preview_enabled = False
            self.preview_check.setChecked(False)

        # Check if the removed layer was our original layer
        if removed_layer == self.original_layer:
            logger.debug("Original layer was removed")
            self.original_layer = None
            if self.preview_layer is not None:
                self.viewer.layers.remove(self.preview_layer)
                self.preview_layer = None
            self.preview_enabled = False
            self.preview_check.setChecked(False)

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
                        self.preview_check.setChecked(False)
                        return

                # Always create a new preview layer if enabled and not existing
                if self.preview_layer is None:
                    preview_data = (self.original_layer.data[0] if self.original_layer.data.ndim == 3
                                    else self.original_layer.data)
                    self.preview_layer = self.viewer.add_image(
                        np.zeros_like(preview_data),
                        name='Preview',
                        visible=True
                    )

                # Update preview
                self.update_preview_frame()
            else:
                # Remove preview layer
                if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.preview_layer)
                self.preview_layer = None

        except Exception as e:
            logger.error(f"Error toggling preview: {e}")
            self.preview_check.setChecked(False)
            self.preview_enabled = False
            if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
                self.viewer.layers.remove(self.preview_layer)
            self.preview_layer = None

    def setup_ui(self):
        """Initialize the user interface with aligned controls and odd-only median filter"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Image Preprocessing")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Parameters form
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        def create_parameter_group(
                label: str,
                min_val: float,
                max_val: float,
                default_val: float,
                step: float = 1.0,
                checkbox: bool = True,
                double: bool = False,
                odd_only: bool = False
        ) -> tuple:
            control_layout = QHBoxLayout()
            control_layout.setContentsMargins(0, 0, 0, 0)

            # Add checkbox if needed
            if checkbox:
                check = QCheckBox("")
                check.setFixedWidth(30)
                control_layout.addWidget(check)
            else:
                check = None
                # Add spacer for alignment when no checkbox
                spacer = QWidget()
                spacer.setFixedWidth(30)
                control_layout.addWidget(spacer)

            # Add spinbox
            if double:
                spin = QDoubleSpinBox()
                spin.setSingleStep(step)
            else:
                spin = QSpinBox()
                spin.setSingleStep(2 if odd_only else int(step))

            spin.setRange(min_val, max_val)
            spin.setValue(default_val)
            if checkbox:
                spin.setEnabled(False)
            spin.setFixedWidth(80)
            control_layout.addWidget(spin)

            # Add slider with stretch
            slider = QSlider(Qt.Horizontal)
            if odd_only:
                slider_range = int((max_val - min_val) / 2)
                slider.setRange(0, slider_range)
                slider.setValue(int((default_val - min_val) / 2))
            else:
                slider.setRange(int(min_val * (1 / step)), int(max_val * (1 / step)))
                slider.setValue(int(default_val * (1 / step)))

            if checkbox:
                slider.setEnabled(False)
            control_layout.addWidget(slider, stretch=1)

            form_layout.addRow(label + ":", control_layout)
            return check, spin, slider

        # Intensity Range (special case with range slider)
        intensity_layout = QVBoxLayout()
        intensity_layout.setSpacing(5)

        # Range slider first
        self.intensity_slider = QRangeSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 255)
        self.intensity_slider.setValue((0, 255))
        intensity_layout.addWidget(self.intensity_slider)

        # Spinboxes below in their own layout
        spin_layout = QHBoxLayout()
        spin_layout.setContentsMargins(0, 0, 0, 0)

        # Add spacer to align with other controls
        spacer = QWidget()
        spacer.setFixedWidth(30)
        spin_layout.addWidget(spacer)

        # Min spinbox
        self.min_spin = QSpinBox()
        self.min_spin.setRange(0, 255)
        self.min_spin.setValue(0)
        self.min_spin.setFixedWidth(80)

        # Max spinbox
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, 255)
        self.max_spin.setValue(255)
        self.max_spin.setFixedWidth(80)

        spin_layout.addWidget(self.min_spin)
        spin_layout.addWidget(self.max_spin)
        spin_layout.addStretch()

        intensity_layout.addLayout(spin_layout)
        form_layout.addRow("Intensity Range:", intensity_layout)

        # Median Filter with odd-only values
        self.median_check, self.median_size_spin, self.median_slider = create_parameter_group(
            "Median Filter", 3, 15, 3, step=2, odd_only=True
        )

        # Gaussian Filter
        self.gaussian_check, self.gaussian_sigma_spin, self.gaussian_slider = create_parameter_group(
            "Gaussian Filter", 0.1, 10.0, 1.0, step=0.1, double=True
        )

        # CLAHE Clip Limit
        self.clahe_check, self.clahe_clip_spin, self.clahe_clip_slider = create_parameter_group(
            "CLAHE Clip Limit", 0.1, 100.0, 16.0, step=0.1, double=True
        )

        # CLAHE Grid Size - maintain alignment with other controls
        self.clahe_grid_control, self.clahe_grid_spin, self.clahe_grid_slider = create_parameter_group(
            "CLAHE Grid Size", 1, 64, 16, checkbox=False
        )

        # Set initial disabled state for grid controls
        self.clahe_grid_spin.setEnabled(False)
        self.clahe_grid_slider.setEnabled(False)

        layout.addLayout(form_layout)

        # Preview checkbox
        preview_layout = QHBoxLayout()
        self.preview_check = QCheckBox("Show Preview")
        preview_layout.addWidget(self.preview_check)
        preview_layout.addStretch()
        layout.addLayout(preview_layout)

        # Action buttons
        button_layout = QHBoxLayout()
        self.preprocess_btn = QPushButton("Run Preprocessing")
        self.reset_btn = QPushButton("Reset Parameters")
        button_layout.addWidget(self.preprocess_btn)
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
        """Connect widget signals with special handling for odd-only values"""
        # Intensity range controls
        self.intensity_slider.valueChanged.connect(self._update_from_intensity_slider)
        self.min_spin.valueChanged.connect(self._update_from_intensity_spinboxes)
        self.max_spin.valueChanged.connect(self._update_from_intensity_spinboxes)

        # Helper function to connect slider and spinbox
        def connect_slider_spin(slider, spinbox, checkbox=None, scale_factor=1.0, odd_only=False):
            def update_from_slider(value):
                spinbox.blockSignals(True)
                if odd_only:
                    # Convert slider value back to odd number
                    actual_value = 3 + (value * 2)  # Starts at 3, increases by 2
                    spinbox.setValue(actual_value)
                else:
                    spinbox.setValue(value * scale_factor)
                spinbox.blockSignals(False)
                self.update_parameters()

            def update_from_spin(value):
                slider.blockSignals(True)
                if odd_only:
                    # Convert odd number to slider value
                    slider_value = int((value - 3) / 2)  # Convert back to 0-based index
                    slider.setValue(slider_value)
                else:
                    slider.setValue(int(value / scale_factor))
                slider.blockSignals(False)
                self.update_parameters()

            slider.valueChanged.connect(update_from_slider)
            spinbox.valueChanged.connect(update_from_spin)

            if checkbox:
                checkbox.toggled.connect(slider.setEnabled)
                checkbox.toggled.connect(spinbox.setEnabled)
                checkbox.toggled.connect(self.update_parameters)

        # Connect parameter controls
        connect_slider_spin(self.median_slider, self.median_size_spin, self.median_check, odd_only=True)
        connect_slider_spin(self.gaussian_slider, self.gaussian_sigma_spin, self.gaussian_check, 0.1)
        connect_slider_spin(self.clahe_clip_slider, self.clahe_clip_spin, self.clahe_check, 0.1)
        connect_slider_spin(self.clahe_grid_slider, self.clahe_grid_spin)

        # CLAHE checkbox controls both clip and grid parameters
        self.clahe_check.toggled.connect(self.clahe_grid_spin.setEnabled)
        self.clahe_check.toggled.connect(self.clahe_grid_slider.setEnabled)

        # Preview
        self.preview_check.toggled.connect(self.toggle_preview)

        # Viewer dims change
        if self.viewer is not None:
            self.viewer.dims.events.current_step.connect(self.update_preview_frame)

        # Buttons
        self.preprocess_btn.clicked.connect(self.run_preprocessing)
        self.reset_btn.clicked.connect(self.reset_parameters)
    def _update_from_intensity_slider(self, values):
        """Update spinboxes when intensity range slider changes"""
        min_val, max_val = values
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)

        self.min_spin.setValue(min_val)
        self.max_spin.setValue(max_val)

        self.current_min_intensity = min_val
        self.current_max_intensity = max_val

        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)

        self.update_parameters()

    def _update_from_intensity_spinboxes(self):
        """Update intensity range slider when spinboxes change"""
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()

        # Ensure min <= max
        if min_val > max_val:
            if self.sender() == self.min_spin:
                max_val = min_val
                self.max_spin.setValue(max_val)
            else:
                min_val = max_val
                self.min_spin.setValue(min_val)

        self.current_min_intensity = min_val
        self.current_max_intensity = max_val

        self.intensity_slider.blockSignals(True)
        self.intensity_slider.setValue((min_val, max_val))
        self.intensity_slider.blockSignals(False)

        self.update_parameters()

    def update_parameters(self):
        """Update preprocessing parameters from UI controls with slider values"""
        try:
            params = PreprocessingParameters(
                min_intensity=self.current_min_intensity,
                max_intensity=self.current_max_intensity,
                enable_median_filter=self.median_check.isChecked(),
                median_filter_size=self.median_size_spin.value(),
                enable_gaussian_filter=self.gaussian_check.isChecked(),
                gaussian_sigma=self.gaussian_sigma_spin.value(),
                enable_clahe=self.clahe_check.isChecked(),
                clahe_clip_limit=self.clahe_clip_spin.value(),
                clahe_grid_size=self.clahe_grid_spin.value()
            )

            params.validate()
            self.preprocessor.update_parameters(params)

            self.preprocess_btn.setEnabled(True)
            self.status_label.setText("Parameters updated")
            self.parameters_updated.emit()

            if self.preview_check.isChecked():
                self.update_preview_frame()

        except ValueError as e:
            self.preprocess_btn.setEnabled(False)
            self.status_label.setText(f"Invalid parameters: {str(e)}")

    def reset_parameters(self):
        """Reset all parameters to defaults, including slider positions"""
        # Reset intensity range
        self.intensity_slider.setValue((0, 255))
        self.min_spin.setValue(0)
        self.max_spin.setValue(255)
        self.current_min_intensity = 0
        self.current_max_intensity = 255

        # Reset median filter
        self.median_check.setChecked(False)
        self.median_size_spin.setValue(3)
        self.median_slider.setValue(3)

        # Reset gaussian filter
        self.gaussian_check.setChecked(False)
        self.gaussian_sigma_spin.setValue(1.0)
        self.gaussian_slider.setValue(10)  # Value of 10 corresponds to 1.0 with scale factor

        # Reset CLAHE
        self.clahe_check.setChecked(False)
        self.clahe_clip_spin.setValue(16.0)
        self.clahe_clip_slider.setValue(160)  # Value of 160 corresponds to 16.0 with scale factor
        self.clahe_grid_spin.setValue(16)
        self.clahe_grid_slider.setValue(16)

        self.status_label.setText("Parameters reset to defaults")
        self.update_parameters()

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all controls including sliders"""
        controls = [
            self.preprocess_btn,
            self.preview_check,
            self.reset_btn,
            self.intensity_slider,
            self.min_spin,
            self.max_spin,
            self.median_size_spin,
            self.median_slider,
            self.gaussian_sigma_spin,
            self.gaussian_slider,
            self.clahe_clip_spin,
            self.clahe_clip_slider,
            self.clahe_grid_spin,
            self.clahe_grid_slider,
            self.median_check,
            self.gaussian_check,
            self.clahe_check
        ]

        for control in controls:
            control.setEnabled(enabled)

        # If enabled, respect the checkbox states
        if enabled:
            self.median_size_spin.setEnabled(self.median_check.isChecked())
            self.median_slider.setEnabled(self.median_check.isChecked())
            self.gaussian_sigma_spin.setEnabled(self.gaussian_check.isChecked())
            self.gaussian_slider.setEnabled(self.gaussian_check.isChecked())

            clahe_enabled = self.clahe_check.isChecked()
            self.clahe_clip_spin.setEnabled(clahe_enabled)
            self.clahe_clip_slider.setEnabled(clahe_enabled)
            self.clahe_grid_spin.setEnabled(clahe_enabled)
            self.clahe_grid_slider.setEnabled(clahe_enabled)

    def update_preview_frame(self, event: Optional[Event] = None):
        """Update the preview for the current frame"""
        if not self.preview_enabled or self.original_layer is None:
            return

        try:
            # Get current frame
            if self.original_layer.data.ndim == 3:
                current_step = self.viewer.dims.current_step[0]
                frame = self.original_layer.data[current_step].copy()
            else:
                frame = self.original_layer.data.copy()

            if frame.ndim != 2:
                raise ValueError(f"Invalid frame dimensions: {frame.shape}")

            # Store original statistics before conversion
            original_min = float(frame.min())
            original_max = float(frame.max())
            original_mean = float(frame.mean())
            original_std = float(frame.std())

            # First convert to 8-bit
            frame_8bit = self.preprocessor.convert_to_8bit(frame)

            # Then apply preprocessing on 8-bit image
            processed_frame, info = self.preprocessor.preprocess_frame(frame_8bit)

            # Update preview layer
            if self.preview_layer is None:
                self.preview_layer = self.viewer.add_image(
                    np.zeros_like(frame_8bit, dtype=np.uint8),
                    name='Preview',
                    visible=True
                )

            self.preview_layer.data = processed_frame
            self.preview_layer.contrast_limits = (0, 255)

            # Display processing info including original and converted ranges
            info_text = (
                f"Preview - Original range: ({original_min:.0f}, {original_max:.0f})\n"
                f"Original mean: {original_mean:.1f}\n"
                f"Original std: {original_std:.1f}\n"
                f"Final mean: {info['final_mean']:.1f}"
            )
            self.status_label.setText(info_text)

        except Exception as e:
            error_msg = f"Preview failed: {str(e)}"
            self.status_label.setText(error_msg)
            self.preview_check.setChecked(False)

    def run_preprocessing(self):
        """Run preprocessing on the entire stack"""
        if self.preview_enabled:
            self.preview_check.setChecked(False)

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

                # Get frame
                frame = stack[frame_idx].copy()

                # First convert to 8-bit
                frame_8bit = self.preprocessor.convert_to_8bit(frame)

                # Then process the 8-bit frame
                try:
                    processed_frame, frame_info = self.preprocessor.preprocess_frame(frame_8bit)
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

            # Add new preprocessed layer
            preprocessed_layer = self.viewer.add_image(
                processed_stack,
                name='Preprocessed',
                visible=True,
                metadata={'preprocessing_info': preprocessing_info}
            )

            # Set consistent contrast limits for 8-bit
            preprocessed_layer.contrast_limits = (0, 255)

            # Store results in data manager
            self.data_manager.preprocessed_data = processed_stack
            self.data_manager.preprocessing_info = preprocessing_info

            self._update_status("Preprocessing complete", 100)
            self.preprocessing_completed.emit(processed_stack, preprocessing_info)

        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            self._update_status(error_msg, 0)
            self.preprocessing_failed.emit(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
        finally:
            self._set_controls_enabled(True)


    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and progress bar"""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)

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
