import logging
from typing import Optional

import napari
import numpy as np
from napari.utils.events import Event
from qtpy.QtCore import Signal, Qt, QObject, QThread
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QGroupBox, QSizePolicy, QProgressBar, QLabel,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QFormLayout
)
from qtrangeslider import QRangeSlider

from .base_widget import BaseAnalysisWidget, ProcessingError
from .data_manager import DataManager
from .preprocessing import PreprocessingParameters, ImagePreprocessor
from .visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class PreprocessingWorker(QObject):
    """Worker object to run preprocessing in background thread"""
    progress = Signal(int, str)
    finished = Signal(tuple)  # Emits (processed_stack, preprocessing_info)
    error = Signal(Exception)

    def __init__(self, preprocessor: ImagePreprocessor, stack: np.ndarray):
        super().__init__()
        self.preprocessor = preprocessor
        self.stack = stack

    def run(self):
        try:
            total_frames = len(self.stack)
            processed_frames = []
            preprocessing_info = []

            # Process each frame
            for frame_idx in range(total_frames):
                progress = int(5 + (90 * frame_idx / total_frames))
                self.progress.emit(progress, f"Processing frame {frame_idx + 1}/{total_frames}")

                # Get frame
                frame = self.stack[frame_idx].copy()

                # Convert to 8-bit and process
                try:
                    frame_8bit = self.preprocessor.convert_to_8bit(frame)
                    processed_frame, frame_info = self.preprocessor.preprocess_frame(frame_8bit)
                    processed_frames.append(processed_frame)
                    preprocessing_info.append(frame_info)
                except Exception as e:
                    raise ProcessingError(
                        f"Error processing frame {frame_idx}",
                        str(e)
                    )

            # Combine processed frames
            processed_stack = np.stack(processed_frames, axis=0)
            self.finished.emit((processed_stack, preprocessing_info))

        except Exception as e:
            self.error.emit(e)


class PreprocessingWidget(BaseAnalysisWidget):
    """Widget for controlling image preprocessing parameters"""

    preprocessing_completed = Signal(np.ndarray, list)  # Processed stack and info

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

        self.preprocessor = ImagePreprocessor()

        # Track preview state
        self.preview_enabled = False
        self.original_layer = None
        self.preview_layer = None

        # Track intensity range
        self.current_min_intensity = 0
        self.current_max_intensity = 255

        # Initialize thread management
        self._preprocessing_thread = None
        self._preprocessing_worker = None

        # Initialize all controls
        self._initialize_controls()

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

        # Add layer events handlers
        self.viewer.layers.events.removed.connect(self._update_ui_state)
        self.viewer.layers.events.inserted.connect(self._update_ui_state)
        self.viewer.layers.selection.events.changed.connect(self._update_ui_state)

        # Initial UI state update
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

        # Create and add groups
        right_layout.addWidget(self._create_intensity_group())
        right_layout.addWidget(self._create_filter_group())

        # Add preprocess button
        right_layout.addWidget(self.preprocess_btn)

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

        # Add to the main layout
        self.main_layout.addWidget(right_container)
        self.main_layout.addStretch(1)

        # Register controls
        self._register_controls()

    def _create_filter_controls(self, form_layout: QFormLayout):
        """Create filter parameter controls"""
        # Add tooltip about disabling filters
        tooltip_label = QLabel("Note: Set parameter to 0 to disable a filter")
        tooltip_label.setStyleSheet("color: gray; font-style: italic;")
        form_layout.addRow(tooltip_label)

        def create_parameter_group(
                label: str,
                min_val: float,
                max_val: float,
                default_val: float,
                step: float = 1.0,
                double: bool = False,
                odd_only: bool = False,
                tooltip: str = ""
        ):
            control_layout = QHBoxLayout()

            spin = QDoubleSpinBox() if double else QSpinBox()
            spin.setRange(0, max_val)
            spin.setValue(default_val)
            spin.setSingleStep(step if not odd_only else 2)
            spin.setFixedWidth(80)
            if double:
                spin.setDecimals(1)
            spin.setToolTip(tooltip)
            control_layout.addWidget(spin)

            slider = QSlider(Qt.Horizontal)
            if odd_only:
                # For median filter, allow full range including 0
                slider_range = ((max_val - 1) // 2)  # Convert max value to slider steps
                slider.setRange(0, slider_range)
                if default_val == 0:
                    slider.setValue(0)
                else:
                    slider.setValue((default_val - 1) // 2)
            else:
                slider.setRange(0, int(max_val * (1 / step)))
                slider.setValue(int(default_val * (1 / step)))
            slider.setToolTip(tooltip)
            control_layout.addWidget(slider, stretch=1)

            # Create label with tooltip
            label_widget = QLabel(label + ":")
            label_widget.setToolTip(tooltip)
            form_layout.addRow(label_widget, control_layout)
            return spin, slider

        # Create controls with tooltips
        self.median_size_spin, self.median_slider = create_parameter_group(
            "Median Filter Size", 3, 15, 0, step=2, odd_only=True,
            tooltip="Size of median filter kernel (odd numbers only).\nReduces noise while preserving edges.\nSet to 0 to disable."
        )

        self.gaussian_sigma_spin, self.gaussian_slider = create_parameter_group(
            "Gaussian Sigma", 0.1, 10.0, 0, step=0.1, double=True,
            tooltip="Standard deviation of Gaussian blur.\nLarger values create more blur.\nSet to 0 to disable."
        )

        self.clahe_clip_spin, self.clahe_clip_slider = create_parameter_group(
            "CLAHE Clip Limit", 0.1, 100.0, 0, step=0.1, double=True,
            tooltip="Contrast limit for histogram equalization.\nHigher values allow more contrast enhancement.\nSet to 0 to disable CLAHE."
        )

        self.clahe_grid_spin, self.clahe_grid_slider = create_parameter_group(
            "CLAHE Grid Size", 1, 64, 16,
            tooltip="Size of grid for adaptive histogram equalization.\nSmaller values give more local contrast enhancement."
        )

        # Add preview checkbox with tooltip
        preview_layout = QHBoxLayout()
        self.preview_check.setText("")
        self.preview_check.setFixedWidth(30)
        self.preview_check.setToolTip("Show live preview of preprocessing results.\nUpdates automatically when parameters change.")
        preview_layout.addWidget(self.preview_check)
        preview_label = QLabel("")
        preview_layout.addWidget(preview_label)
        preview_layout.addStretch()
        form_layout.addRow("Show Preview:", preview_layout)

    def _create_intensity_group(self) -> QGroupBox:
        """Create intensity range controls group"""
        group = QGroupBox("Intensity Range")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Configure intensity slider with tooltip
        self.intensity_slider.setRange(0, 255)
        self.intensity_slider.setValue((0, 255))
        self.intensity_slider.setToolTip(
            "Adjust minimum and maximum intensity values.\n"
            "Values outside this range will be clipped."
        )
        layout.addWidget(self.intensity_slider)

        # Configure spinboxes
        spin_layout = QHBoxLayout()
        spin_layout.setContentsMargins(0, 0, 0, 0)

        # Configure individual spinboxes with tooltips
        for spin in (self.min_spin, self.max_spin):
            spin.setRange(0, 255)
            spin.setFixedWidth(80)
            spin.setToolTip(
                "Minimum/maximum intensity value.\n"
                "Values outside this range will be clipped."
            )

        # Set initial values explicitly
        self.min_spin.setValue(0)
        self.max_spin.setValue(255)

        # Left spinbox
        spin_layout.addWidget(self.min_spin)

        # Stretch to push max spinbox right
        spin_layout.addStretch(1)

        # Right spinbox - explicitly specify alignment
        spin_layout.addWidget(self.max_spin, alignment=Qt.AlignRight)

        layout.addLayout(spin_layout)
        group.setLayout(layout)

        # Add tooltip to the group itself
        group.setToolTip(
            "Set the intensity range for preprocessing.\n"
            "Values below minimum will be set to 0.\n"
            "Values above maximum will be set to 255."
        )

        return group

    def _create_filter_group(self) -> QGroupBox:
        """Create parameters group with all controls"""
        group = QGroupBox("Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Create form layout for parameters
        form_layout = QFormLayout()
        form_layout.setSpacing(4)

        # Create filter controls
        self._create_filter_controls(form_layout)

        # Add form layout to main layout
        layout.addLayout(form_layout)

        # Add reset button spanning full width
        layout.addSpacing(8)  # Add some space before the reset button
        layout.addWidget(self.reset_btn)

        group.setLayout(layout)
        return group

    def _connect_signals(self):
        """Connect widget signals"""
        self.preprocess_btn.clicked.connect(self.run_preprocessing)
        self.reset_btn.clicked.connect(self.reset_parameters)

        # Intensity range
        self.intensity_slider.valueChanged.connect(self._update_from_intensity_slider)
        self.min_spin.valueChanged.connect(self._update_from_intensity_spinboxes)
        self.max_spin.valueChanged.connect(self._update_from_intensity_spinboxes)

        def connect_slider_spin(slider, spinbox, scale_factor=1.0, odd_only=False):
            def update_from_slider(value):
                spinbox.blockSignals(True)
                if odd_only:
                    if value == 0:
                        spinbox.setValue(0)
                    else:
                        actual_value = 1 + (value * 2)  # Start from 3,5,7,... for non-zero values
                        spinbox.setValue(actual_value)
                else:
                    spinbox.setValue(value * scale_factor)
                spinbox.blockSignals(False)
                self.update_parameters()

            def update_from_spin(value):
                slider.blockSignals(True)
                if odd_only:
                    if value == 0:
                        slider.setValue(0)
                    else:
                        slider_value = (value - 1) // 2  # Convert back to slider value
                        slider.setValue(slider_value)
                else:
                    slider.setValue(int(value / scale_factor))
                slider.blockSignals(False)
                self.update_parameters()

            slider.valueChanged.connect(update_from_slider)
            spinbox.valueChanged.connect(update_from_spin)

        # Connect parameter controls
        connect_slider_spin(self.median_slider, self.median_size_spin, odd_only=True)
        connect_slider_spin(self.gaussian_slider, self.gaussian_sigma_spin, 0.1)
        connect_slider_spin(self.clahe_clip_slider, self.clahe_clip_spin, 0.1)
        connect_slider_spin(self.clahe_grid_slider, self.clahe_grid_spin)

        # Preview
        self.preview_check.toggled.connect(self.toggle_preview)

        # Viewer dims change
        if self.viewer is not None:
            self.viewer.dims.events.current_step.connect(self.update_preview_frame)

    def _initialize_controls(self):
        """Initialize all UI controls"""
        # Intensity controls
        self.intensity_slider = QRangeSlider(Qt.Horizontal)
        self.min_spin = QSpinBox()
        self.max_spin = QSpinBox()

        # Filter controls
        self.median_size_spin = QSpinBox()
        self.median_slider = QSlider(Qt.Horizontal)

        self.gaussian_sigma_spin = QDoubleSpinBox()
        self.gaussian_slider = QSlider(Qt.Horizontal)

        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_slider = QSlider(Qt.Horizontal)
        self.clahe_grid_spin = QSpinBox()
        self.clahe_grid_slider = QSlider(Qt.Horizontal)

        # Preview control
        self.preview_check = QCheckBox("")

        # Action buttons
        self.preprocess_btn = QPushButton("Run Preprocessing")
        self.reset_btn = QPushButton("Reset Parameters")

    def _register_controls(self):
        """Register all controls with base widget"""
        for control in [
            self.intensity_slider, self.min_spin, self.max_spin,
            self.median_size_spin, self.median_slider,
            self.gaussian_sigma_spin, self.gaussian_slider,
            self.clahe_clip_spin, self.clahe_clip_slider,
            self.clahe_grid_spin, self.clahe_grid_slider,
            self.preview_check,
            self.preprocess_btn,
            self.reset_btn
        ]:
            self.register_control(control)

    def _update_ui_state(self, event=None):
        """Update UI based on current state"""
        active_layer = self._get_active_image_layer()
        has_valid_image = (active_layer is not None and
                           isinstance(active_layer, napari.layers.Image) and
                           active_layer.data.ndim in [2, 3])

        # Update button states
        self.preprocess_btn.setEnabled(has_valid_image)
        self.preview_check.setEnabled(has_valid_image)

        # Disable preview if the original layer is gone or no longer selected
        if self.preview_enabled and self.original_layer is not None:
            if self.original_layer not in self.viewer.layers:
                self.preview_check.setChecked(False)
                self.preview_enabled = False
                if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.preview_layer)
                self.preview_layer = None
                self.original_layer = None

    def update_parameters(self):
        """Update preprocessing parameters from UI controls"""
        try:
            # Get values from controls
            median_size = self.median_size_spin.value()
            gaussian_sigma = self.gaussian_sigma_spin.value()
            clahe_clip = self.clahe_clip_spin.value()
            clahe_grid = self.clahe_grid_spin.value()

            # Create parameters with appropriate enabled states and valid values
            params = PreprocessingParameters(
                min_intensity=self.current_min_intensity,
                max_intensity=self.current_max_intensity,

                # For median filter: if size is 0, disable and use minimum valid size
                enable_median_filter=median_size > 0,
                median_filter_size=max(3, median_size) if median_size > 0 else 3,

                # For Gaussian: if sigma is 0, disable and use minimum valid sigma
                enable_gaussian_filter=gaussian_sigma > 0,
                gaussian_sigma=max(0.1, gaussian_sigma) if gaussian_sigma > 0 else 0.1,

                # For CLAHE: if either clip limit or grid size is 0, disable the filter
                enable_clahe=(clahe_clip > 0 and clahe_grid > 0),
                clahe_clip_limit=max(0.1, clahe_clip) if clahe_clip > 0 else 0.1,
                clahe_grid_size=max(1, clahe_grid) if clahe_grid > 0 else 1
            )

            params.validate()
            self.preprocessor.update_parameters(params)

            self._update_status("Parameters updated")
            self.parameters_updated.emit()

            if self.preview_enabled:
                self.update_preview_frame()

        except ValueError as e:
            raise ProcessingError("Invalid parameters", str(e))

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        # Reset intensity range
        self.intensity_slider.setValue((0, 255))
        self.min_spin.setValue(0)
        self.max_spin.setValue(255)
        self.current_min_intensity = 0
        self.current_max_intensity = 255

        # Reset filters to disabled state (0)
        self.median_size_spin.setValue(0)
        self.median_slider.setValue(0)

        self.gaussian_sigma_spin.setValue(0)
        self.gaussian_slider.setValue(0)

        self.clahe_clip_spin.setValue(0)
        self.clahe_clip_slider.setValue(0)
        self.clahe_grid_spin.setValue(16)  # Keep a valid grid size even when CLAHE is disabled
        self.clahe_grid_slider.setValue(16)

        self._update_status("Parameters reset to defaults")
        self.update_parameters()

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

    def toggle_preview(self, enabled: bool):
        """Toggle preview mode"""
        self.preview_enabled = enabled

        try:
            if enabled:
                # Always get the currently active layer when preview is enabled
                current_layer = self._get_active_image_layer()
                if current_layer is None:
                    raise ProcessingError("No image layer found")

                # If we're switching to a different layer, clean up the old preview
                if self.original_layer != current_layer:
                    if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
                        self.viewer.layers.remove(self.preview_layer)
                    self.preview_layer = None

                # Update the original layer reference
                self.original_layer = current_layer

                if self.preview_layer is None:
                    preview_data = (self.original_layer.data[0] if self.original_layer.data.ndim == 3
                                    else self.original_layer.data)
                    # Store current selection before adding preview
                    current_selected = list(self.viewer.layers.selection)

                    self.preview_layer = self.viewer.add_image(
                        np.zeros_like(preview_data),
                        name='Preview',
                        visible=True
                    )

                    # Restore original selection
                    self.viewer.layers.selection.clear()
                    for layer in current_selected:
                        self.viewer.layers.selection.add(layer)

                self.update_preview_frame()
            else:
                if self.preview_layer is not None:
                    # Store current selection before removing preview
                    current_selected = list(self.viewer.layers.selection)

                    self.viewer.layers.remove(self.preview_layer)

                    # Restore original selection, excluding the preview layer
                    self.viewer.layers.selection.clear()
                    for layer in current_selected:
                        if layer != self.preview_layer:
                            self.viewer.layers.selection.add(layer)

                    self.preview_layer = None
                    self.original_layer = None  # Also clear the original layer reference

        except Exception as e:
            self.preview_check.setChecked(False)
            self.preview_enabled = False
            if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
                self.viewer.layers.remove(self.preview_layer)
            self.preview_layer = None
            self.original_layer = None  # Clear the original layer reference on error
            raise ProcessingError("Preview failed", str(e))

    def update_preview_frame(self, event: Optional[Event] = None):
        """Update the preview for the current frame"""
        if not self.preview_enabled or self.original_layer is None:
            return

        try:
            if self.original_layer.data.ndim == 3:
                current_step = self.viewer.dims.current_step[0]
                frame = self.original_layer.data[current_step].copy()
            else:
                frame = self.original_layer.data.copy()

            if frame.ndim != 2:
                raise ProcessingError(f"Invalid frame dimensions: {frame.shape}")

            # Process frame
            frame_8bit = self.preprocessor.convert_to_8bit(frame)
            processed_frame, info = self.preprocessor.preprocess_frame(frame_8bit)

            # Update preview
            if self.preview_layer is None:
                # Store current selection before adding preview
                current_selected = list(self.viewer.layers.selection)

                self.preview_layer = self.viewer.add_image(
                    np.zeros_like(frame_8bit, dtype=np.uint8),
                    name='Preview',
                    visible=True
                )

                # Restore original selection
                self.viewer.layers.selection.clear()
                for layer in current_selected:
                    self.viewer.layers.selection.add(layer)
            else:
                # Update preview data without changing selection
                current_selected = list(self.viewer.layers.selection)
                self.preview_layer.data = processed_frame
                self.viewer.layers.selection.clear()
                for layer in current_selected:
                    self.viewer.layers.selection.add(layer)

            self.preview_layer.contrast_limits = (0, 255)

            # Update status
            info_text = (
                f"Preview - Original range: ({frame.min():.0f}, {frame.max():.0f})\n"
                f"Original mean: {frame.mean():.1f}, std: {frame.std():.1f}\n"
                f"Final mean: {info['final_mean']:.1f}"
            )
            self._update_status(info_text)

        except Exception as e:
            raise ProcessingError("Preview failed", str(e))

    def run_preprocessing(self):
        """Run preprocessing on the entire stack"""
        if self.preview_enabled:
            self.preview_check.setChecked(False)

        try:
            # Get active layer
            active_layer = self._get_active_image_layer()
            if active_layer is None:
                raise ProcessingError("No image layer selected")

            # Store the data and name from the active layer before processing
            original_data = active_layer.data.copy()
            original_name = active_layer.name

            # Disable controls during processing
            self._set_controls_enabled(False)
            self._update_status("Starting preprocessing...", 0)

            # Get and validate the image data
            stack = self._ensure_stack_format(original_data)
            if not self._validate_input_data(stack):
                raise ProcessingError("Invalid input data format")

            # Create worker and thread
            self._preprocessing_thread = QThread()
            self._preprocessing_worker = PreprocessingWorker(self.preprocessor, stack)
            self._preprocessing_worker.moveToThread(self._preprocessing_thread)

            # Connect signals
            self._preprocessing_thread.started.connect(self._preprocessing_worker.run)
            self._preprocessing_worker.progress.connect(self._handle_preprocessing_progress)
            self._preprocessing_worker.finished.connect(lambda results: self._handle_preprocessing_complete(results, original_name))
            self._preprocessing_worker.error.connect(self._handle_preprocessing_error)
            self._preprocessing_worker.finished.connect(self._preprocessing_thread.quit)
            self._preprocessing_worker.finished.connect(self._preprocessing_worker.deleteLater)
            self._preprocessing_thread.finished.connect(self._preprocessing_thread.deleteLater)

            # Start preprocessing
            self._preprocessing_thread.start()

        except Exception as e:
            self._handle_error(ProcessingError(
                message="Failed to start preprocessing",
                details=str(e),
                component=self.__class__.__name__
            ))
            self._set_controls_enabled(True)

    def _handle_preprocessing_progress(self, progress: int, message: str):
        """Handle progress updates from worker"""
        self._update_status(message, progress)

    def _handle_preprocessing_complete(self, results: tuple, original_name: str):
        """Handle completion of preprocessing"""
        try:
            processed_stack, preprocessing_info = results

            self._update_status("Updating visualization...", 95)

            # Store current selection before adding new layer
            current_selected = list(self.viewer.layers.selection)

            # Add new preprocessed layer - napari will handle duplicate names automatically
            preprocessed_layer = self.viewer.add_image(
                processed_stack,
                name='Preprocessed',
                visible=True,
                metadata={'preprocessing_info': preprocessing_info}
            )

            # Set consistent contrast limits for 8-bit
            preprocessed_layer.contrast_limits = (0, 255)

            # Restore original selection
            self.viewer.layers.selection.clear()
            for layer in current_selected:
                self.viewer.layers.selection.add(layer)

            # Store results
            self.data_manager.preprocessed_data = processed_stack
            self.data_manager.preprocessing_info = preprocessing_info

            self._update_status("Preprocessing complete", 100)
            self.preprocessing_completed.emit(processed_stack, preprocessing_info)

        except Exception as e:
            self._handle_preprocessing_error(e)
        finally:
            self._set_controls_enabled(True)

    def _handle_preprocessing_error(self, error):
        """Handle errors from worker"""
        if isinstance(error, ProcessingError):
            self._handle_error(error)
        else:
            self._handle_error(ProcessingError(
                message="Error during preprocessing",
                details=str(error),
                component=self.__class__.__name__
            ))
        self._set_controls_enabled(True)

    def cleanup(self):
        """Clean up resources"""
        # Clean up preprocessing thread
        if self._preprocessing_thread and self._preprocessing_thread.isRunning():
            self._preprocessing_thread.quit()
            self._preprocessing_thread.wait()

        if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
            self.viewer.layers.remove(self.preview_layer)
        self.preview_layer = None
        self.original_layer = None
        super().cleanup()
