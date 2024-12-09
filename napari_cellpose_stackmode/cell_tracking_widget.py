from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QProgressDialog,
                            QProgressBar, QFileDialog, QMessageBox, QHBoxLayout,
                            QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox)
from qtpy.QtCore import Signal, Qt
import napari
import numpy as np
from pathlib import Path
import logging
from typing import Optional
from napari_cellpose_stackmode.data_manager import DataManager
from napari_cellpose_stackmode.error_handling import ProcessingError, ErrorSignals
from napari_cellpose_stackmode.visualization_manager import VisualizationManager
from napari_cellpose_stackmode.cell_tracking import CellTracker
from napari_cellpose_stackmode.structure import AnalysisConfig, TrackingParameters

logger = logging.getLogger(__name__)


class CellTrackingWidget(QWidget):
    """Enhanced widget for cell tracking operations with robust error handling."""

    tracking_completed = Signal(np.ndarray)  # Emits tracked data
    tracking_failed = Signal(str)  # Error message
    parameters_updated = Signal()  # Parameters changed

    def __init__(self, viewer: "napari.Viewer", data_manager: DataManager,
                 visualization_manager: VisualizationManager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.visualization_manager = visualization_manager
        self.tracking_params = TrackingParameters()
        self.tracker = CellTracker(AnalysisConfig())

        # Initialize error handling
        self.error_signals = ErrorSignals()
        self.error_signals.processing_error.connect(self._handle_processing_error)
        self.error_signals.critical_error.connect(self._handle_critical_error)
        self.error_signals.warning.connect(self._handle_warning)

        # Track processing state
        self._processing = False
        self._batch_processing = False

        # Setup UI and connect signals
        self.setup_ui()
        self.connect_signals()

    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and progress"""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)
        logger.info(message)

    def setup_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Cell Tracking")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Parameters section with all the existing parameter controls
        param_group = QFormLayout()

        # Create all UI elements and store them as instance variables
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 1.0)
        self.overlap_spin.setSingleStep(0.1)
        self.overlap_spin.setValue(self.tracking_params.min_overlap_ratio)
        self.overlap_spin.setToolTip("Minimum overlap ratio between frames for cell identity assignment (0-1)")
        param_group.addRow("Min Overlap Ratio:", self.overlap_spin)

        self.displacement_spin = QDoubleSpinBox()
        self.displacement_spin.setRange(0.0, float('inf'))
        self.displacement_spin.setMaximum(1e9)
        self.displacement_spin.setSingleStep(5.0)
        self.displacement_spin.setValue(self.tracking_params.max_displacement)
        self.displacement_spin.setToolTip("Maximum allowed cell movement between frames (pixels)")
        param_group.addRow("Max Displacement:", self.displacement_spin)

        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setRange(0, int(1e9))
        self.cell_size_spin.setSingleStep(10)
        self.cell_size_spin.setValue(self.tracking_params.min_cell_size)
        self.cell_size_spin.setToolTip("Minimum cell size in pixels (0 to disable filtering)")
        param_group.addRow("Min Cell Size:", self.cell_size_spin)

        self.gap_closing_check = QCheckBox()
        self.gap_closing_check.setChecked(self.tracking_params.enable_gap_closing)
        self.gap_closing_check.setToolTip("Enable tracking across gaps in segmentation")
        param_group.addRow("Enable Gap Closing:", self.gap_closing_check)

        self.gap_frames_spin = QSpinBox()
        self.gap_frames_spin.setRange(1, int(1e4))
        self.gap_frames_spin.setSingleStep(1)
        self.gap_frames_spin.setValue(self.tracking_params.max_frame_gap)
        self.gap_frames_spin.setToolTip("Maximum number of frames to look ahead for gap closing")
        param_group.addRow("Max Frame Gap:", self.gap_frames_spin)

        layout.addLayout(param_group)

        # Button layout - simplified to only include track cells button
        button_layout = QHBoxLayout()

        self.track_btn = QPushButton("Track Cells")
        self.track_btn.clicked.connect(self.run_tracking)
        button_layout.addWidget(self.track_btn)

        self.reset_btn = QPushButton("Reset Parameters")
        self.reset_btn.clicked.connect(self.reset_parameters)
        button_layout.addWidget(self.reset_btn)

        layout.addLayout(button_layout)

        # Progress and status
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def connect_signals(self):
        """Connect all parameter control signals"""
        # Make sure all UI elements exist before connecting
        if hasattr(self, 'overlap_spin'):
            self.overlap_spin.valueChanged.connect(self.update_parameters)
        if hasattr(self, 'displacement_spin'):
            self.displacement_spin.valueChanged.connect(self.update_parameters)
        if hasattr(self, 'cell_size_spin'):
            self.cell_size_spin.valueChanged.connect(self.update_parameters)
        if hasattr(self, 'gap_closing_check'):
            self.gap_closing_check.stateChanged.connect(self.update_parameters)
        if hasattr(self, 'gap_frames_spin'):
            self.gap_frames_spin.valueChanged.connect(self.update_parameters)

    def _handle_processing_error(self, error: ProcessingError):
        """Handle recoverable processing errors"""
        self._update_status(f"Error: {error.message}")
        self.progress_bar.setValue(0)
        QMessageBox.warning(self, "Processing Error",
                            f"{error.message}\n\nDetails: {error.details}")
        self._set_controls_enabled(True)
        self._processing = False
        self._batch_processing = False

    def _handle_critical_error(self, error: ProcessingError):
        """Handle non-recoverable errors"""
        self._update_status(f"Critical Error: {error.message}")
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Critical Error",
                             f"{error.message}\n\nDetails: {error.details}")
        self._set_controls_enabled(False)
        self._processing = False
        self._batch_processing = False

    def _handle_warning(self, message: str):
        """Handle warning messages"""
        self.status_label.setText(f"Warning: {message}")

    def run_tracking(self):
        """Run cell tracking with robust error handling"""
        if self._processing:
            self.error_signals.warning.emit("Tracking already in progress")
            return

        try:
            self._processing = True
            self._set_controls_enabled(False)
            self._update_status("Starting tracking...", 0)

            # Get selected layer
            selected = self.viewer.layers.selection.active
            if selected is None:
                raise ProcessingError(
                    message="No layer selected",
                    details="Please select a layer containing cell segmentation",
                    component="CellTrackingWidget"
                )

            # Validate input data
            stack = self._ensure_stack_format(selected.data)
            if not self._validate_input_data(stack):
                raise ProcessingError(
                    message="Invalid input data",
                    details="Input data must be a 3D stack of segmentation masks",
                    component="CellTrackingWidget"
                )

            self._update_status("Running cell tracking...", 30)

            # Run tracking with current parameters
            tracked_labels = self.tracker.track_cells(stack)

            # Validate output
            if not self._validate_output_data(tracked_labels):
                raise ProcessingError(
                    message="Invalid tracking results",
                    details="Tracking produced invalid output data",
                    component="CellTrackingWidget"
                )

            # Update data manager and visualization
            self._update_status("Updating visualization...", 70)
            self.data_manager.tracked_data = tracked_labels
            self.visualization_manager.update_tracking_visualization(tracked_labels)

            self._update_status("Cell tracking complete", 100)
            self.tracking_completed.emit(tracked_labels)

        except ProcessingError as e:
            self.error_signals.processing_error.emit(e)
        except Exception as e:
            self.error_signals.critical_error.emit(
                ProcessingError(
                    message="Unexpected error during tracking",
                    details=str(e),
                    component="CellTrackingWidget",
                    recoverable=False
                )
            )
        finally:
            self._processing = False
            self._set_controls_enabled(True)

    def _validate_input_data(self, data: np.ndarray) -> bool:
        """Validate input data format and content"""
        try:
            if data.ndim != 3:
                return False
            if not np.issubdtype(data.dtype, np.integer):
                return False
            if data.min() < 0:
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating input data: {e}")
            return False

    def _validate_output_data(self, data: np.ndarray) -> bool:
        """Validate tracking output data"""
        try:
            if data is None:
                return False
            if data.ndim != 3:
                return False
            if not np.issubdtype(data.dtype, np.integer):
                return False
            if data.min() < 0:
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating output data: {e}")
            return False

    def update_parameters(self):
        """Update tracking parameters with validation"""
        try:
            new_params = TrackingParameters(
                min_overlap_ratio=self.overlap_spin.value(),
                max_displacement=self.displacement_spin.value(),
                min_cell_size=self.cell_size_spin.value(),
                enable_gap_closing=self.gap_closing_check.isChecked(),
                max_frame_gap=self.gap_frames_spin.value()
            )

            # Validate parameters
            new_params.validate()

            # Update tracker with new parameters
            self.tracking_params = new_params
            self.tracker.update_parameters(new_params)

            self.track_btn.setEnabled(True)
            self.status_label.setText("Parameters updated")
            self.parameters_updated.emit()

        except ValueError as e:
            self.error_signals.processing_error.emit(
                ProcessingError(
                    message="Invalid parameters",
                    details=str(e),
                    component="CellTrackingWidget"
                )
            )
            self.track_btn.setEnabled(False)

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all controls"""
        controls = [
            self.track_btn, self.reset_btn, self.overlap_spin,
            self.displacement_spin, self.cell_size_spin,
            self.gap_closing_check, self.gap_frames_spin
        ]
        for control in controls:
            control.setEnabled(enabled)

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.tracking_params = TrackingParameters()
        self.overlap_spin.setValue(self.tracking_params.min_overlap_ratio)
        self.displacement_spin.setValue(self.tracking_params.max_displacement)
        self.cell_size_spin.setValue(self.tracking_params.min_cell_size)
        self.gap_closing_check.setChecked(self.tracking_params.enable_gap_closing)
        self.gap_frames_spin.setValue(self.tracking_params.max_frame_gap)
        self.status_label.setText("Parameters reset to defaults")

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
