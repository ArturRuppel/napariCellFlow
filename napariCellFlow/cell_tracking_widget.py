import logging
import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QSizePolicy, QProgressBar, QLabel,
    QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton
)

from .base_widget import BaseAnalysisWidget, ProcessingError
from .cell_tracking import CellTracker
from .structure import AnalysisConfig, TrackingParameters

logger = logging.getLogger(__name__)


class CellTrackingWidget(BaseAnalysisWidget):
    """Widget for cell tracking operations with interactive parameters."""

    tracking_completed = Signal(object)  # Tracked data signal

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: "DataManager",
            visualization_manager: "VisualizationManager"
    ):
        super().__init__(
            viewer=viewer,
            data_manager=data_manager,
            visualization_manager=visualization_manager
        )

        # Initialize tracker and parameters
        self.tracking_params = TrackingParameters()
        self.tracker = CellTracker(AnalysisConfig())

        # Initialize controls
        self._initialize_controls()

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

        logger.debug("CellTrackingWidget initialized")

    def _initialize_controls(self):
        """Initialize all UI controls"""
        # Overlap controls
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 1.0)
        self.overlap_spin.setSingleStep(0.1)
        self.overlap_spin.setValue(self.tracking_params.min_overlap_ratio)
        self.overlap_spin.setToolTip("Minimum overlap ratio between frames for cell identity assignment (0-1)")

        # Displacement controls
        self.displacement_spin = QDoubleSpinBox()
        self.displacement_spin.setRange(0.0, float('inf'))
        self.displacement_spin.setMaximum(1e9)
        self.displacement_spin.setSingleStep(5.0)
        self.displacement_spin.setValue(self.tracking_params.max_displacement)
        self.displacement_spin.setToolTip("Maximum allowed cell movement between frames (pixels)")

        # Cell size controls
        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setRange(0, int(1e9))
        self.cell_size_spin.setSingleStep(10)
        self.cell_size_spin.setValue(self.tracking_params.min_cell_size)
        self.cell_size_spin.setToolTip("Minimum cell size in pixels (0 to disable filtering)")

        # Gap closing controls
        self.gap_closing_check = QCheckBox()
        self.gap_closing_check.setChecked(self.tracking_params.enable_gap_closing)
        self.gap_closing_check.setToolTip("Enable tracking across gaps in segmentation")

        self.gap_frames_spin = QSpinBox()
        self.gap_frames_spin.setRange(1, int(1e4))
        self.gap_frames_spin.setSingleStep(1)
        self.gap_frames_spin.setValue(self.tracking_params.max_frame_gap)
        self.gap_frames_spin.setToolTip("Maximum number of frames to look ahead for gap closing")
        self.gap_frames_spin.setEnabled(self.gap_closing_check.isChecked())

        # Action buttons
        self.track_btn = QPushButton("Track Cells")
        self.reset_btn = QPushButton("Reset Parameters")

        logger.debug("Controls initialized with default parameters")

    def _validate_stack(self, stack: np.ndarray) -> None:
        """Validate the input stack thoroughly"""
        if stack is None:
            raise ProcessingError("Input stack is None")

        if not isinstance(stack, np.ndarray):
            raise ProcessingError(
                f"Invalid input type: expected numpy array, got {type(stack)}"
            )

        if stack.ndim not in [2, 3]:
            raise ProcessingError(
                f"Invalid dimensions: expected 2D or 3D array, got {stack.ndim}D"
            )

        if stack.size == 0:
            raise ProcessingError("Empty input array")

        if not np.issubdtype(stack.dtype, np.integer):
            raise ProcessingError(
                f"Invalid data type: expected integer labels, got {stack.dtype}"
            )

        if np.any(stack < 0):
            raise ProcessingError("Negative values found in labels")

        logger.debug(f"Stack validation passed: shape={stack.shape}, dtype={stack.dtype}")

    def run_analysis(self):
        """Run cell tracking with current parameters"""
        try:
            logger.debug("Starting cell tracking analysis")
            self.processing_started.emit()

            # Get and validate active layer
            active_layer = self._get_active_labels_layer()
            if active_layer is None:
                raise ProcessingError(
                    "No labels layer selected",
                    "Please select a layer containing cell segmentation"
                )

            stack = active_layer.data
            if stack is None:
                raise ProcessingError("Empty layer data")

            self._validate_stack(stack)
            self._set_controls_enabled(False)

            # Set up progress callback
            def update_progress(progress: float, message: str = ""):
                self._update_status(message, int(progress))

            # Configure tracker with progress callback
            self.tracker.set_progress_callback(update_progress)

            # Ensure proper data format
            try:
                stack = self._ensure_stack_format(stack)
            except Exception as e:
                logger.error(f"Failed to format stack: {str(e)}", exc_info=True)
                raise ProcessingError(
                    "Data formatting failed",
                    f"Error formatting input data: {str(e)}"
                )

            # Run tracking
            try:
                tracked_labels = self.tracker.track_cells(stack)
                if tracked_labels is None:
                    raise ProcessingError("Tracking produced no results")
            except Exception as e:
                logger.error(f"Tracking algorithm failed: {str(e)}", exc_info=True)
                raise ProcessingError(
                    "Tracking algorithm failed",
                    f"Error during cell tracking: {str(e)}"
                )

            # Store and visualize results
            self.data_manager.tracked_data = tracked_labels
            self.visualization_manager.update_tracking_visualization(tracked_labels)

            self._update_status("Cell tracking complete", 100)
            self.processing_completed.emit(tracked_labels)
            logger.debug("Cell tracking workflow completed successfully")

        except ProcessingError as e:
            logger.error(f"Processing error: {e.message}", exc_info=True)
            self._handle_error(e)
        except Exception as e:
            logger.error(f"Unexpected error during tracking: {str(e)}", exc_info=True)
            error = ProcessingError(
                "Cell tracking failed",
                f"Unexpected error: {str(e)}",
                self.__class__.__name__
            )
            self._handle_error(error)
        finally:
            self._set_controls_enabled(True)
            logger.debug("Controls re-enabled")
    def _create_parameter_group(self) -> QGroupBox:
        """Create tracking parameters group"""
        group = QGroupBox("Tracking Parameters")
        layout = QFormLayout()
        layout.setSpacing(4)

        # Add parameter controls
        layout.addRow("Min Overlap Ratio:", self.overlap_spin)
        layout.addRow("Max Displacement:", self.displacement_spin)
        layout.addRow("Min Cell Size:", self.cell_size_spin)

        # Gap closing section
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(self.gap_closing_check)
        gap_layout.addWidget(self.gap_frames_spin)
        gap_layout.addStretch()
        layout.addRow("Enable Gap Closing:", gap_layout)

        group_widget = QWidget()
        group_widget.setLayout(layout)

        group_layout = QVBoxLayout()
        group_layout.addWidget(group_widget)
        group.setLayout(group_layout)

        return group

    def _create_action_group(self) -> QGroupBox:
        """Create action buttons group"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.addWidget(self.track_btn)
        layout.addWidget(self.reset_btn)
        group.setLayout(layout)
        return group

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
        right_layout.addWidget(self._create_parameter_group())
        right_layout.addWidget(self._create_action_group())

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

    def _register_controls(self):
        """Register all controls with base widget"""
        controls = [
            self.overlap_spin,
            self.displacement_spin,
            self.cell_size_spin,
            self.gap_closing_check,
            self.gap_frames_spin,
            self.track_btn,
            self.reset_btn
        ]

        for control in controls:
            self.register_control(control)

        # Connect gap frames spin enabled state to checkbox
        self.gap_closing_check.toggled.connect(self.gap_frames_spin.setEnabled)

    def _connect_signals(self):
        """Connect widget signals"""
        self.track_btn.clicked.connect(self.run_analysis)
        self.reset_btn.clicked.connect(self.reset_parameters)

        # Parameter update signals
        self.overlap_spin.valueChanged.connect(self.update_parameters)
        self.displacement_spin.valueChanged.connect(self.update_parameters)
        self.cell_size_spin.valueChanged.connect(self.update_parameters)
        self.gap_closing_check.toggled.connect(self.update_parameters)
        self.gap_frames_spin.valueChanged.connect(self.update_parameters)

    def update_parameters(self):
        """Update tracking parameters from UI controls"""
        try:
            self.tracking_params = TrackingParameters(
                min_overlap_ratio=self.overlap_spin.value(),
                max_displacement=self.displacement_spin.value(),
                min_cell_size=self.cell_size_spin.value(),
                enable_gap_closing=self.gap_closing_check.isChecked(),
                max_frame_gap=self.gap_frames_spin.value()
            )

            self.tracking_params.validate()
            self.tracker.update_parameters(self.tracking_params)

            self._update_status("Parameters updated")
            self.parameters_updated.emit()

        except ValueError as e:
            raise ProcessingError("Invalid parameters", str(e))

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.tracking_params = TrackingParameters()

        self.overlap_spin.setValue(self.tracking_params.min_overlap_ratio)
        self.displacement_spin.setValue(self.tracking_params.max_displacement)
        self.cell_size_spin.setValue(self.tracking_params.min_cell_size)
        self.gap_closing_check.setChecked(self.tracking_params.enable_gap_closing)
        self.gap_frames_spin.setValue(self.tracking_params.max_frame_gap)

        self._update_status("Parameters reset to defaults")
        self.update_parameters()

    def cleanup(self):
        """Clean up resources"""
        super().cleanup()




