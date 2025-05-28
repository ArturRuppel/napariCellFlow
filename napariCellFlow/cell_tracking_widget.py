import logging

import napari
import numpy as np
from qtpy.QtCore import Signal, QThread, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QSizePolicy, QProgressBar, QLabel,
    QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton
)

from .base_widget import BaseAnalysisWidget, ProcessingError
from .cell_tracking import CellTracker
from .structure import AnalysisConfig, TrackingParameters

logger = logging.getLogger(__name__)


class CellTrackingWorker(QObject):
    """Worker object to run cell tracking in background thread"""
    progress = Signal(float, str)  # Emits progress percentage and status message
    finished = Signal(object)  # Emits tracked labels
    error = Signal(Exception)

    def __init__(self, tracker: CellTracker, stack: np.ndarray):
        super().__init__()
        self.tracker = tracker
        self.stack = stack

    def run(self):
        """Execute cell tracking in background thread"""
        try:
            # Configure progress callback
            def update_progress(progress: float, message: str = ""):
                self.progress.emit(progress, message)

            self.tracker.set_progress_callback(update_progress)

            # Run tracking
            tracked_labels = self.tracker.track_cells(self.stack)

            if tracked_labels is None:
                raise ProcessingError("Tracking produced no results")

            self.finished.emit(tracked_labels)

        except Exception as e:
            self.error.emit(e)
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

        # Initialize thread management
        self._tracking_thread = None
        self._tracking_worker = None

        # Initialize controls
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

        logger.debug("CellTrackingWidget initialized")

    def _handle_tracking_progress(self, progress: float, message: str):
        """Handle progress updates from worker"""
        self._update_status(message, int(progress))

    def _handle_tracking_complete(self, tracked_labels: np.ndarray):
        """Handle completion of tracking"""
        try:
            # Store and visualize results
            self.data_manager.tracked_data = tracked_labels
            self.visualization_manager.update_tracking_visualization(tracked_labels)

            self._update_status("Cell tracking complete", 100)
            self.processing_completed.emit(tracked_labels)
            logger.debug("Cell tracking workflow completed successfully")

        except Exception as e:
            logger.error(f"Error handling tracking completion: {str(e)}", exc_info=True)
            self._handle_error(ProcessingError(
                "Failed to process tracking results",
                str(e),
                self.__class__.__name__
            ))
        finally:
            self._set_controls_enabled(True)

    def _handle_tracking_error(self, error: Exception):
        """Handle errors from worker"""
        if isinstance(error, ProcessingError):
            self._handle_error(error)
        else:
            self._handle_error(ProcessingError(
                "Cell tracking failed",
                str(error),
                self.__class__.__name__
            ))
        self._set_controls_enabled(True)

    def cleanup(self):
        """Clean up resources"""
        # Clean up tracking thread
        if self._tracking_thread and self._tracking_thread.isRunning():
            self._tracking_thread.quit()
            self._tracking_thread.wait()

        super().cleanup()

    def _update_ui_state(self, event=None):
        """Update UI based on current state"""
        active_layer = self._get_active_labels_layer()
        has_valid_labels = (active_layer is not None and
                            isinstance(active_layer, napari.layers.Labels) and
                            active_layer.data.ndim in [2, 3])

        # Update button states
        self.track_btn.setEnabled(has_valid_labels)

        logger.debug(f"UI state updated: has_valid_labels={has_valid_labels}")

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

        # Gap closing controls (modified)
        self.gap_frames_spin = QSpinBox()
        self.gap_frames_spin.setRange(0, int(1e4))
        self.gap_frames_spin.setSingleStep(1)
        self.gap_frames_spin.setValue(0)  # Default value set to 0
        self.gap_frames_spin.setToolTip("Maximum number of frames to look ahead for gap closing (0 to disable)")

        # Action buttons
        self.track_btn = QPushButton("Run Tracking")
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

    def _create_parameter_group(self) -> QGroupBox:
        """Create tracking parameters group"""
        group = QGroupBox("Parameters")
        layout = QFormLayout()
        layout.setSpacing(4)

        # Add parameter controls
        layout.addRow("Min Overlap Ratio:", self.overlap_spin)
        layout.addRow("Max Displacement:", self.displacement_spin)
        layout.addRow("Min Cell Size:", self.cell_size_spin)
        layout.addRow("Gap Closing:", self.gap_frames_spin)
        layout.addWidget(self.reset_btn)

        group_widget = QWidget()
        group_widget.setLayout(layout)

        group_layout = QVBoxLayout()
        group_layout.addWidget(group_widget)
        group_layout.addWidget(self.reset_btn)

        group.setLayout(group_layout)

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
        right_layout.addWidget(self.track_btn)

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
            self.gap_frames_spin,
            self.track_btn,
            self.reset_btn
        ]

        for control in controls:
            self.register_control(control)

    def _connect_signals(self):
        """Connect widget signals"""
        self.track_btn.clicked.connect(self.run_analysis)
        self.reset_btn.clicked.connect(self.reset_parameters)

        # Parameter update signals
        self.overlap_spin.valueChanged.connect(self.update_parameters)
        self.displacement_spin.valueChanged.connect(self.update_parameters)
        self.cell_size_spin.valueChanged.connect(self.update_parameters)
        self.gap_frames_spin.valueChanged.connect(self.update_parameters)

    def update_parameters(self):
        """Update tracking parameters from UI controls"""
        try:
            # If gap closing value is 0, don't enable gap closing and set max_frame_gap to 1
            # to satisfy validation, but it won't be used since gap closing is disabled
            gap_frames = self.gap_frames_spin.value()
            enable_gap_closing = gap_frames > 0
            max_frame_gap = gap_frames if enable_gap_closing else 1

            self.tracking_params = TrackingParameters(
                min_overlap_ratio=self.overlap_spin.value(),
                max_displacement=self.displacement_spin.value(),
                min_cell_size=self.cell_size_spin.value(),
                enable_gap_closing=enable_gap_closing,
                max_frame_gap=max_frame_gap
            )

            self.tracking_params.validate()
            self.tracker.update_parameters(self.tracking_params)

            self._update_status("Parameters updated")
            self.parameters_updated.emit()

        except ValueError as e:
            raise ProcessingError("Invalid parameters", str(e))

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        # Create new default parameters
        self.tracking_params = TrackingParameters()

        # Temporarily disconnect signals
        self.overlap_spin.valueChanged.disconnect(self.update_parameters)
        self.displacement_spin.valueChanged.disconnect(self.update_parameters)
        self.cell_size_spin.valueChanged.disconnect(self.update_parameters)
        self.gap_frames_spin.valueChanged.disconnect(self.update_parameters)

        # Set all values without triggering updates
        self.overlap_spin.setValue(self.tracking_params.min_overlap_ratio)
        self.displacement_spin.setValue(self.tracking_params.max_displacement)
        self.cell_size_spin.setValue(self.tracking_params.min_cell_size)
        self.gap_frames_spin.setValue(0)  # Default value set to 0

        # Reconnect signals
        self.overlap_spin.valueChanged.connect(self.update_parameters)
        self.displacement_spin.valueChanged.connect(self.update_parameters)
        self.cell_size_spin.valueChanged.connect(self.update_parameters)
        self.gap_frames_spin.valueChanged.connect(self.update_parameters)

        # Update once with all reset values
        self._update_status("Parameters reset to defaults")
        self.update_parameters()

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

            # Ensure proper data format
            try:
                stack = self._ensure_stack_format(stack)
            except Exception as e:
                logger.error(f"Failed to format stack: {str(e)}", exc_info=True)
                raise ProcessingError(
                    "Data formatting failed",
                    f"Error formatting input data: {str(e)}"
                )

            # Create worker and thread
            self._tracking_thread = QThread()
            self._tracking_worker = CellTrackingWorker(self.tracker, stack)
            self._tracking_worker.moveToThread(self._tracking_thread)

            # Connect signals
            self._tracking_thread.started.connect(self._tracking_worker.run)
            self._tracking_worker.progress.connect(self._handle_tracking_progress)
            self._tracking_worker.finished.connect(self._handle_tracking_complete)
            self._tracking_worker.error.connect(self._handle_tracking_error)
            self._tracking_worker.finished.connect(self._tracking_thread.quit)
            self._tracking_worker.finished.connect(self._tracking_worker.deleteLater)
            self._tracking_thread.finished.connect(self._tracking_thread.deleteLater)

            # Start tracking
            self._tracking_thread.start()

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
