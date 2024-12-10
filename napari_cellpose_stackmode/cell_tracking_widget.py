import logging
from qtpy.QtWidgets import (
    QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QHBoxLayout
)

from napari_cellpose_stackmode.base_widget import BaseAnalysisWidget, ProcessingError
from napari_cellpose_stackmode.cell_tracking import CellTracker
from napari_cellpose_stackmode.structure import AnalysisConfig, TrackingParameters

logger = logging.getLogger(__name__)


class CellTrackingWidget(BaseAnalysisWidget):
    """Widget for cell tracking operations with interactive parameters."""

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: "DataManager",
            visualization_manager: "VisualizationManager"
    ):
        super().__init__(viewer, data_manager, visualization_manager, "Cell Tracking")
        self.tracking_params = TrackingParameters()
        self.tracker = CellTracker(AnalysisConfig())
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize the tracking-specific UI elements"""
        # Parameters section
        param_layout = QFormLayout()

        # Create parameter controls
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 1.0)
        self.overlap_spin.setSingleStep(0.1)
        self.overlap_spin.setValue(self.tracking_params.min_overlap_ratio)
        self.overlap_spin.setToolTip("Minimum overlap ratio between frames for cell identity assignment (0-1)")
        self.register_control(self.overlap_spin)
        param_layout.addRow("Min Overlap Ratio:", self.overlap_spin)

        self.displacement_spin = QDoubleSpinBox()
        self.displacement_spin.setRange(0.0, float('inf'))
        self.displacement_spin.setMaximum(1e9)
        self.displacement_spin.setSingleStep(5.0)
        self.displacement_spin.setValue(self.tracking_params.max_displacement)
        self.displacement_spin.setToolTip("Maximum allowed cell movement between frames (pixels)")
        self.register_control(self.displacement_spin)
        param_layout.addRow("Max Displacement:", self.displacement_spin)

        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setRange(0, int(1e9))
        self.cell_size_spin.setSingleStep(10)
        self.cell_size_spin.setValue(self.tracking_params.min_cell_size)
        self.cell_size_spin.setToolTip("Minimum cell size in pixels (0 to disable filtering)")
        self.register_control(self.cell_size_spin)
        param_layout.addRow("Min Cell Size:", self.cell_size_spin)

        self.gap_closing_check = QCheckBox()
        self.gap_closing_check.setChecked(self.tracking_params.enable_gap_closing)
        self.gap_closing_check.setToolTip("Enable tracking across gaps in segmentation")
        self.register_control(self.gap_closing_check)
        param_layout.addRow("Enable Gap Closing:", self.gap_closing_check)

        self.gap_frames_spin = QSpinBox()
        self.gap_frames_spin.setRange(1, int(1e4))
        self.gap_frames_spin.setSingleStep(1)
        self.gap_frames_spin.setValue(self.tracking_params.max_frame_gap)
        self.gap_frames_spin.setToolTip("Maximum number of frames to look ahead for gap closing")
        self.register_control(self.gap_frames_spin)
        param_layout.addRow("Max Frame Gap:", self.gap_frames_spin)

        # Add parameter layout to main layout
        self.main_layout.insertLayout(1, param_layout)  # Insert after title

        # Add button layout
        button_layout = QHBoxLayout()

        # Track cells button
        self.track_btn = QPushButton("Track Cells")
        self.track_btn.clicked.connect(self.run_analysis)
        self.register_control(self.track_btn)
        button_layout.addWidget(self.track_btn)

        # Reset parameters button
        self.reset_btn = QPushButton("Reset Parameters")
        self.reset_btn.clicked.connect(self.reset_parameters)
        self.register_control(self.reset_btn)
        button_layout.addWidget(self.reset_btn)

        # Add button layout after parameters but before the stretch
        self.main_layout.insertLayout(2, button_layout)

    def _connect_signals(self):
        """Connect parameter control signals"""
        self.overlap_spin.valueChanged.connect(self.update_parameters)
        self.displacement_spin.valueChanged.connect(self.update_parameters)
        self.cell_size_spin.valueChanged.connect(self.update_parameters)
        self.gap_closing_check.stateChanged.connect(self.update_parameters)
        self.gap_frames_spin.valueChanged.connect(self.update_parameters)

    def run_analysis(self):
        """Run cell tracking with current parameters"""
        try:
            active_layer = self._get_active_labels_layer()
            if active_layer is None:
                raise ProcessingError(
                    "No labels layer selected",
                    "Please select a layer containing cell segmentation"
                )

            stack = active_layer.data
            if not self._validate_input_data(stack):
                raise ProcessingError(
                    "Invalid input data",
                    "Data must be a 2D or 3D numpy array"
                )

            self._processing = True
            self._set_controls_enabled(False)

            # More granular progress updates
            self._update_status("Preparing image stack...", 10)
            stack = self._ensure_stack_format(stack)

            self._update_status("Initializing cell tracker...", 20)
            self._update_status("Analyzing cell movements...", 40)
            tracked_labels = self.tracker.track_cells(stack)

            self._update_status("Storing results...", 80)
            self.data_manager.tracked_data = tracked_labels

            self._update_status("Updating visualization...", 90)
            self.vis_manager.update_tracking_visualization(tracked_labels)

            self._update_status("Cell tracking complete", 100)
            # Emit both sets of signals consistently
            self.processing_completed.emit(tracked_labels)

        except ProcessingError as e:
            self._handle_error(e)
        except Exception as e:
            error = ProcessingError(
                message="Cell tracking failed",
                details=str(e),
                component=self.__class__.__name__
            )
            self._handle_error(error)
        finally:
            self._processing = False
            self._set_controls_enabled(True)

    def update_parameters(self):
        """Update tracking parameters from UI controls"""
        try:
            self.tracking_params.min_overlap_ratio = self.overlap_spin.value()
            self.tracking_params.max_displacement = self.displacement_spin.value()
            self.tracking_params.min_cell_size = self.cell_size_spin.value()
            self.tracking_params.enable_gap_closing = self.gap_closing_check.isChecked()
            self.tracking_params.max_frame_gap = self.gap_frames_spin.value()

            self.tracking_params.validate()
            self.tracker.update_parameters(self.tracking_params)
            self._update_status("Parameters updated")
            self.parameters_updated.emit()

        except ValueError as e:
            self._update_status(f"Invalid parameters: {str(e)}")
            logger.warning(f"Invalid parameter combination: {str(e)}")

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.tracking_params = TrackingParameters()
        self.overlap_spin.setValue(self.tracking_params.min_overlap_ratio)
        self.displacement_spin.setValue(self.tracking_params.max_displacement)
        self.cell_size_spin.setValue(self.tracking_params.min_cell_size)
        self.gap_closing_check.setChecked(self.tracking_params.enable_gap_closing)
        self.gap_frames_spin.setValue(self.tracking_params.max_frame_gap)
        self._update_status("Parameters reset to defaults")

    # def get_parameters(self) -> TrackingParameters:
    #     """Get current tracking parameters"""
    #     return TrackingParameters(
    #         min_overlap_ratio=self.overlap_spin.value(),
    #         max_displacement=self.displacement_spin.value(),
    #         min_cell_size=self.cell_size_spin.value(),
    #         enable_gap_closing=self.gap_closing_check.isChecked(),
    #         max_frame_gap=self.gap_frames_spin.value()
    #     )
    #
    # def set_parameters(self, params: TrackingParameters):
    #     """Set tracking parameters and update UI"""
    #     try:
    #         params.validate()
    #         logger.debug(f"Setting tracking parameters: {params}")
    #
    #         self.overlap_spin.setValue(params.min_overlap_ratio)
    #         self.displacement_spin.setValue(params.max_displacement)
    #         self.cell_size_spin.setValue(params.min_cell_size)
    #         self.gap_closing_check.setChecked(params.enable_gap_closing)
    #         self.gap_frames_spin.setValue(params.max_frame_gap)
    #
    #         self.tracking_params = params
    #         self.tracker.update_parameters(params)
    #         self.parameters_updated.emit()
    #
    #     except ValueError as e:
    #         logger.error(f"Invalid parameters: {e}")
    #         raise
