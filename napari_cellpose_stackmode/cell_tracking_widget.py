from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                            QProgressBar, QFileDialog, QMessageBox, QHBoxLayout,
                            QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox)
from qtpy.QtCore import Signal
import napari
import numpy as np
from pathlib import Path
import logging
from typing import Optional
from napari_cellpose_stackmode.data_manager import DataManager
from napari_cellpose_stackmode.visualization_manager import VisualizationManager
from napari_cellpose_stackmode.cell_tracking import CellTracker
from napari_cellpose_stackmode.structure import AnalysisConfig, TrackingParameters

logger = logging.getLogger(__name__)


class CellTrackingWidget(QWidget):
    """Enhanced widget for cell tracking operations with interactive parameters."""

    tracking_completed = Signal(np.ndarray)
    tracking_failed = Signal(str)
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
        self.tracking_params = TrackingParameters()
        self.tracker = CellTracker(AnalysisConfig())

        # First create all UI elements
        self.setup_ui()
        # Then connect signals
        self.connect_signals()

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

    def run_tracking(self):
        """Run cell tracking with current parameters"""
        selected = self.viewer.layers.selection.active

        if selected is None:
            QMessageBox.warning(self, "Warning", "Please select a layer containing cell segmentation")
            return

        try:
            # Disable all controls during processing
            self._set_controls_enabled(False)
            self._update_status("Processing image stack...", 10)

            # Convert input to proper format
            stack = self._ensure_stack_format(selected.data)
            logger.info(f"Input stack shape: {stack.shape}")

            self._update_status("Running cell tracking...", 30)

            # Run tracking with current parameters
            tracked_labels = self.tracker.track_cells(stack)
            logger.info(f"Tracking complete. Output shape: {tracked_labels.shape}")

            # Store results
            self.data_manager.tracked_data = tracked_labels

            # Explicitly update visualization
            self.visualization_manager.update_tracking_visualization(tracked_labels)

            self._update_status("Cell tracking complete - use Save button to store results", 100)
            self.tracking_completed.emit(tracked_labels)


        except Exception as e:
            error_msg = f"Cell tracking failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
            self._update_status("Cell tracking failed", 0)
            self.tracking_failed.emit(error_msg)

        finally:
            self._set_controls_enabled(True)

    def save_tracking_results(self):
        """Save the current tracking results"""
        try:
            if self.data_manager.tracked_data is None:
                QMessageBox.warning(self, "Warning", "No tracking results to save")
                return

            save_path = self._get_save_path()
            if save_path:
                self.data_manager.save_tracking_results(save_path)
                self._update_status(f"Results saved to {save_path.name}", 100)

        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
            self._update_status("Saving failed", 0)

    def load_tracking_results(self):
        """Load previously saved tracking results"""
        try:
            file_path = self._get_load_path()
            if file_path is None:
                return

            self._update_status("Loading tracked cells...", 20)

            # Load data through data manager
            self.data_manager.load_tracking_results(file_path)
            tracked_data = self.data_manager.tracked_data

            self._update_status("Updating visualization...", 80)

            # Update visualization
            self.visualization_manager.update_tracking_visualization(tracked_data)

            self._update_status(f"Loaded tracked cells from {file_path.name}", 100)


            # Emit completion signal
            self.tracking_completed.emit(tracked_data)

        except Exception as e:
            error_msg = f"Failed to load tracked cells: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
            self._update_status("Loading failed", 0)
            self.tracking_failed.emit(error_msg)

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all controls except the save button"""
        self.track_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)
        self.overlap_spin.setEnabled(enabled)
        self.displacement_spin.setEnabled(enabled)
        self.cell_size_spin.setEnabled(enabled)
        self.gap_closing_check.setEnabled(enabled)
        self.gap_frames_spin.setEnabled(enabled)

    def update_parameters(self):
        """Update tracking parameters from UI controls"""
        try:
            self.tracking_params.min_overlap_ratio = self.overlap_spin.value()
            self.tracking_params.max_displacement = self.displacement_spin.value()
            self.tracking_params.min_cell_size = self.cell_size_spin.value()
            self.tracking_params.enable_gap_closing = self.gap_closing_check.isChecked()
            self.tracking_params.max_frame_gap = self.gap_frames_spin.value()

            # Validate new parameters
            self.tracking_params.validate()

            # Update tracker with new parameters
            self.tracker.update_parameters(self.tracking_params)

            # Enable tracking button if parameters are valid
            self.track_btn.setEnabled(True)
            self.status_label.setText("Parameters updated")
            self.parameters_updated.emit()

        except ValueError as e:
            self.track_btn.setEnabled(False)
            self.status_label.setText(f"Invalid parameters: {str(e)}")
            logger.warning(f"Invalid parameter combination: {str(e)}")

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.tracking_params = TrackingParameters()
        self.overlap_spin.setValue(self.tracking_params.min_overlap_ratio)
        self.displacement_spin.setValue(self.tracking_params.max_displacement)
        self.cell_size_spin.setValue(self.tracking_params.min_cell_size)
        self.gap_closing_check.setChecked(self.tracking_params.enable_gap_closing)
        self.gap_frames_spin.setValue(self.tracking_params.max_frame_gap)
        self.status_label.setText("Parameters reset to defaults")

    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and optionally progress bar"""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)
        logger.info(message)

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

    def _get_save_path(self) -> Optional[Path]:
        """Show file dialog for saving results"""
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Save Tracking Results")
        dialog.setNameFilter("TIFF files (*.tif *.tiff)")
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)

        # Start in last used directory if available
        if self.data_manager.last_directory:
            dialog.setDirectory(str(self.data_manager.last_directory))

        if dialog.exec_():
            return Path(dialog.selectedFiles()[0])
        return None

    def _get_load_path(self) -> Optional[Path]:
        """Show file dialog for loading results"""
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Load Tracking Results")
        dialog.setNameFilter("TIFF files (*.tif *.tiff)")
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFile)

        # Start in last used directory if available
        if self.data_manager.last_directory:
            dialog.setDirectory(str(self.data_manager.last_directory))

        if dialog.exec_():
            return Path(dialog.selectedFiles()[0])
        return None

    def run_batch_analysis(self):
        """Run tracking analysis in batch mode"""
        try:
            if not self.data_manager.batch_mode:
                logger.error("Not in batch mode")
                return

            current_path = self.data_manager.input_paths[self.data_manager.current_batch_index - 1]
            logger.info(f"Processing tracking for {current_path.name}")

            # Load and process image
            stack = self._load_image_stack(current_path)
            stack = self._ensure_stack_format(stack)

            tracked_labels = self.tracker.track_cells(stack)

            # Store results
            self.data_manager.tracked_data = tracked_labels
            self.data_manager.store_batch_results(current_path)

            # Save results automatically
            save_path = current_path.parent / f"{current_path.stem}_tracked.tif"
            self.data_manager.save_tracking_results(save_path)

            logger.info(f"Completed tracking for {current_path.name}")

        except Exception as e:
            error_msg = f"Tracking batch processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def _load_image_stack(self, path: Path) -> np.ndarray:
        """Load image stack from file"""
        try:
            return self.data_manager._data_io.load_tiff_stack(path)
        except Exception as e:
            raise ValueError(f"Failed to load {path.name}: {str(e)}")

    def get_parameters(self) -> TrackingParameters:
        """Get current tracking parameters"""
        return TrackingParameters(
            min_overlap_ratio=self.overlap_spin.value(),
            max_displacement=self.displacement_spin.value(),
            min_cell_size=self.cell_size_spin.value(),
            enable_gap_closing=self.gap_closing_check.isChecked(),
            max_frame_gap=self.gap_frames_spin.value()
        )

    def set_parameters(self, params: TrackingParameters):
        """Set tracking parameters and update UI"""
        try:
            params.validate()  # Validate before setting
            logger.debug(f"Setting tracking parameters: {params}")

            # Update UI controls
            self.overlap_spin.setValue(params.min_overlap_ratio)
            self.displacement_spin.setValue(params.max_displacement)
            self.cell_size_spin.setValue(params.min_cell_size)
            self.gap_closing_check.setChecked(params.enable_gap_closing)
            self.gap_frames_spin.setValue(params.max_frame_gap)

            # Update internal parameters
            self.tracking_params = params
            self.tracker.update_parameters(params)

            # Emit signal that parameters have changed
            self.parameters_updated.emit()

            logger.debug("Tracking parameters updated successfully")

        except ValueError as e:
            logger.error(f"Invalid parameters: {e}")
            raise
