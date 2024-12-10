import logging
from pathlib import Path
from typing import Optional

import napari
from qtpy.QtWidgets import (QPushButton, QFileDialog, QMessageBox, QHBoxLayout, QFormLayout,
                            QSpinBox, QDoubleSpinBox, QCheckBox)

from .base_widget import BaseAnalysisWidget, ProcessingError
from .data_manager import DataManager
from .edge_analysis import EdgeAnalyzer
from .structure import EdgeAnalysisParams, CellBoundary
from .visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class EdgeAnalysisWidget(BaseAnalysisWidget):
    """Widget for edge detection and analysis operations."""

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: DataManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(viewer, data_manager, visualization_manager, "Edge Analysis")

        self.params = EdgeAnalysisParams()
        self.analyzer = EdgeAnalyzer()
        self.analyzer.update_parameters(self.params)

        self._current_results = None
        self._current_boundaries = None

        self._setup_ui()

    def _setup_ui(self):
        """Initialize the edge analysis-specific UI elements"""
        # Parameters section
        param_layout = QFormLayout()

        # Edge detection parameters
        self.dilation_spin = QSpinBox()
        self.dilation_spin.setRange(1, 10)
        self.dilation_spin.setValue(self.params.dilation_radius)
        self.dilation_spin.setToolTip("Radius for morphological dilation when finding boundaries")
        self.register_control(self.dilation_spin)
        param_layout.addRow("Dilation Radius:", self.dilation_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(1, 100)
        self.overlap_spin.setValue(self.params.min_overlap_pixels)
        self.overlap_spin.setToolTip("Minimum number of overlapping pixels to consider cells as neighbors")
        self.register_control(self.overlap_spin)
        param_layout.addRow("Min Overlap Pixels:", self.overlap_spin)

        self.min_length_spin = QDoubleSpinBox()
        self.min_length_spin.setRange(0.0, 1000.0)
        self.min_length_spin.setValue(self.params.min_edge_length)
        self.min_length_spin.setSingleStep(0.5)
        self.min_length_spin.setToolTip("Minimum edge length in pixels (0 to disable)")
        self.register_control(self.min_length_spin)
        param_layout.addRow("Min Edge Length:", self.min_length_spin)

        self.filter_isolated_check = QCheckBox()
        self.filter_isolated_check.setChecked(self.params.filter_isolated)
        self.filter_isolated_check.setToolTip("Filter out edges that don't connect to others")
        self.register_control(self.filter_isolated_check)
        param_layout.addRow("Filter Isolated:", self.filter_isolated_check)

        # Intercalation parameters
        self.temporal_window_spin = QSpinBox()
        self.temporal_window_spin.setRange(1, 10)
        self.temporal_window_spin.setValue(self.params.temporal_window)
        self.temporal_window_spin.setToolTip("Number of frames to consider for temporal analysis")
        self.register_control(self.temporal_window_spin)
        param_layout.addRow("Temporal Window:", self.temporal_window_spin)

        self.min_contact_spin = QSpinBox()
        self.min_contact_spin.setRange(1, 10)
        self.min_contact_spin.setValue(self.params.min_contact_frames)
        self.min_contact_spin.setToolTip("Minimum frames of contact required")
        self.register_control(self.min_contact_spin)
        param_layout.addRow("Min Contact Frames:", self.min_contact_spin)

        # Add parameter layout to main layout
        self.main_layout.addLayout(param_layout)

        # Button layout
        button_layout = QHBoxLayout()

        self.analyze_btn = QPushButton("Analyze Edges")
        self.analyze_btn.clicked.connect(self.run_analysis)
        button_layout.addWidget(self.analyze_btn)

        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Results")
        self.load_btn.clicked.connect(self.load_results)
        button_layout.addWidget(self.load_btn)

        self.reset_btn = QPushButton("Reset Parameters")
        self.reset_btn.clicked.connect(self.reset_parameters)
        button_layout.addWidget(self.reset_btn)

        self.main_layout.addLayout(button_layout)

        # Connect parameter control signals
        self.dilation_spin.valueChanged.connect(self.update_parameters)
        self.overlap_spin.valueChanged.connect(self.update_parameters)
        self.min_length_spin.valueChanged.connect(self.update_parameters)
        self.filter_isolated_check.stateChanged.connect(self.update_parameters)
        self.temporal_window_spin.valueChanged.connect(self.update_parameters)
        self.min_contact_spin.valueChanged.connect(self.update_parameters)

    def update_parameters(self):
        try:
            new_params = EdgeAnalysisParams(
                dilation_radius=self.dilation_spin.value(),
                min_overlap_pixels=self.overlap_spin.value(),
                min_edge_length=self.min_length_spin.value(),
                filter_isolated=self.filter_isolated_check.isChecked(),
                temporal_window=self.temporal_window_spin.value(),
                min_contact_frames=self.min_contact_spin.value()
            )
            new_params.validate()

            self.params = new_params
            self.analyzer.update_parameters(new_params)

            self.parameters_updated.emit()

        except ValueError as e:
            raise ProcessingError(
                message=f"Invalid parameters: {str(e)}",
                component=self.__class__.__name__
            )

    def run_analysis(self):
        selected = self._get_active_labels_layer()

        if selected is None:
            raise ProcessingError(
                message="Please select a label layer for edge analysis",
                component=self.__class__.__name__
            )

        try:
            self._processing = True
            self._set_controls_enabled(False)
            self._update_status("Starting edge analysis...", 10)

            self.vis_manager.clear_edge_layers()

            results = self.analyzer.analyze_sequence(selected.data)
            self._current_results = results

            self._update_status("Processing results...", 50)

            self.data_manager.analysis_results = results

            boundaries_by_frame = self._extract_boundaries(results)
            self._current_boundaries = boundaries_by_frame

            self.vis_manager.update_edge_visualization(boundaries_by_frame)
            self.vis_manager.update_intercalation_visualization(results)
            self.vis_manager.update_edge_analysis_visualization(results)

            self.processing_completed.emit(results)

            intercalation_count = sum(
                len(edge.intercalations)
                for edge in results.edges.values()
                if edge.intercalations
            )

            self._update_status(
                f"Analysis complete - found {len(results.edges)} edges and {intercalation_count} intercalations",
                100
            )

        except ProcessingError as e:
            self._handle_error(e)
        except Exception as e:
            self._handle_error(ProcessingError(
                message="Unexpected error during edge analysis",
                details=str(e),
                component=self.__class__.__name__
            ))
        finally:
            self._processing = False
            self._set_controls_enabled(True)

    def _extract_boundaries(self, results):
        boundaries_by_frame = {}
        for edge_id, edge in results.edges.items():
            for frame_idx, frame in enumerate(edge.frames):
                if frame not in boundaries_by_frame:
                    boundaries_by_frame[frame] = []

                boundary = CellBoundary(
                    cell_ids=edge.cell_pairs[frame_idx],
                    coordinates=edge.coordinates[frame_idx],
                    endpoint1=edge.coordinates[frame_idx][0],
                    endpoint2=edge.coordinates[frame_idx][-1],
                    length=edge.lengths[frame_idx]
                )
                boundaries_by_frame[frame].append(boundary)

        return boundaries_by_frame

    def connect_signals(self):
        """Connect parameter control signals"""
        self.dilation_spin.valueChanged.connect(self.update_parameters)
        self.overlap_spin.valueChanged.connect(self.update_parameters)
        self.min_length_spin.valueChanged.connect(self.update_parameters)
        self.filter_isolated_check.stateChanged.connect(self.update_parameters)
        self.temporal_window_spin.valueChanged.connect(self.update_parameters)
        self.min_contact_spin.valueChanged.connect(self.update_parameters)
        self.edges_detected.connect(self.visualization_manager.update_edge_visualization)
        self.analysis_completed.connect(self.visualization_manager.update_edge_analysis_visualization)

    def save_results(self):
        """Save the current analysis results"""
        try:
            if self.data_manager.analysis_results is None:
                QMessageBox.warning(self, "Warning", "No analysis results to save")
                return

            save_path = self._get_save_path()
            if save_path:
                self.data_manager.save_analysis_results(save_path)
                self._update_status(f"Results saved to {save_path.name}", 100)

        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
            self._update_status("Saving failed", 0)

    def load_results(self):
        """Load previously saved analysis results"""
        try:
            self._set_controls_enabled(False)

            file_path = self._get_load_path()
            if file_path is None:
                return

            self._update_status("Loading analysis results...", 20)

            # Clear previous visualizations
            self.visualization_manager.clear_edge_layers()

            # Load through data manager
            self.data_manager.load_analysis_results(file_path)
            loaded_data = self.data_manager.analysis_results
            self._current_results = loaded_data

            # Reconstruct boundaries from loaded data
            boundaries_by_frame = {}
            for edge_id, edge in loaded_data.edges.items():
                for frame_idx, frame in enumerate(edge.frames):
                    if frame not in boundaries_by_frame:
                        boundaries_by_frame[frame] = []

                    boundary = CellBoundary(
                        cell_ids=edge.cell_pairs[frame_idx],
                        coordinates=edge.coordinates[frame_idx],
                        endpoint1=edge.coordinates[frame_idx][0],
                        endpoint2=edge.coordinates[frame_idx][-1],
                        length=edge.lengths[frame_idx]
                    )
                    boundaries_by_frame[frame].append(boundary)

            self._current_boundaries = boundaries_by_frame

            # Update visualizations in sequence - this creates three separate layers
            self.visualization_manager.update_edge_visualization(boundaries_by_frame)  # Initial edge detection
            self.visualization_manager.update_intercalation_visualization(loaded_data)  # Intercalation events
            self.visualization_manager.update_edge_analysis_visualization(loaded_data)  # Final analyzed edges

            self._update_status("Analysis results loaded", 100)
            self.save_btn.setEnabled(True)

            # Emit signals
            self.edges_detected.emit(boundaries_by_frame)
            self.analysis_completed.emit(loaded_data)

        except Exception as e:
            error_msg = f"Failed to load results: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
            self._update_status("Loading failed", 0)
            self.analysis_failed.emit(error_msg)

        finally:
            self._set_controls_enabled(True)

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.params = EdgeAnalysisParams()
        self.analyzer.update_parameters(self.params)

        self.dilation_spin.setValue(self.params.dilation_radius)
        self.overlap_spin.setValue(self.params.min_overlap_pixels)
        self.min_length_spin.setValue(self.params.min_edge_length)
        self.filter_isolated_check.setChecked(self.params.filter_isolated)
        self.temporal_window_spin.setValue(self.params.temporal_window)
        self.min_contact_spin.setValue(self.params.min_contact_frames)

        self.status_label.setText("Parameters reset to defaults")

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all controls except save button"""
        self.analyze_btn.setEnabled(enabled)
        self.load_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)
        self.dilation_spin.setEnabled(enabled)
        self.overlap_spin.setEnabled(enabled)
        self.min_length_spin.setEnabled(enabled)
        self.filter_isolated_check.setEnabled(enabled)
        self.temporal_window_spin.setEnabled(enabled)
        self.min_contact_spin.setEnabled(enabled)

    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and optionally progress bar"""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)
        logger.info(message)

    def _get_save_path(self) -> Optional[Path]:
        """Show file dialog for saving results"""
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Save Edge Analysis Results")
        dialog.setNameFilter("Pickle files (*.pkl)")
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)

        if self.data_manager.last_directory:
            dialog.setDirectory(str(self.data_manager.last_directory))

        if dialog.exec_():
            return Path(dialog.selectedFiles()[0])
        return None

    def _get_load_path(self) -> Optional[Path]:
        """Show file dialog for loading results"""
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Load Edge Analysis Results")
        dialog.setNameFilter("Pickle files (*.pkl)")
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFile)

        if self.data_manager.last_directory:
            dialog.setDirectory(str(self.data_manager.last_directory))

        if dialog.exec_():
            return Path(dialog.selectedFiles()[0])
        return None
