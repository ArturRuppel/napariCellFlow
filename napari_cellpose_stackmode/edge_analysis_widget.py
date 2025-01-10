import logging
from pathlib import Path
from typing import Optional

import napari
import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QProgressBar, QLabel,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QSizePolicy
)

from .base_widget_new import BaseAnalysisWidget, ProcessingError
from .data_manager import DataManager
from .edge_analysis import EdgeAnalyzer
from .structure import EdgeAnalysisParams, CellBoundary
from .visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class EdgeAnalysisWidget(BaseAnalysisWidget):
    """Widget for edge detection and analysis operations."""

    edges_detected = Signal(dict)  # Boundaries by frame
    analysis_completed = Signal(object)  # Analysis results

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

        # Use same attribute name as original
        self.vis_manager = visualization_manager

        # Initialize visualization manager's edge layers if they don't exist
        if not hasattr(self.vis_manager, '_edge_layer'):
            self.vis_manager._edge_layer = None
        if not hasattr(self.vis_manager, '_intercalation_layer'):
            self.vis_manager._intercalation_layer = None
        if not hasattr(self.vis_manager, '_analysis_layer'):
            self.vis_manager._analysis_layer = None

        # Initialize color cycle
        if not hasattr(self.vis_manager, '_color_cycle'):
            self.vis_manager._color_cycle = np.random.RandomState(0)  # Use seeded random for reproducible colors

        self.params = EdgeAnalysisParams()
        self.analyzer = EdgeAnalyzer()
        self.analyzer.update_parameters(self.params)

        self._current_results = None
        self._current_boundaries = None

        self._initialize_controls()
        self._setup_ui()
        self._connect_signals()
    def _connect_signals(self):
        # Parameter signals
        self.dilation_spin.valueChanged.connect(self.update_parameters)
        self.overlap_spin.valueChanged.connect(self.update_parameters)
        self.min_length_spin.valueChanged.connect(self.update_parameters)
        self.filter_isolated_check.stateChanged.connect(self.update_parameters)
        self.temporal_window_spin.valueChanged.connect(self.update_parameters)
        self.min_contact_spin.valueChanged.connect(self.update_parameters)

        # Connect visualization signals (even though we'll primarily use direct calls)
        self.edges_detected.connect(self.vis_manager.update_edge_visualization)
        self.analysis_completed.connect(self.vis_manager.update_edge_analysis_visualization)

        # Action buttons
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.save_btn.clicked.connect(self.save_results)
        self.load_btn.clicked.connect(self.load_results)
        self.reset_btn.clicked.connect(self.reset_parameters)

    def run_analysis(self):
        selected = self._get_active_labels_layer()
        if selected is None:
            raise ProcessingError(
                message="Please select a label layer for edge analysis",
                component=self.__class__.__name__
            )

        try:
            self._set_controls_enabled(False)
            self._update_status("Starting edge analysis...", 10)

            self.vis_manager.clear_edge_layers()

            results = self.analyzer.analyze_sequence(selected.data)
            self._current_results = results

            self._update_status("Processing results...", 50)

            self.data_manager.analysis_results = results

            boundaries_by_frame = self._extract_boundaries(results)
            self._current_boundaries = boundaries_by_frame

            # Direct visualization updates
            self.vis_manager.update_edge_visualization(boundaries_by_frame)
            self.vis_manager.update_intercalation_visualization(results)
            self.vis_manager.update_edge_analysis_visualization(results)

            self.save_btn.setEnabled(True)
            self.processing_completed.emit(results)  # Only emit this signal

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
            self._set_controls_enabled(True)

    def _initialize_controls(self):
        """Initialize all UI controls"""
        # Edge detection parameters
        self.dilation_spin = QSpinBox()
        self.overlap_spin = QSpinBox()
        self.min_length_spin = QDoubleSpinBox()
        self.filter_isolated_check = QCheckBox()

        # Temporal parameters
        self.temporal_window_spin = QSpinBox()
        self.min_contact_spin = QSpinBox()

        # Action buttons
        self.analyze_btn = QPushButton("Run Analysis")
        self.save_btn = QPushButton("Save Results")
        self.load_btn = QPushButton("Load Results")
        self.reset_btn = QPushButton("Reset Parameters")

    def _create_detection_group(self) -> QGroupBox:
        """Create edge detection parameters group"""
        group = QGroupBox("Edge Detection Parameters")
        layout = QFormLayout()
        layout.setSpacing(4)

        # Configure dilation control
        self.dilation_spin.setRange(1, 10)
        self.dilation_spin.setValue(self.params.dilation_radius)
        self.dilation_spin.setToolTip("Radius for morphological dilation when finding boundaries")
        layout.addRow("Dilation Radius:", self.dilation_spin)

        # Configure overlap control
        self.overlap_spin.setRange(1, 100)
        self.overlap_spin.setValue(self.params.min_overlap_pixels)
        self.overlap_spin.setToolTip("Minimum number of overlapping pixels to consider cells as neighbors")
        layout.addRow("Min Overlap Pixels:", self.overlap_spin)

        # Configure length control
        self.min_length_spin.setRange(0.0, 1000.0)
        self.min_length_spin.setValue(self.params.min_edge_length)
        self.min_length_spin.setSingleStep(0.5)
        self.min_length_spin.setToolTip("Minimum edge length in pixels (0 to disable)")
        layout.addRow("Min Edge Length:", self.min_length_spin)

        # Configure filter control
        self.filter_isolated_check.setChecked(self.params.filter_isolated)
        self.filter_isolated_check.setToolTip("Filter out edges that don't connect to others")
        layout.addRow("Filter Isolated:", self.filter_isolated_check)

        group_widget = QWidget()
        group_widget.setLayout(layout)

        group_layout = QVBoxLayout()
        group_layout.addWidget(group_widget)
        group.setLayout(group_layout)

        return group

    def _create_temporal_group(self) -> QGroupBox:
        """Create temporal parameters group"""
        group = QGroupBox("Temporal Parameters")
        layout = QFormLayout()
        layout.setSpacing(4)

        # Configure temporal window control
        self.temporal_window_spin.setRange(1, 10)
        self.temporal_window_spin.setValue(self.params.temporal_window)
        self.temporal_window_spin.setToolTip("Number of frames to consider for temporal analysis")
        layout.addRow("Temporal Window:", self.temporal_window_spin)

        # Configure contact frames control
        self.min_contact_spin.setRange(1, 10)
        self.min_contact_spin.setValue(self.params.min_contact_frames)
        self.min_contact_spin.setToolTip("Minimum frames of contact required")
        layout.addRow("Min Contact Frames:", self.min_contact_spin)

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

        layout.addWidget(self.analyze_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.reset_btn)

        self.save_btn.setEnabled(False)

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
        right_layout.addWidget(self._create_detection_group())
        right_layout.addWidget(self._create_temporal_group())
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
        for control in [
            self.dilation_spin,
            self.overlap_spin,
            self.min_length_spin,
            self.filter_isolated_check,
            self.temporal_window_spin,
            self.min_contact_spin,
            self.analyze_btn,
            self.save_btn,
            self.load_btn,
            self.reset_btn
        ]:
            self.register_control(control)

    def update_parameters(self):
        """Update edge analysis parameters from UI controls"""
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

            self._update_status("Parameters updated")
            self.parameters_updated.emit()

        except ValueError as e:
            raise ProcessingError(
                message=f"Invalid parameters: {str(e)}",
                component=self.__class__.__name__
            )

    def save_results(self):
        """Save the current analysis results"""
        try:
            if self.data_manager.analysis_results is None:
                QMessageBox.warning(self, "Warning", "No analysis results to save")
                return

            file_path = self._get_save_path()
            if file_path:
                self.data_manager.save_analysis_results(file_path)
                self._update_status(f"Results saved to {file_path.name}", 100)

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

            # Extract boundaries
            boundaries_by_frame = self._extract_boundaries(loaded_data)
            self._current_boundaries = boundaries_by_frame

            # Update visualizations
            self.visualization_manager.update_edge_visualization(boundaries_by_frame)
            self.visualization_manager.update_intercalation_visualization(loaded_data)
            self.visualization_manager.update_edge_analysis_visualization(loaded_data)

            self._update_status("Analysis results loaded", 100)
            self.save_btn.setEnabled(True)

            # Emit signals
            self.edges_detected.emit(boundaries_by_frame)
            self.processing_completed.emit(loaded_data)

        except Exception as e:
            self._handle_error(ProcessingError(
                message="Failed to load results",
                details=str(e),
                component=self.__class__.__name__
            ))
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

        self._update_status("Parameters reset to defaults")

    def _extract_boundaries(self, results):
        """Extract cell boundaries from analysis results"""
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

    def cleanup(self):
        """Clean up resources"""
        if self.visualization_manager:
            self.visualization_manager.clear_edge_layers()

        self._current_results = None
        self._current_boundaries = None

        super().cleanup()