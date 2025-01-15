import logging
from pathlib import Path
from typing import Optional

import napari
import numpy as np
from qtpy.QtCore import Signal, QObject, QThread
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QProgressBar, QLabel,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QHBoxLayout,
    QFileDialog, QMessageBox, QGroupBox, QSizePolicy
)

from .base_widget import BaseAnalysisWidget, ProcessingError
from .data_manager import DataManager
from .edge_analysis import EdgeAnalyzer
from .edge_analysis_visualization import Visualizer
from .structure import EdgeAnalysisParams, CellBoundary, VisualizationConfig
from .visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class AnalysisWorker(QObject):
    """Worker object to run analysis in background thread"""
    progress = Signal(int, str)
    finished = Signal(object)  # Emits results
    error = Signal(Exception)

    def __init__(self, analyzer, data):
        super().__init__()
        self.analyzer = analyzer
        self.data = data

    def run(self):
        try:
            results = self.analyzer.analyze_sequence(
                self.data,
                lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(e)

class VisualizationWorker(QObject):
    """Worker object to run visualization generation in background thread"""
    progress = Signal(int, str)
    finished = Signal()
    error = Signal(Exception)

    def __init__(self, visualizer, results, output_dir):
        super().__init__()
        self.visualizer = visualizer
        self.results = results
        self.output_dir = output_dir

    def run(self):
        try:
            def progress_callback(stage: str, progress: float):
                self.progress.emit(int(progress), stage)

            self.visualizer.set_progress_callback(progress_callback)
            self.visualizer.output_dir = self.output_dir
            self.visualizer.create_visualizations(self.results)
            self.finished.emit()
        except Exception as e:
            self.error.emit(e)

class EdgeAnalysisWidget(BaseAnalysisWidget):
    """Widget for edge detection, analysis, and visualization operations."""

    edges_detected = Signal(dict)  # Boundaries by frame
    analysis_completed = Signal(object)  # Analysis results
    visualization_config_changed = Signal(object)  # Emits VisualizationConfig object
    visualization_completed = Signal()  # Signal when visualization is complete
    visualization_failed = Signal(str)  # Signal when visualization fails

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: DataManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(
            viewer=viewer,
            data_manager=data_manager,
            visualization_manager=visualization_manager,
            widget_title="Edge Analysis and Visualization"
        )

        self.vis_manager = visualization_manager

        # Initialize visualization manager's layers
        if not hasattr(self.vis_manager, '_edge_layer'):
            self.vis_manager._edge_layer = None
        if not hasattr(self.vis_manager, '_intercalation_layer'):
            self.vis_manager._intercalation_layer = None
        if not hasattr(self.vis_manager, '_analysis_layer'):
            self.vis_manager._analysis_layer = None
        if not hasattr(self.vis_manager, '_color_cycle'):
            self.vis_manager._color_cycle = np.random.RandomState(0)

        # Initialize analysis components
        self.analysis_params = EdgeAnalysisParams()
        self.visualization_config = VisualizationConfig()
        self.analyzer = EdgeAnalyzer()
        self.analyzer.update_parameters(self.analysis_params)
        self.visualizer = Visualizer(config=self.visualization_config, napari_viewer=self.viewer)

        # Initialize results storage
        self._current_results = None
        self._current_boundaries = None
        self._loading_config = False

        # Initialize thread management
        self._analysis_thread = None
        self._analysis_worker = None
        self._visualization_thread = None
        self._visualization_worker = None

        # Initialize UI components
        self._initialize_controls()
        self._setup_ui()
        self._connect_signals()

        # Add layer events handlers
        self.viewer.layers.events.removed.connect(self._update_ui_state)
        self.viewer.layers.events.inserted.connect(self._update_ui_state)
        self.viewer.layers.selection.events.changed.connect(self._update_ui_state)

        self._update_ui_state()

    def _create_parameters_group(self) -> QGroupBox:
        """Create unified parameters group"""
        group = QGroupBox("Analysis Parameters")
        layout = QFormLayout()
        layout.setSpacing(4)

        # Configure dilation control
        self.dilation_spin.setRange(1, 10)
        self.dilation_spin.setValue(self.analysis_params.dilation_radius)
        self.dilation_spin.setToolTip("Radius for morphological dilation when finding boundaries")
        layout.addRow("Dilation Radius:", self.dilation_spin)

        # Configure overlap control
        self.overlap_spin.setRange(1, 100)
        self.overlap_spin.setValue(self.analysis_params.min_overlap_pixels)
        self.overlap_spin.setToolTip("Minimum number of overlapping pixels to consider cells as neighbors")
        layout.addRow("Min Overlap Pixels:", self.overlap_spin)

        # Configure length control
        self.min_length_spin.setRange(0.0, 1000.0)
        self.min_length_spin.setValue(self.analysis_params.min_edge_length)
        self.min_length_spin.setSingleStep(0.5)
        self.min_length_spin.setToolTip("Minimum edge length in pixels (0 to disable)")
        layout.addRow("Min Edge Length (px):", self.min_length_spin)

        # Add pixel size control
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.001, 1000.0)
        self.pixel_size_spin.setValue(1.0)
        self.pixel_size_spin.setSingleStep(0.1)
        self.pixel_size_spin.setDecimals(3)
        self.pixel_size_spin.setToolTip("Pixel size in micrometers")
        layout.addRow("Pixel Size (µm):", self.pixel_size_spin)

        # Add frame length control
        self.frame_length_spin = QDoubleSpinBox()
        self.frame_length_spin.setRange(0.1, 1000.0)
        self.frame_length_spin.setValue(1.0)
        self.frame_length_spin.setSingleStep(0.1)
        self.frame_length_spin.setDecimals(1)
        self.frame_length_spin.setToolTip("Time between frames in minutes")
        layout.addRow("Frame Length (min):", self.frame_length_spin)

        # Configure filter control
        self.filter_isolated_check.setChecked(self.analysis_params.filter_isolated)
        self.filter_isolated_check.setToolTip("Filter out edges that don't connect to others")
        layout.addRow("Filter Isolated:", self.filter_isolated_check)

        # Add reset button at the bottom of parameters
        layout.addRow(self.reset_btn)

        group.setLayout(layout)
        return group

    def _initialize_controls(self):
        """Initialize all UI controls"""
        # Analysis parameters controls
        self.dilation_spin = QSpinBox()
        self.overlap_spin = QSpinBox()
        self.min_length_spin = QDoubleSpinBox()
        self.filter_isolated_check = QCheckBox()

        # Physical units controls
        self.pixel_size_spin = QDoubleSpinBox()
        self.frame_length_spin = QDoubleSpinBox()

        # Visualization controls - all checked by default
        self.tracking_checkbox = QCheckBox("Cell tracking plots")
        self.tracking_checkbox.setChecked(True)

        self.edge_checkbox = QCheckBox("Edge detection overlays")
        self.edge_checkbox.setChecked(True)

        self.intercalation_checkbox = QCheckBox("Intercalation event plots")
        self.intercalation_checkbox.setChecked(True)

        self.edge_length_checkbox = QCheckBox("Edge length evolution plots")
        self.edge_length_checkbox.setChecked(True)

        self.example_gifs_checkbox = QCheckBox("Create example GIFs")
        self.example_gifs_checkbox.setChecked(True)

        self.max_gifs_spinbox = QSpinBox()

        # Action buttons
        self.analyze_btn = QPushButton("Run Edge Analysis")
        self.generate_vis_btn = QPushButton("Generate Visualizations")
        self.save_btn = QPushButton("Save Results")
        self.load_btn = QPushButton("Load Results")
        self.reset_btn = QPushButton("Reset Parameters")

    def _register_controls(self):
        """Register all controls with base widget"""
        for control in [
            self.dilation_spin,
            self.overlap_spin,
            self.min_length_spin,
            self.filter_isolated_check,
            self.pixel_size_spin,  # Add new control
            self.frame_length_spin,  # Add new control
            self.tracking_checkbox,
            self.edge_checkbox,
            self.intercalation_checkbox,
            self.edge_length_checkbox,
            self.example_gifs_checkbox,
            self.max_gifs_spinbox,
            self.analyze_btn,
            self.generate_vis_btn,
            self.save_btn,
            self.load_btn,
            self.reset_btn,
            self.vis_reset_btn
        ]:
            self.register_control(control)

    def _handle_analysis_complete(self, results):
        """Handle completion of analysis"""
        try:
            self._current_results = results

            # Add metadata about units and parameters
            results.update_metadata('pixel_size_um', self.pixel_size_spin.value())
            results.update_metadata('frame_length_min', self.frame_length_spin.value())
            results.update_metadata('units_info', {
                'Coordinates are stored in pixels, edge lengths are stored in µm'
            })

            # Convert lengths to physical units
            pixel_size = self.pixel_size_spin.value()
            for edge_data in results.edges.values():
                edge_data.lengths = [length * pixel_size for length in edge_data.lengths]

            # Add segmentation data to results
            results.set_segmentation_data(self.segmentation_data)

            self.data_manager.analysis_results = results

            boundaries_by_frame = self._extract_boundaries(results)
            self._current_boundaries = boundaries_by_frame

            # Update visualizations
            self.vis_manager.update_edge_visualization(boundaries_by_frame)
            self.vis_manager.update_intercalation_visualization(results)
            self.vis_manager.update_edge_analysis_visualization(results)

            self.save_btn.setEnabled(True)
            self.processing_completed.emit(results)

            # Calculate final statistics
            intercalation_count = sum(
                len(edge.intercalations)
                for edge in results.edges.values()
                if edge.intercalations
            )

            self._update_status(
                f"Analysis complete - found {len(results.edges)} edges and {intercalation_count} intercalations",
                100
            )

        except Exception as e:
            self._handle_analysis_error(e)
        finally:
            self._set_controls_enabled(True)

    def reset_parameters(self):
        """Reset all parameters to defaults simultaneously"""
        # Reset analysis parameters to defaults
        self.dilation_spin.setValue(2)  # Default dilation radius
        self.overlap_spin.setValue(5)  # Default minimum overlap pixels
        self.min_length_spin.setValue(0.0)  # Default minimum edge length
        self.filter_isolated_check.setChecked(True)  # Default filter isolated setting

        # Reset physical unit parameters
        self.pixel_size_spin.setValue(1.0)  # Default pixel size in micrometers
        self.frame_length_spin.setValue(1.0)  # Default frame length in minutes

        # Update the analysis parameters object
        self.analysis_params = EdgeAnalysisParams(
            dilation_radius=2,
            min_overlap_pixels=5,
            min_edge_length=0.0,
            filter_isolated=True
        )

        # Update analyzer with new parameters
        self.analyzer.update_parameters(self.analysis_params)

        # Emit signal that parameters have been updated
        self.parameters_updated.emit()

        # Update status
        self._update_status("Parameters reset to defaults")
    def _generate_visualizations(self):
        """Generate visualizations based on current configuration"""
        try:
            # Check if any visualizations are enabled
            enabled_vis = []
            if self.visualization_config.tracking_plots_enabled:
                enabled_vis.append("tracking plots")
            if self.visualization_config.edge_detection_overlay:
                enabled_vis.append("edge detection overlays")
            if self.visualization_config.intercalation_events:
                enabled_vis.append("intercalation events")
            if self.visualization_config.edge_length_evolution:
                enabled_vis.append("edge length evolution")
            if self.visualization_config.create_example_gifs:
                enabled_vis.append("example GIFs")

            if not enabled_vis:
                raise ProcessingError(
                    "No visualizations enabled",
                    "Please enable at least one visualization type."
                )

            # Get output directory and convert to absolute path
            output_dir = self._get_output_directory()
            if output_dir is None:
                return

            # Convert to absolute path
            output_dir = output_dir.resolve()
            self.visualization_config.output_dir = output_dir
            self.visualizer.config = self.visualization_config

            # Disable controls during processing
            self._set_controls_enabled(False)
            self._update_status(f"Starting visualization generation for: {', '.join(enabled_vis)}...", 0)

            # Create and setup visualization worker and thread
            self._visualization_thread = QThread()
            self._visualization_worker = VisualizationWorker(
                self.visualizer,
                self._current_results,
                output_dir
            )
            self._visualization_worker.moveToThread(self._visualization_thread)

            # Connect signals
            self._visualization_thread.started.connect(self._visualization_worker.run)
            self._visualization_worker.progress.connect(self._handle_visualization_progress)
            self._visualization_worker.finished.connect(self._handle_visualization_complete)
            self._visualization_worker.error.connect(self._handle_visualization_error)
            self._visualization_worker.finished.connect(self._visualization_thread.quit)
            self._visualization_worker.finished.connect(self._visualization_worker.deleteLater)
            self._visualization_thread.finished.connect(self._visualization_thread.deleteLater)

            # Start visualization
            self._visualization_thread.start()

        except Exception as e:
            self._handle_error(ProcessingError(
                message="Failed to start visualization generation",
                details=str(e),
                component=self.__class__.__name__
            ))
            self._set_controls_enabled(True)

    def _handle_visualization_progress(self, progress: int, message: str):
        """Handle progress updates from visualization worker"""
        self._update_status(message, progress)

    def _handle_visualization_complete(self):
        """Handle completion of visualization generation"""
        try:
            # Format success message with enabled visualization types
            enabled_vis = []
            if self.visualization_config.tracking_plots_enabled:
                enabled_vis.append("tracking plots")
            if self.visualization_config.edge_detection_overlay:
                enabled_vis.append("edge detection overlays")
            if self.visualization_config.intercalation_events:
                enabled_vis.append("intercalation events")
            if self.visualization_config.edge_length_evolution:
                enabled_vis.append("edge length evolution")
            if self.visualization_config.create_example_gifs:
                enabled_vis.append("example GIFs")

            vis_count = len(enabled_vis)
            vis_types = "visualization" if vis_count == 1 else "visualizations"

            self._update_status(
                f"Successfully generated {vis_count} {vis_types} in {self.visualizer.output_dir}\n"
                f"Types: {', '.join(enabled_vis)}",
                100
            )
            self.visualization_completed.emit()

        except Exception as e:
            self._handle_visualization_error(e)
        finally:
            self._set_controls_enabled(True)

    def _handle_visualization_error(self, error):
        """Handle errors from visualization worker"""
        if isinstance(error, ProcessingError):
            self._handle_error(error)
        else:
            self._handle_error(ProcessingError(
                message="Error during visualization generation",
                details=str(error),
                component=self.__class__.__name__
            ))
        self._set_controls_enabled(True)

    def _create_analysis_actions_group(self) -> QGroupBox:
        """Create analysis and results actions group"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        layout.addWidget(self.analyze_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.load_btn)

        self.save_btn.setEnabled(False)

        group.setLayout(layout)
        return group

    def _create_visualization_group(self) -> QGroupBox:
        """Create visualization options group"""
        group = QGroupBox("Visualization Options")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        form_layout = QFormLayout()
        form_layout.setSpacing(4)

        # Add visualization controls
        form_layout.addRow(self.tracking_checkbox)
        form_layout.addRow(self.edge_checkbox)
        form_layout.addRow(self.intercalation_checkbox)
        form_layout.addRow(self.edge_length_checkbox)
        form_layout.addRow(self.example_gifs_checkbox)

        # Configure and add GIF controls
        self.max_gifs_spinbox.setRange(1, 10)
        self.max_gifs_spinbox.setValue(self.visualization_config.max_example_gifs)
        gif_layout = QHBoxLayout()
        gif_label = QLabel("Maximum example GIFs:")
        gif_layout.addWidget(gif_label)
        gif_layout.addWidget(self.max_gifs_spinbox)
        gif_layout.addStretch()
        form_layout.addRow(gif_layout)

        # Create visualization reset button
        self.vis_reset_btn = QPushButton("Reset Parameters")

        # Add layouts to main group layout
        layout.addLayout(form_layout)
        layout.addWidget(self.vis_reset_btn)

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

        # Add groups in new order
        right_layout.addWidget(self._create_parameters_group())
        right_layout.addWidget(self._create_analysis_actions_group())
        right_layout.addWidget(self._create_visualization_group())
        right_layout.addWidget(self.generate_vis_btn)

        self.generate_vis_btn.setEnabled(False)

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

        # Add to main layout
        self.main_layout.addWidget(right_container)
        self.main_layout.addStretch(1)

        # Register all controls
        self._register_controls()

    def _create_action_group(self) -> QGroupBox:
        """Create action buttons group"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        layout.addWidget(self.analyze_btn)
        layout.addWidget(self.generate_vis_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.load_btn)

        self.save_btn.setEnabled(False)
        self.generate_vis_btn.setEnabled(False)

        group.setLayout(layout)
        return group

    def _connect_signals(self):
        # Analysis parameter signals
        self.dilation_spin.valueChanged.connect(self.update_parameters)
        self.overlap_spin.valueChanged.connect(self.update_parameters)
        self.min_length_spin.valueChanged.connect(self.update_parameters)
        self.filter_isolated_check.stateChanged.connect(self.update_parameters)

        # Visualization parameter signals
        for checkbox in [
            self.tracking_checkbox,
            self.edge_checkbox,
            self.intercalation_checkbox,
            self.edge_length_checkbox,
            self.example_gifs_checkbox
        ]:
            checkbox.toggled.connect(self._config_changed)

        self.max_gifs_spinbox.valueChanged.connect(self._config_changed)
        self.example_gifs_checkbox.stateChanged.connect(
            lambda state: self.max_gifs_spinbox.setEnabled(bool(state))
        )

        # Action button signals
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.generate_vis_btn.clicked.connect(self._generate_visualizations)
        self.save_btn.clicked.connect(self.save_results)
        self.load_btn.clicked.connect(self.load_results)
        self.reset_btn.clicked.connect(self.reset_parameters)
        self.vis_reset_btn.clicked.connect(self.reset_visualization_parameters)

        # Result signals
        self.edges_detected.connect(self.vis_manager.update_edge_visualization)
        self.analysis_completed.connect(self.vis_manager.update_edge_analysis_visualization)

    def _config_changed(self):
        """Update visualization config when any setting changes"""
        if self._loading_config:
            return

        logger.debug("Updating visualization config from UI")

        self.visualization_config.tracking_plots_enabled = self.tracking_checkbox.isChecked()
        self.visualization_config.edge_detection_overlay = self.edge_checkbox.isChecked()
        self.visualization_config.intercalation_events = self.intercalation_checkbox.isChecked()
        self.visualization_config.edge_length_evolution = self.edge_length_checkbox.isChecked()
        self.visualization_config.create_example_gifs = self.example_gifs_checkbox.isChecked()
        self.visualization_config.max_example_gifs = self.max_gifs_spinbox.value()

        self.visualization_config_changed.emit(self.visualization_config)

    def reset_visualization_parameters(self):
        """Reset visualization parameters to defaults"""
        self._loading_config = True
        try:
            # Reset checkboxes
            self.tracking_checkbox.setChecked(True)
            self.edge_checkbox.setChecked(True)
            self.intercalation_checkbox.setChecked(True)
            self.edge_length_checkbox.setChecked(True)
            self.example_gifs_checkbox.setChecked(True)

            # Reset GIF spinbox
            self.max_gifs_spinbox.setValue(3)  # Default value

            # Update configuration
            self.visualization_config = VisualizationConfig()
            self.visualization_config_changed.emit(self.visualization_config)

            self._update_status("Visualization parameters reset to defaults")
        finally:
            self._loading_config = False

    def _get_output_directory(self) -> Optional[Path]:
        """Show directory dialog for selecting output location"""
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Output Directory for Visualizations")
        dialog.setFileMode(QFileDialog.Directory)

        if hasattr(self.data_manager, 'last_directory') and self.data_manager.last_directory:
            dialog.setDirectory(str(self.data_manager.last_directory))

        if dialog.exec_():
            output_dir = Path(dialog.selectedFiles()[0])
            if hasattr(self.data_manager, 'last_directory'):
                self.data_manager.last_directory = output_dir.parent
            return output_dir
        return None

    def _update_ui_state(self, event=None):
        """Update UI based on current state"""
        # Check for valid input: active labels layer
        active_layer = self._get_active_labels_layer()
        has_valid_input = (active_layer is not None and
                           isinstance(active_layer, napari.layers.Labels) and
                           active_layer.data.ndim in [2, 3])

        # Check for valid results data
        has_valid_results = (self._current_results is not None and
                             hasattr(self._current_results, 'edges') and
                             bool(self._current_results.edges))

        # Update button states
        self.analyze_btn.setEnabled(has_valid_input)
        self.save_btn.setEnabled(has_valid_results)
        self.generate_vis_btn.setEnabled(has_valid_results)

    def run_analysis(self):
        selected = self._get_active_labels_layer()
        if selected is None:
            raise ProcessingError(
                message="Please select a label layer for edge analysis",
                component=self.__class__.__name__
            )

        try:
            self._set_controls_enabled(False)
            self._update_status("Initializing edge analysis...", 5)

            self.vis_manager.clear_edge_layers()

            # Store the segmentation data
            self.segmentation_data = selected.data

            # Create worker and thread
            self._analysis_thread = QThread()
            self._analysis_worker = AnalysisWorker(self.analyzer, self.segmentation_data)
            self._analysis_worker.moveToThread(self._analysis_thread)

            # Connect signals
            self._analysis_thread.started.connect(self._analysis_worker.run)
            self._analysis_worker.progress.connect(self._handle_progress)
            self._analysis_worker.finished.connect(self._handle_analysis_complete)
            self._analysis_worker.error.connect(self._handle_analysis_error)
            self._analysis_worker.finished.connect(self._analysis_thread.quit)
            self._analysis_worker.finished.connect(self._analysis_worker.deleteLater)
            self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)

            # Start analysis
            self._analysis_thread.start()

        except Exception as e:
            self._handle_error(ProcessingError(
                message="Failed to start analysis",
                details=str(e),
                component=self.__class__.__name__
            ))
            self._set_controls_enabled(True)

    def _handle_progress(self, progress: int, message: str):
        """Handle progress updates from worker"""
        self._update_status(message, progress)

    def _handle_analysis_error(self, error):
        """Handle errors from worker"""
        if isinstance(error, ProcessingError):
            self._handle_error(error)
        else:
            self._handle_error(ProcessingError(
                message="Error during analysis",
                details=str(error),
                component=self.__class__.__name__
            ))
        self._set_controls_enabled(True)

    def update_parameters(self):
        """Update edge analysis parameters from UI controls"""
        try:
            new_params = EdgeAnalysisParams(
                dilation_radius=self.dilation_spin.value(),
                min_overlap_pixels=self.overlap_spin.value(),
                min_edge_length=self.min_length_spin.value(),
                filter_isolated=self.filter_isolated_check.isChecked()
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
            # Validate we have results to save
            if self._current_results is None:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No analysis results to save. Please run analysis first."
                )
                return

            # Get save path
            file_path = self._get_save_path()
            if file_path is None:
                return

            self._set_controls_enabled(False)
            self._update_status("Saving analysis results...", 20)

            # Ensure file has .pkl extension
            if file_path.suffix.lower() != '.pkl':
                file_path = file_path.with_suffix('.pkl')

            # Ensure segmentation data is included in results
            if self.segmentation_data is not None and self._current_results.get_segmentation_data() is None:
                self._current_results.set_segmentation_data(self.segmentation_data)

            # Update data manager and save
            self.data_manager.analysis_results = self._current_results
            self.data_manager.save_analysis_results(file_path)

            self._update_status(
                f"Results saved to {file_path.name} "
                f"({len(self._current_results.edges)} edges)",
                100
            )

        except Exception as e:
            # Use base widget's error handling
            self._handle_error(ProcessingError(
                message="Failed to save results",
                details=str(e),
                component=self.__class__.__name__
            ))
        finally:
            self._set_controls_enabled(True)

    def load_results(self):
        """Load previously saved analysis results"""
        try:
            file_path = self._get_load_path()
            if file_path is None:
                # User cancelled the file dialog
                self._update_ui_state()  # Ensure buttons are in correct state
                return

            if not file_path.exists():
                raise ProcessingError(
                    message="File not found",
                    details=str(file_path),
                    component=self.__class__.__name__
                )

            self._set_controls_enabled(False)
            self._update_status("Loading analysis results...", 20)

            # Clear previous visualizations
            self.vis_manager.clear_edge_layers()

            # Load and validate data
            self.data_manager.load_analysis_results(file_path)
            loaded_data = self.data_manager.analysis_results

            if loaded_data is None:
                raise ProcessingError(
                    message="No valid analysis results found in file",
                    component=self.__class__.__name__
                )

            # Validate essential attributes
            if not hasattr(loaded_data, 'edges') or not loaded_data.edges:
                raise ProcessingError(
                    message="Invalid analysis results: missing edge data",
                    component=self.__class__.__name__
                )

            # Validate segmentation data
            if not hasattr(loaded_data, 'get_segmentation_data') or loaded_data.get_segmentation_data() is None:
                raise ProcessingError(
                    message="Invalid analysis results: missing segmentation data",
                    component=self.__class__.__name__
                )

            # Store current results and segmentation data
            self._current_results = loaded_data
            self.segmentation_data = loaded_data.get_segmentation_data()

            # Extract and validate boundaries
            boundaries_by_frame = self._extract_boundaries(loaded_data)
            if not boundaries_by_frame:
                raise ProcessingError(
                    message="No valid boundary data found in results",
                    component=self.__class__.__name__
                )
            self._current_boundaries = boundaries_by_frame

            # Update visualizations
            self._update_status("Updating visualizations...", 60)
            self.vis_manager.update_edge_visualization(boundaries_by_frame)
            self.vis_manager.update_intercalation_visualization(loaded_data)
            self.vis_manager.update_edge_analysis_visualization(loaded_data)

            # Optionally, restore the segmentation layer in napari
            self._restore_segmentation_layer()

            # Emit signals for other components
            self.edges_detected.emit(boundaries_by_frame)
            self.analysis_completed.emit(loaded_data)

            self._update_status(
                f"Loaded analysis results from {file_path.name} "
                f"({len(loaded_data.edges)} edges)",
                100
            )

        except Exception as e:
            if isinstance(e, ProcessingError):
                self._handle_error(e)  # Pass through existing ProcessingErrors
            else:
                # Use base widget's error handling
                self._handle_error(ProcessingError(
                    message="Failed to load results",
                    details=str(e),
                    component=self.__class__.__name__
                ))

            # Clear any partial results
            self._current_results = None
            self._current_boundaries = None
            self.segmentation_data = None

        finally:
            self._set_controls_enabled(True)
            self._update_ui_state()  # Ensure buttons are in correct state

    def _restore_segmentation_layer(self):
        """Restore the segmentation layer in napari viewer"""
        if self.segmentation_data is not None:
            # Check if a layer with the same data already exists
            existing_layers = [layer for layer in self.viewer.layers
                               if isinstance(layer, napari.layers.Labels)
                               and np.array_equal(layer.data, self.segmentation_data)]

            if not existing_layers:
                # Add new layer if it doesn't exist
                self.viewer.add_labels(
                    self.segmentation_data,
                    name='Loaded Segmentation',
                    opacity=0.5
                )

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
        # Clean up analysis thread
        if self._analysis_thread and self._analysis_thread.isRunning():
            self._analysis_thread.quit()
            self._analysis_thread.wait()

        # Clean up visualization thread
        if self._visualization_thread and self._visualization_thread.isRunning():
            self._visualization_thread.quit()
            self._visualization_thread.wait()

        if self.visualization_manager:
            self.visualization_manager.clear_edge_layers()

        self._current_results = None
        self._current_boundaries = None
        self._update_ui_state()

        super().cleanup()
