import logging
from pathlib import Path
from typing import Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QSizePolicy,
    QCheckBox, QSpinBox, QFormLayout, QPushButton, QFileDialog,
    QLabel
)

from napariCellFlow.base_widget import BaseAnalysisWidget, ProcessingError
from napariCellFlow.data_manager import DataManager
from napariCellFlow.structure import VisualizationConfig
from napariCellFlow.edge_analysis_visualization import Visualizer
from napariCellFlow.visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class VisualizationWidget(BaseAnalysisWidget):
    """Widget for controlling visualization output generation"""

    # Signals
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
            widget_title="Visualization Controls"
        )

        self.config = VisualizationConfig()
        self.visualizer = Visualizer(config=self.config, napari_viewer=self.viewer)
        self._loading_config = False

        # Initialize controls
        self._initialize_controls()

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

    def _initialize_controls(self):
        """Initialize all UI controls"""
        # Visualization type controls
        self.tracking_checkbox = QCheckBox("Cell tracking plots")
        self.edge_checkbox = QCheckBox("Edge detection overlays")
        self.intercalation_checkbox = QCheckBox("Intercalation event plots")
        self.edge_length_checkbox = QCheckBox("Edge length evolution plots")
        self.example_gifs_checkbox = QCheckBox("Create example GIFs")

        # GIF controls
        self.max_gifs_spinbox = QSpinBox()
        self.max_gifs_spinbox.setRange(1, 10)
        self.max_gifs_spinbox.setValue(self.config.max_example_gifs)

        # Action buttons
        self.generate_button = QPushButton("Generate Visualizations")
        self.reset_button = QPushButton("Reset Parameters")

    def _create_visualization_group(self) -> QGroupBox:
        """Create visualization options group"""
        group = QGroupBox("Output Visualizations")
        layout = QFormLayout()
        layout.setSpacing(4)

        # Add visualization controls
        layout.addRow(self.tracking_checkbox)
        layout.addRow(self.edge_checkbox)
        layout.addRow(self.intercalation_checkbox)
        layout.addRow(self.edge_length_checkbox)
        layout.addRow(self.example_gifs_checkbox)

        # Add GIF controls
        gif_layout = QHBoxLayout()
        gif_label = QLabel("Maximum example GIFs:")
        gif_layout.addWidget(gif_label)
        gif_layout.addWidget(self.max_gifs_spinbox)
        gif_layout.addStretch()
        layout.addRow(gif_layout)

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
        layout.addWidget(self.generate_button)
        layout.addWidget(self.reset_button)
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

        # Add visualization and action groups
        right_layout.addWidget(self._create_visualization_group())
        right_layout.addWidget(self._create_action_group())

        # Add stretch before status section
        right_layout.addStretch()

        right_container.setLayout(right_layout)

        # Add to main layout from base widget
        self.main_layout.addWidget(right_container)
        self.main_layout.addStretch(1)

        # Register all controls
        self._register_controls()

    def _register_controls(self):
        """Register all controls with base widget"""
        for control in [
            self.tracking_checkbox,
            self.edge_checkbox,
            self.intercalation_checkbox,
            self.edge_length_checkbox,
            self.example_gifs_checkbox,
            self.max_gifs_spinbox,
            self.generate_button,
            self.reset_button
        ]:
            self.register_control(control)

    def _connect_signals(self):
        """Connect widget signals"""
        self.generate_button.clicked.connect(self._generate_visualizations)
        self.reset_button.clicked.connect(self.reset_parameters)

        # Connect checkboxes
        for checkbox in [
            self.tracking_checkbox,
            self.edge_checkbox,
            self.intercalation_checkbox,
            self.edge_length_checkbox,
            self.example_gifs_checkbox
        ]:
            checkbox.toggled.connect(self._config_changed)

        # Connect spinbox
        self.max_gifs_spinbox.valueChanged.connect(self._config_changed)

        # Connect example GIFs checkbox to spinbox enabled state
        self.example_gifs_checkbox.stateChanged.connect(
            lambda state: self.max_gifs_spinbox.setEnabled(bool(state))
        )

    def _config_changed(self):
        """Update config when any setting changes"""
        if self._loading_config:
            return

        logger.debug("Updating visualization config from UI")

        self.config.tracking_plots_enabled = self.tracking_checkbox.isChecked()
        self.config.edge_detection_overlay = self.edge_checkbox.isChecked()
        self.config.intercalation_events = self.intercalation_checkbox.isChecked()
        self.config.edge_length_evolution = self.edge_length_checkbox.isChecked()
        self.config.create_example_gifs = self.example_gifs_checkbox.isChecked()
        self.config.max_example_gifs = self.max_gifs_spinbox.value()

        self.visualization_config_changed.emit(self.config)

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self._loading_config = True

        self.tracking_checkbox.setChecked(False)
        self.edge_checkbox.setChecked(False)
        self.intercalation_checkbox.setChecked(False)
        self.edge_length_checkbox.setChecked(False)
        self.example_gifs_checkbox.setChecked(False)
        self.max_gifs_spinbox.setValue(3)

        self._loading_config = False
        self._config_changed()
        self._update_status("Parameters reset to defaults")

    def _generate_visualizations(self):
        """Generate visualizations based on current configuration"""
        try:
            logger.debug("Starting visualization generation")

            # Check if any visualizations are enabled
            any_vis_enabled = (
                    self.config.tracking_plots_enabled or
                    self.config.edge_detection_overlay or
                    self.config.intercalation_events or
                    self.config.edge_length_evolution
            )

            if not any_vis_enabled:
                raise ProcessingError(
                    "No visualizations enabled",
                    "Please enable at least one visualization type."
                )

            # Disable controls during processing
            self._set_controls_enabled(False)
            self._update_status("Starting visualization generation...", 0)

            # Get output directory
            output_dir = self._get_output_directory()
            if output_dir is None:
                raise ProcessingError("No output directory selected")

            logger.debug(f"Selected output directory: {output_dir}")

            # Update config with output directory
            self.config.output_dir = output_dir

            # Check for analysis results
            if not hasattr(self.data_manager, 'analysis_results') or self.data_manager.analysis_results is None:
                raise ProcessingError(
                    "No analysis results available",
                    "Please complete edge analysis first."
                )

            # Generate visualizations
            self._update_status("Generating visualizations...", 30)
            self.visualizer.create_visualizations(self.data_manager.analysis_results)

            self._update_status(f"Visualizations generated successfully in {output_dir}", 100)
            self.visualization_completed.emit()

        except ProcessingError as e:
            self._handle_error(e)
        except Exception as e:
            self._handle_error(ProcessingError(
                "Visualization generation failed",
                str(e),
                self.__class__.__name__
            ))
        finally:
            self._set_controls_enabled(True)

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

    def get_config(self) -> VisualizationConfig:
        """Get the current visualization configuration"""
        return self.config

    def set_config(self, config: VisualizationConfig):
        """Set the visualization configuration"""
        self._loading_config = True

        self.config = config
        self.tracking_checkbox.setChecked(config.tracking_plots_enabled)
        self.edge_checkbox.setChecked(config.edge_detection_overlay)
        self.intercalation_checkbox.setChecked(config.intercalation_events)
        self.edge_length_checkbox.setChecked(config.edge_length_evolution)
        self.example_gifs_checkbox.setChecked(config.create_example_gifs)
        self.max_gifs_spinbox.setValue(config.max_example_gifs)
        self.max_gifs_spinbox.setEnabled(config.create_example_gifs)

        self._loading_config = False
        self._config_changed()