import numpy as np
from PyQt5.QtWidgets import QGridLayout
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, QSpinBox,
    QGroupBox, QFormLayout, QPushButton, QMessageBox, QFileDialog
)
from qtpy.QtCore import Signal
import napari
import logging
from pathlib import Path
from typing import Dict, Optional, List

from napari_cellpose_stackmode.data_manager import DataManager
from napari_cellpose_stackmode.structure import VisualizationConfig, EdgeData
from napari_cellpose_stackmode.visualization import Visualizer
from napari_cellpose_stackmode.visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class VisualizationWidget(QWidget):
    """Widget for controlling visualization output generation"""

    # Signals
    visualization_config_changed = Signal(object)  # Emits VisualizationConfig object
    visualization_completed = Signal()  # Signal when visualization is complete
    visualization_failed = Signal(str)  # Signal when visualization fails

    def __init__(self, viewer: "napari.Viewer", data_manager: DataManager, visualization_manager: VisualizationManager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.visualization_manager = visualization_manager
        self.config = VisualizationConfig()
        self.visualizer = Visualizer(config=self.config, napari_viewer=self.viewer)
        self._loading_config = False
        self.setup_ui()

    def setup_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create group box for visualization options
        group_box = QGroupBox("Output Visualizations")
        group_layout = QGridLayout()
        group_box.setLayout(group_layout)

        # Create checkboxes for different plot types
        self.tracking_checkbox = QCheckBox("Cell tracking plots")
        self.tracking_checkbox.setChecked(self.config.tracking_plots_enabled)
        self.tracking_checkbox.stateChanged.connect(self._config_changed)

        self.edge_checkbox = QCheckBox("Edge detection overlays")
        self.edge_checkbox.setChecked(self.config.edge_detection_overlay)
        self.edge_checkbox.stateChanged.connect(self._config_changed)

        self.intercalation_checkbox = QCheckBox("Intercalation event plots")
        self.intercalation_checkbox.setChecked(self.config.intercalation_events)
        self.intercalation_checkbox.stateChanged.connect(self._config_changed)

        self.edge_length_checkbox = QCheckBox("Edge length evolution plots")
        self.edge_length_checkbox.setChecked(self.config.edge_length_evolution)
        self.edge_length_checkbox.stateChanged.connect(self._config_changed)

        self.example_gifs_checkbox = QCheckBox("Create example GIFs")
        self.example_gifs_checkbox.setChecked(self.config.create_example_gifs)
        self.example_gifs_checkbox.stateChanged.connect(self._config_changed)

        # Create spinbox for max example GIFs
        spinbox_label = QLabel("Maximum example GIFs:")
        self.max_gifs_spinbox = QSpinBox()
        self.max_gifs_spinbox.setRange(1, 10)
        self.max_gifs_spinbox.setValue(self.config.max_example_gifs)
        self.max_gifs_spinbox.setEnabled(self.config.create_example_gifs)
        self.max_gifs_spinbox.valueChanged.connect(self._config_changed)

        # Add widgets to grid layout (2x3)
        group_layout.addWidget(self.tracking_checkbox, 0, 0)
        group_layout.addWidget(self.edge_checkbox, 1, 0)
        group_layout.addWidget(self.intercalation_checkbox, 2, 0)
        group_layout.addWidget(self.edge_length_checkbox, 0, 1)
        group_layout.addWidget(self.example_gifs_checkbox, 1, 1)

        # Add spinbox with label to the last grid position
        spinbox_widget = QWidget()
        spinbox_layout = QFormLayout()
        spinbox_layout.setContentsMargins(0, 0, 0, 0)
        spinbox_layout.addRow(spinbox_label, self.max_gifs_spinbox)
        spinbox_widget.setLayout(spinbox_layout)
        group_layout.addWidget(spinbox_widget, 2, 1)

        layout.addWidget(group_box)

        # Add Generate button
        self.generate_button = QPushButton("Generate Visualizations")
        self.generate_button.clicked.connect(self._generate_visualizations)
        layout.addWidget(self.generate_button)

        # Add status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Connect example GIFs checkbox to spinbox enabled state
        self.example_gifs_checkbox.stateChanged.connect(
            lambda state: self.max_gifs_spinbox.setEnabled(bool(state))
        )

    def _config_changed(self):
        """Update config when any setting changes"""
        # Check if the config is being loaded programmatically
        if self._loading_config:
            return

        logger.debug("Updating visualization config from UI")
        logger.debug(f"tracking_plots_enabled: {self.tracking_checkbox.isChecked()}")
        logger.debug(f"edge_detection_overlay: {self.edge_checkbox.isChecked()}")
        logger.debug(f"intercalation_events: {self.intercalation_checkbox.isChecked()}")
        logger.debug(f"edge_length_evolution: {self.edge_length_checkbox.isChecked()}")
        logger.debug(f"create_example_gifs: {self.example_gifs_checkbox.isChecked()}")
        logger.debug(f"max_example_gifs: {self.max_gifs_spinbox.value()}")

        self.config.tracking_plots_enabled = self.tracking_checkbox.isChecked()
        self.config.edge_detection_overlay = self.edge_checkbox.isChecked()
        self.config.intercalation_events = self.intercalation_checkbox.isChecked()
        self.config.edge_length_evolution = self.edge_length_checkbox.isChecked()
        self.config.create_example_gifs = self.example_gifs_checkbox.isChecked()
        self.config.max_example_gifs = self.max_gifs_spinbox.value()

        # Emit the updated config
        self.visualization_config_changed.emit(self.config)

    def _on_edges_detected(self, boundaries):
        self.edge_analysis_widget.setEnabled(True)

    def _disable_interactive_widgets(self):
        for widget in [self.tracking_widget, self.edge_analysis_widget]:  # Remove edge_widget reference
            widget.setEnabled(False)

    def _enable_interactive_widgets(self):
        self.tracking_widget.setEnabled(True)
        self.edge_analysis_widget.setEnabled(True)
        self.visualization_widget.setEnabled(True)

    def _convert_analysis_results_to_trajectories(self) -> Dict[int, EdgeData]:
        """Convert stored analysis results to EdgeData objects"""
        if self.data_manager.analysis_results is None:
            return None

        edge_data_dict = {}
        results = self.data_manager.analysis_results

        # Each edge in the results should already be an EdgeData object
        for edge_id, edge_data in results.edges.items():
            edge_data_dict[edge_id] = edge_data

        return edge_data_dict

    def create_edge_evolution_animation(self, segmentation_stack: np.ndarray,
                                        edge_data: EdgeData,
                                        boundaries_by_frame: Dict[int, List['CellBoundary']],
                                        output_path: Path) -> None:
        """Create animation showing edge evolution and length plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.patch.set_facecolor('white')

        def update(frame):
            ax1.clear()
            ax2.clear()
            for ax in (ax1, ax2):
                ax.set_facecolor('white')
                ax.spines['bottom'].set_color('black')
                ax.spines['top'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.spines['right'].set_color('black')
                ax.tick_params(colors='black')

            # Plot segmentation and edge
            ax1.imshow(segmentation_stack[frame], cmap='gray')
            if frame in edge_data.frames:
                idx = edge_data.frames.index(frame)
                cell_pair = edge_data.cell_pairs[idx]

                for boundary in boundaries_by_frame[frame]:
                    if tuple(sorted(int(x) for x in boundary.cell_ids)) == cell_pair:
                        coords = boundary.coordinates
                        ax1.plot(coords[:, 1], coords[:, 0], 'r-',
                                 linewidth=self.config.line_width)
                        break

            ax1.set_title(f'Frame {frame}', color='black')
            ax1.axis('off')

            # Plot length trajectory
            ax2.plot(edge_data.frames, edge_data.lengths, 'b-')
            if frame in edge_data.frames:
                idx = edge_data.frames.index(frame)
                ax2.plot(frame, edge_data.lengths[idx], 'ro')

            # Plot intercalation events
            intercalation_frames = [event.frame for event in edge_data.intercalations]
            for f in intercalation_frames:
                ax2.axvline(x=f, color='r', linestyle='--', alpha=0.5)

            ax2.set_xlabel('Frame', color='black')
            ax2.set_ylabel('Edge Length (µm)', color='black')
            ax2.tick_params(colors='black')
            ax2.grid(True, alpha=0.3)

        anim = FuncAnimation(fig, update, frames=len(segmentation_stack),
                             interval=self.config.animation_interval)
        anim.save(str(output_path), writer='pillow',
                  savefig_kwargs={'facecolor': 'white'})
        plt.close()

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
                QMessageBox.warning(self, "Warning", "No visualizations are enabled. Please enable at least one visualization type.")
                return

            # Disable button during processing
            self.generate_button.setEnabled(False)
            self._update_status("Generating visualizations...")

            # Get output directory
            output_dir = self._get_output_directory()
            if output_dir is None:
                logger.debug("No output directory selected")
                return

            logger.debug(f"Selected output directory: {output_dir}")

            # Create new config with output directory
            new_config = VisualizationConfig(
                tracking_plots_enabled=self.config.tracking_plots_enabled,
                edge_detection_overlay=self.config.edge_detection_overlay,
                intercalation_events=self.config.intercalation_events,
                edge_length_evolution=self.config.edge_length_evolution,
                create_example_gifs=self.config.create_example_gifs,
                max_example_gifs=self.config.max_example_gifs,
                output_dir=output_dir
            )

            # Create visualizer with viewer
            visualizer = Visualizer(config=new_config, napari_viewer=self.viewer)
            logger.debug("Created new visualizer with config and viewer")

            # Get analysis results or create from edge analysis widget data
            if not hasattr(self.data_manager, 'analysis_results') or self.data_manager.analysis_results is None:
                # Try to get data from edge analysis widget if available
                if hasattr(self, 'edge_analysis_widget') and hasattr(self.edge_analysis_widget, 'get_analysis_data'):
                    boundaries, edge_data, events = self.edge_analysis_widget.get_analysis_data()
                    if boundaries and edge_data:
                        self.data_manager.set_analysis_results(boundaries, edge_data, events or [])
                else:
                    raise ValueError("No analysis results available. Please complete edge analysis first.")

            if self.data_manager.analysis_results is None:
                raise ValueError("No analysis results available. Please complete edge analysis first.")

            # Generate visualizations
            logger.debug(f"Analysis results available: {self.data_manager.analysis_results is not None}")
            if self.data_manager.analysis_results:
                logger.debug(f"Number of edges in results: {len(self.data_manager.analysis_results.edges)}")

            logger.debug("Starting visualization creation")
            visualizer.create_visualizations(self.data_manager.analysis_results)

            self._update_status(f"Visualizations generated successfully in {output_dir}")
            self.visualization_completed.emit()

        except Exception as e:
            error_msg = f"Error generating visualizations: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
            self._update_status("Visualization generation failed")
            self.visualization_failed.emit(error_msg)

        finally:
            self.generate_button.setEnabled(True)
    def _get_output_directory(self) -> Optional[Path]:
        """Show directory dialog for selecting output location"""
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Output Directory for Visualizations")
        dialog.setFileMode(QFileDialog.DirectoryOnly)

        # Start in last used directory if available
        if hasattr(self.data_manager, 'last_directory') and self.data_manager.last_directory:
            dialog.setDirectory(str(self.data_manager.last_directory))

        if dialog.exec_():
            output_dir = Path(dialog.selectedFiles()[0])
            # Update last used directory
            if hasattr(self.data_manager, 'last_directory'):
                self.data_manager.last_directory = output_dir.parent
            return output_dir
        return None

    def get_config(self) -> VisualizationConfig:
        """Get the current visualization configuration"""
        return self.config

    def set_config(self, config: VisualizationConfig):
        """Set the visualization configuration"""
        self.config = config

        logger.debug(f"Loaded visualization config: {config}")
        logger.debug(f"tracking_plots_enabled: {config.tracking_plots_enabled}")
        logger.debug(f"edge_detection_overlay: {config.edge_detection_overlay}")
        logger.debug(f"intercalation_events: {config.intercalation_events}")
        logger.debug(f"edge_length_evolution: {config.edge_length_evolution}")
        logger.debug(f"create_example_gifs: {config.create_example_gifs}")
        logger.debug(f"max_example_gifs: {config.max_example_gifs}")

        # Set a flag to indicate that the config is being loaded programmatically
        self._loading_config = True

        # Update UI to match config
        self.tracking_checkbox.setChecked(config.tracking_plots_enabled)
        self.edge_checkbox.setChecked(config.edge_detection_overlay)
        self.intercalation_checkbox.setChecked(config.intercalation_events)
        self.edge_length_checkbox.setChecked(config.edge_length_evolution)
        self.example_gifs_checkbox.setChecked(config.create_example_gifs)
        self.max_gifs_spinbox.setValue(config.max_example_gifs)
        self.max_gifs_spinbox.setEnabled(config.create_example_gifs)

        # Reset the flag after loading the config
        self._loading_config = False

    def enable_generation(self, enable: bool = True):
        """Enable or disable the generate button"""
        self.generate_button.setEnabled(enable)

    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and optionally progress"""
        self.status_label.setText(message)
        if progress is not None:
            # Add progress handling if needed
            pass
        logger.info(message)

    def update_ui_from_config(self):
        """Update UI controls to reflect current configuration"""
        if not hasattr(self, 'config'):
            return

        # Update checkboxes
        self.tracking_checkbox.setChecked(self.config.tracking_plots_enabled)
        self.edge_checkbox.setChecked(self.config.edge_detection_overlay)
        self.intercalation_checkbox.setChecked(self.config.intercalation_events)
        self.edge_length_checkbox.setChecked(self.config.edge_length_evolution)
        self.example_gifs_checkbox.setChecked(self.config.create_example_gifs)

        # Update spinbox
        self.max_gifs_spinbox.setValue(self.config.max_example_gifs)
        self.max_gifs_spinbox.setEnabled(self.config.create_example_gifs)

        # Emit signal that configuration has changed
        self.visualization_config_changed.emit(self.config)
