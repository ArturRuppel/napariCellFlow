import logging
from pathlib import Path
from typing import List

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox
)

from napari_cellpose_stackmode.cell_tracking_widget import CellTrackingWidget
from napari_cellpose_stackmode.data_manager import DataManager
from napari_cellpose_stackmode.visualization_manager import VisualizationManager


logger = logging.getLogger(__name__)


class CellposeStackmodeWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer

        # Create scroll area for widgets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create container widget for scroll area
        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        # Add title
        title = QLabel("Cellpose Stackmode")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        container_layout.addWidget(title)

        # Initialize managers with proper viewer instance
        self.visualization_manager = VisualizationManager(self.viewer)  # Pass viewer here
        self.data_manager = DataManager()

        # Initialize component widgets
        self.tracking_widget = CellTrackingWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )

        # Add widgets to container
        container_layout.addWidget(self.tracking_widget)

        container_layout.addStretch()

        # Set container as scroll area widget
        scroll.setWidget(container)

        # Add scroll area to main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self.connect_signals()
    def _create_data_manager(self):
        """Temporary minimal DataManager"""
        class DataManager:
            def __init__(self):
                self.tracked_data = None
                self.last_directory = None
                self.batch_mode = False
        return DataManager()

    def _create_visualization_manager(self):
        """Temporary minimal VisualizationManager"""
        class VisualizationManager:
            def __init__(self):
                pass
            def update_tracking_visualization(self, tracked_data):
                pass
        return VisualizationManager()

    def connect_signals(self):
        """Connect signals between components"""
        self.tracking_widget.tracking_completed.connect(self._on_tracking_completed)
        self.tracking_widget.tracking_failed.connect(self._on_tracking_failed)

    def _on_tracking_completed(self, tracked_data):
        """Handle completion of tracking"""
        # Will implement visualization and data storage logic here
        pass

    def _on_tracking_failed(self, error_msg):
        """Handle tracking failure"""
        QMessageBox.critical(self, "Error", f"Tracking failed: {error_msg}")