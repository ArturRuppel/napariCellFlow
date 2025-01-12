import logging
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QTabWidget
)

from napariCellFlow.preprocessing_widget import PreprocessingWidget
from napariCellFlow.cell_tracking_widget import CellTrackingWidget
from napariCellFlow.segmentation_widget import SegmentationWidget
from napariCellFlow.data_manager import DataManager
from napariCellFlow.visualization_manager import VisualizationManager
from napariCellFlow.edge_analysis_widget import EdgeAnalysisWidget

logger = logging.getLogger(__name__)


class napariCellFlow(QWidget):
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
        title = QLabel("napariCellFlow")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        container_layout.addWidget(title)

        # Initialize managers first
        self.data_manager = DataManager()
        self.visualization_manager = VisualizationManager(self.viewer, self.data_manager)

        # Create tab widget for different components
        tabs = QTabWidget()

        # Initialize component widgets with the managers
        self.preprocessing_widget = PreprocessingWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )

        self.tracking_widget = CellTrackingWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )

        self.segmentation_widget = SegmentationWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )

        self.edge_analysis_widget = EdgeAnalysisWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )


        # Add widgets to tabs
        tabs.addTab(self.preprocessing_widget, "Preprocessing")
        tabs.addTab(self.segmentation_widget, "Segmentation")
        tabs.addTab(self.tracking_widget, "Cell Tracking")
        tabs.addTab(self.edge_analysis_widget, "Edge Analysis")

        # Add tabs to container
        container_layout.addWidget(tabs)
        container_layout.addStretch()

        # Set container as scroll area widget
        scroll.setWidget(container)

        # Add scroll area to main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self.connect_signals()

    def connect_signals(self):
        """Connect signals between components"""
        # Connect preprocessing signals
        self.preprocessing_widget.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.preprocessing_widget.processing_failed.connect(self._on_preprocessing_failed)

        # Connect tracking signals
        self.tracking_widget.processing_completed.connect(self._on_tracking_completed)
        self.tracking_widget.processing_failed.connect(self._on_tracking_failed)

        # Connect segmentation signals
        self.segmentation_widget.processing_completed.connect(self._on_segmentation_completed)
        self.segmentation_widget.processing_failed.connect(self._on_segmentation_failed)



    def _on_preprocessing_completed(self, processed_stack, preprocessing_info):
        """Handle completion of preprocessing"""
        logger.info("Preprocessing completed successfully")
        self.data_manager.preprocessed_data = processed_stack

    def _on_preprocessing_failed(self, error_msg):
        """Handle preprocessing failure"""
        logger.error(f"Preprocessing failed: {error_msg}")
        QMessageBox.critical(self, "Error", f"Preprocessing failed: {error_msg}")

    def _on_segmentation_completed(self, result):
        """Handle completion of segmentation"""
        logger.info("Segmentation completed successfully")
        self.data_manager.segmentation_data = result

    def _on_segmentation_failed(self, error_msg):
        """Handle segmentation failure"""
        logger.error(f"Segmentation failed: {error_msg}")
        QMessageBox.critical(self, "Error", f"Segmentation failed: {error_msg}")

    def _on_tracking_completed(self, tracked_data):
        """Handle completion of tracking"""
        logger.info("Tracking completed successfully")
        self.data_manager.tracked_data = tracked_data

    def _on_tracking_failed(self, error_msg):
        """Handle tracking failure"""
        logger.error(f"Tracking failed: {error_msg}")
        QMessageBox.critical(self, "Error", f"Tracking failed: {error_msg}")


