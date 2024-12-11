from typing import Optional, List
import logging
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QProgressBar,
    QDoubleSpinBox, QSpinBox, QCheckBox, QProgressDialog, QGroupBox,
    QMessageBox
)
from qtpy.QtCore import Signal, Qt
import napari

from napari_cellpose_stackmode.error_handling import ErrorHandlingMixin, ApplicationError, ErrorSeverity

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Custom error for processing operations"""

    def __init__(self, message: str, details: str = None, component: str = None):
        self.message = message
        self.details = details or message
        self.component = component
        super().__init__(message)


class BaseAnalysisWidget(QWidget, ErrorHandlingMixin):
    # Common signals
    processing_completed = Signal(object)  # Generic completion signal
    processing_failed = Signal(str)  # Error message signal
    parameters_updated = Signal()  # Parameters changed signal

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: "DataManager",
            visualization_manager: "VisualizationManager",
            widget_title: str
    ):
        super().__init__()

        # Store main components
        self.viewer = viewer
        self.data_manager = data_manager
        self.vis_manager = visualization_manager

        # Initialize state
        self._processing = False
        self._controls: List[QWidget] = []

        # Set up UI
        self._setup_base_ui(widget_title)
    def _handle_error(self, error):
        # Convert ProcessingError to ApplicationError if needed
        if isinstance(error, ProcessingError):
            app_error = self.create_error(
                message=error.message,
                details=error.details,
                component=error.component or self.__class__.__name__
            )
        elif isinstance(error, ApplicationError):
            app_error = error
        else:
            app_error = self.create_error(
                message=str(error),
                severity=ErrorSeverity.ERROR
            )

        self.handle_error(app_error)


    def _setup_base_ui(self, title: str):
        """Initialize the base user interface"""
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.main_layout.addWidget(self.title_label)

        # Progress bar
        self.progress_bar = QProgressBar()

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        # Create bottom section for progress and status
        self.bottom_layout = QVBoxLayout()
        self.bottom_layout.addWidget(self.progress_bar)
        self.bottom_layout.addWidget(self.status_label)

        # Add stretch to push bottom section down
        self.main_layout.addStretch()
        self.main_layout.addLayout(self.bottom_layout)

    def _ensure_stack_format(self, data: np.ndarray) -> np.ndarray:
        """Ensure data is in [t, y, x] format"""
        try:
            if data.ndim == 2:
                return data[np.newaxis, :, :]
            elif data.ndim == 3:
                if data.shape[-1] < data.shape[0] and data.shape[-1] < data.shape[1]:
                    return np.moveaxis(data, -1, 0)
                return data
            else:
                raise ProcessingError(
                    message=f"Unexpected data dimensions: {data.shape}",
                    component=self.__class__.__name__
                )
        except Exception as e:
            raise ProcessingError(
                message="Failed to format data stack",
                details=str(e),
                component=self.__class__.__name__
            )

    def _create_parameter_group(self, title: str) -> QGroupBox:
        """Create a standard parameter group"""
        group = QGroupBox(title)
        group_layout = QVBoxLayout()
        group.setLayout(group_layout)
        return group

    def _create_progress_dialog(self, max_value: int, title: str = "Processing...") -> QProgressDialog:
        """Create a progress dialog"""
        progress = QProgressDialog(title, "Cancel", 0, max_value, self)
        progress.setWindowModality(Qt.WindowModal)
        return progress

    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and progress"""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)
        logger.info(f"{self.__class__.__name__}: {message}")

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all controls"""
        for control in self._controls:
            if isinstance(control, (QCheckBox, QPushButton)):
                control.setEnabled(enabled)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                # Respect checkbox dependencies
                parent_checkbox = getattr(control, '_parent_checkbox', None)
                control.setEnabled(enabled and (not parent_checkbox or parent_checkbox.isChecked()))

    def register_control(self, control: QWidget, parent_checkbox: Optional[QCheckBox] = None):
        """Register a control for state management"""
        self._controls.append(control)
        if parent_checkbox:
            setattr(control, '_parent_checkbox', parent_checkbox)
            # Connect checkbox state to control enabled state
            parent_checkbox.stateChanged.connect(
                lambda state: control.setEnabled(bool(state))
            )

    def _validate_input_data(self, data: np.ndarray) -> bool:
        """Validate input data format and content"""
        try:
            if not isinstance(data, np.ndarray):
                return False
            if data.size == 0:
                return False
            if data.ndim not in [2, 3]:
                return False
            return True
        except Exception as e:
            logger.warning(f"Data validation failed: {str(e)}")
            return False

    def _get_active_image_layer(self) -> Optional["napari.layers.Image"]:
        """Get currently active image layer"""
        active_layer = self.viewer.layers.selection.active
        if isinstance(active_layer, napari.layers.Image):
            return active_layer

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                return layer
        return None

    def _get_active_labels_layer(self) -> Optional["napari.layers.Labels"]:
        """Get currently active labels layer or first available labels layer"""
        active_layer = self.viewer.layers.selection.active
        if isinstance(active_layer, napari.layers.Labels):
            return active_layer

        # If no labels layer is active, look for the first available labels layer
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                return layer
        return None

    def cleanup(self):
        """Clean up resources"""
        self._controls.clear()