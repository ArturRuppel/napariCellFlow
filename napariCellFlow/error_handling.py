from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any
import logging
from qtpy.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ApplicationError:
    """Unified error type for application-wide error handling"""
    message: str
    details: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.ERROR
    component: Optional[str] = None
    recovery_hint: Optional[str] = None
    original_error: Optional[Exception] = None

    def log(self):
        """Log the error with appropriate severity"""
        log_message = f"{self.component}: {self.message}" if self.component else self.message
        if self.details:
            log_message += f"\nDetails: {self.details}"

        if self.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_message, exc_info=self.original_error)
        else:  # CRITICAL
            logger.critical(log_message, exc_info=self.original_error)


class ErrorHandler:
    """Central error handling system"""

    @staticmethod
    def handle(error: ApplicationError, parent_widget: Optional[Any] = None) -> None:
        """Handle an error with appropriate UI feedback and logging"""
        # Log the error
        error.log()

        # Show UI feedback if we have a parent widget
        if parent_widget:
            if error.severity == ErrorSeverity.INFO:
                QMessageBox.information(
                    parent_widget,
                    "Information",
                    error.message + (f"\n\nHint: {error.recovery_hint}" if error.recovery_hint else "")
                )
            elif error.severity == ErrorSeverity.WARNING:
                QMessageBox.warning(
                    parent_widget,
                    "Warning",
                    error.message + (f"\n\nHint: {error.recovery_hint}" if error.recovery_hint else "")
                )
            elif error.severity == ErrorSeverity.ERROR:
                QMessageBox.critical(
                    parent_widget,
                    "Error",
                    f"{error.message}\n\n" +
                    (f"Details: {error.details}\n\n" if error.details else "") +
                    (f"Hint: {error.recovery_hint}" if error.recovery_hint else "")
                )
            else:  # CRITICAL
                QMessageBox.critical(
                    parent_widget,
                    "Critical Error",
                    f"{error.message}\n\n" +
                    (f"Details: {error.details}\n\n" if error.details else "") +
                    "The application may be in an unstable state. Please save your work if possible."
                )

    @staticmethod
    def create_and_handle(
            message: str,
            severity: ErrorSeverity,
            parent_widget: Optional[Any] = None,
            **kwargs
    ) -> None:
        """Convenience method to create and handle an error in one step"""
        error = ApplicationError(message=message, severity=severity, **kwargs)
        ErrorHandler.handle(error, parent_widget)


class ErrorHandlingMixin:
    """Mixin to add error handling capabilities to widgets"""

    def handle_error(self, error: ApplicationError) -> None:
        """Handle an error in the context of this widget"""
        ErrorHandler.handle(error, self)

        # Update widget state if applicable
        if hasattr(self, 'update_status'):
            self.update_status(f"Error: {error.message}")

        if hasattr(self, '_set_controls_enabled'):
            self._set_controls_enabled(True)

        # Emit error signal if available
        if hasattr(self, 'processing_failed'):
            self.processing_failed.emit(error.message)


    def create_error(
            self,
            message: str,
            severity: ErrorSeverity = ErrorSeverity.ERROR,
            **kwargs
    ) -> ApplicationError:
        """Create an error with this widget's context"""
        error_params = {
            'message': message,
            'severity': severity,
        }

        # Only add component if not already in kwargs
        if 'component' not in kwargs:
            error_params['component'] = self.__class__.__name__

        # Add any additional kwargs
        error_params.update(kwargs)

        return ApplicationError(**error_params)