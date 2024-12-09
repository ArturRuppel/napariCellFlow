from dataclasses import dataclass
from typing import Optional
from qtpy.QtCore import QObject, Signal

@dataclass
class ProcessingError:
    """Standardized error information"""
    message: str
    details: Optional[str] = None
    component: Optional[str] = None
    recoverable: bool = True

class ErrorSignals(QObject):
    """Centralized error signals"""
    processing_error = Signal(ProcessingError)  # For recoverable processing errors
    critical_error = Signal(ProcessingError)    # For non-recoverable errors
    warning = Signal(str)                       # For warnings/notifications