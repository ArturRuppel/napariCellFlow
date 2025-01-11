"""
pytest configuration file for napariCellFlow.

This file contains test configuration and fixtures that are shared across
multiple test files in the project.
"""

import pytest


def pytest_configure(config):
    """Register custom markers to avoid warnings."""
    config.addinivalue_line(
        "markers",
        "gpu: mark tests that require GPU support"
    )


@pytest.fixture(autouse=True)
def _skip_gpu_tests(request):
    """Skip GPU tests if GPU is not available."""
    if request.node.get_closest_marker('gpu'):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip('GPU not available')
        except ImportError:
            from cellpose.core import use_gpu
            if not use_gpu():
                pytest.skip('GPU not available')