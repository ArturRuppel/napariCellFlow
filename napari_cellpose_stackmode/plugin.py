from napari_plugin_engine import napari_hook_implementation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    """Provide TissueDynamicsWidget dock widget to napari."""
    from ._widget import TissueDynamicsWidget
    return TissueDynamicsWidget

# Optional: Provide a description that will show up in the plugins menu
@napari_hook_implementation
def napari_get_reader(path):
    """Optional reader implementation"""
    return None