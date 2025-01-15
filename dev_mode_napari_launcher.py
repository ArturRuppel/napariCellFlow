import sys
import logging
from pathlib import Path
import napari
from qtpy.QtWidgets import QMessageBox

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
###################################
# check https://napari.org/0.5.4/plugins/advanced_topics/npe2_migration_guide.html
###################################
def setup_debug_environment():
    """Setup the debug environment and verify imports"""
    try:
        # Add the parent directory to Python path
        current_dir = Path(__file__).parent.absolute()
        parent_dir = current_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
            logger.debug(f"Added {parent_dir} to Python path")

        # Import the widget
        try:
            from napariCellFlow._widget import napariCellFlow
            logger.debug("Successfully imported napariCellFlow")
            return napariCellFlow
        except ImportError as e:
            logger.error(f"Failed to import napariCellFlow: {e}")
            raise

    except Exception as e:
        logger.error(f"Error setting up debug environment: {e}")
        raise

def create_viewer_with_widget():
    """Create the napari viewer and add the widget"""
    try:
        # Create the viewer
        viewer = napari.Viewer()
        logger.debug("Created napari viewer")

        # Get the widget class
        napariCellFlow = setup_debug_environment()

        # Create and add the widget
        widget = napariCellFlow(viewer)
        dock_widget = viewer.window.add_dock_widget(
            widget,
            name="napariCellFlow",
            area='right'
        )
        logger.debug("Added widget to viewer")

        return viewer

    except Exception as e:
        logger.error(f"Error creating viewer with widget: {e}")
        raise

def main():
    """Main entry point with error handling"""
    try:
        logger.debug("Starting napari launcher")
        viewer = create_viewer_with_widget()
        napari.run()

    except Exception as e:
        logger.error(f"Fatal error in napari launcher: {e}")
        # Show error dialog
        QMessageBox.critical(None, "Fatal Error",
                           f"Error starting napari: {str(e)}\n\n"
                           f"Check the console for more details.")
        raise

if __name__ == "__main__":
    main()