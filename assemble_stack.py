import os

import numpy as np
from pathlib import Path
import tifffile
import logging
import re


def process_segmentation_frames(input_dir: str, output_path: str) -> None:
    """
    Load .npy files from segmentation frames directory, extract masks, and save as a stack.

    Args:
        input_dir (str): Directory containing segmentation frame .npy files
        output_path (str): Path to save the output tiff stack
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Convert paths to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_path)

    # Find all segmentation .npy files
    npy_files = list(input_path.glob("*_seg.npy"))
    if not npy_files:
        raise ValueError(f"No segmentation .npy files found in {input_dir}")

    # Sort files numerically by frame number
    npy_files.sort(key=lambda x: int(re.search(r'img_(\d+)_seg\.npy', x.name).group(1)))

    logger.info(f"Found {len(npy_files)} segmentation files")

    # Load first file to get dimensions
    first_data = np.load(npy_files[0], allow_pickle=True).item()
    mask_shape = first_data['masks'].shape

    # Initialize stack
    mask_stack = np.zeros((len(npy_files), *mask_shape), dtype=np.uint16)

    # Process each file
    for i, file_path in enumerate(npy_files):
        try:
            # Load the .npy file
            seg_data = np.load(file_path, allow_pickle=True).item()

            # Extract masks
            masks = seg_data['masks']

            # Add to stack
            mask_stack[i] = masks

            logger.info(f"Processed frame {i + 1}/{len(npy_files)}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    # Save the stack
    logger.info(f"Saving stack to {output_path}")
    tifffile.imwrite(output_path, mask_stack)
    logger.info("Processing complete")


if __name__ == "__main__":
    # Example usage
    base_dir = "D:/2024-11-27/position14/"

    input_directory = os.path.join(base_dir, "cellpose_export")
    output_file = os.path.join(base_dir, "segmentation.tif")

    process_segmentation_frames(input_directory, output_file)