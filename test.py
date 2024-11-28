import os
import tifffile
from skimage.transform import resize
import numpy as np

base_dir = r"D:\2024-11-27"

# Loop through position folders
for i in range(23):  # 0 to 22 inclusive
    position_folder = f"position{i}"
    folder_path = os.path.join(base_dir, position_folder)

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Skipping {folder_path} - folder not found")
        continue

    input_file = os.path.join(folder_path, "registered_membrane_slice.tif")
    output_file = os.path.join(folder_path, "registered_membrane_slice_downsized.tif")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Skipping {input_file} - file not found")
        continue

    try:
        # Read the TIF file
        img = tifffile.imread(input_file)

        # Get dimensions
        t, h, w = img.shape

        # Calculate crop coordinates for center 2000x2000
        start_y = h // 2 - 1000
        start_x = w // 2 - 1000

        # Crop the image while keeping time dimension
        cropped = img[:, start_y:start_y + 2000, start_x:start_x + 2000]

        # Resize to t,500,500
        resized = np.zeros((t, 500, 500), dtype=img.dtype)
        for time_point in range(t):
            resized[time_point] = resize(cropped[time_point], (500, 500),
                                         preserve_range=True).astype(img.dtype)

        # Save the resized image
        tifffile.imwrite(output_file, resized)
        print(f"Processed {position_folder}")

    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

print("Processing complete!")