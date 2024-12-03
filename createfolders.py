import os


def create_position_folders():
    # List of subfolder names to create in each position folder
    subfolders = ['raw_data', 'registered_data', 'edge_analysis_data', "edge_analysis_visualizations"]

    # Iterate through position folders (0 to 21)
    for position in range(22):  # range(22) gives us 0-21
        # Create the main position folder name
        base_dir = "D:/2024-11-27/"
        position_folder = base_dir + f'position{position}'

        # Create the position folder if it doesn't exist
        if not os.path.exists(position_folder):
            os.makedirs(position_folder)

        # Create each subfolder within the position folder
        for subfolder in subfolders:
            subfolder_path = os.path.join(position_folder, subfolder)

            # Create the subfolder if it doesn't exist
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                print(f'Created: {subfolder_path}')
            else:
                print(f'Already exists: {subfolder_path}')


if __name__ == '__main__':
    create_position_folders()