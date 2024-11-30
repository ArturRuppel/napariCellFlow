from setuptools import setup, find_packages

setup(
    name="napari-cellpose-stackmode",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "napari",
        "numpy",
        "qtpy",
        "scikit-image",
        "tifffile",
        "pyyaml",
    ],
    entry_points={
        "napari.plugin": [
            "napari-cellpose-stackmode = napari_cellpose_stackmode.plugin:napari_experimental_provide_dock_widget",
        ],
    },
    description="A napari plugin for running Cellpose model on image stacks",
    author="aruppel",
    license="MIT",
    url="https://github.com/ArturRuppel/napari_cellpose_stackmode",
)