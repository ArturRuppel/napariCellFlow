from setuptools import setup, find_packages

setup(
    name="napariCellFlow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "cellpose>=3.1.0",
        "tifffile>=2024.8.30",
        "napari>=0.5.4",
        "qtpy>=2.4.2",
        "PyYAML>=6.0.2",
        "scikit-image>=0.24.0",
        "networkx>=3.2.1",
        "scipy>=1.13.1",
        "matplotlib>=3.9.3",
        "tqdm>=4.67.1",
        "qtrangeslider>=0.1.5",
        "opencv-python>=4.9.0",
        "imageio>=2.34.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-qt>=4.4.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
        ]
    },
    entry_points={
        "napari.manifest": [
            "napariCellFlow = napariCellFlow",
        ],
    },
    python_requires=">=3.9",
    author="Artur Ruppel, Claude AI",
    author_email="",
    description="A napari plugin for cell segmentation, tracking and analysis of cell-cell contacts in microscopy data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Framework :: napari",
    ],
)