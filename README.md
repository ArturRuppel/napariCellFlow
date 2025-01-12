# napariCellFlow User Guide
Welcome to napariCellFlow! This tool helps you analyze cellular dynamics in microscopy data through an intuitive, step-by-step workflow. By combining Cellpose's powerful cell segmentation with dedicated tracking and edge analysis capabilities, napariCellFlow makes it straightforward to study cell behavior, track movements, and identify intercalation events in your time-lapse images. As a napari plugin, it provides interactive visualizations and immediate feedback at each step of your analysis pipeline, from initial preprocessing through to final results.

## Installation
[Installation instructions to be added]

## Overview

napariCellFlow is a comprehensive tool for analyzing cell dynamics in microscopy data. It provides a complete pipeline for:

- Image preprocessing and enhancement
- Cell segmentation using Cellpose
- Cell tracking across time frames
- Cell boundary and edge analysis
- Detection and analysis of cell intercalation events

The tool is integrated into napari as a plugin, providing an intuitive interface for visualizing and analyzing your data.

## Interface Layout

napariCellFlow adds a dockable widget to napari's right sidebar with four main tabs:
- Preprocessing
- Segmentation
- Cell Tracking
- Edge Analysis

Each tab provides specialized tools while maintaining a consistent workflow from raw data to final analysis.

## Preprocessing Widget

### Purpose
The preprocessing widget helps prepare your microscopy data for analysis by enhancing image quality and normalizing intensity values.

### Features
- **Intensity Range Adjustment**: Set minimum and maximum intensity values with an interactive dual slider
- **Filter Options**:
  - Median Filter: Remove noise while preserving edges
  - Gaussian Filter: General smoothing
  - CLAHE (Contrast Limited Adaptive Histogram Equalization): Enhance local contrast
- **Live Preview**: Toggle real-time preview of preprocessing effects

### Usage Tips
1. Load your image data in napari
2. Adjust intensity range using the slider to optimize contrast
3. Enable and configure filters as needed:
   - Start with minimal filtering and gradually increase if needed
   - Use Median filter for salt-and-pepper noise
   - Use Gaussian filter for general smoothing
   - Use CLAHE for improving contrast in low-contrast regions
4. Use the preview feature to check results before processing
5. Click "Run Preprocessing" to apply settings to entire dataset

## Segmentation Widget

### Purpose
The segmentation widget interfaces with Cellpose to identify and label individual cells in your images.

### Features
- **Model Selection**:
  - Cyto3: Optimized for cytoplasm detection
  - Nuclei: Specialized for nucleus detection
  - Custom: Load your own trained models
- **Parameter Controls**:
  - Cell Diameter: Set expected cell size
  - Flow Threshold: Control cell separation sensitivity
  - Cell Probability: Adjust detection confidence threshold
  - Minimum Size: Filter out small objects
- **Advanced Options**:
  - GPU acceleration toggle
  - Image normalization toggle
  - Automatic diameter computation
- **Visual Aids**:
  - Scale disk overlay for size reference
  - Manual correction tools for fixing segmentation errors
- **Integration**:
  - Export to Cellpose GUI for manual editing or model training
  - Import processed results back

### Usage Tips
1. Start with the Cyto3 model for general cell detection
2. Set cell diameter:
   - Use the scale disk overlay to verify size
   - Enable auto-computation for varying sizes
3. Adjust flow threshold:
   - Increase to separate touching cells
   - Decrease if cells are over-split
4. Use the probability threshold to balance detection:
   - Higher values for more confident detection
   - Lower values to detect more potential cells
5. Enable GPU acceleration if available for faster processing
6. Use manual correction tools to fix any errors

## Cell Tracking Widget

### Purpose
The tracking widget connects cell identities across time frames to analyze cell movement and behavior.

### Features
- **Tracking Parameters**:
  - Minimum Overlap Ratio: Control cell matching between frames
  - Maximum Displacement: Limit cell movement distance
  - Minimum Cell Size: Filter tracking targets
  - Gap Closing: Handle temporary cell disappearance
- **Process Controls**:
  - Single frame or full stack processing
  - Progress monitoring
  - Results visualization
- **Track Visualization**:
  - Color-coded cell tracks
  - Cell ID labels
  - Motion trajectories

### Usage Tips
1. Adjust overlap ratio based on cell movement speed:
   - Higher values for slow-moving cells
   - Lower values for fast-moving cells
2. Set maximum displacement based on expected cell speed
3. Enable gap closing if cells temporarily disappear
4. Start with a few frames to verify settings
5. Run full stack processing once settings are optimized

## Edge Analysis Widget

### Purpose
The edge analysis widget detects and analyzes cell boundaries and intercalation events.

### Features
- **Edge Detection**:
  - Boundary detection between cells
  - Length measurements
  - Contact time analysis
- **Intercalation Analysis**:
  - T1 transition detection
  - Neighbor exchange tracking
  - Event timeline visualization
- **Visualization Options**:
  - Edge overlay display
  - Intercalation event highlighting
  - Time-series analysis plots
- **Data Export**:
  - Save analysis results
  - Export visualizations
  - Statistical summaries

### Usage Tips
1. Configure edge detection parameters:
   - Adjust dilation radius for boundary detection
   - Set minimum overlap for cell contacts
2. Enable edge visualization to verify detection
3. Run analysis on full dataset
4. Review intercalation events:
   - Use timeline view to track changes
   - Verify detected events
5. Export results for further analysis

## Best Practices

### General Workflow
1. Start with preprocessing to optimize image quality
2. Verify segmentation results before tracking
3. Adjust tracking parameters with a subset of data first
4. Run edge analysis after confirming tracking accuracy
5. Save intermediate results regularly

### Performance Tips
- Enable GPU acceleration when available
- Process smaller datasets first to verify settings
- Use appropriate cell size filters to reduce noise
- Save projects regularly
- Export results for backup

### Troubleshooting
- If segmentation fails, try adjusting cell diameter
- For tracking errors, check maximum displacement
- If edge detection misses boundaries, adjust dilation
- Use manual correction tools for persistent errors
- Check GPU memory if processing large datasets

## Getting Help
- Check the [GitHub repository](https://github.com/ArturRuppel/napariCellFlow) for updates
- Report issues through the GitHub issue tracker
- Consult the API documentation for advanced usage
- Join the napari community for general support