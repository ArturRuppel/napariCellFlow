# Cell Segmentation Tool Specification
## Project Overview
Development of a tool for automated cell segmentation and tracking in fluorescence microscopy images, with a focus on handling 3D movement and temporal consistency. The tool will use a two-pass approach with Cellpose and allow for manual corrections.
## Goals
- Process image stacks (60 frames) in under 30 minutes
- Improve temporal consistency in segmentation
- Allow for efficient manual correction
- Enable generation of training data for model improvement
## Development Phases
### Phase 1: Script-Based Prototype
#### Core Features
1. Two-pass Cellpose segmentation:
   - Conservative initial segmentation
   - Refined segmentation using temporal information
2. Basic preprocessing pipeline:
   - Denoising options
   - Contrast enhancement
   - Parameter adjustment capabilities
3. Integration with Cellpose GUI for manual corrections
4. Cell tracking after corrections
5. Basic save/load functionality for intermediate results
#### Implementation Approach
- Command-line interface for rapid development
- Step-by-step pipeline with intervention points
- Use of Cellpose's native format for training data
- Basic visualization between steps
#### Optional Tools
- Parameter testing script:
  - Test different preprocessing parameters
  - Test different Cellpose parameters
  - Visual comparison of results
### Phase 2: Napari Integration
#### Features
1. Convert successful elements from Phase 1 into napari plugin
2. Enhanced visualization:
   - Configurable temporal context viewing
   - Side-by-side comparison views
3. Advanced features:
   - Undo/redo functionality
   - Real-time tracking updates
   - Batch processing capabilities
4. Custom segmentation correction widget (replacing Cellpose GUI)
## Technical Considerations
### Data Handling
- Input: Multiple movies (22) with 60 frames each
- Format: Compatible with standard microscopy formats
- Memory management for larger datasets
- Intermediate result storage
### Preprocessing Options
- Median filtering
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Local contrast normalization
- Cellpose 3.0 built-in denoising
### Cellpose Parameters
#### Conservative Pass
- Higher flow_threshold (≈0.5)
- More selective cellprob_threshold
- Adjusted min_size
- omni=True, cluster=True
#### Refinement Pass
- Relaxed flow_threshold
- Standard cellprob_threshold
- Temporal consistency constraints
### Integration Points
- Cellpose GUI for initial manual corrections
- Existing tracking code integration
- Future napari plugin ecosystem
## Future Enhancements
1. Automated parameter optimization
2. Advanced temporal consistency checking
3. Real-time tracking feedback during corrections
4. Custom correction interface in napari
5. Batch processing improvements
6. Extended training data export options
## Success Criteria
1. Processing time under 30 minutes per stack
2. Improved temporal consistency compared to frame-by-frame processing
3. Efficient manual correction workflow
4. Successful generation of training data for model improvement
## Development Priorities
1. Core segmentation pipeline
2. Basic manual correction workflow
3. Parameter testing capabilities
4. Data export functionality
5. Napari integration
6. Advanced features and optimizations