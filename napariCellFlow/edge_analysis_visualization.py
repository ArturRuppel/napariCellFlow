import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import imageio
import napari
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, hsv_to_rgb, Normalize
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from napariCellFlow.structure import VisualizationConfig, CellBoundary

logger = logging.getLogger(__name__)


class Visualizer:
    """Handles all visualization tasks for cell tracking and analysis"""

    def __init__(self, config: Optional[VisualizationConfig] = None, napari_viewer: Optional["napari.Viewer"] = None):
        self.config = config or VisualizationConfig()
        self.viewer = napari_viewer
        self.progress_callback = None
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'axes.edgecolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black'
        })
        logger.debug(f"Visualizer initialized with viewer: {napari_viewer is not None}")

    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback

    def _update_progress(self, stage: str, progress: float):
        """Update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(stage, progress)

    def create_visualizations(self, results: 'EdgeAnalysisResults', input_path: Path = None):
        """Create all visualizations with progress tracking"""
        total_stages = 0
        current_stage = 0

        # Count enabled visualizations
        if self.config.tracking_plots_enabled:
            total_stages += 2  # trajectories and animation
        if self.config.edge_detection_overlay:
            total_stages += 1
        if self.config.intercalation_events:
            total_stages += 1
        if self.config.edge_length_evolution:
            total_stages += 1
            if self.config.create_example_gifs:
                total_stages += 1

        if total_stages == 0:
            logger.info("No visualizations enabled")
            return

        try:
            # Get segmentation data
            segmentation_stack = self._get_segmentation_data()
            self._update_progress("Initializing", 0)

            # Create base directory
            base_dir = self.config.output_dir
            if not base_dir:
                raise ValueError("Output directory must be specified in VisualizationConfig.")
            base_dir.mkdir(parents=True, exist_ok=True)


            # Generate tracking visualizations
            if self.config.tracking_plots_enabled and segmentation_stack is not None:
                tracking_dir = base_dir
                tracking_dir.mkdir(exist_ok=True)

                # Cell trajectories plot
                self._update_progress("Generating cell trajectory plot", current_stage / total_stages * 100)
                trajectory_path = tracking_dir / "cell_trajectories.png"
                self.visualize_cell_tracks(segmentation_stack, trajectory_path)
                current_stage += 1

                # Tracking animation
                self._update_progress("Creating tracking animation", current_stage / total_stages * 100)
                animation_path = tracking_dir / "tracking_animation.gif"
                self.create_tracking_animation(segmentation_stack, animation_path)
                current_stage += 1

            # Edge detection visualization
            if self.config.edge_detection_overlay:
                self._update_progress("Generating edge detection overlays", current_stage / total_stages * 100)
                edge_dir = base_dir
                edge_dir.mkdir(exist_ok=True)

                boundaries_by_frame = self._extract_boundaries(results)
                if segmentation_stack is not None and boundaries_by_frame:
                    output_path = edge_dir / "edge_detection.gif"
                    self.visualize_boundaries(segmentation_stack, boundaries_by_frame, output_path)
                current_stage += 1

            # Intercalation events
            if self.config.intercalation_events:
                self._update_progress("Creating intercalation visualizations", current_stage / total_stages * 100)
                intercalation_dir = base_dir
                intercalation_dir.mkdir(exist_ok=True)

                all_events = self._collect_intercalation_events(results)
                if all_events:
                    boundaries_by_frame = self._extract_boundaries(results)
                    output_path = intercalation_dir / "intercalations.gif"
                    self.create_intercalation_animation(segmentation_stack, boundaries_by_frame, all_events, output_path)
                current_stage += 1

            # Edge length evolution
            if self.config.edge_length_evolution:
                self._update_progress("Generating edge length evolution plots", current_stage / total_stages * 100)
                edge_analysis_dir = base_dir
                edge_analysis_dir.mkdir(exist_ok=True)

                current_stage += 1

                # Example GIFs for edges with intercalations
                if self.config.create_example_gifs:
                    self._update_progress("Creating example visualizations", current_stage / total_stages * 100)
                    self._create_example_visualizations(results, edge_analysis_dir)
                    current_stage += 1

            self._update_progress("Completed", 100)

        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}")
            raise

    def _extract_boundaries(self, results: 'EdgeAnalysisResults') -> Dict[int, List['CellBoundary']]:
        """Extract cell boundaries from analysis results"""
        boundaries_by_frame = {}
        for edge_id, edge_data in results.edges.items():
            for frame_idx, frame in enumerate(edge_data.frames):
                if frame not in boundaries_by_frame:
                    boundaries_by_frame[frame] = []

                # Create CellBoundary object from edge data
                boundary = CellBoundary(
                    cell_ids=edge_data.cell_pairs[frame_idx],
                    coordinates=edge_data.coordinates[frame_idx],
                    endpoint1=edge_data.coordinates[frame_idx][0],
                    endpoint2=edge_data.coordinates[frame_idx][-1],
                    length=edge_data.lengths[frame_idx]
                )
                boundaries_by_frame[frame].append(boundary)

        return boundaries_by_frame

    def _collect_intercalation_events(self, results: 'EdgeAnalysisResults') -> List['IntercalationEvent']:
        """Collect all intercalation events from the results"""
        all_events = []
        for edge_data in results.edges.values():
            if hasattr(edge_data, 'intercalations'):
                all_events.extend(edge_data.intercalations)
        return all_events

    def create_tracking_animation(self, segmentation_stack: np.ndarray, output_path: Path) -> None:
        """Create an animation showing cell tracking over time."""
        logger.debug(f"Starting tracking animation with stack shape: {segmentation_stack.shape}")
        self._update_progress("Tracking Animation: Initializing...", 0)

        try:
            # Find all unique cell IDs across all frames
            all_cell_ids = set()
            for frame in segmentation_stack:
                all_cell_ids.update(np.unique(frame))
            cell_ids = sorted([cid for cid in all_cell_ids if cid != 0])  # Remove background
            logger.debug(f"Found {len(cell_ids)} unique cell IDs")

            # Generate colors for tracks
            colors = {
                int(cell_id): color
                for cell_id, color in zip(cell_ids, plt.cm.rainbow(np.linspace(0, 1, len(cell_ids))))
            }

            # Pre-calculate centroids
            self._update_progress("Tracking Animation: Calculating cell positions...", 5)
            all_centroids = {int(cell_id): [] for cell_id in cell_ids}
            for frame_idx, frame in enumerate(segmentation_stack):
                for cell_id in cell_ids:
                    mask = frame == cell_id
                    if np.any(mask):
                        y, x = np.where(mask)
                        if len(x) > 0 and len(y) > 0:
                            all_centroids[int(cell_id)].append((np.mean(x), np.mean(y)))
                        else:
                            all_centroids[int(cell_id)].append(None)
                    else:
                        all_centroids[int(cell_id)].append(None)

            # Process frames in batches
            BATCH_SIZE = 10
            frames = []
            total_frames = len(segmentation_stack)
            self._update_progress("Tracking Animation: Starting frame generation...", 10)

            for batch_start in range(0, total_frames, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_frames)

                # Create figure for this batch
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                fig.patch.set_facecolor('black')

                for frame_idx in range(batch_start, batch_end):
                    ax.clear()
                    ax.set_facecolor('black')
                    ax.imshow(segmentation_stack[frame_idx], cmap='gray', alpha=1)

                    for cell_id in cell_ids:
                        cell_id_int = int(cell_id)
                        try:
                            current_centroids = [c for c in all_centroids[cell_id_int][:frame_idx + 1] if c is not None]

                            if current_centroids:
                                track = np.array(current_centroids)
                                if len(track) > 0:
                                    color = colors[cell_id_int]
                                    ax.plot(track[:, 0], track[:, 1], '-',
                                            color=color, linewidth=self.config.line_width,
                                            alpha=self.config.alpha)
                                    ax.plot(track[:, 0], track[:, 1], 'o',
                                            color=color, markersize=3)
                                    ax.text(track[-1, 0], track[-1, 1], str(cell_id_int),
                                            color=color, fontsize=self.config.font_size,
                                            ha='left', va='bottom')
                        except (IndexError, ValueError) as e:
                            logger.debug(f"Skipping plot for cell {cell_id_int} due to error: {e}")
                            continue

                    ax.set_title(f'Frame {frame_idx}', color='white')
                    ax.axis('off')

                    # Convert plot to image
                    fig.canvas.draw()
                    frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(frame_image)

                    # Update progress once per frame
                    progress = 10 + (frame_idx / total_frames * 70)
                    self._update_progress(
                        f"Tracking Animation: Processing frame {frame_idx + 1}/{total_frames}",
                        progress
                    )

                # Clean up batch figures
                plt.close(fig)
                plt.close('all')

            # Save the animation
            self._update_progress("Tracking Animation: Saving...", 80)

            output_path = Path(output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fps = 1000 / self.config.animation_interval

            imageio.mimsave(
                str(output_path),
                frames,
                fps=fps,
                optimize=False,
                palettesize=256,
                loop=0
            )

            self._update_progress("Tracking Animation: Complete", 100)
            logger.debug("Tracking animation completed")

        except Exception as e:
            logger.error(f"Error in tracking animation: {str(e)}")
            raise
        finally:
            plt.close('all')

    def create_intercalation_animation(self, tracked_stack: np.ndarray,
                                       boundaries_by_frame: Dict[int, List['CellBoundary']],
                                       events: List['IntercalationEvent'],
                                       output_path: Path) -> None:
        """Create animation showing intercalation events with frame-by-frame visualization."""
        logger.debug("Starting intercalation animation")
        self._update_progress("Intercalation Animation: Initializing...", 0)

        try:
            if not events or tracked_stack is None:
                logger.warning("Missing required data for animation")
                return

            # Validate frame ranges
            max_frame = len(tracked_stack) - 1
            valid_events = [event for event in events
                            if hasattr(event, 'frame') and
                            event.frame >= 0 and
                            event.frame <= max_frame]

            if not valid_events:
                logger.warning("No valid events within frame range")
                return

            self._update_progress(f"Intercalation Animation: Found {len(valid_events)} valid events", 5)

            # Process frames in batches
            BATCH_SIZE = 10
            frames = []
            total_frames = len(tracked_stack)
            self._update_progress("Intercalation Animation: Starting frame generation", 10)

            for batch_start in range(0, total_frames, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_frames)

                # Create figure for this batch
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                fig.patch.set_facecolor('black')

                for frame_idx in range(batch_start, batch_end):
                    ax.clear()
                    ax.set_facecolor('black')
                    ax.imshow(tracked_stack[frame_idx], cmap='gray')

                    try:
                        # Plot events
                        for event in valid_events:
                            if frame_idx == event.frame and hasattr(event, 'losing_cells'):
                                losing_pair = set(int(x) for x in event.losing_cells)
                                if frame_idx in boundaries_by_frame:
                                    for boundary in boundaries_by_frame[frame_idx]:
                                        if set(int(x) for x in boundary.cell_ids) == losing_pair:
                                            coords = boundary.coordinates
                                            if len(coords) > 0:
                                                ax.plot(coords[:, 1], coords[:, 0],
                                                        color='red',
                                                        linewidth=self.config.line_width * 2)

                            if frame_idx > 0 and frame_idx - 1 == event.frame and hasattr(event, 'gaining_cells'):
                                gaining_pair = set(int(x) for x in event.gaining_cells)
                                if frame_idx in boundaries_by_frame:
                                    for boundary in boundaries_by_frame[frame_idx]:
                                        if set(int(x) for x in boundary.cell_ids) == gaining_pair:
                                            coords = boundary.coordinates
                                            if len(coords) > 0:
                                                ax.plot(coords[:, 1], coords[:, 0],
                                                        color='red',
                                                        linewidth=self.config.line_width * 2)

                        ax.set_title(f'Frame {frame_idx}', color='white')
                        ax.axis('off')

                        # Convert plot to image
                        fig.canvas.draw()
                        frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                        frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        frames.append(frame_image)

                        # Update progress once per frame
                        progress = 10 + (frame_idx / total_frames * 70)
                        self._update_progress(
                            f"Intercalation Animation: Processing frame {frame_idx + 1}/{total_frames}",
                            progress
                        )

                    except Exception as e:
                        logger.debug(f"Error processing frame {frame_idx}: {e}")

                # Clean up batch figures
                plt.close(fig)
                plt.close('all')

            # Save the animation
            self._update_progress("Intercalation Animation: Saving...", 80)

            output_path = Path(output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fps = 1000 / self.config.animation_interval

            imageio.mimsave(
                str(output_path),
                frames,
                fps=fps,
                optimize=False,
                palettesize=256,
                loop=0
            )

            self._update_progress("Intercalation Animation: Complete", 100)
            logger.debug("Intercalation animation completed")

        except Exception as e:
            logger.error(f"Error in intercalation animation: {str(e)}")
            raise
        finally:
            plt.close('all')

    def visualize_boundaries(self, segmented_stack: np.ndarray,
                             boundaries_by_frame: Dict[int, List['CellBoundary']],
                             output_path: Optional[Path] = None) -> None:
        """Create animation of detected boundaries across all frames."""
        logger.debug("Starting boundary visualization")
        self._update_progress("Edge Detection Animation: Initializing...", 0)

        try:
            if not boundaries_by_frame:
                logger.warning("No boundaries provided for visualization")
                return

            # Get actual frame range
            max_frame = max(boundaries_by_frame.keys())
            if max_frame >= len(segmented_stack):
                logger.warning(f"Frame mismatch: max boundary frame {max_frame} >= stack length {len(segmented_stack)}")
                max_frame = len(segmented_stack) - 1

            # Calculate color map
            self._update_progress("Edge Detection Animation: Analyzing cell pairs...", 5)
            all_cell_pairs = set()
            for boundaries in boundaries_by_frame.values():
                for boundary in boundaries:
                    all_cell_pairs.add(tuple(sorted(boundary.cell_ids)))

            color_map = {
                cell_pair: plt.cm.rainbow(i / max(1, len(all_cell_pairs) - 1))
                for i, cell_pair in enumerate(sorted(all_cell_pairs))
            }

            # Process frames in batches
            BATCH_SIZE = 10
            frames = []
            total_frames = min(len(segmented_stack), max_frame + 1)
            self._update_progress("Edge Detection Animation: Starting frame generation...", 10)

            for batch_start in range(0, total_frames, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_frames)

                # Create figure for this batch
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                fig.patch.set_facecolor('black')

                for frame_idx in range(batch_start, batch_end):
                    ax.clear()
                    ax.set_facecolor('black')
                    ax.imshow(segmented_stack[frame_idx], cmap='gray')

                    if frame_idx in boundaries_by_frame:
                        for boundary in boundaries_by_frame[frame_idx]:
                            try:
                                coords = boundary.coordinates
                                cell_pair = tuple(sorted(boundary.cell_ids))
                                color = color_map.get(cell_pair)

                                if color is not None and len(coords) > 0:
                                    ax.plot(coords[:, 1], coords[:, 0],
                                            c=color, linewidth=self.config.line_width,
                                            alpha=self.config.alpha)

                                    if len(coords) > len(coords) // 2:
                                        mid_point = coords[len(coords) // 2]
                                        ax.text(mid_point[1], mid_point[0],
                                                f'{boundary.cell_ids[0]}-{boundary.cell_ids[1]}',
                                                color=color, fontsize=self.config.font_size)

                            except (IndexError, AttributeError) as e:
                                logger.debug(f"Error plotting boundary in frame {frame_idx}: {e}")
                                continue

                    ax.set_title(f'Frame {frame_idx}', color='white')
                    ax.axis('off')

                    # Convert plot to image
                    fig.canvas.draw()
                    frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(frame_image)

                    # Update progress once per frame
                    progress = 10 + (frame_idx / total_frames * 70)
                    self._update_progress(
                        f"Edge Detection Animation: Processing frame {frame_idx + 1}/{total_frames}",
                        progress
                    )

                # Clean up batch figures
                plt.close(fig)
                plt.close('all')

            # Save the animation
            if output_path:
                self._update_progress("Edge Detection Animation: Saving...", 80)

                output_path = Path(output_path).resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fps = 1000 / self.config.animation_interval

                imageio.mimsave(
                    str(output_path),
                    frames,
                    fps=fps,
                    optimize=False,
                    palettesize=256,
                    loop=0
                )

            self._update_progress("Edge Detection Animation: Complete", 100)
            logger.debug("Boundary visualization completed")

        except Exception as e:
            logger.error(f"Error in boundary visualization: {str(e)}")
            raise
        finally:
            plt.close('all')

    def visualize_cell_tracks(self, segmentation_stack: np.ndarray, output_path: Path) -> None:
        """Create a visualization showing cell trajectories across all frames."""
        logger.debug(f"Starting cell track visualization with stack shape: {segmentation_stack.shape}")
        self._update_progress("Cell Tracks Visualization: Initializing...", 0)

        # Find unique cell IDs
        cell_ids = np.unique(segmentation_stack)
        cell_ids = cell_ids[cell_ids != 0]  # Remove background
        self._update_progress(f"Cell Tracks Visualization: Found {len(cell_ids)} cells to track", 10)

        # Set up the figure
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Show last frame as background
        plt.imshow(segmentation_stack[-1], cmap='gray')

        # Generate colors for tracks
        colors = plt.cm.rainbow(np.linspace(0, 1, len(cell_ids)))

        # Process cell trajectories
        self._update_progress("Cell Tracks Visualization: Starting trajectory calculation", 20)

        total_cells = len(cell_ids)
        for cell_idx, (cell_id, color) in enumerate(zip(cell_ids, colors)):
            centroids = []
            for frame in segmentation_stack:
                mask = frame == cell_id
                if np.any(mask):
                    y, x = np.where(mask)
                    centroids.append((np.mean(x), np.mean(y)))

            if centroids:
                track = np.array(centroids)
                plt.plot(track[:, 0], track[:, 1], '-',
                         color=color, linewidth=self.config.line_width,
                         alpha=self.config.alpha)
                plt.plot(track[:, 0], track[:, 1], 'o',
                         color=color, markersize=3)
                plt.text(track[-1, 0], track[-1, 1], str(cell_id),
                         color=color, fontsize=self.config.font_size,
                         ha='left', va='bottom')

            # Update progress for each cell
            progress = 20 + (cell_idx / total_cells * 60)
            self._update_progress(
                f"Cell Tracks Visualization: Processing cell {cell_idx + 1}/{total_cells}",
                progress
            )

        plt.title('Cell Trajectories', color='white')
        plt.axis('off')

        # Save the visualization
        self._update_progress("Cell Tracks Visualization: Saving...", 80)

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        plt.close()

        self._update_progress("Cell Tracks Visualization: Complete", 100)
        logger.debug("Cell track visualization completed")

    def create_edge_evolution_animation(self, segmentation_stack: np.ndarray,
                                        trajectory: 'EdgeTrajectory',
                                        boundaries_by_frame: Dict[int, List['CellBoundary']],
                                        output_path: Path,
                                        example_num: int,
                                        total_examples: int) -> None:
        """Create animation showing edge evolution and length plot with memory optimization."""
        logger.debug("Starting edge evolution animation")
        self._update_progress("Edge Evolution Animation: Initializing...", 0)

        if segmentation_stack is None or len(segmentation_stack) == 0:
            logger.warning("No segmentation data provided")
            return

        # Validate trajectory data
        if not hasattr(trajectory, 'frames') or not trajectory.frames:
            logger.warning("No frame data in trajectory")
            return

        try:
            # Ensure frames are within bounds
            max_frame = len(segmentation_stack) - 1
            valid_frames = [f for f in trajectory.frames if 0 <= f <= max_frame]

            if not valid_frames:
                logger.warning("No valid frames in trajectory")
                return

            self._update_progress("Edge Evolution Animation: Validating trajectory data...", 5)

            # Pre-calculate valid frame data
            valid_frame_data = {}
            if hasattr(trajectory, 'lengths') and hasattr(trajectory, 'frames'):
                valid_indices = [i for i, f in enumerate(trajectory.frames)
                                 if 0 <= f < len(segmentation_stack)]
                if valid_indices:
                    valid_frame_data['frames'] = [trajectory.frames[i] for i in valid_indices]
                    valid_frame_data['lengths'] = [trajectory.lengths[i] for i in valid_indices]

            self._update_progress("Edge Evolution Animation: Starting frame generation...", 10)

            # Process frames in batches to manage memory
            BATCH_SIZE = 10
            frames = []
            total_frames = len(segmentation_stack)

            for batch_start in range(0, total_frames, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_frames)

                # Create figure for this batch
                plt.style.use('default')
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                fig.patch.set_facecolor('white')

                for frame_idx in range(batch_start, batch_end):
                    # Clear axes
                    ax1.clear()
                    ax2.clear()

                    for ax in (ax1, ax2):
                        ax.set_facecolor('white')
                        ax.spines['bottom'].set_color('black')
                        ax.spines['top'].set_color('black')
                        ax.spines['left'].set_color('black')
                        ax.spines['right'].set_color('black')
                        ax.tick_params(colors='black')

                    # Plot segmentation
                    ax1.imshow(segmentation_stack[frame_idx], cmap='gray')

                    # Plot edge if present
                    if frame_idx in trajectory.frames:
                        frame_data_idx = trajectory.frames.index(frame_idx)
                        if hasattr(trajectory, 'cell_pairs') and frame_data_idx < len(trajectory.cell_pairs):
                            cell_pair = trajectory.cell_pairs[frame_data_idx]
                            if frame_idx in boundaries_by_frame:
                                for boundary in boundaries_by_frame[frame_idx]:
                                    if (tuple(sorted(int(x) for x in boundary.cell_ids)) ==
                                            tuple(sorted(int(x) for x in cell_pair))):
                                        coords = boundary.coordinates
                                        if len(coords) > 0:
                                            ax1.plot(coords[:, 1], coords[:, 0], 'r-',
                                                     linewidth=self.config.line_width)
                                        break

                    ax1.set_title(f'Frame {frame_idx}', color='black')
                    ax1.axis('off')

                    # Plot length trajectory
                    if valid_frame_data:
                        ax2.plot(valid_frame_data['frames'], valid_frame_data['lengths'], 'b-')
                        if frame_idx in valid_frame_data['frames']:
                            idx = valid_frame_data['frames'].index(frame_idx)
                            ax2.plot(frame_idx, valid_frame_data['lengths'][idx], 'ro')

                    # Add intercalation markers
                    if hasattr(trajectory, 'intercalation_frames'):
                        valid_intercalations = [f for f in trajectory.intercalation_frames
                                                if 0 <= f < len(segmentation_stack)]
                        for f in valid_intercalations:
                            ax2.axvline(x=f, color='r', linestyle='--', alpha=0.5)

                    ax2.set_xlabel('Frame', color='black')
                    ax2.set_ylabel('Edge Length (µm)', color='black')
                    ax2.tick_params(colors='black')
                    ax2.grid(True, alpha=0.3)

                    # Convert plot to image
                    fig.canvas.draw()
                    frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(frame_image)

                    # Update progress once per frame
                    progress = 10 + (frame_idx / total_frames * 70)
                    self._update_progress(
                        f"Edge Evolution Animation (Example {example_num + 1}/{total_examples}): Processing frame {frame_idx + 1}/{total_frames}",
                        progress
                    )

                # Clean up the batch's figure
                plt.close(fig)
                plt.close('all')

            # Save the animation
            self._update_progress("Edge Evolution Animation: Saving...", 80)

            output_path = Path(output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fps = 1000 / self.config.animation_interval

            imageio.mimsave(
                str(output_path),
                frames,
                fps=fps,
                optimize=False,
                palettesize=256,
                loop=0
            )

            self._update_progress("Edge Evolution Animation: Complete", 100)
            logger.debug("Edge evolution animation completed")

        except Exception as e:
            logger.error(f"Error in edge evolution animation: {str(e)}")
            raise
        finally:
            # Ensure all figures are closed
            plt.close('all')

    def _create_example_visualizations(self, results: 'EdgeAnalysisResults', output_dir: Path) -> None:
        """Create example visualizations for edges with intercalations"""
        example_dir = output_dir
        example_dir.mkdir(exist_ok=True)

        # Find edges with intercalations
        edges_with_intercalations = {
            edge_id: edge_data
            for edge_id, edge_data in results.edges.items()
            if hasattr(edge_data, 'intercalations') and edge_data.intercalations
        }

        total_examples = min(len(edges_with_intercalations), self.config.max_example_gifs)
        if total_examples == 0:
            logger.info("No edges with intercalations found for example visualizations")
            return

        # Get segmentation data once
        segmentation_stack = self._get_segmentation_data()
        if segmentation_stack is None:
            logger.warning("No segmentation data available for example visualizations")
            return

        # Extract boundaries once
        boundaries_by_frame = self._extract_boundaries(results)

        # Process each example
        for i, (edge_id, edge_data) in enumerate(edges_with_intercalations.items()):
            if i >= self.config.max_example_gifs:
                break

            try:
                # Create animation
                animation_path = example_dir / f"edge_{edge_id}_evolution.gif"
                self.create_edge_evolution_animation(
                    segmentation_stack,
                    edge_data,
                    boundaries_by_frame,
                    animation_path,
                    i,
                    total_examples
                )
            except Exception as e:
                logger.error(f"Failed to create example visualization for edge {edge_id}: {str(e)}")
                continue
    def _get_segmentation_data(self) -> Optional[np.ndarray]:
        """Get segmentation data from Napari layers"""
        if self.viewer is None:
            logger.debug("No viewer available")
            return None

        logger.debug(f"Available layers: {[layer.name for layer in self.viewer.layers]}")
        for layer in self.viewer.layers:
            logger.debug(f"Checking layer: {layer.name}, type: {type(layer)}")
            if layer.name == "Segmentation" and isinstance(layer, napari.layers.Labels):
                logger.debug(f"Found segmentation layer with shape: {layer.data.shape}")
                return layer.data

        logger.debug("No segmentation layer found")
        return None