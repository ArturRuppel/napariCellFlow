import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import napari
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
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
            base_dir = (self.config.output_dir if self.config.output_dir
                        else (input_path.parent / input_path.stem / "visualizations" if input_path
                              else Path.cwd() / "visualizations"))
            base_dir.mkdir(parents=True, exist_ok=True)

            # Generate tracking visualizations
            if self.config.tracking_plots_enabled and segmentation_stack is not None:
                tracking_dir = base_dir / "tracking_output"
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
                edge_dir = base_dir / "edge_output"
                edge_dir.mkdir(exist_ok=True)

                boundaries_by_frame = self._extract_boundaries(results)
                if segmentation_stack is not None and boundaries_by_frame:
                    output_path = edge_dir / "edge_detection.gif"
                    self.visualize_boundaries(segmentation_stack, boundaries_by_frame, output_path)
                current_stage += 1

            # Intercalation events
            if self.config.intercalation_events:
                self._update_progress("Creating intercalation visualizations", current_stage / total_stages * 100)
                intercalation_dir = base_dir / "intercalation_output"
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
                edge_analysis_dir = base_dir / "edge_analysis_output"
                edge_analysis_dir.mkdir(exist_ok=True)

                # Main length evolution plot
                self.plot_edge_length_tracks(results.edges, edge_analysis_dir / "all_edges_length_evolution.png")
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

    def _create_example_visualizations(self, results: 'EdgeAnalysisResults', output_dir: Path) -> None:
        """Create example visualizations for edges with intercalations"""
        example_dir = output_dir / "examples"
        example_dir.mkdir(exist_ok=True)

        # Find edges with intercalations
        edges_with_intercalations = {
            edge_id: edge_data
            for edge_id, edge_data in results.edges.items()
            if hasattr(edge_data, 'intercalations') and edge_data.intercalations
        }

        # Create visualizations for the specified number of examples
        for i, (edge_id, edge_data) in enumerate(edges_with_intercalations.items()):
            if i >= self.config.max_example_gifs:
                break

            output_path = example_dir / f"edge_{edge_id}_analysis.png"
            self.plot_edge_analysis(edge_data, output_path)

            # If segmentation data is available, create animation
            segmentation_stack = self._get_segmentation_data()
            if segmentation_stack is not None:
                boundaries_by_frame = self._extract_boundaries(results)
                animation_path = example_dir / f"edge_{edge_id}_evolution.gif"
                self.create_edge_evolution_animation(
                    segmentation_stack,
                    edge_data,
                    boundaries_by_frame,
                    animation_path
                )

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

    def visualize_cell_tracks(self, segmentation_stack: np.ndarray, output_path: Path) -> None:
        """Create a visualization showing cell trajectories across all frames."""
        logger.debug(f"Starting cell track visualization with stack shape: {segmentation_stack.shape}")

        cell_ids = np.unique(segmentation_stack)
        cell_ids = cell_ids[cell_ids != 0]  # Remove background
        logger.debug(f"Found {len(cell_ids)} unique cell IDs: {cell_ids}")

        # Set up the figure
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Show last frame as background
        logger.debug("Plotting last frame")
        plt.imshow(segmentation_stack[-1], cmap='gray')

        # Generate colors for tracks
        colors = plt.cm.rainbow(np.linspace(0, 1, len(cell_ids)))

        logger.debug("Processing cell trajectories")
        for cell_id, color in zip(cell_ids, colors):
            centroids = []
            for frame in segmentation_stack:
                mask = frame == cell_id
                if np.any(mask):
                    y, x = np.where(mask)
                    centroids.append((np.mean(x), np.mean(y)))

            if centroids:
                track = np.array(centroids)
                logger.debug(f"Plotting trajectory for cell {cell_id} with {len(centroids)} points")
                plt.plot(track[:, 0], track[:, 1], '-',
                         color=color, linewidth=self.config.line_width,
                         alpha=self.config.alpha)
                plt.plot(track[:, 0], track[:, 1], 'o',
                         color=color, markersize=3)
                plt.text(track[-1, 0], track[-1, 1], str(cell_id),
                         color=color, fontsize=self.config.font_size,
                         ha='left', va='bottom')

        plt.title('Cell Trajectories', color='white')
        plt.axis('off')

        logger.debug(f"Saving figure to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        plt.close()
        logger.debug("Cell track visualization completed")

    def create_tracking_animation(self, segmentation_stack: np.ndarray, output_path: Path) -> None:
        """Create an animation showing cell tracking over time."""
        logger.debug(f"Starting tracking animation with stack shape: {segmentation_stack.shape}")

        # Find all unique cell IDs across all frames
        all_cell_ids = set()
        for frame in segmentation_stack:
            all_cell_ids.update(np.unique(frame))
        cell_ids = sorted([cid for cid in all_cell_ids if cid != 0])  # Remove background
        logger.debug(f"Found {len(cell_ids)} unique cell IDs: {cell_ids}")

        # Generate colors for tracks
        colors = {
            int(cell_id): color
            for cell_id, color in zip(cell_ids, plt.cm.rainbow(np.linspace(0, 1, len(cell_ids))))
        }

        # Pre-calculate all centroids with safety checks
        logger.debug("Pre-calculating cell centroids")
        all_centroids = {int(cell_id): [] for cell_id in cell_ids}
        for frame in segmentation_stack:
            for cell_id in cell_ids:
                mask = frame == cell_id
                if np.any(mask):
                    y, x = np.where(mask)
                    if len(x) > 0 and len(y) > 0:  # Additional safety check
                        all_centroids[int(cell_id)].append((np.mean(x), np.mean(y)))
                    else:
                        all_centroids[int(cell_id)].append(None)
                else:
                    all_centroids[int(cell_id)].append(None)

        # Set up the figure
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        def update(frame_idx):
            ax.clear()
            ax.set_facecolor('black')
            ax.imshow(segmentation_stack[frame_idx], cmap='gray', alpha=1)

            for cell_id in cell_ids:
                cell_id_int = int(cell_id)
                # Get valid centroids up to current frame
                current_centroids = [c for c in all_centroids[cell_id_int][:frame_idx + 1] if c is not None]

                if current_centroids:  # Only plot if we have valid centroids
                    try:
                        track = np.array(current_centroids)
                        if len(track) > 0:  # Additional check before plotting
                            color = colors[cell_id_int]
                            ax.plot(track[:, 0], track[:, 1], '-',
                                    color=color, linewidth=self.config.line_width,
                                    alpha=self.config.alpha)
                            ax.plot(track[:, 0], track[:, 1], 'o',
                                    color=color, markersize=3)
                            # Only add label if we have a valid last position
                            ax.text(track[-1, 0], track[-1, 1], str(cell_id_int),
                                    color=color, fontsize=self.config.font_size,
                                    ha='left', va='bottom')
                    except (IndexError, ValueError) as e:
                        logger.debug(f"Skipping plot for cell {cell_id_int} due to error: {e}")
                        continue

            ax.set_title(f'Frame {frame_idx}', color='white')
            ax.axis('off')

        logger.debug("Creating animation")
        anim = FuncAnimation(fig, update, frames=len(segmentation_stack),
                             interval=self.config.animation_interval, blit=False)

        logger.debug(f"Saving animation to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(output_path), writer='pillow', savefig_kwargs={'facecolor': 'black'})
        plt.close()
        logger.debug("Tracking animation completed")
    def plot_intercalation_event(self, event: 'IntercalationEvent', output_path: Path):
        """Create a static visualization of an intercalation event"""
        plt.figure(figsize=self.config.figure_size)

        # Plot the event coordinates
        plt.scatter(event.coordinates[1], event.coordinates[0], c='red', s=100)

        # Add labels for the cells involved
        plt.annotate(f"Losing cells: {event.losing_cells}",
                     (event.coordinates[1], event.coordinates[0]),
                     xytext=(10, 10), textcoords='offset points')
        plt.annotate(f"Gaining cells: {event.gaining_cells}",
                     (event.coordinates[1], event.coordinates[0]),
                     xytext=(10, -10), textcoords='offset points')

        plt.title(f"Intercalation Event at Frame {event.frame}")
        plt.axis('equal')
        plt.savefig(output_path)
        plt.close()

    def plot_edge_analysis(self, edge_data: 'EdgeData', output_path: Path):
        """Create a comprehensive visualization of edge evolution"""
        plt.figure(figsize=self.config.figure_size)

        # Plot length evolution
        plt.plot(edge_data.frames, edge_data.lengths, '-b', label='Edge length')

        # Mark intercalation events
        if hasattr(edge_data, 'intercalations') and edge_data.intercalations:
            event_frames = [event.frame for event in edge_data.intercalations]
            event_lengths = [edge_data.lengths[edge_data.frames.index(frame)]
                             for frame in event_frames]
            plt.scatter(event_frames, event_lengths, c='red', s=100,
                        label='Intercalation events')

        plt.xlabel('Frame')
        plt.ylabel('Edge Length')
        plt.title(f'Edge {edge_data.edge_id} Analysis')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

    def _create_custom_colormap(self, start_color: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        """Create a custom colormap starting with a specific color"""
        n_colors = 256
        colors = plt.cm.get_cmap(self.config.color_scheme)(np.linspace(0, 1, n_colors))
        colors[0] = start_color
        return ListedColormap(colors)

    def _get_cell_colors(self, cell_ids: np.ndarray) -> Dict[int, Tuple[float, float, float]]:
        """Generate unique colors for each cell ID"""
        cell_ids = sorted([int(cell_id) for cell_id in cell_ids if cell_id != 0])
        n_cells = len(cell_ids)

        return {
            cell_id: hsv_to_rgb((i / max(1, n_cells - 1), 0.8, 0.8))
            for i, cell_id in enumerate(cell_ids)
        }

    def _generate_colors(self, n_cells: int) -> List[Tuple[float, float, float]]:
        """Generate evenly spaced vibrant colors"""
        colors = []
        for i in range(n_cells):
            # Use golden ratio to generate well-distributed hues
            hue = i * 0.618033988749895 % 1.0
            # High saturation and value for vibrant colors
            saturation = 0.9
            value = 1.0
            colors.append(hsv_to_rgb((hue, saturation, value)))
        return colors

    def _create_full_range_normalization(self, image: np.ndarray) -> Normalize:
        """Create normalization that stretches the image to full dynamic range"""
        vmin = np.min(image[image > 0])  # Exclude background (0)
        vmax = np.max(image)
        return Normalize(vmin=vmin, vmax=vmax)

    def visualize_boundaries(self, segmented_stack: np.ndarray,
                             boundaries_by_frame: Dict[int, List['CellBoundary']],
                             output_path: Optional[Path] = None,
                             show_progress: bool = True) -> None:
        """Create animation of detected boundaries across all frames."""
        logger.debug("Starting boundary visualization")

        # Validate inputs
        if not boundaries_by_frame:
            logger.warning("No boundaries provided for visualization")
            return

        # Get actual frame range from the data
        max_frame = max(boundaries_by_frame.keys())
        if max_frame >= len(segmented_stack):
            logger.warning(f"Frame mismatch: max boundary frame {max_frame} >= stack length {len(segmented_stack)}")
            max_frame = len(segmented_stack) - 1

        # Set up the figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Get unique cell pairs and assign colors
        all_cell_pairs = set()
        for frame in range(max_frame + 1):
            if frame in boundaries_by_frame:
                for boundary in boundaries_by_frame[frame]:
                    all_cell_pairs.add(tuple(sorted(boundary.cell_ids)))

        logger.debug(f"Found {len(all_cell_pairs)} unique cell pairs")

        color_map = {
            cell_pair: plt.cm.rainbow(i / max(1, len(all_cell_pairs) - 1))
            for i, cell_pair in enumerate(sorted(all_cell_pairs))
        }

        def update(frame):
            if frame >= len(segmented_stack):
                logger.warning(f"Skipping frame {frame} - beyond stack length")
                return

            ax.clear()
            ax.imshow(segmented_stack[frame], cmap='gray')
            ax.set_title(f'Frame {frame}')

            if frame in boundaries_by_frame:
                for boundary in boundaries_by_frame[frame]:
                    try:
                        coords = boundary.coordinates
                        cell_pair = tuple(sorted(boundary.cell_ids))
                        color = color_map.get(cell_pair)

                        if color is not None and len(coords) > 0:
                            ax.plot(coords[:, 1], coords[:, 0],
                                    c=color, linewidth=self.config.line_width,
                                    alpha=self.config.alpha)

                            # Only add label if we have valid coordinates
                            if len(coords) > len(coords) // 2:  # Use middle point
                                mid_point = coords[len(coords) // 2]
                                ax.text(mid_point[1], mid_point[0],
                                        f'{boundary.cell_ids[0]}-{boundary.cell_ids[1]}',
                                        color=color, fontsize=self.config.font_size)
                    except (IndexError, AttributeError) as e:
                        logger.debug(f"Error plotting boundary in frame {frame}: {e}")
                        continue

            ax.axis('off')

        frames = min(len(segmented_stack), max_frame + 1)
        logger.debug(f"Creating animation with {frames} frames")

        anim = FuncAnimation(fig, update, frames=frames,
                             interval=self.config.animation_interval)

        if show_progress:
            pbar = tqdm(total=frames, desc="Creating animation")

            def update_progress(frame, *args):
                pbar.update(1)

            anim.event_source.add_callback(update_progress)

        if output_path:
            logger.debug(f"Saving animation to {output_path}")
            anim.save(str(output_path), writer='pillow')
            plt.close()
            if show_progress:
                pbar.close()
        else:
            plt.show()

        logger.debug("Boundary visualization completed")

    def create_intercalation_animation(self, tracked_stack: np.ndarray,
                                       boundaries_by_frame: Dict[int, List['CellBoundary']],
                                       events: List['IntercalationEvent'],
                                       output_path: Path) -> None:
        """Create animation showing intercalation events with frame-by-frame visualization."""
        logger.debug("Starting intercalation animation")

        if not events:
            logger.warning("No events provided for animation")
            return

        if tracked_stack is None:
            logger.warning("No tracking data provided for animation")
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

        logger.debug(f"Creating animation with {len(valid_events)} valid events")

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        def update(frame):
            if frame >= len(tracked_stack):
                logger.warning(f"Skipping frame {frame} - beyond stack length")
                return

            ax.clear()
            ax.imshow(tracked_stack[frame], cmap='gray')

            try:
                # Check for events at current frame
                for event_idx, event in enumerate(valid_events, 1):
                    # Check for losing events at current frame
                    if frame == event.frame and hasattr(event, 'losing_cells'):
                        # Find and highlight the boundary between losing cells
                        losing_pair = set(int(x) for x in event.losing_cells)
                        if frame in boundaries_by_frame:
                            for boundary in boundaries_by_frame[frame]:
                                try:
                                    current_pair = set(int(x) for x in boundary.cell_ids)
                                    if current_pair == losing_pair:
                                        # Plot the boundary
                                        coords = boundary.coordinates
                                        if len(coords) > 0:
                                            ax.plot(coords[:, 1], coords[:, 0],
                                                    'r-', linewidth=self.config.line_width * 2)
                                except (ValueError, AttributeError, IndexError) as e:
                                    logger.debug(f"Error processing losing boundary: {e}")
                                    continue

                    # Check for gaining events in the previous frame
                    if frame > 0 and frame - 1 == event.frame and hasattr(event, 'gaining_cells'):
                        # Find and highlight the boundary between gaining cells
                        gaining_pair = set(int(x) for x in event.gaining_cells)
                        if frame in boundaries_by_frame:
                            for boundary in boundaries_by_frame[frame]:
                                try:
                                    current_pair = set(int(x) for x in boundary.cell_ids)
                                    if current_pair == gaining_pair:
                                        # Plot the boundary
                                        coords = boundary.coordinates
                                        if len(coords) > 0:
                                            ax.plot(coords[:, 1], coords[:, 0],
                                                    'r-', linewidth=self.config.line_width * 2)
                                except (ValueError, AttributeError, IndexError) as e:
                                    logger.debug(f"Error processing gaining boundary: {e}")
                                    continue

            except Exception as e:
                logger.debug(f"Error updating frame {frame}: {e}")

            ax.set_title(f'Frame {frame}')
            ax.axis('off')
            plt.tight_layout()

        logger.debug("Setting up animation")
        anim = FuncAnimation(
            fig,
            update,
            frames=range(len(tracked_stack)),
            interval=self.config.animation_interval,
            blit=False
        )

        logger.debug(f"Saving animation to {output_path}")
        anim.save(str(output_path), writer='pillow')
        plt.close()
        logger.debug("Intercalation animation completed")
    def plot_edge_length_tracks(self, trajectories: Dict[int, 'EdgeData'],
                                output_path: Path) -> None:
        """Plot edge length trajectories in groups of 10."""
        edge_ids = sorted(trajectories.keys())
        num_groups = (len(edge_ids) + 9) // 10

        fig, axes = plt.subplots(num_groups, 1, figsize=(15, 5 * num_groups))
        fig.patch.set_facecolor('white')

        if num_groups > 1:
            for ax in axes.flatten():
                ax.set_facecolor('white')
                ax.spines['bottom'].set_color('black')
                ax.spines['top'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.spines['right'].set_color('black')
                ax.tick_params(colors='black')
        else:
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes[0].set_facecolor('white')
            axes[0].spines['bottom'].set_color('black')
            axes[0].spines['top'].set_color('black')
            axes[0].spines['left'].set_color('black')
            axes[0].spines['right'].set_color('black')
            axes[0].tick_params(colors='black')

        for group_idx in range(num_groups):
            ax = axes[group_idx] if num_groups > 1 else axes[0]
            start_idx = group_idx * 10
            group_edges = edge_ids[start_idx:start_idx + 10]

            for edge_id in group_edges:
                edge_data = trajectories[edge_id]
                # Check for intercalations using the intercalations list
                has_intercalations = (hasattr(edge_data, 'intercalations') and
                                      edge_data.intercalations is not None and
                                      len(edge_data.intercalations) > 0)

                color = 'red' if has_intercalations else 'gray'
                ax.plot(edge_data.frames, edge_data.lengths, '-',
                        color=color, label=f'Edge {edge_id}')

                # Plot vertical lines for intercalation events
                if has_intercalations:
                    for event in edge_data.intercalations:
                        ax.axvline(x=event.frame, color='blue',
                                   linestyle='--', alpha=0.3)

                ax.set_xlabel('Frame', color='black')
                ax.set_ylabel('Edge Length (µm)', color='black')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.set_title(f'Edges {start_idx + 1}-{min(start_idx + 10, len(edge_ids))}',
                             color='black')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config.dpi,
                    facecolor='white', edgecolor='none')
        plt.close()

    def create_edge_evolution_animation(self, segmentation_stack: np.ndarray,
                                        trajectory: 'EdgeTrajectory',
                                        boundaries_by_frame: Dict[int, List['CellBoundary']],
                                        output_path: Path) -> None:
        """Create animation showing edge evolution and length plot."""
        logger.debug("Starting edge evolution animation")

        if segmentation_stack is None or len(segmentation_stack) == 0:
            logger.warning("No segmentation data provided")
            return

        # Validate trajectory data
        if not hasattr(trajectory, 'frames') or not trajectory.frames:
            logger.warning("No frame data in trajectory")
            return

        # Ensure frames are within bounds
        max_frame = len(segmentation_stack) - 1
        valid_frames = [f for f in trajectory.frames if 0 <= f <= max_frame]

        if not valid_frames:
            logger.warning("No valid frames in trajectory")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.patch.set_facecolor('white')

        def update(frame):
            if frame >= len(segmentation_stack):
                logger.warning(f"Skipping frame {frame} - beyond stack length")
                return

            for ax in (ax1, ax2):
                ax.clear()
                ax.set_facecolor('white')
                ax.spines['bottom'].set_color('black')
                ax.spines['top'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.spines['right'].set_color('black')
                ax.tick_params(colors='black')

            try:
                # Plot segmentation and edge
                ax1.imshow(segmentation_stack[frame], cmap='gray')

                if frame in trajectory.frames:
                    frame_idx = trajectory.frames.index(frame)
                    if hasattr(trajectory, 'cell_pairs') and frame_idx < len(trajectory.cell_pairs):
                        cell_pair = trajectory.cell_pairs[frame_idx]

                        # Find and plot matching boundary
                        if frame in boundaries_by_frame:
                            for boundary in boundaries_by_frame[frame]:
                                try:
                                    if (tuple(sorted(int(x) for x in boundary.cell_ids)) ==
                                            tuple(sorted(int(x) for x in cell_pair))):
                                        coords = boundary.coordinates
                                        if len(coords) > 0:
                                            ax1.plot(coords[:, 1], coords[:, 0], 'r-',
                                                     linewidth=self.config.line_width)
                                        break
                                except (ValueError, AttributeError, IndexError) as e:
                                    logger.debug(f"Error plotting boundary: {e}")
                                    continue

                ax1.set_title(f'Frame {frame}', color='black')
                ax1.axis('off')

                # Plot length trajectory
                if hasattr(trajectory, 'lengths') and hasattr(trajectory, 'frames'):
                    valid_indices = [i for i, f in enumerate(trajectory.frames)
                                     if 0 <= f < len(segmentation_stack)]

                    if valid_indices:
                        valid_frames = [trajectory.frames[i] for i in valid_indices]
                        valid_lengths = [trajectory.lengths[i] for i in valid_indices]

                        ax2.plot(valid_frames, valid_lengths, 'b-')

                        if frame in valid_frames:
                            idx = valid_frames.index(frame)
                            ax2.plot(frame, valid_lengths[idx], 'ro')

                # Add intercalation markers if available
                if hasattr(trajectory, 'intercalation_frames'):
                    valid_intercalations = [f for f in trajectory.intercalation_frames
                                            if 0 <= f < len(segmentation_stack)]
                    for f in valid_intercalations:
                        ax2.axvline(x=f, color='r', linestyle='--', alpha=0.5)

                ax2.set_xlabel('Frame', color='black')
                ax2.set_ylabel('Edge Length (µm)', color='black')
                ax2.tick_params(colors='black')
                ax2.grid(True, alpha=0.3)

            except Exception as e:
                logger.debug(f"Error updating frame {frame}: {e}")

        logger.debug("Creating animation")
        anim = FuncAnimation(
            fig,
            update,
            frames=range(len(segmentation_stack)),
            interval=self.config.animation_interval
        )

        logger.debug(f"Saving animation to {output_path}")
        anim.save(str(output_path), writer='pillow',
                  savefig_kwargs={'facecolor': 'white'})
        plt.close()
        logger.debug("Edge evolution animation completed")



