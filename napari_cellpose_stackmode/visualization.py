import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, hsv_to_rgb, Normalize
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from scipy.ndimage import binary_dilation
from tqdm import tqdm

from napari_cellpose_stackmode.structure import VisualizationConfig

logger = logging.getLogger(__name__)

class Visualizer:
    """Handles all visualization tasks for cell tracking and analysis"""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        # Set global style for white background and black text
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

    def visualize_cell_tracks(self, segmentation_stack: np.ndarray, output_path: Path) -> None:
        """Create a visualization showing cell trajectories across all frames."""
        cell_ids = np.unique(segmentation_stack)
        cell_ids = cell_ids[cell_ids != 0]  # Remove background

        # Set up the figure with black background
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Show last frame with simple gray colormap
        plt.imshow(segmentation_stack[-1], cmap='gray')

        # Generate colors for tracks using rainbow colormap
        colors = plt.cm.rainbow(np.linspace(0, 1, len(cell_ids)))

        for cell_id, color in zip(cell_ids, colors):
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

        plt.title('Cell Trajectories', color='white')
        plt.axis('off')
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        plt.close()

    def create_tracking_animation(self, segmentation_stack: np.ndarray,
                                  output_path: Path) -> None:
        """Create an animation showing cell tracking over time."""
        cell_ids = np.unique(segmentation_stack)
        cell_ids = cell_ids[cell_ids != 0]  # Remove background

        # Generate colors for tracks
        colors = {
            int(cell_id): color
            for cell_id, color in zip(cell_ids, plt.cm.rainbow(np.linspace(0, 1, len(cell_ids))))
        }

        # Set up the figure with black background
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Pre-calculate all centroids
        all_centroids = {int(cell_id): [] for cell_id in cell_ids}
        for frame in segmentation_stack:
            for cell_id in cell_ids:
                mask = frame == cell_id
                if np.any(mask):
                    y, x = np.where(mask)
                    all_centroids[int(cell_id)].append((np.mean(x), np.mean(y)))
                else:
                    all_centroids[int(cell_id)].append(None)

        def update(frame_idx):
            ax.clear()
            ax.set_facecolor('black')
            # Show current frame with simple gray colormap
            ax.imshow(segmentation_stack[frame_idx], cmap='gray', alpha=1)

            for cell_id in cell_ids:
                cell_id_int = int(cell_id)
                centroids = [c for c in all_centroids[cell_id_int][:frame_idx + 1]
                             if c is not None]
                if centroids:
                    track = np.array(centroids)
                    color = colors[cell_id_int]
                    ax.plot(track[:, 0], track[:, 1], '-',
                            color=color, linewidth=self.config.line_width,
                            alpha=self.config.alpha)
                    ax.plot(track[:, 0], track[:, 1], 'o',
                            color=color, markersize=3)
                    if len(track) > 0:
                        ax.text(track[-1, 0], track[-1, 1], str(cell_id_int),
                                color=color, fontsize=self.config.font_size,
                                ha='left', va='bottom')

            ax.set_title(f'Frame {frame_idx}', color='white')
            ax.axis('off')

        anim = FuncAnimation(fig, update, frames=len(segmentation_stack),
                             interval=self.config.animation_interval, blit=False)

        # Save animation with black background
        anim.save(str(output_path), writer='pillow',
                  savefig_kwargs={'facecolor': 'black'})
        plt.close()

    def visualize_boundaries(self, segmented_stack: np.ndarray,
                             boundaries_by_frame: Dict[int, List['CellBoundary']],
                             output_path: Optional[Path] = None,
                             show_progress: bool = True) -> None:
        """Create animation of detected boundaries across all frames."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Get unique cell pairs and assign colors
        all_cell_pairs = set()
        for boundaries in boundaries_by_frame.values():
            for boundary in boundaries:
                all_cell_pairs.add(tuple(sorted(boundary.cell_ids)))

        color_map = {
            cell_pair: plt.cm.rainbow(i / len(all_cell_pairs))
            for i, cell_pair in enumerate(sorted(all_cell_pairs))
        }

        def update(frame):
            ax.clear()
            ax.imshow(segmented_stack[frame], cmap='gray')
            ax.set_title(f'Frame {frame}')

            if frame in boundaries_by_frame:
                for boundary in boundaries_by_frame[frame]:
                    coords = boundary.coordinates
                    cell_pair = tuple(sorted(boundary.cell_ids))
                    color = color_map[cell_pair]

                    ax.plot(coords[:, 1], coords[:, 0],
                            c=color, linewidth=self.config.line_width,
                            alpha=self.config.alpha)

                    mid_point = coords[len(coords) // 2]
                    ax.text(mid_point[1], mid_point[0],
                            f'{boundary.cell_ids[0]}-{boundary.cell_ids[1]}',
                            color=color, fontsize=self.config.font_size)

            ax.axis('off')

        frames = len(segmented_stack)
        anim = FuncAnimation(fig, update, frames=frames,
                             interval=self.config.animation_interval)

        if show_progress:
            pbar = tqdm(total=frames, desc="Creating animation")

            def update_progress(frame, *args):
                pbar.update(1)

            anim.event_source.add_callback(update_progress)

        if output_path:
            anim.save(str(output_path), writer='pillow')
            plt.close()
            if show_progress:
                pbar.close()
        else:
            plt.show()

    def visualize_intercalation_event(self, event: 'IntercalationEvent',
                                      frame_before: np.ndarray,
                                      frame_after: np.ndarray,
                                      output_path: Optional[Path] = None) -> None:
        """Create visualization of an intercalation event with improved clarity"""
        losing_cells = tuple(int(x) for x in event.losing_cells)
        gaining_cells = tuple(int(x) for x in event.gaining_cells)

        # Create figure with less empty space
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('white')  # Set figure background to white

        # Create custom colormaps for non-participating cells
        other_cells_cmap = ListedColormap(['white'])  # White for non-participating cells

        # Get the valid region bounds to zoom in
        def get_bounds(frame):
            mask = frame > 0
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Add padding
            pad = 10
            rmin, rmax = max(0, rmin - pad), min(frame.shape[0], rmax + pad)
            cmin, cmax = max(0, cmin - pad), min(frame.shape[1], cmax + pad)
            return rmin, rmax, cmin, cmax

        # Plot each frame
        for ax, frame, title in [(ax1, frame_before, 'Before (Frame {})'.format(event.frame)),
                                 (ax2, frame_after, 'After (Frame {})'.format(event.frame + 1))]:
            ax.set_facecolor('white')  # Set axes background to white

            rmin, rmax, cmin, cmax = get_bounds(frame)
            frame_zoomed = frame[rmin:rmax, cmin:cmax]

            # Create base image with white for all cells
            ax.imshow(frame_zoomed > 0, cmap=other_cells_cmap)

            # Plot cells involved in intercalation
            for cell_id in losing_cells:
                mask = frame_zoomed == cell_id
                if np.any(mask):
                    # Red for losing cells
                    ax.imshow(mask, cmap='Reds', alpha=0.7 if ax == ax1 else 0.3)
                    y, x = np.nonzero(mask)
                    ax.text(np.mean(x), np.mean(y), str(cell_id),
                            color='darkred', fontsize=12, fontweight='bold',
                            ha='center', va='center')

            for cell_id in gaining_cells:
                mask = frame_zoomed == cell_id
                if np.any(mask):
                    # Blue for gaining cells
                    ax.imshow(mask, cmap='Blues', alpha=0.3 if ax == ax1 else 0.7)
                    y, x = np.nonzero(mask)
                    ax.text(np.mean(x), np.mean(y), str(cell_id),
                            color='darkblue', fontsize=12, fontweight='bold',
                            ha='center', va='center')

            # Add boundary highlighting
            if ax == ax1:
                # Find boundary between losing cells
                boundary = np.zeros_like(frame_zoomed, dtype=bool)
                for c1 in losing_cells:
                    for c2 in losing_cells:
                        if c1 < c2:
                            mask1 = frame_zoomed == c1
                            mask2 = frame_zoomed == c2
                            boundary |= (binary_dilation(mask1) & mask2)
                # Plot boundary
                if np.any(boundary):
                    ax.imshow(boundary, cmap=ListedColormap(['none', 'red']), alpha=0.8)
            else:
                # Find boundary between gaining cells
                boundary = np.zeros_like(frame_zoomed, dtype=bool)
                for c1 in gaining_cells:
                    for c2 in gaining_cells:
                        if c1 < c2:
                            mask1 = frame_zoomed == c1
                            mask2 = frame_zoomed == c2
                            boundary |= (binary_dilation(mask1) & mask2)
                # Plot boundary
                if np.any(boundary):
                    ax.imshow(boundary, cmap=ListedColormap(['none', 'blue']), alpha=0.8)

            # Add cell edges for all cells
            for cell_id in np.unique(frame_zoomed):
                if cell_id > 0:  # Skip background
                    mask = frame_zoomed == cell_id
                    edges = binary_dilation(mask) & ~mask
                    ax.imshow(edges, cmap=ListedColormap(['none', 'black']), alpha=0.2)

            ax.set_title(title)
            ax.axis('off')

        plt.suptitle(f'Intercalation Event:\nCells {losing_cells} separate, Cells {gaining_cells} connect',
                     y=1.05, color="black")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=self.config.dpi,
                        facecolor='white', edgecolor='none')
            plt.close()
        else:
            plt.show()

    def create_intercalation_animation(self, tracked_stack: np.ndarray,
                                       boundaries_by_frame: Dict[int, List['CellBoundary']],
                                       events: List['IntercalationEvent'],
                                       output_dir: Path) -> None:
        """Create animation showing intercalation events."""
        if not events:
            logger.warning("No events provided for animation")
            return

        output_path = output_dir / "intercalations.gif"
        padding = 2
        start_frame = max(0, min(e.frame for e in events) - padding)
        end_frame = min(len(tracked_stack), max(e.frame for e in events) + padding + 1)

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        def update(frame):
            ax.clear()
            ax.imshow(tracked_stack[frame], cmap='gray')

            # Get active events for this frame
            active_events = []
            for event in events:
                # Show losing cells boundary at event frame
                if frame == event.frame:
                    active_events.append((event, 'losing'))
                # Show gaining cells boundary at frame after event
                elif frame == event.frame + 1:
                    active_events.append((event, 'gaining'))

            # Highlight only the boundaries involved in active events
            for event, event_type in active_events:
                for boundary in boundaries_by_frame[frame]:
                    if event_type == 'losing' and set(int(x) for x in boundary.cell_ids) == set(int(x) for x in event.losing_cells):
                        ax.plot(boundary.coordinates[:, 1],
                                boundary.coordinates[:, 0],
                                'r-', linewidth=self.config.line_width,
                                label='Separating cells')
                    elif event_type == 'gaining' and set(int(x) for x in boundary.cell_ids) == set(int(x) for x in event.gaining_cells):
                        ax.plot(boundary.coordinates[:, 1],
                                boundary.coordinates[:, 0],
                                'g-', linewidth=self.config.line_width,
                                label='Connecting cells')

            # Add legend if there are active events
            # if active_events:
            #     handles, labels = ax.get_legend_handles_labels()
                # by_label = dict(zip(labels, handles))
                # ax.legend(by_label.values(), by_label.keys(), loc='upper right')

            ax.set_title(f'Frame {frame}')
            ax.axis('off')
            plt.tight_layout()

        anim = FuncAnimation(
            fig,
            update,
            frames=range(start_frame, end_frame),
            interval=self.config.animation_interval,
            blit=False
        )

        anim.save(str(output_path), writer='pillow')
        plt.close()

    def plot_edge_length_tracks(self, trajectories: Dict[int, 'EdgeTrajectory'],
                                output_path: Path) -> None:
        """Plot edge length trajectories in groups of 10."""
        edge_ids = sorted(trajectories.keys())
        num_groups = (len(edge_ids) + 9) // 10

        fig, axes = plt.subplots(num_groups, 1, figsize=(15, 5 * num_groups))
        # Set white background and black text/axes

        if num_groups > 1:
            for ax in axes.flatten():
                # Set white background and black text/axes
                ax.set_facecolor('white')
                ax.spines['bottom'].set_color('black')
                ax.spines['top'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.spines['right'].set_color('black')
                ax.tick_params(colors='black')
        else:
            axes.set_facecolor('white')
            axes.spines['bottom'].set_color('black')
            axes.spines['top'].set_color('black')
            axes.spines['left'].set_color('black')
            axes.spines['right'].set_color('black')
            axes.tick_params(colors='black')

        if num_groups == 1:
            axes = [axes]

        for group_idx in range(num_groups):
            ax = axes[group_idx]
            start_idx = group_idx * 10
            group_edges = edge_ids[start_idx:start_idx + 10]

            for edge_id in group_edges:
                traj = trajectories[edge_id]
                color = 'red' if traj.intercalation_frames else 'gray'
                ax.plot(traj.frames, traj.lengths, '-',
                        color=color, label=f'Edge {edge_id}')

                if traj.intercalation_frames:
                    for frame in traj.intercalation_frames:
                        ax.axvline(x=frame, color='blue',
                                   linestyle='--', alpha=0.3)

            ax.set_xlabel('Frame')
            ax.set_ylabel('Edge Length (µm)')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_facecolor('white')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config.dpi,
                    facecolor='white', edgecolor='none')
        plt.close()

    def create_edge_evolution_animation(self, segmentation_stack: np.ndarray,
                                        trajectory: 'EdgeTrajectory',
                                        boundaries_by_frame: Dict[int, List['CellBoundary']],
                                        output_path: Path) -> None:
        """Create animation showing edge evolution and length plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.patch.set_facecolor('white')

        def update(frame):
            ax1.clear()
            ax2.clear()
            for ax in (ax1, ax2):
                # Set white background and black text/axes
                ax.set_facecolor('white')
                ax.spines['bottom'].set_color('black')
                ax.spines['top'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.spines['right'].set_color('black')
                ax.tick_params(colors='black')

            # Plot segmentation and edge
            ax1.imshow(segmentation_stack[frame], cmap='gray')
            if frame in trajectory.frames:
                idx = trajectory.frames.index(frame)
                cell_pair = trajectory.cell_pairs[idx]

                for boundary in boundaries_by_frame[frame]:
                    if tuple(sorted(int(x) for x in boundary.cell_ids)) == cell_pair:
                        coords = boundary.coordinates
                        ax1.plot(coords[:, 1], coords[:, 0], 'r-',
                                 linewidth=self.config.line_width)
                        break

            ax1.set_title(f'Frame {frame}', color='black')
            ax1.axis('off')

            # Plot length trajectory
            ax2.plot(trajectory.frames, trajectory.lengths, 'b-')
            if frame in trajectory.frames:
                idx = trajectory.frames.index(frame)
                ax2.plot(frame, trajectory.lengths[idx], 'ro')

            if trajectory.intercalation_frames:
                for f in trajectory.intercalation_frames:
                    ax2.axvline(x=f, color='r', linestyle='--', alpha=0.5)

            ax2.set_xlabel('Frame', color='black')
            ax2.set_ylabel('Edge Length (µm)', color='black')
            ax2.tick_params(colors='black')
            ax2.grid(True, alpha=0.3)

        anim = FuncAnimation(fig, update, frames=len(segmentation_stack),
                             interval=self.config.animation_interval)
        anim.save(str(output_path), writer='pillow',
                  savefig_kwargs={'facecolor': 'white'})
        plt.close()

    # def plot_edge_length_distribution(self, csv_path: Path, output_path: Path) -> None:
    #     """Create distribution plot of edge lengths with log(1/probability) on y-axis."""
    #     df = pd.read_csv(csv_path)
    #     lengths = df['length'].values
    #     hist, bin_edges = np.histogram(lengths, bins=self.config.histogram_bins, density=True)
    #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #     epsilon = 1e-10
    #     log_inverse_prob = np.log(1 / (hist + epsilon))
    #
    #     plt.figure(figsize=self.config.figure_size)
    #
    #     # Set white background and black text/axes
    #     ax = plt.gca()
    #     ax.set_facecolor('white')
    #     plt.gcf().patch.set_facecolor('white')
    #     ax.spines['bottom'].set_color('black')
    #     ax.spines['top'].set_color('black')
    #     ax.spines['left'].set_color('black')
    #     ax.spines['right'].set_color('black')
    #     ax.tick_params(colors='black')
    #
    #     plt.plot(bin_centers, log_inverse_prob, '-o',
    #              color='blue',
    #              alpha=self.config.alpha,
    #              linewidth=self.config.line_width,
    #              markersize=4)
    #
    #     plt.xlabel('Edge Length (μm)', color='black')
    #     plt.ylabel('log(1/probability)', color='black')
    #     plt.title('Edge Length Distribution', color='black')
    #     plt.grid(True, alpha=0.3)
    #
    #     plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight',
    #                 facecolor='white', edgecolor='none')
    #     plt.close()
    #
    #     logger.info(f"Edge length distribution plot saved to {output_path}")

    def create_visualizations(self, results: 'AnalysisResults', input_path: Path = None):
        """Create all visualizations from analysis results"""
        logger.info("Starting visualization generation")

        # Check if any visualizations are enabled before setting up directories
        if not any([
            self.config.tracking_plots_enabled,
            self.config.edge_detection_overlay,
            self.config.intercalation_events,
            self.config.edge_length_evolution
        ]):
            logger.info("No visualizations enabled in configuration. Skipping visualization generation.")
            return

        # Set up base output directory based on input file name
        if input_path:
            base_dir = input_path.parent / input_path.stem / "visualizations"
        else:
            base_dir = self.config.output_dir if self.config.output_dir else Path.cwd()

        # Only create directories for enabled visualizations that have available data
        directories_to_create = []

        if results.tracked_stack is not None and self.config.tracking_plots_enabled:
            directories_to_create.append(base_dir / "tracking_output")
            logger.info("Will create tracking visualizations")

        if (results.tracked_stack is not None and
                results.boundaries is not None and
                self.config.edge_detection_overlay):
            directories_to_create.append(base_dir / "edge_output")
            logger.info("Will create edge detection visualizations")

        if results.events is not None and self.config.intercalation_events:
            directories_to_create.append(base_dir / "intercalation_output")
            logger.info("Will create intercalation visualizations")

        if results.trajectories is not None and self.config.edge_length_evolution:
            directories_to_create.append(base_dir / "edge_analysis_output")
            logger.info("Will create edge analysis visualizations")

        # If no directories to create after checking data availability, return early
        if not directories_to_create:
            logger.info("No visualizations to generate with available data.")
            return

        # Create parent directory and subdirectories
        base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created parent directory: {base_dir}")

        for directory in directories_to_create:
            directory.mkdir(exist_ok=True)
            logger.debug(f"Created directory: {directory}")

        # Generate enabled visualizations
        if results.tracked_stack is not None and self.config.tracking_plots_enabled:
            logger.info("Creating tracking visualizations")
            tracking_dir = base_dir / "tracking_output"
            self.visualize_cell_tracks(
                results.tracked_stack,
                tracking_dir / "cell_trajectories.png"
            )
            self.create_tracking_animation(
                results.tracked_stack,
                tracking_dir / "tracking_animation.gif"
            )

        if (results.tracked_stack is not None and
                results.boundaries is not None and
                self.config.edge_detection_overlay):
            logger.info("Creating edge detection visualizations")
            edge_dir = base_dir / "edge_output"
            self.visualize_boundaries(
                results.tracked_stack,
                results.boundaries,
                edge_dir / "edge_detection.gif"
            )

        if results.events is not None and self.config.intercalation_events:
            logger.info("Creating intercalation visualizations")
            vis_dir = base_dir / "intercalation_output"

            for i, event in enumerate(results.events):
                self.visualize_intercalation_event(
                    event,
                    results.tracked_stack[event.frame],
                    results.tracked_stack[event.frame + 1],
                    vis_dir / f"event_{i:03d}.png"
                )

                if self.config.create_example_gifs and results.boundaries is not None:
                    self.create_intercalation_animation(
                        results.tracked_stack,
                        results.boundaries,
                        results.events,
                        vis_dir
                    )

        if results.trajectories is not None and self.config.edge_length_evolution:
            logger.info("Creating edge analysis visualizations")
            edge_analysis_dir = base_dir / "edge_analysis_output"
            plots_dir = edge_analysis_dir

            # Create basic edge length evolution plot
            self.plot_edge_length_tracks(
                results.trajectories,
                plots_dir / "length_tracks.png"
            )

            if (results.tracked_stack is not None and
                    results.boundaries is not None and
                    self.config.create_example_gifs):
                example_dir = plots_dir / "examples"
                example_dir.mkdir(exist_ok=True)

                # Choose interesting edges (with intercalations)
                interesting_edges = [
                                        edge_id for edge_id, traj in results.trajectories.items()
                                        if traj.intercalation_frames
                                    ][:self.config.max_example_gifs]

                # Add some regular edges if we don't have enough interesting ones
                if len(interesting_edges) < self.config.max_example_gifs:
                    regular_edges = [
                                        edge_id for edge_id, traj in results.trajectories.items()
                                        if not traj.intercalation_frames
                                    ][:self.config.max_example_gifs - len(interesting_edges)]
                    interesting_edges.extend(regular_edges)

                for edge_id in interesting_edges:
                    self.create_edge_evolution_animation(
                        results.tracked_stack,
                        results.trajectories[edge_id],
                        results.boundaries,
                        example_dir / f"edge_{edge_id}_evolution.gif"
                    )

        logger.info(f"Visualization generation completed. Output saved to {base_dir}")


if __name__ == "__main__":
    # Example usage
    config = VisualizationConfig()
    visualizer = Visualizer(config)
    logger.info("Visualization module loaded successfully")