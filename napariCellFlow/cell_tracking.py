import logging
from typing import Dict, List, Tuple, Set
import cProfile
import pstats
import io
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from skimage.measure import regionprops
import tifffile

from napariCellFlow.structure import TrackingParameters, AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class RegionCache:
    """Cache for region properties to avoid recomputation"""
    label: int
    centroid: np.ndarray
    area: float
    coords: np.ndarray


class CellTracker:
    def __init__(self, config: 'AnalysisConfig'):
        self.config = config
        self.params = TrackingParameters()
        self._region_cache = {}
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self._progress_callback = callback

    def _update_progress(self, progress: float, message: str = ""):
        """Update progress through callback if available"""
        if self._progress_callback:
            self._progress_callback(progress, message)

    def track_cells(self, segmentation_stack: np.ndarray) -> np.ndarray:
        """Track cells across frames using current parameters."""
        if len(segmentation_stack.shape) != 3:
            raise ValueError("Expected 3D stack (t, x, y)")

        # Fix duplicate logging
        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.handlers = [root_logger.handlers[0]]

        total_frames = len(segmentation_stack)
        tracked_segmentation = np.zeros_like(segmentation_stack)
        all_used_ids: Set[int] = set()
        future_assignments = {}

        logger.debug("=== Starting tracking process ===")
        logger.debug(f"Input stack shape: {segmentation_stack.shape}")
        logger.debug(f"Input first frame unique IDs: {set(np.unique(segmentation_stack[0])) - {0} }")

        self._update_progress(0, "Initializing tracking...")

        # Filter small cells if needed
        if self.params.min_cell_size > 0:
            self._update_progress(5, "Filtering small cells...")
            mask = np.zeros_like(segmentation_stack, dtype=bool)
            for t in range(total_frames):
                regions = self._cache_regions(segmentation_stack[t], t)
                for region in regions:
                    if region.area >= self.params.min_cell_size:
                        mask[t][region.coords[:, 0], region.coords[:, 1]] = True
                self._update_progress(5 + (15 * t / total_frames), "Filtering small cells...")
            segmentation_stack = segmentation_stack * mask

        # Process first frame
        self._update_progress(20, "Processing first frame...")
        first_frame = segmentation_stack[0]
        regions = self._cache_regions(first_frame, 0)
        logger.debug(f"Number of regions found in first frame: {len(regions)}")
        logger.debug(f"Region labels: {[r.label for r in regions]}")

        sorted_regions = sorted(regions, key=lambda r: np.linalg.norm(r.centroid))
        logger.debug(f"Regions after sorting: {[r.label for r in sorted_regions]}")

        id_mapping = {region.label: idx for idx, region in enumerate(sorted_regions, start=1)}
        logger.debug(f"ID mapping for first frame: {id_mapping}")

        tracked_segmentation[0] = np.zeros_like(first_frame)
        for old_id, new_id in id_mapping.items():
            tracked_segmentation[0][first_frame == old_id] = new_id
            all_used_ids.add(new_id)
            logger.debug(f"Mapped ID {old_id} -> {new_id}")

        logger.debug(f"Unique IDs in tracked first frame: {set(np.unique(tracked_segmentation[0])) - {0} }")

        # Process subsequent frames
        for t in range(total_frames - 1):
            progress = 20 + (70 * t / (total_frames - 1))
            self._update_progress(progress, f"Tracking frame {t + 1}/{total_frames - 1}...")

            current_frame = tracked_segmentation[t]
            next_frame = segmentation_stack[t + 1]

            overlap_matrix = self._calculate_overlap_matrix(
                current_frame,
                next_frame,
                t,
                t + 1,
                max_displacement=self.params.max_displacement,
                min_overlap_ratio=self.params.min_overlap_ratio
            )

            tracked_segmentation[t + 1] = np.zeros_like(next_frame)
            assigned_ids: Set[int] = set()

            # Handle gap closing if enabled
            if self.params.enable_gap_closing and t >= self.params.max_frame_gap:
                logger.debug(f"Running gap closing for frame {t}")
                gap_assignments = self._handle_gap_closing(
                    tracked_segmentation=tracked_segmentation,
                    current_frame=t,
                    overlap_matrix=overlap_matrix,
                    assigned_ids=assigned_ids,
                    segmentation_stack=segmentation_stack
                )

                # Merge new assignments with existing future assignments
                for frame_idx, assignments in gap_assignments.items():
                    if frame_idx not in future_assignments:
                        future_assignments[frame_idx] = {}
                    future_assignments[frame_idx].update(assignments)
                    logger.debug(f"Added assignments for frame {frame_idx}: {assignments.keys()}")

            # Process normal tracking for next frame
            self._process_frame_regions_optimized(
                next_frame,
                tracked_segmentation[t + 1],
                overlap_matrix,
                assigned_ids,
                all_used_ids
            )

            # Apply any assignments for this frame
            if t + 1 in future_assignments:
                logger.debug(f"Applying stored assignments for frame {t + 1}")
                frame_assignments = future_assignments[t + 1]
                for cell_id, (next_id, mask) in frame_assignments.items():
                    logger.debug(f"Applying assignment: cell {cell_id} -> {next_id}")
                    tracked_segmentation[t + 1][mask] = cell_id
                    logger.debug(f"After applying assignment, unique IDs in frame {t + 1}: "
                                 f"{set(np.unique(tracked_segmentation[t + 1])) - {0} }")
                del future_assignments[t + 1]

            logger.debug(f"Completed processing frame {t + 1}")
            logger.debug(f"Unique IDs in frame {t + 1}: {set(np.unique(tracked_segmentation[t + 1])) - {0} }")

        self._update_progress(100, "Tracking complete")
        return tracked_segmentation

    def _handle_gap_closing(
            self,
            tracked_segmentation: np.ndarray,
            current_frame: int,
            overlap_matrix: Dict[int, List[Tuple[int, int, float]]],
            assigned_ids: Set[int],
            segmentation_stack: np.ndarray
    ) -> Dict[int, Dict[int, Tuple[int, np.ndarray]]]:
        """Gap closing that returns assignments for future frames."""
        logger.debug(f"=== Starting gap closing for frame {current_frame} ===")
        logger.debug(f"Track matrix shape: {tracked_segmentation.shape}")
        logger.debug(f"Max frame gap: {self.params.max_frame_gap}")

        # Get labels from current frame
        current_labels = set(np.unique(tracked_segmentation[current_frame]))
        current_labels.discard(0)

        # Look back to find disappeared cells
        prev_frame = current_frame - 1
        if prev_frame < 0:
            return {}

        prev_labels = set(np.unique(tracked_segmentation[prev_frame]))
        prev_labels.discard(0)

        # Find cells that disappeared
        disappeared_cells = prev_labels - current_labels
        if not disappeared_cells:
            return {}

        # Check up to max_frame_gap frames ahead
        max_look_ahead = min(self.params.max_frame_gap, len(tracked_segmentation) - current_frame - 1)
        logger.debug(f"Looking ahead up to {max_look_ahead} frames")

        # Store assignments for each frame
        frame_assignments = {}

        for cell_id in disappeared_cells:
            if cell_id in assigned_ids:
                logger.debug(f"Cell {cell_id} already assigned, skipping")
                continue

            logger.debug(f"Processing disappeared cell {cell_id}")

            disappeared_cell_frame = np.zeros_like(tracked_segmentation[prev_frame])
            disappeared_cell_mask = tracked_segmentation[prev_frame] == cell_id
            disappeared_cell_frame[disappeared_cell_mask] = cell_id

            if not np.any(disappeared_cell_mask):
                continue

            best_score = -1
            best_match = None

            for frame_offset in range(1, max_look_ahead + 1):
                future_frame = current_frame + frame_offset
                logger.debug(f"Checking frame {future_frame} for cell {cell_id}")

                next_frame_data = segmentation_stack[future_frame]

                adjusted_max_displacement = self.params.max_displacement * (frame_offset + 1)
                adjusted_min_overlap = self.params.min_overlap_ratio * (0.9 ** frame_offset)

                gap_matrix = self._calculate_overlap_matrix(
                    disappeared_cell_frame,
                    next_frame_data,
                    prev_frame,
                    future_frame,
                    max_displacement=adjusted_max_displacement,
                    min_overlap_ratio=adjusted_min_overlap
                )

                if cell_id in gap_matrix and gap_matrix[cell_id]:
                    next_id, overlap, score = gap_matrix[cell_id][0]
                    logger.debug(f"Frame {future_frame}: Found potential match "
                                 f"next_id={next_id}, overlap={overlap}, score={score}")

                    if score > best_score:
                        mask = next_frame_data == next_id
                        best_score = score
                        best_match = (future_frame, next_id, mask)

            if best_match:
                future_frame, next_id, mask = best_match
                if future_frame not in frame_assignments:
                    frame_assignments[future_frame] = {}
                frame_assignments[future_frame][cell_id] = (next_id, mask)
                logger.debug(f"Storing assignment for frame {future_frame}: cell {cell_id} -> {next_id}")
                logger.debug(f"Mask sum: {np.sum(mask)}")

        logger.debug(f"Returning assignments for frames: {list(frame_assignments.keys())}")
        return frame_assignments

    def update_parameters(self, new_params: TrackingParameters):
        """Update tracking parameters"""
        self.params = new_params
        self._region_cache.clear()

    def _cache_regions(self, frame: np.ndarray, frame_idx: int) -> List[RegionCache]:
        """Cache region properties for a frame"""
        cache_key = (frame_idx, hash(frame.tobytes()))
        if cache_key in self._region_cache:
            return self._region_cache[cache_key]

        regions = []
        for region in regionprops(frame):
            cached_region = RegionCache(
                label=region.label,
                centroid=np.array(region.centroid),
                area=region.area,
                coords=region.coords
            )
            regions.append(cached_region)

        self._region_cache[cache_key] = regions
        return regions

    def _process_frame_regions_optimized(
            self,
            current_frame: np.ndarray,
            output_frame: np.ndarray,
            overlap_matrix: Dict[int, List[Tuple[int, int, float]]],
            assigned_ids: Set[int],
            all_used_ids: Set[int]
    ) -> None:
        """Optimized region processing avoiding expensive unique operations"""
        # Process matched cells first
        for current_id, candidates in overlap_matrix.items():
            for next_id, _, score in candidates:
                if next_id not in assigned_ids:
                    output_frame[current_frame == next_id] = current_id
                    assigned_ids.add(next_id)
                    break

        # Find unmatched cells more efficiently
        current_labels = set()
        # Use a small buffer to process the array in chunks
        chunk_size = 1000000  # Adjust based on available memory
        flat_frame = current_frame.ravel()
        for i in range(0, len(flat_frame), chunk_size):
            chunk = flat_frame[i:i + chunk_size]
            current_labels.update(np.unique(chunk[chunk > 0]))

        unmatched_labels = current_labels - assigned_ids

        if unmatched_labels:
            start_id = max(all_used_ids) + 1 if all_used_ids else 1
            new_ids = range(start_id, start_id + len(unmatched_labels))

            for new_id, label in zip(new_ids, unmatched_labels):
                output_frame[current_frame == label] = new_id
                all_used_ids.add(new_id)
                assigned_ids.add(label)

    def _calculate_overlap_matrix(
            self,
            current_frame: np.ndarray,
            next_frame: np.ndarray,
            current_idx: int,
            next_idx: int,
            max_displacement: float,
            min_overlap_ratio: float
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        """Optimized overlap calculation using cached region properties"""
        overlap_matrix = defaultdict(list)

        # Get regions for both frames
        current_regions = self._cache_regions(current_frame, current_idx)
        next_regions = self._cache_regions(next_frame, next_idx)

        # If either frame is empty, return empty matrix
        if not current_regions or not next_regions:
            logger.debug(f"Empty regions detected - current: {len(current_regions)}, next: {len(next_regions)}")
            return overlap_matrix

        # Create lookup dictionaries
        next_lookup = {r.label: r for r in next_regions}

        # Pre-calculate arrays for vectorized operations
        next_labels = np.array([r.label for r in next_regions])
        next_centroids = np.array([r.centroid for r in next_regions])

        for current_region in current_regions:
            current_id = current_region.label
            candidates = []

            # Vectorized distance calculation
            current_centroid = current_region.centroid
            try:
                distances = np.linalg.norm(next_centroids - current_centroid, axis=1)
            except ValueError as e:
                logger.error(f"Distance calculation failed: {e}")
                logger.debug(f"Current centroid shape: {current_centroid.shape}")
                logger.debug(f"Next centroids shape: {next_centroids.shape}")
                continue

            # Filter by distance
            valid_indices = distances <= max_displacement
            valid_next_labels = next_labels[valid_indices]

            if len(valid_next_labels) > 0:
                # Calculate overlaps for valid candidates
                coords = current_region.coords
                next_values = next_frame[coords[:, 0], coords[:, 1]]

                for next_id in valid_next_labels:
                    next_region = next_lookup[next_id]

                    # Vectorized overlap calculation
                    overlap = np.sum(next_values == next_id)

                    if overlap > 0:
                        overlap_ratio_current = overlap / current_region.area
                        overlap_ratio_next = overlap / next_region.area
                        effective_overlap = min(overlap_ratio_current, overlap_ratio_next)

                        if effective_overlap > 0:
                            displacement = distances[next_labels == next_id][0]
                            displacement_score = 1.0 - (displacement / max_displacement)
                            combined_score = (0.7 * effective_overlap + 0.3 * displacement_score)

                            if combined_score >= min_overlap_ratio:
                                candidates.append((next_id, overlap, combined_score))

                if candidates:
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    overlap_matrix[current_id] = candidates

        return overlap_matrix

    def _process_frame_regions(
            self,
            current_frame: np.ndarray,
            output_frame: np.ndarray,
            overlap_matrix: Dict[int, List[Tuple[int, int, float]]],
            assigned_ids: Set[int],
            all_used_ids: Set[int]
    ) -> None:
        """Optimized region processing using vectorized operations"""
        # Get unique labels once
        next_labels = np.unique(current_frame)
        next_labels = next_labels[next_labels > 0]

        processed_ids = set()

        # Process matched cells
        for current_id, candidates in overlap_matrix.items():
            for next_id, _, score in candidates:  # Changed to explicitly use score
                if next_id not in assigned_ids:
                    output_frame[current_frame == next_id] = current_id
                    assigned_ids.add(next_id)
                    processed_ids.add(current_id)
                    break

        # Process unmatched cells
        unmatched_mask = ~np.isin(current_frame, list(assigned_ids))
        unmatched_labels = np.unique(current_frame[unmatched_mask])
        unmatched_labels = unmatched_labels[unmatched_labels > 0]

        if len(unmatched_labels) > 0:
            start_id = max(all_used_ids) + 1 if all_used_ids else 1
            new_ids = np.arange(start_id, start_id + len(unmatched_labels))

            for new_id, label in zip(new_ids, unmatched_labels):
                output_frame[current_frame == label] = new_id
                all_used_ids.add(new_id)
                assigned_ids.add(label)


if __name__ == "__main__":
    # Set up detailed logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # This ensures our configuration takes precedence
    )

    # Create a console handler with the same formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the console handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)

    # Load the segmentation stack
    segmentation_path = Path(r"C:\Users\aruppel\Desktop\segmentation.tif")
    segmentation_stack = tifffile.imread(str(segmentation_path))

    # Create dummy config
    config = AnalysisConfig()

    # Initialize tracker with gap closing enabled
    tracker = CellTracker(config)
    params = TrackingParameters()
    params.enable_gap_closing = True
    params.max_frame_gap = 2
    tracker.update_parameters(params)

    # Run tracking
    logger.info("Starting cell tracking...")
    tracked_stack = tracker.track_cells(segmentation_stack)

    # Save results
    output_path = segmentation_path.parent / "tracked_segmentation.tif"
    tifffile.imwrite(str(output_path), tracked_stack)
    logger.info(f"Saved tracked segmentation to {output_path}")
