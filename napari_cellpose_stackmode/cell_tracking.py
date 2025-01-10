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

from structure import TrackingParameters, AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class RegionCache:
    """Cache for region properties to avoid recomputation"""
    label: int
    centroid: np.ndarray
    area: float
    coords: np.ndarray


class CellTracker:
    """Enhanced cell tracking with configurable parameters."""

    def __init__(self, config: 'AnalysisConfig'):
        self.config = config
        self.params = TrackingParameters()
        self._region_cache = {}

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

    def track_cells(self, segmentation_stack: np.ndarray) -> np.ndarray:
        """Track cells across frames using current parameters."""
        if len(segmentation_stack.shape) != 3:
            raise ValueError("Expected 3D stack (t, x, y)")

        tracked_segmentation = np.zeros_like(segmentation_stack)
        all_used_ids: Set[int] = set()

        # Filter small cells only if min_cell_size > 0
        if self.params.min_cell_size > 0:
            mask = np.zeros_like(segmentation_stack, dtype=bool)
            for t in range(len(segmentation_stack)):
                regions = self._cache_regions(segmentation_stack[t], t)
                for region in regions:
                    if region.area >= self.params.min_cell_size:
                        mask[t][region.coords[:, 0], region.coords[:, 1]] = True
            segmentation_stack = segmentation_stack * mask

        # Process first frame
        first_frame = segmentation_stack[0]
        regions = self._cache_regions(first_frame, 0)
        sorted_regions = sorted(regions, key=lambda r: np.linalg.norm(r.centroid))

        # Initialize with sequential IDs using vectorized operations
        labels = np.unique(first_frame)
        labels = labels[labels > 0]
        id_mapping = {region.label: idx for idx, region in enumerate(sorted_regions, start=1)}

        # Vectorized relabeling of first frame
        tracked_segmentation[0] = np.zeros_like(first_frame)
        for old_id, new_id in id_mapping.items():
            tracked_segmentation[0][first_frame == old_id] = new_id
            all_used_ids.add(new_id)

        # Process subsequent frames
        for t in range(len(segmentation_stack) - 1):
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

            if self.params.enable_gap_closing and t >= self.params.max_frame_gap:
                self._handle_gap_closing_optimized(
                    tracked_segmentation,
                    t,
                    overlap_matrix,
                    assigned_ids
                )

            self._process_frame_regions(
                next_frame,
                tracked_segmentation[t + 1],
                overlap_matrix,
                assigned_ids,
                all_used_ids
            )

        return tracked_segmentation

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

        current_regions = self._cache_regions(current_frame, current_idx)
        next_regions = self._cache_regions(next_frame, next_idx)

        # Create lookup dictionaries
        next_lookup = {r.label: r for r in next_regions}

        # Pre-calculate label sets for vectorized operations
        next_labels = np.array([r.label for r in next_regions])

        for current_region in current_regions:
            current_id = current_region.label
            candidates = []

            # Vectorized distance calculation
            current_centroid = current_region.centroid
            next_centroids = np.array([r.centroid for r in next_regions])
            distances = np.linalg.norm(next_centroids - current_centroid, axis=1)

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

    def _handle_gap_closing_optimized(
            self,
            tracked_segmentation: np.ndarray,
            current_frame: int,
            overlap_matrix: Dict[int, List[Tuple[int, int]]],
            assigned_ids: Set[int]
    ) -> None:
        """Optimized gap closing using cached region properties"""
        # Pre-calculate unique labels for current frame
        current_labels = set(np.unique(tracked_segmentation[current_frame]))
        current_labels.discard(0)

        # Process gaps in reverse order for efficiency
        for gap in range(self.params.max_frame_gap, 0, -1):
            prev_frame = current_frame - gap
            if prev_frame < 0:
                continue

            # Vectorized set operations for disappearing cells
            prev_labels = set(np.unique(tracked_segmentation[prev_frame]))
            prev_labels.discard(0)
            disappearing_cells = prev_labels - current_labels

            if not disappearing_cells:
                continue

            # Cache regions for both frames
            prev_regions = {r.label: r for r in self._cache_regions(tracked_segmentation[prev_frame], prev_frame)}
            current_regions = {r.label: r for r in self._cache_regions(tracked_segmentation[current_frame], current_frame)}

            # Process disappearing cells
            for cell_id in disappearing_cells:
                if cell_id in assigned_ids:
                    continue

                prev_region = prev_regions.get(cell_id)
                if prev_region is None:
                    continue

                # Vectorized distance and size comparison
                prev_centroid = prev_region.centroid
                prev_area = prev_region.area

                best_match = None
                best_score = float('inf')

                # Calculate all distances at once
                current_centroids = np.array([r.centroid for r in current_regions.values()])
                distances = np.linalg.norm(current_centroids - prev_centroid, axis=1)

                valid_indices = distances <= (self.params.max_displacement * gap)
                if not np.any(valid_indices):
                    continue

                # Process valid candidates
                valid_regions = [r for i, r in enumerate(current_regions.values()) if valid_indices[i]]
                for region in valid_regions:
                    if region.label not in assigned_ids:
                        displacement = distances[list(current_regions.values()).index(region)]
                        size_diff = abs(region.area - prev_area) / prev_area
                        score = displacement + size_diff * 100

                        if score < best_score:
                            best_score = score
                            best_match = region.label

                if best_match is not None:
                    # Add a dummy overlap value and use score as the combined_score
                    overlap_matrix[cell_id].append((best_match, prev_area, 1.0 - (best_score / (self.params.max_displacement * gap))))

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
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the segmentation stack
    segmentation_path = Path(r"D:\2024-11-27\position0\registered_data\segmentation.tif")
    segmentation_stack = tifffile.imread(str(segmentation_path))

    # Create dummy config
    config = AnalysisConfig()

    # Initialize tracker
    tracker = CellTracker(config)

    # Set up profiler
    pr = cProfile.Profile()
    pr.enable()

    # Run tracking
    logger.info("Starting cell tracking...")
    tracked_stack = tracker.track_cells(segmentation_stack)

    # Stop profiler
    pr.disable()

    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    logger.info(f"\nProfiling results:\n{s.getvalue()}")

    # Save results
    output_path = segmentation_path.parent / "tracked_segmentation.tif"
    tifffile.imwrite(str(output_path), tracked_stack)
    logger.info(f"Saved tracked segmentation to {output_path}")