import logging
from typing import Dict, List, Tuple, Set

import numpy as np
from skimage.measure import regionprops

from structure import TrackingParameters, AnalysisConfig

logger = logging.getLogger(__name__)



class CellTracker:
    """Enhanced cell tracking with configurable parameters."""

    def __init__(self, config: 'AnalysisConfig'):
        self.config = config
        self.params = TrackingParameters()

    def update_parameters(self, new_params: TrackingParameters):
        """Update tracking parameters"""
        self.params = new_params

    def track_cells(self, segmentation_stack: np.ndarray) -> np.ndarray:
        """Track cells across frames using current parameters."""
        logger.info("Starting cell tracking with parameters: "
                    f"overlap_ratio={self.params.min_overlap_ratio}, "
                    f"max_displacement={self.params.max_displacement}, "
                    f"min_cell_size={self.params.min_cell_size}")

        if len(segmentation_stack.shape) != 3:
            raise ValueError("Expected 3D stack (t, x, y)")

        tracked_segmentation = np.zeros_like(segmentation_stack)
        all_used_ids: Set[int] = set()

        # Filter small cells only if min_cell_size > 0
        if self.params.min_cell_size > 0:
            for t in range(len(segmentation_stack)):
                regions = regionprops(segmentation_stack[t])
                for region in regions:
                    if region.area < self.params.min_cell_size:
                        segmentation_stack[t][segmentation_stack[t] == region.label] = 0

        # Process first frame
        first_frame = segmentation_stack[0]
        regions = regionprops(first_frame)
        sorted_regions = sorted(regions, key=lambda r: np.linalg.norm(r.centroid))

        # Initialize with sequential IDs
        id_mapping = {}
        for idx, region in enumerate(sorted_regions, start=1):
            id_mapping[region.label] = idx
            all_used_ids.add(idx)

        # Apply initial mapping
        tracked_segmentation[0] = first_frame.copy()
        for old_id, new_id in id_mapping.items():
            tracked_segmentation[0][first_frame == old_id] = new_id

        # Process subsequent frames
        for t in range(len(segmentation_stack) - 1):
            current_frame = tracked_segmentation[t]
            next_frame = segmentation_stack[t + 1]

            # Calculate overlaps with consideration for max displacement
            overlap_matrix = self._calculate_overlap_matrix(
                current_frame,
                next_frame,
                max_displacement=self.params.max_displacement,
                min_overlap_ratio=self.params.min_overlap_ratio
            )

            tracked_segmentation[t + 1] = np.zeros_like(next_frame)
            assigned_ids: Set[int] = set()

            # Handle gap closing if enabled
            if self.params.enable_gap_closing and t >= self.params.max_frame_gap:
                self._handle_gap_closing(
                    tracked_segmentation,
                    t,
                    overlap_matrix,
                    assigned_ids
                )

            # Process remaining regions
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
            max_displacement: float,
            min_overlap_ratio: float
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        """
        Calculate overlap between cells with combined displacement and overlap scoring.
        Returns dictionary mapping current cell IDs to list of (next_id, overlap, score) tuples.
        """
        from collections import defaultdict
        import logging
        logger = logging.getLogger(__name__)

        overlap_matrix = defaultdict(list)
        current_regions = regionprops(current_frame)
        next_regions = regionprops(next_frame)

        logger.debug(f"Processing frame with {len(current_regions)} current regions and {len(next_regions)} next regions")

        # Build spatial lookup for efficiency
        next_centroids = {r.label: np.array(r.centroid) for r in next_regions}
        next_areas = {r.label: r.area for r in next_regions}

        for current_region in current_regions:
            current_id = current_region.label
            current_centroid = np.array(current_region.centroid)
            current_area = current_region.area

            logger.debug(f"Processing current_id {current_id}")
            candidates = []

            for next_id, next_centroid in next_centroids.items():
                displacement = np.linalg.norm(current_centroid - next_centroid)

                # Only consider cells within maximum displacement
                if displacement <= max_displacement:
                    # Calculate overlap
                    coords = current_region.coords
                    overlap = np.sum(next_frame[coords[:, 0], coords[:, 1]] == next_id)

                    # Calculate overlap ratios in both directions
                    overlap_ratio_current = overlap / current_area if current_area > 0 else 0
                    overlap_ratio_next = overlap / next_areas[next_id] if next_areas[next_id] > 0 else 0

                    # Use minimum of both overlap ratios
                    effective_overlap = min(overlap_ratio_current, overlap_ratio_next)

                    if effective_overlap > 0:  # Only consider cells with any overlap
                        # Calculate combined score:
                        overlap_score = effective_overlap
                        displacement_score = 1.0 - (displacement / max_displacement)
                        combined_score = (0.7 * overlap_score + 0.3 * displacement_score)

                        if combined_score >= min_overlap_ratio:
                            candidates.append((next_id, overlap, combined_score))
                            logger.debug(f"Added candidate {next_id} with score {combined_score}")

            if candidates:
                candidates.sort(key=lambda x: x[2], reverse=True)
                overlap_matrix[current_id] = candidates
                logger.debug(f"Added {len(candidates)} candidates for current_id {current_id}")

        return overlap_matrix

    def _process_frame_regions(
            self,
            current_frame: np.ndarray,
            output_frame: np.ndarray,
            overlap_matrix: Dict[int, List[Tuple[int, int, float]]],
            assigned_ids: Set[int],
            all_used_ids: Set[int]
    ) -> None:
        """Process regions using combined scoring approach."""
        import logging
        logger = logging.getLogger(__name__)

        # Get all regions from current frame
        next_labels = np.unique(current_frame)
        next_labels = next_labels[next_labels > 0]  # Remove background
        logger.debug(f"Processing frame with {len(next_labels)} regions")

        # First handle all matched cells
        processed_ids = set()

        for current_id, candidates in overlap_matrix.items():
            logger.debug(f"Processing current_id {current_id} with {len(candidates)} candidates")

            next_label = None
            best_candidate = None

            for candidate in candidates:
                if candidate[0] not in assigned_ids:
                    next_label = candidate[0]
                    best_candidate = candidate
                    break

            if next_label is not None and best_candidate is not None:
                logger.debug(f"Assigning current_id {current_id} to next_label {next_label}")
                output_frame[current_frame == next_label] = current_id
                assigned_ids.add(next_label)
                processed_ids.add(current_id)

        # Then handle any unmatched cells
        for label in next_labels:
            if label not in assigned_ids:
                new_id = max(all_used_ids) + 1 if all_used_ids else 1
                logger.debug(f"Assigning new_id {new_id} to unmatched label {label}")
                output_frame[current_frame == label] = new_id
                all_used_ids.add(new_id)
                assigned_ids.add(label)

        logger.debug(f"Processed {len(processed_ids)} matched cells and {len(next_labels) - len(processed_ids)} unmatched cells")

    def _handle_gap_closing(
            self,
            tracked_segmentation: np.ndarray,
            current_frame: int,
            overlap_matrix: Dict[int, List[Tuple[int, int]]],
            assigned_ids: Set[int]
    ) -> None:
        """Handle tracking across gaps in segmentation."""
        # Look back through previous frames
        for gap in range(1, self.params.max_frame_gap + 1):
            prev_frame = current_frame - gap
            if prev_frame < 0:
                continue

            # Find disappearing cells
            disappearing_cells = set(np.unique(tracked_segmentation[prev_frame])) - \
                                 set(np.unique(tracked_segmentation[current_frame]))
            disappearing_cells.discard(0)  # Remove background

            for cell_id in disappearing_cells:
                if cell_id in assigned_ids:
                    continue

                # Find best match based on position and size
                prev_region = None
                for region in regionprops(tracked_segmentation[prev_frame]):
                    if region.label == cell_id:
                        prev_region = region
                        break

                if prev_region is None:
                    continue

                prev_centroid = np.array(prev_region.centroid)
                prev_area = prev_region.area

                best_match = None
                best_score = float('inf')

                # Check unassigned regions in current frame
                for region in regionprops(tracked_segmentation[current_frame]):
                    if region.label not in assigned_ids:
                        current_centroid = np.array(region.centroid)
                        displacement = np.linalg.norm(current_centroid - prev_centroid)

                        if displacement <= self.params.max_displacement * gap:
                            # Score based on displacement and size difference
                            size_diff = abs(region.area - prev_area) / prev_area
                            score = displacement + size_diff * 100  # Weight size difference more heavily

                            if score < best_score:
                                best_score = score
                                best_match = region.label

                if best_match is not None:
                    overlap_matrix[cell_id].append((best_match, prev_area))

