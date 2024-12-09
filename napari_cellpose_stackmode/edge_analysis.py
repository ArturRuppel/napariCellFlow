import cv2
import numpy as np
import networkx as nx
import logging
from skimage.morphology import skeletonize
from typing import Dict, List, Set, Tuple, Optional, Union, FrozenSet
from collections import defaultdict
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist

from .structure import CellBoundary, EdgeData, EdgeAnalysisResults, EdgeAnalysisParams, IntercalationEvent

logger = logging.getLogger(__name__)


@dataclass
class EdgeGroup:
    """Tracks a group of related edges through intercalations"""
    group_id: int
    edge_ids: Set[int] = field(default_factory=set)
    cell_pairs: Set[FrozenSet[int]] = field(default_factory=set)
    frames: List[int] = field(default_factory=list)
    active: bool = True


class EdgeAnalyzer:
    """Unified analyzer for edge dynamics and intercalations"""

    def __init__(self, params: Optional[EdgeAnalysisParams] = None):
        self.params = params or EdgeAnalysisParams()
        self.next_edge_id = 1
        self.next_group_id = 1
        self._edge_history = {}
        self._edge_groups = {}
        self._cell_pair_to_edge_id = {}
        self._edge_to_group = {}
        self._frame_graphs = {}

    def _normalize_cell_pair(self, cells: Union[Tuple[int, int], List[int], np.ndarray]) -> Tuple[Union[FrozenSet[int], Tuple[int, int]], Tuple[int, int]]:
        """
        Convert cell pair to both frozenset and ordered tuple representations

        Returns:
            Tuple containing (frozenset_representation, ordered_tuple_representation)
        """
        if isinstance(cells, (tuple, list, np.ndarray)) and len(cells) == 2:
            # Create ordered tuple (for visualization)
            ordered = tuple(sorted(int(c) for c in cells))
            # Create frozenset (for tracking)
            frozen = frozenset(ordered)
            return frozen, ordered
        else:
            raise ValueError("Cell pair must be a tuple/list/array of exactly 2 cells")

    def _create_edge_data(self, edge_id: int, frame: int, boundary: CellBoundary) -> EdgeData:
        """Initialize new edge tracking data"""
        _, ordered_pair = self._normalize_cell_pair(boundary.cell_ids)
        return EdgeData(
            edge_id=edge_id,
            frames=[frame],
            cell_pairs=[ordered_pair],  # Store ordered tuple
            lengths=[boundary.length],
            coordinates=[boundary.coordinates]
        )

    def _update_edge_data(self, edge_data: EdgeData, frame: int, boundary: CellBoundary) -> None:
        """Add new frame data to existing edge"""
        _, ordered_pair = self._normalize_cell_pair(boundary.cell_ids)
        edge_data.frames.append(frame)
        edge_data.cell_pairs.append(ordered_pair)  # Store ordered tuple
        edge_data.lengths.append(boundary.length)
        edge_data.coordinates.append(boundary.coordinates)

    def _update_edge_histories(self, frame: int, boundaries: List[CellBoundary]):
        """Update edge tracking histories"""
        for boundary in boundaries:
            frozen_pair, _ = self._normalize_cell_pair(boundary.cell_ids)
            edge_id = self._get_or_create_edge_id(frozen_pair)

            if edge_id in self._edge_history:
                self._update_edge_data(self._edge_history[edge_id], frame, boundary)
            else:
                self._edge_history[edge_id] = self._create_edge_data(edge_id, frame, boundary)

    def _create_frame_graph(self, boundaries: List[CellBoundary]) -> nx.Graph:
        """Create NetworkX graph representing cell connectivity"""
        G = nx.Graph()
        for boundary in boundaries:
            _, ordered_pair = self._normalize_cell_pair(boundary.cell_ids)
            G.add_edge(*ordered_pair, boundary=boundary)
        return G

    def _get_or_create_edge_id(self, frozen_pair: FrozenSet[int]) -> int:
        """Get existing edge ID or create new one"""
        if frozen_pair in self._cell_pair_to_edge_id:
            return self._cell_pair_to_edge_id[frozen_pair]

        edge_id = self.next_edge_id
        self.next_edge_id += 1
        self._cell_pair_to_edge_id[frozen_pair] = edge_id
        return edge_id

    def _validate_t1_transition(
            self,
            lost_edge: FrozenSet[int],
            gained_edge: FrozenSet[int],
            G1: nx.Graph,
            G2: nx.Graph
    ) -> bool:
        """Validate that edge changes represent a true T1 transition"""
        # Check for four unique cells
        all_cells = lost_edge | gained_edge
        if len(all_cells) != 4:
            return False

        # Convert to sorted tuples for consistent comparison
        lost_cells = tuple(sorted(lost_edge))
        gained_cells = tuple(sorted(gained_edge))

        # Check if required connecting edges exist
        edges_to_check = [
            tuple(sorted((lost_cells[0], gained_cells[0]))),
            tuple(sorted((lost_cells[0], gained_cells[1]))),
            tuple(sorted((lost_cells[1], gained_cells[0]))),
            tuple(sorted((lost_cells[1], gained_cells[1])))
        ]

        edges_t = set(tuple(sorted(e)) for e in G1.edges())
        edges_t_plus_1 = set(tuple(sorted(e)) for e in G2.edges())

        # Check if connecting edges exist in both frames
        # Note: The exact edges that should be present depends on the transition timing
        connecting_edges_exist = all(
            e in edges_t or e in edges_t_plus_1
            for e in edges_to_check
            if e != lost_cells and e != gained_cells
        )

        return connecting_edges_exist

    def _create_edge_trajectories(self, boundaries_by_frame: Dict[int, List[CellBoundary]],
                                  intercalations: List[IntercalationEvent]) -> Dict[int, EdgeData]:
        logger.debug(f"Processing {len(intercalations)} total intercalation events")
        trajectories = {}

        logger.debug(f"Number of edge groups: {len(self._edge_groups)}")
        logger.debug(f"All intercalation events: {[(e.frame, e.losing_cells, e.gaining_cells) for e in intercalations]}")

        # Process each edge group
        for group in self._edge_groups.values():
            if not group.active:
                logger.debug(f"Skipping inactive group {group.group_id}")
                continue

            logger.debug(f"Processing group {group.group_id} with {len(group.edge_ids)} edges")
            logger.debug(f"Group {group.group_id} cell pairs: {group.cell_pairs}")

            # Combine all edge data in group
            combined_data = None
            for edge_id in group.edge_ids:
                logger.debug(f"Processing edge {edge_id} in group {group.group_id}")
                if edge_id in self._edge_history:
                    if combined_data is None:
                        base = self._edge_history[edge_id]
                        combined_data = EdgeData(
                            edge_id=base.edge_id,
                            frames=base.frames.copy(),
                            cell_pairs=base.cell_pairs.copy(),
                            lengths=base.lengths.copy(),
                            coordinates=base.coordinates.copy()
                        )
                        logger.debug(f"Created initial combined data from edge {edge_id} with frames {base.frames}")
                    else:
                        self._merge_edge_data(combined_data, self._edge_history[edge_id])
                        logger.debug(f"Merged data from edge {edge_id}")
                else:
                    logger.debug(f"Edge {edge_id} not found in edge history")

            if combined_data is not None:
                # Add intercalation information
                frame_set = set(combined_data.frames)
                relevant_events = []

                for event in intercalations:
                    logger.debug(f"Checking event at frame {event.frame} with cells "
                                 f"losing: {event.losing_cells}, gaining: {event.gaining_cells}")

                    # Convert the cell pairs to normalized form
                    losing_pair = self._normalize_cell_pair(event.losing_cells)[0]
                    gaining_pair = self._normalize_cell_pair(event.gaining_cells)[0]

                    logger.debug(f"Normalized pairs - losing: {losing_pair}, gaining: {gaining_pair}")
                    logger.debug(f"Group cell pairs: {group.cell_pairs}")

                    # Check if event belongs to this group
                    if losing_pair in group.cell_pairs or gaining_pair in group.cell_pairs:
                        logger.debug(f"Found matching intercalation for group {group.group_id}")
                        relevant_events.append(event)

                if relevant_events:
                    logger.debug(f"Adding {len(relevant_events)} intercalations to edge {combined_data.edge_id}")
                    combined_data.intercalations = relevant_events

                trajectories[combined_data.edge_id] = combined_data
                logger.debug(f"Added trajectory for edge {combined_data.edge_id}")

        # Add edges not in any group
        logger.debug("Processing edges not in groups")
        ungrouped_count = 0
        for edge_id, data in self._edge_history.items():
            if edge_id not in self._edge_to_group:
                logger.debug(f"Adding ungrouped edge {edge_id}")
                trajectories[edge_id] = EdgeData(
                    edge_id=data.edge_id,
                    frames=data.frames.copy(),
                    cell_pairs=data.cell_pairs.copy(),
                    lengths=data.lengths.copy(),
                    coordinates=data.coordinates.copy()
                )
                ungrouped_count += 1

        logger.debug(f"Added {ungrouped_count} ungrouped edges")

        # Final counts
        edges_with_intercalations = sum(1 for edge in trajectories.values() if hasattr(edge, 'intercalations') and edge.intercalations)
        total_intercalations = sum(len(edge.intercalations) for edge in trajectories.values()
                                   if hasattr(edge, 'intercalations') and edge.intercalations)

        logger.debug(f"Created {len(trajectories)} total trajectories")
        logger.debug(f"Final trajectories contain {edges_with_intercalations} edges with intercalations")
        logger.debug(f"Total intercalation events in trajectories: {total_intercalations}")

        return trajectories

    def _merge_edge_groups(self, lost_edge: FrozenSet[int], gained_edge: FrozenSet[int], frame: int):
        # Find all groups that contain either edge or any historically related edges
        related_groups = set()
        for group in self._edge_groups.values():
            if lost_edge in group.cell_pairs or gained_edge in group.cell_pairs:
                related_groups.add(group.group_id)
                # Also add groups containing any historically related edges
                for cell_pair in group.cell_pairs:
                    for other_group in self._edge_groups.values():
                        if cell_pair in other_group.cell_pairs:
                            related_groups.add(other_group.group_id)

        if not related_groups:
            # Create new group if no existing groups found
            group = EdgeGroup(self.next_group_id)
            self.next_group_id += 1
            group.edge_ids.update({self._cell_pair_to_edge_id[lost_edge],
                                   self._cell_pair_to_edge_id[gained_edge]})
            group.cell_pairs.update({lost_edge, gained_edge})
            group.frames.append(frame)
            self._edge_groups[group.group_id] = group
        else:
            # Merge all related groups into the first one
            base_group_id = min(related_groups)
            merged = EdgeGroup(base_group_id)

            # Combine all data from related groups
            for group_id in related_groups:
                group = self._edge_groups[group_id]
                merged.edge_ids.update(group.edge_ids)
                merged.cell_pairs.update(group.cell_pairs)
                merged.frames.extend(group.frames)
                if group_id != base_group_id:
                    del self._edge_groups[group_id]

            # Add new edges
            merged.edge_ids.update({self._cell_pair_to_edge_id[lost_edge],
                                    self._cell_pair_to_edge_id[gained_edge]})
            merged.cell_pairs.update({lost_edge, gained_edge})
            merged.frames.append(frame)
            merged.frames = sorted(set(merged.frames))

            self._edge_groups[base_group_id] = merged

            # Update all edge to group mappings
            for edge_id in merged.edge_ids:
                self._edge_to_group[edge_id] = base_group_id
    def _detect_edges(self, frame_data: np.ndarray) -> List[CellBoundary]:
        """Detect edges in a single frame"""
        boundaries = []
        cell_ids = np.unique(frame_data)
        cell_ids = cell_ids[cell_ids != 0]  # Remove background

        # Detect edges between all cell pairs
        for i, cell1 in enumerate(cell_ids):
            for cell2 in cell_ids[i + 1:]:
                boundary = self._find_shared_boundary(frame_data, cell1, cell2)
                if boundary is not None:
                    boundaries.append(boundary)

        # Filter isolated edges if enabled
        if self.params.filter_isolated:
            boundaries = self._filter_isolated_edges(boundaries)

        return boundaries

    def _merge_edge_data(self, base: EdgeData, other: EdgeData):
        """Merge two EdgeData objects"""
        # Combine and sort by frame
        frames = base.frames + other.frames
        lengths = base.lengths + other.lengths
        cell_pairs = base.cell_pairs + other.cell_pairs
        coordinates = base.coordinates + other.coordinates

        # Sort everything by frame
        sorted_indices = np.argsort(frames)
        base.frames = [frames[i] for i in sorted_indices]
        base.lengths = [lengths[i] for i in sorted_indices]
        base.cell_pairs = [cell_pairs[i] for i in sorted_indices]
        base.coordinates = [coordinates[i] for i in sorted_indices]

    def process_frame(self, frame_number: int, frame_data: np.ndarray) -> List[CellBoundary]:
        """Process a single frame to detect edges"""
        boundaries = []
        cell_ids = np.unique(frame_data)
        cell_ids = cell_ids[cell_ids != 0]  # Remove background

        # Detect edges between all cell pairs
        for i, cell1 in enumerate(cell_ids):
            for cell2 in cell_ids[i + 1:]:
                boundary = self._find_shared_boundary(frame_data, cell1, cell2)
                if boundary is not None:
                    boundaries.append(boundary)

        # Filter isolated edges if enabled
        if self.params.filter_isolated:
            boundaries = self._filter_isolated_edges(boundaries)

        # Track edges and update history
        current_edges = {self._normalize_cell_pair(b.cell_ids): b for b in boundaries}

        if frame_number > 0:
            prev_edges = set(self._cell_pair_to_edge_id.keys())
            new_edges = set(current_edges.keys()) - prev_edges

            # Assign new edge IDs
            for cell_pair in new_edges:
                self._cell_pair_to_edge_id[cell_pair] = self.next_edge_id
                self.next_edge_id += 1

        # Update edge history
        for boundary in boundaries:
            cell_pair = self._normalize_cell_pair(boundary.cell_ids)
            edge_id = self._cell_pair_to_edge_id.get(cell_pair)

            if edge_id is None:
                edge_id = self.next_edge_id
                self.next_edge_id += 1
                self._cell_pair_to_edge_id[cell_pair] = edge_id

            if edge_id in self._edge_history:
                self._update_edge_data(self._edge_history[edge_id], frame_number, boundary)
            else:
                self._edge_history[edge_id] = self._create_edge_data(edge_id, frame_number, boundary)

        return boundaries

    def _filter_isolated_edges(self, boundaries: List[CellBoundary]) -> List[CellBoundary]:
        """Filter out edges that don't connect to others"""
        # Build connectivity graph
        connections = defaultdict(set)
        for boundary in boundaries:
            cell1, cell2 = boundary.cell_ids
            connections[int(cell1)].add(int(cell2))
            connections[int(cell2)].add(int(cell1))

        # Keep only edges where both cells have multiple connections
        return [b for b in boundaries
                if len(connections[int(b.cell_ids[0])]) > 1
                and len(connections[int(b.cell_ids[1])]) > 1]

    def update_parameters(self, params: EdgeAnalysisParams) -> None:
        """Update analysis parameters and reset state"""
        self.params = params
        # Clear all internal state
        self._edge_history.clear()
        self._edge_groups.clear()
        self._cell_pair_to_edge_id.clear()
        self._edge_to_group.clear()
        self._frame_graphs.clear()
        self.next_edge_id = 1
        self.next_group_id = 1

    def _find_shared_boundary(self, frame: np.ndarray, cell1_id: int, cell2_id: int) -> Optional[CellBoundary]:
        """Find the shared boundary between two cells"""
        # Create cell masks
        mask1 = (frame == cell1_id).astype(np.uint8)
        mask2 = (frame == cell2_id).astype(np.uint8)

        # Create structuring element for dilation
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * self.params.dilation_radius + 1, 2 * self.params.dilation_radius + 1)
        )

        # Dilate cell masks
        dilated1 = cv2.dilate(mask1, kernel)
        dilated2 = cv2.dilate(mask2, kernel)

        # Find overlap region
        overlap = dilated1 & dilated2

        if not np.any(overlap):
            return None

        # Create thin dilation for precise boundary detection
        thin_kernel = np.ones((3, 3), np.uint8)
        thin_dilated1 = cv2.dilate(mask1, thin_kernel)
        thin_dilated2 = cv2.dilate(mask2, thin_kernel)
        boundary_region = thin_dilated1 & thin_dilated2

        if np.sum(boundary_region) < self.params.min_overlap_pixels:
            return None

        # Skeletonize the boundary
        skeleton = skeletonize(boundary_region)
        if not np.any(skeleton):
            return None

        # Order boundary pixels
        ordered_coords = self._order_boundary_pixels(skeleton)
        if len(ordered_coords) < 2:
            return None

        # Calculate boundary length
        length = self._calculate_edge_length(ordered_coords)

        # Apply minimum length filter if set
        if self.params.min_edge_length > 0 and length < self.params.min_edge_length:
            return None

        # Create boundary object
        boundary = CellBoundary(
            cell_ids=(cell1_id, cell2_id),
            coordinates=ordered_coords,
            endpoint1=ordered_coords[0],
            endpoint2=ordered_coords[-1],
            length=length
        )

        return boundary

    def _order_boundary_pixels(self, skeleton: np.ndarray) -> np.ndarray:
        """Order pixels along the boundary from one endpoint to another"""
        points = np.column_stack(np.where(skeleton))
        if len(points) <= 2:
            return points

        # Find endpoints
        endpoint_indices = []
        for i, point in enumerate(points):
            y, x = point
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if skeleton[ny, nx]:
                            neighbors += 1
            if neighbors == 1:
                endpoint_indices.append(i)

        if len(endpoint_indices) != 2:
            # If we don't have exactly 2 endpoints, use the points furthest apart
            distances = cdist(points, points)
            i, j = np.unravel_index(np.argmax(distances), distances.shape)
            endpoint_indices = [i, j]

        # Order points from one endpoint to the other
        ordered = [points[endpoint_indices[0]]]
        remaining = list(range(len(points)))
        remaining.remove(endpoint_indices[0])

        while remaining:
            current = ordered[-1]
            min_dist = float('inf')
            next_idx = None

            for idx in remaining:
                dist = np.sum((points[idx] - current) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    next_idx = idx

            if next_idx is None:
                break

            ordered.append(points[next_idx])
            remaining.remove(next_idx)

        return np.array(ordered)

    def _calculate_edge_length(self, coords: np.ndarray) -> float:
        """Calculate physical length of edge"""
        if len(coords) < 2:
            return 0.0
        diffs = np.diff(coords, axis=0)
        segments = np.sqrt(np.sum(diffs ** 2, axis=1))
        return float(np.sum(segments))

    def _detect_topology_changes(self, frame: int, next_frame: int) -> List[IntercalationEvent]:
        """Detect T1 transitions between consecutive frames using stable detection logic"""
        G1 = self._frame_graphs[frame]
        G2 = self._frame_graphs[next_frame]

        # Convert edges to sorted tuples for consistent comparison
        edges_t = set(tuple(sorted((u, v))) for u, v in G1.edges())
        edges_t_plus_1 = set(tuple(sorted((u, v))) for u, v in G2.edges())

        removed_edges = edges_t - edges_t_plus_1
        added_edges = edges_t_plus_1 - edges_t

        logger.debug(f"Removed edges: {removed_edges}")
        logger.debug(f"Added edges: {added_edges}")

        events = []
        for removed_edge in removed_edges:
            for added_edge in added_edges:
                u1, v1 = removed_edge  # Lost edge cells
                u2, v2 = added_edge  # Gained edge cells

                # Validate the T1 transition
                if self._validate_t1_transition(
                        frozenset(removed_edge),
                        frozenset(added_edge),
                        G1,
                        G2
                ):
                    # Get coordinates from the boundary data
                    boundary = G1.edges[removed_edge]['boundary']
                    coords = np.mean(boundary.coordinates, axis=0)

                    # Create frozenset representations
                    lost_pair = frozenset(removed_edge)
                    gained_pair = frozenset(added_edge)

                    # Ensure both edges have IDs
                    if gained_pair not in self._cell_pair_to_edge_id:
                        # Assign new ID for gained edge
                        self._cell_pair_to_edge_id[gained_pair] = self.next_edge_id
                        self.next_edge_id += 1

                    # Now we can safely merge the groups
                    self._merge_edge_groups(lost_pair, gained_pair, frame)

                    # Create event
                    event = IntercalationEvent(
                        frame=frame,
                        losing_cells=removed_edge,
                        gaining_cells=added_edge,
                        coordinates=coords
                    )
                    events.append(event)
                    logger.debug(f"Found intercalation at frame {frame}: {removed_edge} → {added_edge}")

        logger.debug(f"Found {len(events)} events between frames {frame} and {next_frame}")
        return events
    def analyze_sequence(self, tracked_data: np.ndarray, progress_callback=None) -> EdgeAnalysisResults:
        """Process complete sequence of frames"""
        logger.info("Starting edge analysis sequence...")

        # Reset state
        self._edge_history.clear()
        self._edge_groups.clear()
        self._cell_pair_to_edge_id.clear()
        self._edge_to_group.clear()
        self._frame_graphs.clear()
        self.next_edge_id = 1
        self.next_group_id = 1

        # Create results container
        results = EdgeAnalysisResults(self.params)
        results.update_metadata('total_frames', len(tracked_data))
        results.update_metadata('frame_ids', list(range(len(tracked_data))))

        # Process each frame
        boundaries_by_frame = {}
        all_intercalations = []


        total_frames = len(tracked_data)
        for frame_num in range(total_frames):
            if progress_callback:
                progress_callback(int(100 * frame_num / total_frames))

        for frame_num in range(len(tracked_data)):
            logger.debug(f"Processing frame {frame_num}")

            # Detect edges
            boundaries = self._detect_edges(tracked_data[frame_num])
            boundaries_by_frame[frame_num] = boundaries

            # Create frame graph
            self._frame_graphs[frame_num] = self._create_frame_graph(boundaries)

            # Detect intercalations with previous frame
            if frame_num > 0:
                events = self._detect_topology_changes(frame_num - 1, frame_num)
                all_intercalations.extend(events)

            # Update edge histories
            self._update_edge_histories(frame_num, boundaries)

        logger.info(f"Found total of {len(all_intercalations)} intercalations")

        # Process edge groups and create final trajectories
        edge_trajectories = self._create_edge_trajectories(boundaries_by_frame, all_intercalations)

        # Add trajectories to results
        results.edges = edge_trajectories
        results.validate()

        return results


