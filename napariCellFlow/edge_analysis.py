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
    """Tracks a group of related edges through intercalations.

    This class maintains the relationships between edges that are connected through
    T1 transitions, allowing tracking of edge identity through topology changes.

    Attributes:
        group_id (int): Unique identifier for this edge group
        edge_ids (Set[int]): Set of edge IDs that belong to this group
        cell_pairs (Set[FrozenSet[int]]): Set of cell pairs involved in these edges
        frames (List[int]): List of frames where edges in this group appear
        active (bool): Flag indicating if group is still active in tracking
    """
    group_id: int
    edge_ids: Set[int] = field(default_factory=set)
    cell_pairs: Set[FrozenSet[int]] = field(default_factory=set)
    frames: List[int] = field(default_factory=list)
    active: bool = True


class EdgeAnalyzer:
    """Unified analyzer for edge dynamics and intercalations in cell tissues.

    This class provides comprehensive analysis of cell boundaries and topology
    changes in segmented microscopy data. It can detect and track edges between
    cells, identify T1 transitions, and maintain edge identity through topology
    changes.

    The analyzer uses a combination of image processing and graph theory approaches
    to robustly track edge dynamics and detect intercalation events.

    Attributes:
        params (EdgeAnalysisParams): Configuration parameters for analysis
        next_edge_id (int): Counter for generating unique edge IDs
        next_group_id (int): Counter for generating unique group IDs
        _edge_history (dict): Internal tracking of edge data
        _edge_groups (dict): Internal tracking of edge groups
        _cell_pair_to_edge_id (dict): Mapping of cell pairs to edge IDs
        _edge_to_group (dict): Mapping of edges to their groups
        _frame_graphs (dict): NetworkX graphs for each frame
    """

    def __init__(self, params: Optional[EdgeAnalysisParams] = None):
        """Initialize edge analyzer with given parameters.

        Args:
            params: Configuration parameters for analysis. If None, uses defaults.
        """
        self.params = params or EdgeAnalysisParams()
        self.next_edge_id = 1
        self.next_group_id = 1
        self._edge_history = {}
        self._edge_groups = {}
        self._cell_pair_to_edge_id = {}
        self._edge_to_group = {}
        self._frame_graphs = {}

    def _normalize_cell_pair(self, cells: Union[Tuple[int, int], List[int], np.ndarray]) -> Tuple[Union[FrozenSet[int], Tuple[int, int]], Tuple[int, int]]:
        """Convert cell pair to both frozenset and ordered tuple representations.

        Creates standardized representations of cell pairs for both tracking (using
        frozenset) and visualization (using ordered tuple).

        Args:
            cells: Cell pair in any supported format (tuple, list, or array)

        Returns:
            Tuple containing (frozenset_representation, ordered_tuple_representation)

        Raises:
            ValueError: If input is not exactly 2 cells
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
        """Initialize new edge tracking data.

        Args:
            edge_id: Unique identifier for the edge
            frame: Frame number where edge first appears
            boundary: Detected boundary between cells

        Returns:
            New EdgeData object with initial tracking information
        """
        _, ordered_pair = self._normalize_cell_pair(boundary.cell_ids)
        return EdgeData(
            edge_id=edge_id,
            frames=[frame],
            cell_pairs=[ordered_pair],  # Store ordered tuple
            lengths=[boundary.length],
            coordinates=[boundary.coordinates]
        )

    def _update_edge_data(self, edge_data: EdgeData, frame: int, boundary: CellBoundary) -> None:
        """Add new frame data to existing edge tracking information.

        Args:
            edge_data: Existing tracking data to update
            frame: Current frame number
            boundary: New boundary information
        """
        _, ordered_pair = self._normalize_cell_pair(boundary.cell_ids)
        edge_data.frames.append(frame)
        edge_data.cell_pairs.append(ordered_pair)  # Store ordered tuple
        edge_data.lengths.append(boundary.length)
        edge_data.coordinates.append(boundary.coordinates)

    def _update_edge_histories(self, frame: int, boundaries: List[CellBoundary]):
        """Update edge tracking histories with new frame data.

        Processes all boundaries in a frame to maintain edge tracking information.
        Creates new tracking entries for new edges and updates existing ones.

        Args:
            frame: Current frame number
            boundaries: List of detected boundaries in the frame
        """
        for boundary in boundaries:
            frozen_pair, _ = self._normalize_cell_pair(boundary.cell_ids)
            edge_id = self._get_or_create_edge_id(frozen_pair)

            if edge_id in self._edge_history:
                self._update_edge_data(self._edge_history[edge_id], frame, boundary)
            else:
                self._edge_history[edge_id] = self._create_edge_data(edge_id, frame, boundary)

    def _create_frame_graph(self, boundaries: List[CellBoundary]) -> nx.Graph:
        """Create NetworkX graph representing cell connectivity in a frame.

        Args:
            boundaries: List of detected boundaries

        Returns:
            NetworkX graph where nodes are cells and edges represent boundaries
        """
        G = nx.Graph()
        for boundary in boundaries:
            _, ordered_pair = self._normalize_cell_pair(boundary.cell_ids)
            G.add_edge(*ordered_pair, boundary=boundary)
        return G

    def _get_or_create_edge_id(self, frozen_pair: FrozenSet[int]) -> int:
        """Get existing edge ID or create new one for a cell pair.

        Args:
            frozen_pair: Frozenset representation of cell pair

        Returns:
            Edge ID (existing or new)
        """
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
        """Get existing edge ID or create new one for a cell pair.

        Args:
            frozen_pair: Frozenset representation of cell pair

        Returns:
            Edge ID (existing or new)
        """
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

        # Changed this part: now check that connecting edges exist in BOTH frames
        connecting_edges_exist = all(
            (e in edges_t and e in edges_t_plus_1)
            for e in edges_to_check
            if e != lost_cells and e != gained_cells
        )

        return connecting_edges_exist

    def _create_edge_trajectories(self, boundaries_by_frame: Dict[int, List[CellBoundary]],
                                  intercalations: List[IntercalationEvent]) -> Dict[int, EdgeData]:
        """Create edge trajectories with forward-time merging logic.

        Processes boundaries and intercalations to create complete edge trajectories
        that maintain identity through topology changes.

        Args:
            boundaries_by_frame: Dictionary mapping frames to detected boundaries
            intercalations: List of detected intercalation events

        Returns:
            Dictionary mapping edge IDs to EdgeData objects
        """
        logger.info("Creating edge trajectories with forward-time merging...")

        # Sort intercalations chronologically
        sorted_events = sorted(intercalations, key=lambda x: x.frame)

        # Track edge chains: maps cell pair to current trajectory ID
        edge_trajectories = {}  # final container for trajectories
        cell_pair_to_trajectory = {}  # maps frozen cell pairs to trajectory IDs
        next_trajectory_id = 1

        # First pass: Process all intercalation events chronologically
        logger.debug(f"Processing {len(sorted_events)} intercalation events")
        for event in sorted_events:
            losing_pair = frozenset(event.losing_cells)
            gaining_pair = frozenset(event.gaining_cells)

            # Check if either edge is already part of a trajectory
            losing_traj_id = cell_pair_to_trajectory.get(losing_pair)

            if losing_traj_id is None:
                # Create new trajectory for the losing edge
                losing_traj_id = next_trajectory_id
                next_trajectory_id += 1
                edge_trajectories[losing_traj_id] = EdgeData(
                    edge_id=losing_traj_id,
                    frames=[],
                    cell_pairs=[],
                    lengths=[],
                    coordinates=[],
                    intercalations=[event]
                )
                cell_pair_to_trajectory[losing_pair] = losing_traj_id
            else:
                # Add event to existing trajectory
                edge_trajectories[losing_traj_id].intercalations.append(event)

            # Map gaining edge to same trajectory
            cell_pair_to_trajectory[gaining_pair] = losing_traj_id

            logger.debug(f"Merged trajectory {losing_traj_id}: {event.losing_cells} → {event.gaining_cells}")

        # Second pass: Process all boundaries chronologically
        logger.debug("Processing boundaries frame by frame")
        for frame, boundaries in sorted(boundaries_by_frame.items()):
            for boundary in boundaries:
                cell_pair = frozenset(boundary.cell_ids)

                # Get or create trajectory ID for this edge
                if cell_pair not in cell_pair_to_trajectory:
                    # This is a non-intercalating edge or first occurrence
                    traj_id = next_trajectory_id
                    next_trajectory_id += 1
                    cell_pair_to_trajectory[cell_pair] = traj_id
                    edge_trajectories[traj_id] = EdgeData(
                        edge_id=traj_id,
                        frames=[],
                        cell_pairs=[],
                        lengths=[],
                        coordinates=[],
                        intercalations=[]
                    )

                traj_id = cell_pair_to_trajectory[cell_pair]
                trajectory = edge_trajectories[traj_id]

                # Determine sign based on intercalation events
                sign = -1
                for event in trajectory.intercalations:
                    if frame > event.frame:
                        sign *= -1

                # Add boundary data to trajectory
                trajectory.frames.append(frame)
                trajectory.cell_pairs.append(tuple(sorted(boundary.cell_ids)))
                trajectory.lengths.append(sign * boundary.length)
                trajectory.coordinates.append(boundary.coordinates)

        # Count statistics
        intercalating_edges = sum(1 for t in edge_trajectories.values() if t.intercalations)
        total_intercalations = sum(len(t.intercalations) for t in edge_trajectories.values())

        logger.info(f"Created {len(edge_trajectories)} edge trajectories")
        logger.debug(f"Found {intercalating_edges} edges involved in intercalations")
        logger.debug(f"Total intercalation events processed: {total_intercalations}")

        return edge_trajectories

    def _merge_edge_groups(self, lost_edge: FrozenSet[int], gained_edge: FrozenSet[int], frame: int):
        """Merge edge groups involved in a topology change.

        Combines edge groups when edges are related through T1 transitions,
        maintaining continuity of edge identity through topology changes.

        Args:
            lost_edge: Cell pair that disappears in transition
            gained_edge: Cell pair that appears in transition
            frame: Frame number where transition occurs

        Note:
            Updates the internal group mappings and edge relationships.
            Creates new group if no existing groups are found.
        """
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
        """Detect cell boundaries in a single frame.

        Identifies and characterizes boundaries between all pairs of adjacent cells
        in the segmented frame.

        Args:
            frame_data: 2D numpy array of segmented cells where each cell has a unique ID

        Returns:
            List of CellBoundary objects representing detected boundaries

        Note:
            Applies filtering based on configuration parameters including:
            - Minimum overlap pixels
            - Minimum edge length
            - Isolation filtering (optional)
        """
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

    def _filter_isolated_edges(self, boundaries: List[CellBoundary]) -> List[CellBoundary]:
        """Filter out edges that don't form part of a connected network.

        Removes boundaries where either cell only has one connection, as these
        are likely artifacts or non-biological configurations.

        Args:
            boundaries: List of detected cell boundaries

        Returns:
            Filtered list of boundaries, excluding isolated edges
        """
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
        """Update analysis parameters and reset internal state.

        Args:
            params: New parameters to use for analysis

        Note:
            This resets all internal tracking state, requiring reanalysis
            of any sequence that needs to use the new parameters.
        """
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
        """Find and characterize the shared boundary between two cells.

        Uses image processing techniques to identify, measure, and validate
        the boundary between adjacent cells.

        Args:
            frame: 2D array of segmented cells
            cell1_id: ID of first cell
            cell2_id: ID of second cell

        Returns:
            CellBoundary object if valid boundary found, None otherwise

        Note:
            Applies multiple validation steps:
            1. Checks for sufficient overlap after dilation
            2. Validates boundary pixel count
            3. Applies minimum length threshold
            4. Ensures proper skeletonization
        """
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
        """Order pixels along the boundary from one endpoint to another.

        Creates an ordered sequence of coordinates representing the boundary
        between cells, ensuring proper connectivity and endpoint selection.

        Args:
            skeleton: Binary array representing skeletonized boundary

        Returns:
            Nx2 array of ordered (y,x) coordinates along the boundary

        Note:
            If exactly 2 endpoints aren't found, uses points furthest apart
            to ensure consistent ordering.
        """
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
        """Calculate the physical length of an edge.

        Computes the total Euclidean length of the boundary by summing
        the lengths of each segment between consecutive coordinates.

        Args:
            coords: Nx2 array of ordered (y,x) coordinates along the boundary

        Returns:
            Total length of the boundary in pixel units
        """
        if len(coords) < 2:
            return 0.0
        diffs = np.diff(coords, axis=0)
        segments = np.sqrt(np.sum(diffs ** 2, axis=1))
        return float(np.sum(segments))

    def _detect_topology_changes(self, frame: int, next_frame: int) -> List[IntercalationEvent]:
        """Detect T1 transitions between consecutive frames.

        Identifies and validates potential intercalation events by comparing
        cell connectivity graphs between frames.

        Args:
            frame: Index of first frame
            next_frame: Index of second frame

        Returns:
            List of validated IntercalationEvent objects

        Note:
            Applies rigorous validation to ensure detected events represent
            true T1 transitions rather than segmentation artifacts or other
            topology changes.
        """
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
        """Process and analyze a complete sequence of segmented frames.

        This is the main analysis entry point that processes a sequence of segmented
        cell frames to detect edges, track their evolution, and identify topology
        changes (T1 transitions/intercalations).

        Args:
            tracked_data: 3D numpy array (frames, height, width) containing segmented
                cell data where each cell has a unique integer ID. Background should
                be labeled as 0.
            progress_callback: Optional callback function(progress: int, message: str)
                to report analysis progress. Progress ranges from 0-100.

        Returns:
            EdgeAnalysisResults object containing:
                - Edge trajectories with lengths and coordinates
                - Detected intercalation events
                - Analysis metadata
                - Original segmentation data

        Raises:
            ValueError: If tracked_data is not 3D or contains invalid cell IDs
            TypeError: If tracked_data is not a numpy array

        Note:
            Processing steps:
            1. Reset internal tracking state
            2. Create edge graphs for each frame
            3. Detect edges and update histories
            4. Identify topology changes between consecutive frames
            5. Build edge trajectories with identity tracking through T1s
            6. Compile final results with metadata

        Example:
            >>> analyzer = EdgeAnalyzer()
            >>> data = load_segmented_sequence()  # (frames, height, width)
            >>> def progress(percent, message):
            ...     print(f"{percent}%: {message}")
            >>> results = analyzer.analyze_sequence(data, progress)
            >>> print(f"Found {len(results.edges)} edge trajectories")
        """
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

        # Store the segmentation data
        results.set_segmentation_data(tracked_data)

        # Process each frame
        boundaries_by_frame = {}
        all_intercalations = []

        total_frames = len(tracked_data)

        for frame_num in range(total_frames):
            # Calculate progress percentage using full range for frame processing
            if progress_callback:
                progress = int(100 * frame_num / total_frames)
                progress_callback(progress, f"Processing frame {frame_num + 1}/{total_frames}")

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

        if progress_callback:
            progress_callback(100, "Finalizing analysis...")

        logger.info(f"Found total of {len(all_intercalations)} intercalations")

        # Process edge groups and create final trajectories
        edge_trajectories = self._create_edge_trajectories(boundaries_by_frame, all_intercalations)

        # Add trajectories to results
        results.edges = edge_trajectories
        results.validate()

        return results