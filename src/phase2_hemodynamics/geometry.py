"""
Coronary Vessel Geometry Extraction and Physical Discretization Module.

This module processes 3D Computed Tomography Angiography (CTA) vessel mask
volumes and converts them into physical spatial domains for Physics-Informed
Neural Network (PINN) hemodynamic simulations.

Mathematical & Physical Principles
----------------------------------
1. Voxel to Physical Coordinate Transformation:
   Affine map mapping discrete grid indices i = (i, j, k)^T in Z^3 to continuous
   physical coordinates x = (x, y, z)^T in R^3 (in millimeters or meters):

       x_phys = R . (i * Delta_x) + x_origin

   where R in SO(3) is the directional orientation matrix, Delta_x is the
   anisotropic voxel spacing vector (dx, dy, dz), and x_origin is the physical
   spatial origin.

2. Physical Euclidean Distance Transform (EDT) & Signed Distance Field (SDF):
   Let Omega subset R^3 denote the arterial lumen domain and dOmega_wall denote
   the vessel wall boundary. The exact Euclidean distance transform phi(x) is:

       phi(x) = min_{y in dOmega_wall} ||x - y||_2

   The spatial gradient of the continuous distance field grad(phi(x)) defines
   the unit vector pointing towards the interior axis. The unit outward surface
   normal vector n(x) at any boundary point x in dOmega_wall is given by:

       n(x) = - grad(phi(x)) / ||grad(phi(x))||_2

3. Centerline Medial Axis & Graph Theory:
   The 1D medial skeleton M subset Omega is extracted using 3D topological thinning
   (Lee's thinning / parallel skeletonization). A spatial graph G = (V, E) is built
   where vertices V represent physical 3D skeleton coordinates and edges E represent
   26-connected spatial adjacency weighted by Euclidean distance.
   - Inlets / Outlets: Identified as terminal graph nodes with degree deg(v) = 1.
   - Bifurcations: Identified as branch junction nodes with degree deg(v) >= 3.

4. Murray's Law Branching Analysis:
   Physiological coronary bifurcation geometry follows Murray's Law, which minimizes
   the biological work required for blood transport and maintenance:

       r_parent^gamma = sum_{k=1}^N r_child_k^gamma

   where r denotes local inscribed vessel radius and gamma in [2.0, 3.0] is the
   physiological branching exponent. We fit the optimal empirical exponent gamma*
   via non-linear bounded optimization across all detected coronary bifurcations:

       gamma* = argmin_{gamma} sum_{b in Bifurcations} ( r_{0,b}^gamma - sum_k r_{k,b}^gamma )^2

5. Flow Domain Point Cloud Discretization:
   The spatial domain is sampled into PyTorch tensor point clouds:
   - Interior Collocation Points: Omega_int in R^(N_int x 3)
   - Wall Boundary Points: dOmega_wall in R^(N_wall x 3) with outward normals N_wall in R^(N_wall x 3)
   - Inlet Surface Points: dOmega_inlet in R^(N_inlet x 3) with inlet normals N_inlet in R^(N_inlet x 3)
   - Outlet Surface Points: dOmega_outlet in R^(N_outlet x 3) with outlet normals N_outlet in R^(N_outlet x 3)

Author: Computational Scientist
Language: Python 3.11
Framework: PyTorch 2.x / NumPy / SciPy / NetworkX / Scikit-Image
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import scipy.spatial as spatial
from skimage.morphology import skeletonize
import torch


class NodeType(Enum):
    """Classification of topological nodes along vessel centerline."""
    INLET = auto()
    OUTLET = auto()
    BIFURCATION = auto()
    CONTINUOUS = auto()


@dataclass(frozen=True)
class VoxelMetadata:
    """
    Metadata for 3D CTA image volume affine transformation.

    Attributes
    ----------
    spacing : Tuple[float, float, float]
        Anisotropic voxel spacing (dx, dy, dz) in physical length units (mm).
    origin : Tuple[float, float, float]
        Physical coordinates of voxel index (0, 0, 0) in mm.
    direction : Tuple[Tuple[float, float, float], ...]
        3x3 Direction cosine matrix R describing image orientation in physical space.
    """

    spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    @property
    def direction_matrix(self) -> np.ndarray:
        """3x3 Direction cosine orientation matrix R."""
        return np.array(self.direction, dtype=np.float64)

    @property
    def spacing_vector(self) -> np.ndarray:
        """3D Anisotropic voxel spacing vector Delta_x = (dx, dy, dz)."""
        return np.array(self.spacing, dtype=np.float64)

    @property
    def origin_vector(self) -> np.ndarray:
        """3D Physical origin vector x_0 = (x_0, y_0, z_0)."""
        return np.array(self.origin, dtype=np.float64)


@dataclass(frozen=True)
class CenterlineNode:
    """
    A single spatial node along the extracted coronary centerline.

    Attributes
    ----------
    node_id : int
        Unique integer identifier for graph node.
    position : Tuple[float, float, float]
        Physical coordinates (x, y, z) in physical length units (mm).
    radius : float
        Inscribed lumen sphere radius R at this node in physical length units (mm).
    node_type : NodeType
        Classification of node (INLET, OUTLET, BIFURCATION, CONTINUOUS).
    degree : int
        Topological node degree (number of connected incident edges).
    """

    node_id: int
    position: Tuple[float, float, float]
    radius: float
    node_type: NodeType
    degree: int


@dataclass(frozen=True)
class MurrayBranch:
    """
    Geometric details for a single bifurcation node evaluating Murray's Law.

    Attributes
    ----------
    bifurcation_id : int
        Node ID of the bifurcation junction vertex.
    parent_radius : float
        Inscribed radius r_0 of parent incoming vessel branch (mm).
    child_radii : Tuple[float, ...]
        Inscribed radii (r_1, r_2, ...) of daughter outgoing vessel branches (mm).
    deviation_index : float
        Ratio epsilon = (sum_k r_child_k^gamma) / (r_parent^gamma) indicating Murray fit error.
    """

    bifurcation_id: int
    parent_radius: float
    child_radii: Tuple[float, ...]
    deviation_index: float


@dataclass(frozen=True)
class FlowDomain:
    """
    Immutable container storing discretized physical spatial domain point clouds
    and normal vectors formatted as PyTorch Tensors for PINN training.

    Attributes
    ----------
    interior_points : torch.Tensor
        Collocation coordinates in vessel lumen volume Omega, shape (N_int, 3).
    wall_points : torch.Tensor
        Coordinates on arterial wall boundary dOmega_wall, shape (N_wall, 3).
    wall_normals : torch.Tensor
        Unit outward surface normal vectors on dOmega_wall, shape (N_wall, 3).
    inlet_points : torch.Tensor
        Coordinates on inlet cross-sectional profile dOmega_inlet, shape (N_inlet, 3).
    inlet_normals : torch.Tensor
        Unit outward normal vectors pointing out of inlet domain, shape (N_inlet, 3).
    outlet_points : torch.Tensor
        Coordinates on outlet cross-sectional profiles dOmega_outlet, shape (N_outlet, 3).
    outlet_normals : torch.Tensor
        Unit outward normal vectors pointing out of outlet domain, shape (N_outlet, 3).
    centerline_graph : nx.Graph
        NetworkX graph representation of 3D vessel centerline topology.
    murray_branches : List[MurrayBranch]
        List of Murray's Law bifurcation metric objects.
    estimated_murray_exponent : float
        Fitted physiological branching exponent gamma* across all bifurcations.
    voxel_metadata : VoxelMetadata
        Affine transformation metadata of source CTA image.
    """

    interior_points: torch.Tensor
    wall_points: torch.Tensor
    wall_normals: torch.Tensor
    inlet_points: torch.Tensor
    inlet_normals: torch.Tensor
    outlet_points: torch.Tensor
    outlet_normals: torch.Tensor
    centerline_graph: nx.Graph
    murray_branches: List[MurrayBranch]
    estimated_murray_exponent: float
    voxel_metadata: VoxelMetadata


class CoordinateTransformer:
    """
    Handles bidirectional mapping between discrete voxel grid indices and continuous physical space.
    """

    def __init__(self, metadata: VoxelMetadata) -> None:
        """
        Parameters
        ----------
        metadata : VoxelMetadata
            Affine matrix, spacing, and origin parameters.
        """
        self.meta = metadata
        self.R = metadata.direction_matrix
        self.R_inv = np.linalg.inv(self.R)
        self.spacing = metadata.spacing_vector
        self.origin = metadata.origin_vector

    def voxel_to_physical(self, voxel_coords: np.ndarray) -> np.ndarray:
        """
        Maps discrete grid indices i = (i, j, k)^T to physical space coordinates x = (x, y, z)^T.

        Mathematical Formula
        --------------------
            x_phys = R . (i * Delta_x) + x_origin

        Parameters
        ----------
        voxel_coords : np.ndarray
            Voxel indices of shape (N, 3).

        Returns
        -------
        np.ndarray
            Physical coordinates in mm of shape (N, 3).
        """
        scaled_voxels = voxel_coords * self.spacing  # Shape (N, 3)
        physical_coords = np.dot(scaled_voxels, self.R.T) + self.origin
        return physical_coords

    def physical_to_voxel(self, physical_coords: np.ndarray) -> np.ndarray:
        """
        Maps physical coordinates x = (x, y, z)^T to continuous voxel grid indices.

        Mathematical Formula
        --------------------
            i_voxel = (R^-1 . (x_phys - x_origin)) / Delta_x

        Parameters
        ----------
        physical_coords : np.ndarray
            Physical coordinates in mm of shape (N, 3).

        Returns
        -------
        np.ndarray
            Continuous voxel coordinates of shape (N, 3).
        """
        translated = physical_coords - self.origin
        unrotated = np.dot(translated, self.R_inv.T)
        voxel_coords = unrotated / self.spacing
        return voxel_coords


class EuclideanDistanceTransformer:
    """
    Computes anisotropic 3D Euclidean Distance Transform (EDT) and Signed Distance
    Field (SDF) gradient vectors for smooth normal estimation.
    """

    def __init__(self, metadata: VoxelMetadata) -> None:
        """
        Parameters
        ----------
        metadata : VoxelMetadata
            Spacing parameters for physical anisotropic EDT.
        """
        self.metadata = metadata
        self.transformer = CoordinateTransformer(metadata)

    def compute_distance_field(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Calculates exact physical distance from each interior voxel to the nearest boundary voxel.

        Mathematical Formula
        --------------------
            phi(i) = min_{j in dOmega} || (i - j) * Delta_x ||_2

        Parameters
        ----------
        binary_mask : np.ndarray
            3D binary lumen volume mask (1 for fluid lumen, 0 for exterior wall/tissue).

        Returns
        -------
        np.ndarray
            3D distance field array in physical length units (mm), same shape as binary_mask.
        """
        # H-6 FIX: VoxelMetadata.spacing is stored as (dx, dy, dz).
        # scipy.ndimage.distance_transform_edt interprets the sampling tuple as
        # (axis-0 spacing, axis-1 spacing, axis-2 spacing). For a 3D array
        # indexed as (z, y, x), axis-0 = z, axis-1 = y, axis-2 = x.
        # Passing spacing=(dx, dy, dz) directly assigns dx to the z-axis and
        # dz to the x-axis — correct only for isotropic voxels.
        # Fix: unpack and pass in (dz, dy, dx) order for (z, y, x) axes.
        dx_s, dy_s, dz_s = self.metadata.spacing  # unpack (dx, dy, dz)
        distance_map = ndimage.distance_transform_edt(
            binary_mask, sampling=(dz_s, dy_s, dx_s)  # pass as (z, y, x) axis spacings
        )
        return distance_map

    def compute_outward_normals(
        self, distance_map: np.ndarray, wall_voxel_indices: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates smooth unit outward surface normal vectors on the vessel boundary.

        Mathematical Formula
        --------------------
        The gradient of the distance field inside the lumen points inward towards the
        medial centerline axis. Therefore, the outward unit normal vector is:

            n(x) = - grad(phi(x)) / ||grad(phi(x))||_2

        Parameters
        ----------
        distance_map : np.ndarray
            3D physical distance transform field.
        wall_voxel_indices : np.ndarray
            Integer voxel indices of vessel boundary points, shape (N, 3).

        Returns
        -------
        np.ndarray
            Unit outward normal vectors in physical space, shape (N, 3).
        """
        # H-6 FIX: spacing is stored as (dx, dy, dz); unpack correctly so that
        # np.gradient receives the spacings in (axis-0, axis-1, axis-2) = (z, y, x) order.
        # Previous code `dz, dy, dx = self.metadata.spacing` assigned dz=dx_actual
        # and dx=dz_actual, silently swapping x/z spacings for anisotropic voxels.
        dx_s, dy_s, dz_s = self.metadata.spacing  # (dx, dy, dz) convention
        grad_z, grad_y, grad_x = np.gradient(distance_map, dz_s, dy_s, dx_s)  # (z, y, x) axes

        # Interpolate gradients at discrete wall voxel locations
        gz = grad_z[wall_voxel_indices[:, 0], wall_voxel_indices[:, 1], wall_voxel_indices[:, 2]]
        gy = grad_y[wall_voxel_indices[:, 0], wall_voxel_indices[:, 1], wall_voxel_indices[:, 2]]
        gx = grad_x[wall_voxel_indices[:, 0], wall_voxel_indices[:, 1], wall_voxel_indices[:, 2]]

        # Inward gradient vectors in voxel space
        inward_grads_voxel = np.column_stack([gz, gy, gx])

        # Convert gradient direction to physical orientation space
        inward_grads_phys = np.dot(inward_grads_voxel, self.metadata.direction_matrix.T)

        # Outward unit normal: negative of inward gradient
        outward_normals = -inward_grads_phys
        norm_mag = np.linalg.norm(outward_normals, axis=1, keepdims=True)
        norm_mag[norm_mag < 1e-8] = 1.0  # Safeguard against division by zero

        outward_normals = outward_normals / norm_mag
        return outward_normals


class CenterlineExtractor:
    """
    Extracts 1D medial axis skeleton, builds topological spatial graph,
    prunes spurious spurs, and labels boundary inlet/outlet/bifurcation nodes.
    """

    def __init__(self, metadata: VoxelMetadata) -> None:
        """
        Parameters
        ----------
        metadata : VoxelMetadata
            Voxel affine and spacing metadata.
        """
        self.metadata = metadata
        self.transformer = CoordinateTransformer(metadata)

    def extract_centerline_graph(
        self, binary_mask: np.ndarray, distance_map: np.ndarray
    ) -> Tuple[nx.Graph, List[CenterlineNode]]:
        """
        Runs 3D topological thinning, builds 26-connectivity NetworkX spatial graph,
        and classifies node types.

        Parameters
        ----------
        binary_mask : np.ndarray
            3D binary lumen volume.
        distance_map : np.ndarray
            3D physical distance transform field.

        Returns
        -------
        Tuple[nx.Graph, List[CenterlineNode]]
            - NetworkX centerline graph with physical edge lengths and node positions.
            - List of classified CenterlineNode objects.
        """
        # 1. 3D Parallel Skeletonization
        skeleton_mask = skeletonize(binary_mask.astype(np.uint8)).astype(bool)
        skel_voxels = np.argwhere(skeleton_mask)

        if len(skel_voxels) == 0:
            raise ValueError("Skeleton extraction failed: Binary mask contains no active lumen voxels.")

        # 2. Build Voxel Adjacency Graph (26-connectivity)
        graph = nx.Graph()
        voxel_to_id = {tuple(v): idx for idx, v in enumerate(skel_voxels)}

        for idx, voxel in enumerate(skel_voxels):
            phys_pos = self.transformer.voxel_to_physical(voxel.reshape(1, 3))[0]
            radius = float(distance_map[voxel[0], voxel[1], voxel[2]])
            graph.add_node(idx, voxel=tuple(voxel), pos=tuple(phys_pos), radius=radius)

        # Add 26-connected spatial edges weighted by physical Euclidean distance
        for idx, v in enumerate(skel_voxels):
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        neighbor = (v[0] + dz, v[1] + dy, v[2] + dx)
                        if neighbor in voxel_to_id:
                            n_idx = voxel_to_id[neighbor]
                            if not graph.has_edge(idx, n_idx):
                                pos_a = np.array(graph.nodes[idx]["pos"])
                                pos_b = np.array(graph.nodes[n_idx]["pos"])
                                edge_len = float(np.linalg.norm(pos_a - pos_b))
                                graph.add_edge(idx, n_idx, weight=edge_len)

        # 3. Prune Spurious Skeleton Branches (Spurs shorter than local vessel diameter)
        graph = self._prune_graph_spurs(graph)

        # 4. Classify Node Types
        nodes_list = self._classify_graph_nodes(graph)

        return graph, nodes_list

    def _prune_graph_spurs(self, graph: nx.Graph, min_spur_length_factor: float = 1.5) -> nx.Graph:
        """
        Removes short non-physical skeleton spurs caused by surface voxel irregularities.
        """
        pruned = graph.copy()
        has_pruned = True

        while has_pruned:
            has_pruned = False
            leaf_nodes = [node for node, degree in pruned.degree() if degree == 1]

            for leaf in leaf_nodes:
                neighbors = list(pruned.neighbors(leaf))
                if not neighbors:
                    continue
                neighbor = neighbors[0]

                edge_len = pruned[leaf][neighbor]["weight"]
                local_radius = pruned.nodes[leaf]["radius"]

                # If terminal branch length is less than factor * radius, prune it
                if edge_len < min_spur_length_factor * local_radius and pruned.degree(neighbor) > 2:
                    pruned.remove_node(leaf)
                    has_pruned = True

        return pruned

    def _classify_graph_nodes(self, graph: nx.Graph) -> List[CenterlineNode]:
        """
        Classifies graph nodes into INLET, OUTLET, BIFURCATION, and CONTINUOUS.
        """
        degree_dict = dict(graph.degree())
        terminal_nodes = [node for node, deg in degree_dict.items() if deg == 1]

        if not terminal_nodes:
            raise ValueError("Centerline graph contains no terminal leaf nodes.")

        # Identify primary inlet as the terminal node with maximum axial coordinate (z_phys)
        inlet_node_id = max(
            terminal_nodes, key=lambda n: graph.nodes[n]["pos"][2]
        )

        centerline_nodes = []
        for node_id, deg in degree_dict.items():
            pos = graph.nodes[node_id]["pos"]
            radius = graph.nodes[node_id]["radius"]

            if node_id == inlet_node_id:
                n_type = NodeType.INLET
            elif deg == 1:
                n_type = NodeType.OUTLET
            elif deg >= 3:
                n_type = NodeType.BIFURCATION
            else:
                n_type = NodeType.CONTINUOUS

            graph.nodes[node_id]["node_type"] = n_type

            centerline_nodes.append(
                CenterlineNode(
                    node_id=node_id,
                    position=pos,
                    radius=radius,
                    node_type=n_type,
                    degree=deg,
                )
            )

        return centerline_nodes


class MurraysLawAnalyzer:
    """
    Evaluates physiological branching conservation laws (Murray's Law) at vessel bifurcations.
    """

    def analyze_bifurcations(
        self, graph: nx.Graph
    ) -> Tuple[List[MurrayBranch], float]:
        """
        Identifies parent and child branches for every bifurcation node and fits
        the optimal Murray exponent gamma*.

        Mathematical Formula
        --------------------
            r_0^gamma = r_1^gamma + r_2^gamma + ...

        Parameters
        ----------
        graph : nx.Graph
            Pruned centerline graph with classified node types.

        Returns
        -------
        Tuple[List[MurrayBranch], float]
            - List of MurrayBranch metric instances.
            - Optimal physiological Murray exponent gamma* fitted across domain.
        """
        bifurcation_nodes = [
            node for node, data in graph.nodes(data=True) if data.get("node_type") == NodeType.BIFURCATION
        ]

        if not bifurcation_nodes:
            # Fallback for single non-bifurcating vessel segment
            return [], 3.0

        murray_branches = []
        bifurcation_data = []

        for b_node in bifurcation_nodes:
            neighbors = list(graph.neighbors(b_node))
            if len(neighbors) < 3:
                continue

            # Sort neighbors by axial distance to identify incoming parent vs outgoing children
            b_pos = np.array(graph.nodes[b_node]["pos"])
            neighbor_radii = [graph.nodes[n]["radius"] for n in neighbors]
            neighbor_z = [graph.nodes[n]["pos"][2] for n in neighbors]

            # Parent branch comes from higher axial z (towards inlet)
            parent_idx = int(np.argmax(neighbor_z))
            parent_r = neighbor_radii[parent_idx]

            child_radii = tuple(
                [neighbor_radii[i] for i in range(len(neighbors)) if i != parent_idx]
            )

            bifurcation_data.append((parent_r, child_radii))

        # Fit optimal gamma using non-linear bounded scalar minimization
        def murray_residual(gamma: float) -> float:
            total_error = 0.0
            for p_r, c_radii in bifurcation_data:
                p_term = p_r ** gamma
                c_term = sum(c_r ** gamma for c_r in c_radii)
                total_error += (p_term - c_term) ** 2
            return total_error

        res = optimize.minimize_scalar(murray_residual, bounds=(1.5, 4.0), method="bounded")
        optimal_gamma = float(res.x) if res.success else 3.0

        # Build MurrayBranch objects using optimal gamma
        for b_node, (parent_r, child_radii) in zip(bifurcation_nodes, bifurcation_data):
            p_term = parent_r ** optimal_gamma
            c_term = sum(c_r ** optimal_gamma for c_r in child_radii)
            dev_index = float(c_term / p_term) if p_term > 1e-8 else 1.0

            murray_branches.append(
                MurrayBranch(
                    bifurcation_id=b_node,
                    parent_radius=parent_r,
                    child_radii=child_radii,
                    deviation_index=dev_index,
                )
            )

        return murray_branches, optimal_gamma


class FlowDomainGenerator:
    """
    Discretizes the CTA vessel volume into interior, wall boundary, inlet, and outlet
    point clouds with exact outward normal vectors.
    """

    def __init__(self, metadata: VoxelMetadata) -> None:
        """
        Parameters
        ----------
        metadata : VoxelMetadata
            Voxel spacing and affine parameters.
        """
        self.metadata = metadata
        self.transformer = CoordinateTransformer(metadata)
        self.distance_engine = EuclideanDistanceTransformer(metadata)

    def generate_flow_domain(
        self,
        binary_mask: np.ndarray,
        num_interior_points: int = 20000,
        num_wall_points: int = 8000,
        num_boundary_disk_points: int = 1000,
    ) -> FlowDomain:
        """
        Generates full physical FlowDomain dataclass with PyTorch tensor point clouds.

        Parameters
        ----------
        binary_mask : np.ndarray
            3D binary CTA vessel mask volume.
        num_interior_points : int
            Number of interior collocation points to sample in Omega.
        num_wall_points : int
            Number of wall boundary points to sample on dOmega_wall.
        num_boundary_disk_points : int
            Number of points to sample on inlet/outlet boundary cross-sections.

        Returns
        -------
        FlowDomain
            Complete spatial discretization ready for PINN solver.
        """
        # 1. Compute Distance Field & Extract Centerline
        dist_field = self.distance_engine.compute_distance_field(binary_mask)
        centerline_engine = CenterlineExtractor(self.metadata)
        graph, nodes_list = centerline_engine.extract_centerline_graph(binary_mask, dist_field)

        # 2. Analyze Murray's Law at Bifurcations
        murray_engine = MurraysLawAnalyzer()
        murray_branches, opt_gamma = murray_engine.analyze_bifurcations(graph)

        # 3. Sample Interior Collocation Points (Lumen Omega)
        interior_voxels = np.argwhere(binary_mask > 0)
        if len(interior_voxels) > num_interior_points:
            idx_choice = np.random.choice(len(interior_voxels), size=num_interior_points, replace=False)
            sampled_int_voxels = interior_voxels[idx_choice]
        else:
            sampled_int_voxels = interior_voxels

        interior_phys = self.transformer.voxel_to_physical(sampled_int_voxels)

        # 4. Extract Wall Boundary Points & Normals (dOmega_wall)
        eroded_mask = ndimage.binary_erosion(binary_mask)
        wall_mask = binary_mask.astype(bool) & (~eroded_mask)
        wall_voxels = np.argwhere(wall_mask)

        if len(wall_voxels) > num_wall_points:
            wall_choice = np.random.choice(len(wall_voxels), size=num_wall_points, replace=False)
            sampled_wall_voxels = wall_voxels[wall_choice]
        else:
            sampled_wall_voxels = wall_voxels

        wall_phys = self.transformer.voxel_to_physical(sampled_wall_voxels)
        wall_normals = self.distance_engine.compute_outward_normals(dist_field, sampled_wall_voxels)

        # 5. Extract Inlet and Outlet Boundary Cross-Sections
        inlet_nodes = [n for n in nodes_list if n.node_type == NodeType.INLET]
        outlet_nodes = [n for n in nodes_list if n.node_type == NodeType.OUTLET]

        inlet_pts, inlet_n = self._generate_boundary_disk(
            graph, inlet_nodes[0].node_id, num_boundary_disk_points, is_inlet=True
        )

        all_outlet_pts = []
        all_outlet_normals = []
        
        # Calculate total outlet area to allocate points proportionally
        total_outlet_area = sum(np.pi * graph.nodes[n.node_id]["radius"]**2 for n in outlet_nodes)
        
        for o_node in outlet_nodes:
            area = np.pi * graph.nodes[o_node.node_id]["radius"]**2
            # Allocate proportionally, but ensure at least a minimum number of points
            pts_for_this_outlet = int(num_boundary_disk_points * (area / total_outlet_area))
            pts_for_this_outlet = max(50, pts_for_this_outlet)
            
            o_pts, o_normals = self._generate_boundary_disk(
                graph, o_node.node_id, pts_for_this_outlet, is_inlet=False
            )
            all_outlet_pts.append(o_pts)
            all_outlet_normals.append(o_normals)

        outlet_phys = np.vstack(all_outlet_pts)
        outlet_normals = np.vstack(all_outlet_normals)

        # 6. Package into PyTorch Tensors
        # C-3 FIX: spatial coordinate tensors must have requires_grad=True.
        # All autograd.grad calls in physics.py, boundary_conditions.py,
        # physics_verification.py, and ess.py specify these coordinate tensors
        # as the differentiation inputs. Without requires_grad the autograd
        # engine cannot trace through them and raises RuntimeError.
        # Normal vectors are not differentiated through → requires_grad=False.
        return FlowDomain(
            interior_points=torch.tensor(interior_phys, dtype=torch.float32).requires_grad_(True),
            wall_points=torch.tensor(wall_phys, dtype=torch.float32).requires_grad_(True),
            wall_normals=torch.tensor(wall_normals, dtype=torch.float32),
            inlet_points=torch.tensor(inlet_pts, dtype=torch.float32).requires_grad_(True),
            inlet_normals=torch.tensor(inlet_n, dtype=torch.float32),
            outlet_points=torch.tensor(outlet_phys, dtype=torch.float32).requires_grad_(True),
            outlet_normals=torch.tensor(outlet_normals, dtype=torch.float32),
            centerline_graph=graph,
            murray_branches=murray_branches,
            estimated_murray_exponent=opt_gamma,
            voxel_metadata=self.metadata,
        )

    def _generate_boundary_disk(
        self, graph: nx.Graph, terminal_node_id: int, num_points: int, is_inlet: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs planar disk point cloud perpendicular to local centerline tangent vector.
        """
        center_pos = np.array(graph.nodes[terminal_node_id]["pos"])
        radius = graph.nodes[terminal_node_id]["radius"]

        neighbors = list(graph.neighbors(terminal_node_id))
        if neighbors:
            adj_pos = np.array(graph.nodes[neighbors[0]]["pos"])
            tangent = center_pos - adj_pos
            t_norm = np.linalg.norm(tangent)
            tangent = tangent / t_norm if t_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
        else:
            tangent = np.array([0.0, 0.0, 1.0])

        outward_normal = tangent

        # Construct orthonormal planar basis (u, v) perpendicular to tangent
        if abs(tangent[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])

        u = np.cross(tangent, arbitrary)
        u = u / np.linalg.norm(u)
        v = np.cross(tangent, u)
        v = v / np.linalg.norm(v)

        # Sample uniform polar disk coordinates
        r_samples = np.sqrt(np.random.rand(num_points, 1)) * radius
        theta_samples = np.random.rand(num_points, 1) * 2.0 * math.pi

        disk_pts = (
            center_pos
            + r_samples * np.cos(theta_samples) * u
            + r_samples * np.sin(theta_samples) * v
        )

        normals = np.tile(outward_normal, (num_points, 1))
        return disk_pts, normals


class CTASegmentationLoader:
    """
    Loads CTA vessel mask volume or creates a synthetic 3D bifurcating coronary lumen.
    """

    @staticmethod
    def create_synthetic_coronary_volume(
        grid_shape: Tuple[int, int, int] = (100, 100, 120),
        spacing: Tuple[float, float, float] = (0.2, 0.2, 0.2),
    ) -> Tuple[np.ndarray, VoxelMetadata]:
        """
        Generates 3D synthetic binary coronary volume with stenosis and bifurcation.

        Parameters
        ----------
        grid_shape : Tuple[int, int, int]
            Grid dimension (nz, ny, nx) voxels.
        spacing : Tuple[float, float, float]
            Anisotropic voxel spacing (dz, dy, dx) in mm.

        Returns
        -------
        Tuple[np.ndarray, VoxelMetadata]
            - 3D Binary mask volume (1 for lumen, 0 for tissue).
            - VoxelMetadata affine parameters.
        """
        nz, ny, nx = grid_shape
        mask = np.zeros(grid_shape, dtype=bool)

        z_indices, y_indices, x_indices = np.indices(grid_shape)

        # Transform to physical coordinates
        dz, dy, dx = spacing
        z_phys = z_indices * dz
        y_phys = (y_indices - ny / 2) * dy
        x_phys = (x_indices - nx / 2) * dx

        # Main Trunk (Inlet to Stenosis to Bifurcation)
        r_main_base = 1.8  # mm
        # Apply local stenosis at z = 12 mm
        r_main_profile = r_main_base * (1.0 - 0.35 * np.exp(-((z_phys - 12.0) ** 2) / 4.0))

        trunk_dist = np.sqrt(x_phys**2 + y_phys**2)
        trunk_mask = (trunk_dist <= r_main_profile) & (z_phys <= 16.0)

        # Branch 1 (Left Child, branching off at z = 16 mm)
        r_child1 = 1.2  # mm
        z_bif = 16.0
        x_c1 = (z_phys - z_bif) * 0.4
        c1_dist = np.sqrt((x_phys - x_c1) ** 2 + y_phys ** 2)
        branch1_mask = (c1_dist <= r_child1) & (z_phys > z_bif) & (z_phys <= 22.0)

        # Branch 2 (Right Child, branching off at z = 16 mm)
        r_child2 = 1.3  # mm
        x_c2 = -(z_phys - z_bif) * 0.4
        c2_dist = np.sqrt((x_phys - x_c2) ** 2 + y_phys ** 2)
        branch2_mask = (c2_dist <= r_child2) & (z_phys > z_bif) & (z_phys <= 22.0)

        mask = trunk_mask | branch1_mask | branch2_mask

        meta = VoxelMetadata(
            spacing=spacing,
            origin=(0.0, 0.0, 0.0),
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )

        return mask, meta

    @staticmethod
    def load_nifti_coronary_volume(nifti_path: str) -> Tuple[np.ndarray, VoxelMetadata]:
        """
        Loads 3D coronary volume from a NIfTI file.
        """
        import nibabel as nib
        img = nib.load(nifti_path)
        data = img.get_fdata()
        mask = (data > 0).astype(bool)
        
        spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
        affine = img.affine
        origin = tuple(float(x) for x in affine[:3, 3])
        direction = tuple(tuple(float(v) for v in row) for row in affine[:3, :3])
        
        meta = VoxelMetadata(
            spacing=spacing,
            origin=origin,
            direction=direction,
        )
        return mask, meta


def flow_domain_to_coronary_geometry(domain: FlowDomain):
    """
    Converts FlowDomain (from CTA processing) to CoronaryGeometry (for adaptive sampling).
    Handles:
    - Tensor to numpy conversion
    - Normal convention mapping (FlowDomain uses outward; CoronaryGeometry expects inward)
    - Centerline structure extraction (nx.Graph to numpy arrays)
    """
    import sampling  # Local import to avoid circular dependencies
    
    wall_pts = domain.wall_points.detach().cpu().numpy()
    wall_norms_outward = domain.wall_normals.detach().cpu().numpy()
    wall_norms_inward = -wall_norms_outward  # sampling.py expects inward normals
    
    # Extract centerline positions and radii from graph
    graph = domain.centerline_graph
    if graph and len(graph.nodes) > 0:
        node_order = list(graph.nodes())
        cl_pts = np.array([graph.nodes[n]["pos"] for n in node_order])
        cl_radii = np.array([graph.nodes[n]["radius"] for n in node_order])
    else:
        # Fallback if graph is somehow empty
        cl_pts = np.zeros((0, 3))
        cl_radii = np.zeros((0,))
    
    # Extract bifurcation positions
    if domain.murray_branches:
        bif_pts = np.array([
            graph.nodes[b.bifurcation_id]["pos"]
            for b in domain.murray_branches
        ])
    else:
        bif_pts = None

    return sampling.CoronaryGeometry(
        wall_points=wall_pts,
        wall_normals=wall_norms_inward,
        inlet_points=domain.inlet_points.detach().cpu().numpy(),
        inlet_normals=-domain.inlet_normals.detach().cpu().numpy(),  # outward -> inward
        outlet_points=domain.outlet_points.detach().cpu().numpy(),
        outlet_normals=-domain.outlet_normals.detach().cpu().numpy(),
        centerline_points=cl_pts,
        centerline_radii=cl_radii,
        bifurcation_nodes=bif_pts,
    )


def plot_flow_domain_diagnostics(domain: FlowDomain) -> None:
    """
    Renders 3D diagnostic scatter plot of extracted flow domain discretization.

    Parameters
    ----------
    domain : FlowDomain
        Extracted flow domain containing physical point clouds and centerline graph.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 1. Plot Interior Collocation Points (Subsampled for rendering clarity)
    int_pts = domain.interior_points.numpy()
    sub_idx = np.random.choice(len(int_pts), size=min(3000, len(int_pts)), replace=False)
    ax.scatter(
        int_pts[sub_idx, 0],
        int_pts[sub_idx, 1],
        int_pts[sub_idx, 2],
        c="cyan",
        alpha=0.1,
        s=1,
        label="Interior Domain (Omega)",
    )

    # 2. Plot Arterial Wall Boundary Points
    wall_pts = domain.wall_points.numpy()
    w_sub = np.random.choice(len(wall_pts), size=min(2000, len(wall_pts)), replace=False)
    ax.scatter(
        wall_pts[w_sub, 0],
        wall_pts[w_sub, 1],
        wall_pts[w_sub, 2],
        c="darkgray",
        alpha=0.4,
        s=4,
        label="Arterial Wall (dOmega_wall)",
    )

    # 3. Plot Inlet Boundary Cross-Section
    in_pts = domain.inlet_points.numpy()
    ax.scatter(
        in_pts[:, 0],
        in_pts[:, 1],
        in_pts[:, 2],
        c="red",
        s=15,
        label="Inlet Cross-Section (dOmega_inlet)",
    )

    # 4. Plot Outlet Boundary Cross-Sections
    out_pts = domain.outlet_points.numpy()
    ax.scatter(
        out_pts[:, 0],
        out_pts[:, 1],
        out_pts[:, 2],
        c="blue",
        s=15,
        label="Outlet Cross-Sections (dOmega_outlet)",
    )

    # 5. Plot Centerline Graph Edges
    graph = domain.centerline_graph
    for u, v in graph.edges():
        pos_u = graph.nodes[u]["pos"]
        pos_v = graph.nodes[v]["pos"]
        ax.plot(
            [pos_u[0], pos_v[0]],
            [pos_u[1], pos_v[1]],
            [pos_u[2], pos_v[2]],
            color="black",
            linewidth=2,
        )

    title_str = (
        f"Coronary Artery Geometry Discretization\n"
        f"Fitted Murray Exponent gamma* = {domain.estimated_murray_exponent:.2f}"
    )
    ax.set_title(title_str, fontsize=14, pad=15)
    ax.set_xlabel("Physical X (mm)")
    ax.set_ylabel("Physical Y (mm)")
    ax.set_zlabel("Physical Z (mm)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main execution script demonstrating end-to-end coronary CTA geometry parsing,
    distance transformation, skeletonization, Murray's law fitting, and point cloud sampling.
    """
    print("=" * 70)
    print("CORONARY CTA GEOMETRY PARSING & DOMAIN DISCRETIZATION PIPELINE")
    print("=" * 70)

    # 1. Load / Synthesize 3D CTA Vessel Volume
    print("\n[1/5] Synthesizing 3D Coronary CTA Vessel Volume with Stenosis & Bifurcation...")
    mask, metadata = CTASegmentationLoader.create_synthetic_coronary_volume()
    print(f"      Mask Volume Dimensions : {mask.shape} voxels")
    print(f"      Voxel Spacing (dz,dy,dx): {metadata.spacing} mm")
    print(f"      Total Lumen Voxels      : {np.sum(mask):,}")

    # 2. Execute Domain Discretization Pipeline
    print("\n[2/5] Running Distance Transform, Skeletonization & Point Cloud Sampling...")
    domain_generator = FlowDomainGenerator(metadata)
    flow_domain = domain_generator.generate_flow_domain(
        binary_mask=mask,
        num_interior_points=15000,
        num_wall_points=5000,
        num_boundary_disk_points=500,
    )

    # 3. Print Discretization Statistics
    print("\n[3/5] Discretization Verification Statistics:")
    print(f"      Interior Collocation Points (Omega)      : {flow_domain.interior_points.shape[0]:,}")
    print(f"      Wall Boundary Points (dOmega_wall)       : {flow_domain.wall_points.shape[0]:,}")
    print(f"      Wall Outward Normal Vectors Norm         : {torch.mean(torch.norm(flow_domain.wall_normals, dim=1)).item():.4f}")
    print(f"      Inlet Surface Points (dOmega_inlet)      : {flow_domain.inlet_points.shape[0]:,}")
    print(f"      Outlet Surface Points (dOmega_outlet)    : {flow_domain.outlet_points.shape[0]:,}")

    # 4. Print Murray's Law Analysis Summary
    print("\n[4/5] Murray's Law Branching Analysis:")
    print(f"      Fitted Empirical Murray Exponent (gamma*): {flow_domain.estimated_murray_exponent:.3f}")
    for idx, branch in enumerate(flow_domain.murray_branches, 1):
        print(f"      Bifurcation {idx}: Parent Radius r0 = {branch.parent_radius:.3f} mm | Child Radii = {branch.child_radii} mm | Deviation = {branch.deviation_index:.3f}")

    # 5. Diagnostic Visualization
    print("\n[5/5] Generating 3D Spatial Diagnostic Plot...")
    plot_flow_domain_diagnostics(flow_domain)
    print("\nGeometry processing complete. FlowDomain is fully ready for PINN solver.")


if __name__ == "__main__":
    main()