"""
Centerline extraction for 3D meshes.
Extracts the medial axis/centerline from a 3D surface mesh for camera path animation.
"""

import numpy as np
import pyvista as pv
from scipy import ndimage
from skimage import morphology
from sklearn.neighbors import NearestNeighbors
import networkx as nx


def voxelize_mesh(mesh, resolution=100):
    """
    Convert a PyVista mesh to a binary voxel grid.

    Parameters:
    -----------
    mesh : pv.PolyData
        Input surface mesh
    resolution : int
        Approximate number of voxels along the longest axis

    Returns:
    --------
    voxel_grid : np.ndarray
        3D binary array (1 inside mesh, 0 outside)
    grid_origin : np.ndarray
        Origin point of the voxel grid in world coordinates
    voxel_size : float
        Size of each voxel
    """
    # Get mesh bounds
    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)

    # Calculate voxel size based on longest axis
    x_range = bounds[1] - bounds[0]
    y_range = bounds[3] - bounds[2]
    z_range = bounds[5] - bounds[4]
    max_range = max(x_range, y_range, z_range)

    voxel_size = max_range / resolution

    # Calculate grid dimensions
    grid_x = int(np.ceil(x_range / voxel_size)) + 2
    grid_y = int(np.ceil(y_range / voxel_size)) + 2
    grid_z = int(np.ceil(z_range / voxel_size)) + 2

    # Create structured grid
    grid_origin = np.array([bounds[0] - voxel_size, bounds[2] - voxel_size, bounds[4] - voxel_size])

    # Create voxel grid using PyVista's voxelize
    voxels = pv.voxelize(mesh, density=voxel_size, check_surface=False)

    # Convert to binary array
    # Create empty grid
    voxel_grid = np.zeros((grid_x, grid_y, grid_z), dtype=np.uint8)

    # Get voxel centers and mark them as occupied
    if voxels.n_points > 0:
        points = voxels.points
        indices = ((points - grid_origin) / voxel_size).astype(int)

        # Clip to grid bounds
        indices[:, 0] = np.clip(indices[:, 0], 0, grid_x - 1)
        indices[:, 1] = np.clip(indices[:, 1], 0, grid_y - 1)
        indices[:, 2] = np.clip(indices[:, 2], 0, grid_z - 1)

        # Mark voxels as occupied
        voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    return voxel_grid, grid_origin, voxel_size


def extract_skeleton_points(voxel_grid, grid_origin, voxel_size):
    """
    Extract skeleton points from a binary voxel grid.

    Parameters:
    -----------
    voxel_grid : np.ndarray
        3D binary array
    grid_origin : np.ndarray
        Origin point of the voxel grid
    voxel_size : float
        Size of each voxel

    Returns:
    --------
    skeleton_points : np.ndarray
        Nx3 array of skeleton points in world coordinates
    """
    # Apply 3D skeletonization
    print("  Computing skeleton...")
    skeleton = morphology.skeletonize(voxel_grid)

    # Get skeleton voxel coordinates
    skeleton_coords = np.argwhere(skeleton > 0)

    if len(skeleton_coords) == 0:
        print("  Warning: No skeleton found!")
        return np.array([])

    # Convert to world coordinates
    skeleton_points = skeleton_coords * voxel_size + grid_origin

    print(f"  Found {len(skeleton_points)} skeleton points")
    return skeleton_points


def order_skeleton_points(skeleton_points):
    """
    Order skeleton points to form a path using graph-based approach.
    Finds the longest path through the skeleton, using PCA to determine
    which endpoints represent the logical start and end.

    Parameters:
    -----------
    skeleton_points : np.ndarray
        Nx3 array of unordered skeleton points

    Returns:
    --------
    ordered_path : np.ndarray
        Mx3 array of ordered points forming a path
    """
    if len(skeleton_points) < 2:
        return skeleton_points

    print("  Ordering skeleton points...")

    # Build k-nearest neighbors graph (k=3 to capture branching)
    k = min(3, len(skeleton_points))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(skeleton_points)
    distances, indices = nbrs.kneighbors(skeleton_points)

    # Build graph
    G = nx.Graph()
    for i in range(len(skeleton_points)):
        for j in range(1, k):  # Skip self (index 0)
            neighbor_idx = indices[i, j]
            distance = distances[i, j]
            G.add_edge(i, neighbor_idx, weight=distance)

    # Find endpoints (nodes with degree 1 or 2)
    # Include degree 2 for cases where the skeleton is a simple chain
    endpoints = [node for node, degree in G.degree() if degree <= 2]

    # Use PCA to find the main axis of the structure
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    projected = pca.fit_transform(skeleton_points)

    if len(endpoints) < 2:
        # If no clear endpoints, use PCA to find the two most distant points along main axis
        print("  No clear endpoints, using PCA-based selection")
        min_idx = np.argmin(projected[:, 0])
        max_idx = np.argmax(projected[:, 0])
        endpoints = [min_idx, max_idx]
    elif len(endpoints) > 2:
        # Multiple endpoints - select the two that are furthest apart along the main axis
        print(f"  Found {len(endpoints)} potential endpoints, selecting best pair using PCA")
        endpoint_projections = projected[endpoints, 0]
        min_endpoint_idx = np.argmin(endpoint_projections)
        max_endpoint_idx = np.argmax(endpoint_projections)
        # Get the actual node indices
        start_node = endpoints[min_endpoint_idx]
        end_node = endpoints[max_endpoint_idx]
        endpoints = [start_node, end_node]

    # Find shortest path between the selected endpoints
    if len(endpoints) >= 2:
        try:
            # Use the two endpoints selected by PCA
            start_node = endpoints[0]
            end_node = endpoints[1]

            # Determine which endpoint should be the start based on PCA projection
            # We want to start from the endpoint with lower projection value
            start_proj = projected[start_node, 0]
            end_proj = projected[end_node, 0]

            if start_proj > end_proj:
                # Swap so we start from the lower projection (beginning of main axis)
                start_node, end_node = end_node, start_node

            path = nx.shortest_path(G, start_node, end_node, weight='weight')
            ordered_path = skeleton_points[path]
            print(f"  Created path with {len(ordered_path)} points (PCA-ordered)")
            return ordered_path

        except nx.NetworkXNoPath:
            print("  No path found between endpoints, using PCA fallback")

    # Fallback: return points sorted by first principal component
    print("  Using PCA fallback ordering")
    sorted_indices = np.argsort(projected[:, 0])
    return skeleton_points[sorted_indices]


def smooth_path(path, smoothing_factor=0.5, iterations=3):
    """
    Smooth a path using Gaussian filtering.

    Parameters:
    -----------
    path : np.ndarray
        Nx3 array of path points
    smoothing_factor : float
        Smoothing strength (sigma for Gaussian filter)
    iterations : int
        Number of smoothing iterations

    Returns:
    --------
    smoothed_path : np.ndarray
        Smoothed path
    """
    if len(path) < 3:
        return path

    smoothed = path.copy()
    for _ in range(iterations):
        for dim in range(3):
            smoothed[:, dim] = ndimage.gaussian_filter1d(smoothed[:, dim], sigma=smoothing_factor)

    return smoothed


def extract_centerline(mesh, resolution=100, smooth=True):
    """
    Extract the centerline from a 3D mesh.

    Parameters:
    -----------
    mesh : pv.PolyData
        Input surface mesh
    resolution : int
        Voxelization resolution (higher = more accurate but slower)
    smooth : bool
        Whether to smooth the resulting path

    Returns:
    --------
    centerline_points : np.ndarray
        Nx3 array of ordered centerline points in world coordinates
    """
    print(f"Extracting centerline from mesh with {mesh.n_points} points...")

    # Step 1: Voxelize the mesh
    print("  Voxelizing mesh...")
    voxel_grid, grid_origin, voxel_size = voxelize_mesh(mesh, resolution=resolution)

    # Fill holes in voxel grid to ensure solid interior
    voxel_grid = ndimage.binary_fill_holes(voxel_grid).astype(np.uint8)

    # Step 2: Extract skeleton
    skeleton_points = extract_skeleton_points(voxel_grid, grid_origin, voxel_size)

    if len(skeleton_points) == 0:
        print("  Failed to extract skeleton")
        return np.array([])

    # Step 3: Order points to form a path
    ordered_path = order_skeleton_points(skeleton_points)

    # Step 4: Smooth the path
    if smooth and len(ordered_path) > 2:
        print("  Smoothing path...")
        ordered_path = smooth_path(ordered_path, smoothing_factor=2.0, iterations=3)

    print(f"✓ Centerline extraction complete: {len(ordered_path)} points")
    return ordered_path


def compute_camera_positions(centerline_points, look_ahead_distance=10.0, up_vector=None):
    """
    Compute camera positions and orientations along a centerline.
    Camera is positioned ON the centerline, looking forward along the path.

    Parameters:
    -----------
    centerline_points : np.ndarray
        Nx3 array of centerline points
    look_ahead_distance : float
        Distance ahead to look (in terms of number of points)
    up_vector : np.ndarray or None
        Global up vector for camera orientation (default: [0, 0, 1])

    Returns:
    --------
    camera_positions : list of dict
        List of camera configurations with 'position', 'focal_point', 'up'
    """
    if len(centerline_points) < 2:
        return []

    if up_vector is None:
        up_vector = np.array([0, 0, 1])

    camera_positions = []

    for i in range(len(centerline_points)):
        # Camera is positioned exactly on the centerline
        camera_pos = centerline_points[i]

        # Look ahead along the path to determine focal point
        # Use a point further ahead for smoother camera movement
        look_ahead_idx = min(i + int(look_ahead_distance), len(centerline_points) - 1)
        focal_point = centerline_points[look_ahead_idx]

        # Compute tangent for up vector correction if needed
        if i < len(centerline_points) - 1:
            tangent = centerline_points[i + 1] - centerline_points[i]
        else:
            tangent = centerline_points[i] - centerline_points[i - 1]

        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-10:
            tangent = tangent / tangent_norm

            # Compute right vector by crossing tangent with up
            right = np.cross(tangent, up_vector)
            right_norm = np.linalg.norm(right)

            if right_norm > 1e-10:
                right = right / right_norm
                # Recompute up vector to be perpendicular to both
                camera_up = np.cross(right, tangent)
                camera_up = camera_up / (np.linalg.norm(camera_up) + 1e-10)
            else:
                # Tangent is parallel to up vector, use default
                camera_up = up_vector
        else:
            camera_up = up_vector

        camera_positions.append({
            'position': camera_pos,
            'focal_point': focal_point,
            'up': camera_up
        })

    return camera_positions
