import numpy as np
import torch
from torch_geometric.data import Data


def build_inference_graph(inputs):
    """
    Optimized Vectorized Graph Builder.
    Returns a dictionary of Numpy arrays (Zero-Copy compatible).
    """
    arrangement = inputs['arrangement']
    pitch = inputs['pitch']

    # 1. Identify Nodes (Vectorized)
    # Find indices of non-empty spots
    # rows, cols shape: (N_nodes,)
    rows, cols = np.nonzero(arrangement)
    num_nodes = len(rows)

    # Extract types: 1 (Signal) or -1 (Ground)
    tsv_types = arrangement[rows, cols]

    # 2. Node Features
    # Global features: [Radius, Pitch, Height, Liner, Temperature, Freq]
    global_feats = np.array([
        inputs['radius'], inputs['pitch'], inputs['height'],
        inputs['liner'], inputs.get('temperature', 300.0), inputs['freq']
    ], dtype=np.float32)

    # Create Feature Matrix (N, 7)
    # Col 0: Type, Cols 1-6: Globals
    # We repeat global_feats 'num_nodes' times
    node_globals = np.tile(global_feats, (num_nodes, 1))

    # Combine. Note: We keep type as float in 'x' for the network,
    # but use int logic for masks later.
    x = np.column_stack((tsv_types.astype(np.float32), node_globals))

    # 3. Build Edges (Fully Connected)
    # Generate all pairs (u, v)
    indices = np.arange(num_nodes)
    u_grid, v_grid = np.meshgrid(indices, indices, indexing='ij')

    # Remove self-loops (diagonal)
    mask_no_self = u_grid != v_grid
    u = u_grid[mask_no_self]
    v = v_grid[mask_no_self]

    # 4. Edge Attributes
    # Get coordinates for all edges
    r_u, c_u = rows[u], cols[u]
    r_v, c_v = rows[v], cols[v]

    # Euclidean Distance
    dist_grid = np.sqrt((r_u - r_v)**2 + (c_u - c_v)**2)
    dist_phys = dist_grid * pitch

    # Safe inverse
    dist_safe = dist_phys

    # Stack attributes: [d, 1/d, 1/d^2] -> Shape: (Num_Edges, 3)
    edge_attr = np.column_stack((
        dist_phys,
        1.0 / (dist_safe  + 1e-9),
        1.0 / (dist_safe**2  + 1e-9)
    )).astype(np.float32)

    edge_index = np.stack((u, v)).astype(np.int64)

    # 5. Masks (Vectorized)
    # Node Mask: True where type == 1 (Signal)
    # Note: tsv_types is float, cast comparison
    is_signal = (tsv_types == 1)

    # Node Mask: (N,) boolean
    node_mask = is_signal

    # Edge Mask: True if Source AND Target are signals
    # Use broadcasting on the u/v indices
    edge_mask = is_signal[u] & is_signal[v]

    # Return Dict (Lightweight for Multiprocessing)
    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'node_mask': node_mask,
        'edge_mask': edge_mask,
        'input_id': inputs['id']
    }


def build_inference_data(inputs):
    """
    Convenience wrapper: builds a PyG Data object for direct inference.
    Calls build_inference_graph and wraps the numpy dict into a Data object.
    """
    graph_dict = build_inference_graph(inputs)
    return Data(
        x=torch.from_numpy(graph_dict['x']),
        edge_index=torch.from_numpy(graph_dict['edge_index']),
        edge_attr=torch.from_numpy(graph_dict['edge_attr']),
        node_mask=torch.from_numpy(graph_dict['node_mask']),
        edge_mask=torch.from_numpy(graph_dict['edge_mask']),
        input_id=graph_dict['input_id']
    )
