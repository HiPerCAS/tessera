import numpy as np
import torch
from torch_geometric.data import Data

def convert_pyg_data_to_result(data: Data) -> dict:
    """
    Reconstructs physical input parameters and the complex S-Matrix 
    from a PyG Data object.
    
    Args:
        data (torch_geometric.data.Data): The input graph.
        
    Returns:
        dict: A dictionary containing:
            - 's_matrix': np.array (Complex), shape [2*N_signals, 2*N_signals]
            - 'params': dict of physical scalars (radius, pitch, etc.)
            - 'node_types': list of types (1.0 for Signal, -1.0 for Ground) for active nodes.
    """
    # Ensure data is on CPU and numpy-accessible
    if hasattr(data, 'cpu'):
        data = data.cpu()
    
    x = data.x.numpy()
    
    # --- 1. Extract Global Parameters ---
    # Parameters are identical for all nodes, so we take the first row.
    # Feature indices (7-D, matches convert_result_to_pyg_data):
    # [0:Type, 1:Radius, 2:Pitch, 3:Height, 4:Liner, 5:Temperature, 6:Freq]
    params = {
        "radius":      float(x[0, 1]),
        "pitch":       float(x[0, 2]),
        "height":      float(x[0, 3]),
        "liner":       float(x[0, 4]),
        "temperature": float(x[0, 5]),
        "freq":        float(x[0, 6])
    }
    
    node_types = x[:, 0].astype(int)
    
    # --- 2. Setup S-Matrix Structure ---
    # Count signals (Type == 1)
    num_signals = np.sum(node_types == 1)
    num_ports = num_signals * 2
    
    # Initialize Complex S-Matrix
    s_matrix = np.zeros((num_ports, num_ports), dtype=np.complex128)
    
    # Create mapping: Graph Node Index -> Signal Index (0, 1, 2...)
    node_idx_to_signal_idx = {}
    current_sig_idx = 0
    
    for node_idx, t_type in enumerate(node_types):
        if t_type == 1:
            node_idx_to_signal_idx[node_idx] = current_sig_idx
            current_sig_idx += 1
        else:
            node_idx_to_signal_idx[node_idx] = -1

    # --- 3. Fill Node-Level S-Parameters (S21, S11) ---
    # y_node format: [Re(S21), Im(S21), Re(S11), Im(S11)]
    y_node = data.y_node.numpy()
    node_mask = data.node_mask.numpy()
    
    for node_idx in range(len(node_types)):
        if node_mask[node_idx]:
            # Get Signal Index
            sig_idx = node_idx_to_signal_idx[node_idx]
            
            # Define Ports
            # Port In = 2*k, Port Out = 2*k + 1
            p_in = 2 * sig_idx
            p_out = 2 * sig_idx + 1
            
            vals = y_node[node_idx]
            s21 = complex(vals[0], vals[1])
            s11 = complex(vals[2], vals[3])
            
            # Fill S-Matrix (Assuming Symmetry/Reciprocity)
            # Transmission (Through)
            s_matrix[p_in, p_out] = s21
            s_matrix[p_out, p_in] = s21 
            
            # Reflection (Return)
            s_matrix[p_in, p_in] = s11
            s_matrix[p_out, p_out] = s11 # Assuming symmetry S22 ~= S11 due to cylinder shape

    # --- 4. Fill Edge-Level S-Parameters (NEXT, FEXT) ---
    # y_edge format: [Re(NEXT), Im(NEXT), Re(FEXT), Im(FEXT)]
    y_edge = data.y_edge.numpy()
    edge_mask = data.edge_mask.numpy()
    edge_index = data.edge_index.numpy()
    
    num_edges = edge_index.shape[1]
    
    for i in range(num_edges):
        if edge_mask[i]:
            u = edge_index[0, i]
            v = edge_index[1, i]
            
            sig_u = node_idx_to_signal_idx[u]
            sig_v = node_idx_to_signal_idx[v]
            
            vals = y_edge[i]
            val_next = complex(vals[0], vals[1])
            val_fext = complex(vals[2], vals[3])
            
            # 1. NEXT (Near End): Input u <-> Input v
            # S31 logic: Port (2*u) <-> Port (2*v)
            p_u_in = 2 * sig_u
            p_v_in = 2 * sig_v
            
            s_matrix[p_u_in, p_v_in] = val_next
            s_matrix[p_u_in + 1, p_v_in + 1] = val_next
            # Note: Since graph usually contains both edge u->v and v->u, 
            # this will be overwritten consistently. 
            
            # 2. FEXT (Far End): Input u <-> Output v
            # S41 logic: Port (2*u) <-> Port (2*v + 1)
            p_v_out = 2 * sig_v + 1
            
            s_matrix[p_u_in, p_v_out] = val_fext
            s_matrix[p_v_out, p_u_in] = val_fext # Reciprocity

    return {
        "inputs": params,
        "s_matrix": s_matrix,
        "active_nodes": node_types # Returns list of [1, -1, 1...]
    }

def extract_result_from_pyg_data(data, s_matrix):
    """Inverse of convert_pyg_data_to_result: read node (S21, S11) and edge
    (NEXT, FEXT) values back out of a (possibly modified, e.g. passivity-clipped)
    S-matrix, using the same port mapping. Returns torch tensors y_node [N, 4]
    and y_edge [E, 4] (zeros on ground nodes / non-signal edges).
    """
    if hasattr(data, "cpu"):
        data = data.cpu()
    node_types = data.x.numpy()[:, 0].astype(int)
    edge_index = data.edge_index.numpy()
    node_mask = data.node_mask.numpy()
    edge_mask = data.edge_mask.numpy()

    node_idx_to_signal_idx = {}
    cur = 0
    for k, t in enumerate(node_types):
        node_idx_to_signal_idx[k] = -1
        if t == 1:
            node_idx_to_signal_idx[k] = cur
            cur += 1

    y_node = np.zeros((len(node_types), 4), dtype=np.float32)
    y_edge = np.zeros((edge_index.shape[1], 4), dtype=np.float32)

    for k in range(len(node_types)):
        if node_mask[k]:
            s = node_idx_to_signal_idx[k]
            s21, s11 = s_matrix[2 * s, 2 * s + 1], s_matrix[2 * s, 2 * s]
            y_node[k] = [s21.real, s21.imag, s11.real, s11.imag]
    for i in range(edge_index.shape[1]):
        if edge_mask[i]:
            su = node_idx_to_signal_idx[edge_index[0, i]]
            sv = node_idx_to_signal_idx[edge_index[1, i]]
            nxt, fxt = s_matrix[2 * su, 2 * sv], s_matrix[2 * su, 2 * sv + 1]
            y_edge[i] = [nxt.real, nxt.imag, fxt.real, fxt.imag]

    return torch.from_numpy(y_node), torch.from_numpy(y_edge)


# ==========================================
# Example Usage / Verification
# ==========================================
if __name__ == "__main__":
    # Create dummy data to test round-trip
    print("Testing conversion...")
    
    # Mocking a PyG Data Object (similar to what your loader produces)
    # 2 Signals (idx 0, 1) and 1 Ground (idx 2)
    x_mock = torch.tensor([
        [1.0,  5e-6, 10e-6, 50e-6, 1e-6, 300.0, 1e9], # Sig 1
        [1.0,  5e-6, 10e-6, 50e-6, 1e-6, 300.0, 1e9], # Sig 2
        [-1.0, 5e-6, 10e-6, 50e-6, 1e-6, 300.0, 1e9]  # Gnd
    ])
    
    # 2 Edges: Sig1->Sig2 (Index 0), Sig2->Sig1 (Index 1)
    # We ignore Gnd edges for S-matrix reconstruction as they are masked out
    edge_index_mock = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    # Targets
    y_node_mock = torch.tensor([
        [0.9, 0.1, 0.05, 0.01], # Sig 1 S21, S11
        [0.8, 0.2, 0.06, 0.02], # Sig 2 S21, S11
        [0.0, 0.0, 0.00, 0.00]  # Gnd (Ignored)
    ])
    
    y_edge_mock = torch.tensor([
        [0.01, 0.001, 0.02, 0.002], # Edge 0->1 (NEXT, FEXT)
        [0.01, 0.001, 0.02, 0.002]  # Edge 1->0 (Symmetric)
    ])
    
    node_mask_mock = torch.tensor([True, True, False])
    edge_mask_mock = torch.tensor([True, True])
    
    data_mock = Data(x=x_mock, edge_index=edge_index_mock, 
                     y_node=y_node_mock, y_edge=y_edge_mock,
                     node_mask=node_mask_mock, edge_mask=edge_mask_mock)
    
    # --- RUN CONVERSION ---
    res = convert_pyg_data_to_result(data_mock)
    
    print("\nReconstructed Parameters:")
    print(res['inputs'])
    
    print("\nReconstructed S-Matrix (4x4 for 2 signals):")
    # Clean print
    print(np.round(res['s_matrix'], 4))
    
    print("\nVerification:")
    # Check S11 of Sig 1 (Port 0->0) -> Should be 0.05 + 0.01j
    print(f"S11 (True): 0.05+0.01j | (Recon): {res['s_matrix'][0,0]}")
    # Check NEXT of Sig 1->2 (Port 0->2) -> Should be 0.01 + 0.001j
    print(f"S31 (True): 0.01+0.001j | (Recon): {res['s_matrix'][0,2]}")