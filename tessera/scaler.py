import torch


class InputScaler:
    """Z-score normalizer for node features and log-transformed edge features."""

    def __init__(self):
        self.node_mean = None
        self.node_std = None
        self.edge_mean = None
        self.edge_std = None
        self.fitted = False

    def fit(self, data_list):
        """Calculates Mean and Std from a list of Data objects."""
        print("[Scaler] Fitting normalization statistics on Training Data...")
        
        # 1. Collect all Node Features
        all_x = torch.cat([d.x for d in data_list], dim=0)
        
        # We clone to avoid modifying original data in-place during gathering
        x_processed = all_x.clone()

        # Calculate Mean/Std
        # Note: We usually DON'T normalize Categorical data (Type at index 0).
        # But since Type is -1/1, standardizing it is harmless/redundant. 
        # Let's standardize everything for simplicity.
        self.node_mean = x_processed.mean(dim=0)
        self.node_std = x_processed.std(dim=0) + 1e-12 # Epsilon for stability

        # 2. Collect all Edge Features
        all_edges = torch.cat([d.edge_attr for d in data_list], dim=0)
        
        # Log-transform Edge Features
        # Input: [d, 1/d, 1/d^2]
        e_processed = torch.log10(all_edges + 1e-16)
        
        self.edge_mean = e_processed.mean(dim=0)
        self.edge_std = e_processed.std(dim=0) + 1e-12
        
        self.fitted = True
        print("[Scaler] Fit Complete.")
        print(f"   Node Mean: {self.node_mean}")
        print(f"   Edge Mean: {self.edge_mean}")

    def transform(self, data):
        """Applies normalization to a single Data object."""
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform!")

        # --- Nodes ---
        # 1. Z-Score
        data.x = (data.x - self.node_mean) / self.node_std

        # --- Edges ---
        # 1. Apply Log10
        data.edge_attr = torch.log10(data.edge_attr + 1e-16)
        # 2. Z-Score
        data.edge_attr = (data.edge_attr - self.edge_mean) / self.edge_std
        
        return data

    def save(self, path):
        torch.save({
            'node_mean': self.node_mean,
            'node_std': self.node_std,
            'edge_mean': self.edge_mean,
            'edge_std': self.edge_std
        }, path)

    def load(self, path):
        stats = torch.load(path, weights_only=True)
        self.node_mean = stats['node_mean']
        self.node_std = stats['node_std']
        self.edge_mean = stats['edge_mean']
        self.edge_std = stats['edge_std']
        self.fitted = True