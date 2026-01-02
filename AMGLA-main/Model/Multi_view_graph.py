import torch
import torch.nn as nn
import torch.nn.functional as F


def local_intra_view_retrieval(local_query, local_retrieval_db, k):
    """
    Intra-view retrieval (batched).

    Args:
        local_query: (B, C, D)
        local_retrieval_db: (N, C, D)
        k: top-k

    Returns:
        topk_values: (B, C, k)
        topk_indices: (B, C, k)  # indices in [0, N)
    """
    B, C, D = local_query.shape
    N, _, _ = local_retrieval_db.shape

    # Broadcast to compute similarities over N for each (B, C)
    db_expanded = local_retrieval_db.unsqueeze(0)   # (1, N, C, D)
    query_expanded = local_query.unsqueeze(1)       # (B, 1, C, D)

    similarities = torch.einsum('bcsd,bncd->bcn', [query_expanded, db_expanded])  # (B, C, N)
    topk_values, topk_indices = torch.topk(similarities, k, dim=2, largest=True, sorted=True)

    return topk_values, topk_indices


def build_view_mask(C, N):
    """
    Build a mask to exclude same-view entries when database is flattened as (N*C).

    Args:
        C: number of views
        N: number of items

    Returns:
        mask: (C, N*C), mask[c, i] = True means masked out for view c
    """
    mask = torch.zeros((C, N * C), dtype=torch.bool)
    for c in range(C):
        for i in range(N):
            pos = i * C + c  # flattened index for (item i, view c)
            mask[c, pos] = True
    return mask


def global_inter_view_retrieval_with_mask(global_query, global_retrieval_db, view_mask, k):
    """
    Cross-view retrieval (batched), excluding same-view entries via a precomputed mask.

    Args:
        global_query: (B, C, D)
        global_retrieval_db: (N, C, D)
        view_mask: (C, N*C)
        k: top-k

    Returns:
        topk_values: (B, C, k)
        topk_indices: (B, C, k)  # indices in [0, N*C)
    """
    B, C, D = global_query.shape
    N, _, _ = global_retrieval_db.shape

    flat_db = global_retrieval_db.view(N * C, D)      # (N*C, D)
    flat_query = global_query.view(B * C, D)          # (B*C, D)

    similarities = torch.mm(flat_query, flat_db.t())  # (B*C, N*C)
    similarities = similarities.view(B, C, N * C)      # (B, C, N*C)

    mask = view_mask.unsqueeze(0).expand(B, -1, -1)    # (B, C, N*C)
    masked_similarities = similarities.masked_fill(mask, float('-inf'))

    topk_values, topk_indices = torch.topk(masked_similarities, k, dim=2, largest=True, sorted=True)
    return topk_values, topk_indices


def two_hop_retrieval_separate(query, retrieval_db, k1=5, k2=3, mode='intra'):
    """
    Two-hop retrieval with explicit hop-1 and hop-2 outputs.

    Args:
        query: (B, C, D)
        retrieval_db: (N, C, D)
        k1: hop-1 top-k
        k2: hop-2 top-k
        mode: 'intra' (within view) or 'inter' (cross view)

    Returns:
        hop1_sim: (B, C, k1)
        hop1_idx: (B, C, k1)
        hop2_sim: (B, C, k1, k2)
        hop2_idx: (B, C, k1, k2)
    """
    B, C, D = query.shape
    N = retrieval_db.shape[0]

    # Hop-1 retrieval
    if mode == 'intra':
        hop1_sim, hop1_idx = local_intra_view_retrieval(query, retrieval_db, k1)
    else:
        view_mask = build_view_mask(C, N)
        hop1_sim, hop1_idx = global_inter_view_retrieval_with_mask(query, retrieval_db, view_mask, k1)

    # Hop-1 node feature gathering: (B, C, k1, D)
    retrieval_db_expanded = retrieval_db.permute(1, 0, 2)  # (C, N, D)

    if mode == 'intra':
        hop1_features = torch.gather(
            retrieval_db_expanded.unsqueeze(0).expand(B, -1, -1, -1),
            dim=2,
            index=hop1_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        )
    else:
        # hop1_idx is in flattened space [0, N*C)
        sample_idx = hop1_idx // C
        view_idx = hop1_idx % C

        sample_features = torch.gather(
            retrieval_db_expanded.unsqueeze(0).expand(B, -1, -1, -1),
            dim=2,
            index=sample_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        )
        hop1_features = torch.gather(
            sample_features,
            dim=1,
            index=view_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        )

    # Hop-2 retrieval
    if mode == 'intra':
        sim = torch.einsum('bckd,ncd->bckn', hop1_features, retrieval_db)  # (B, C, k1, N)

        # Mask out the hop-1 node itself in the N dimension
        mask = torch.zeros((B, C, k1, N), dtype=torch.bool, device=query.device)
        mask.scatter_(3, hop1_idx.unsqueeze(-1), torch.ones_like(hop1_idx, dtype=torch.bool).unsqueeze(-1))
        sim = sim.masked_fill(mask, float('-inf'))

        hop2_sim, hop2_idx = torch.topk(sim, k2, dim=-1)

    else:
        view_mask = build_view_mask(C, N)

        flat_hop1 = hop1_features.view(-1, D)        # (B*C*k1, D)
        flat_db = retrieval_db.view(-1, D)           # (N*C, D)

        sim = torch.matmul(flat_hop1, flat_db.t())   # (B*C*k1, N*C)
        sim = sim.view(B, C, k1, N * C)

        # Combined mask: exclude same-view entries and the hop-1 node itself
        view_mask_expanded = view_mask.unsqueeze(0).unsqueeze(2)  # (1, C, 1, N*C)
        self_mask = torch.zeros((B, C, k1, N * C), dtype=torch.bool, device=sim.device)
        self_mask.scatter_(3, hop1_idx.unsqueeze(-1), torch.ones_like(hop1_idx, dtype=torch.bool).unsqueeze(-1))

        sim = sim.masked_fill(view_mask_expanded | self_mask, float('-inf'))
        hop2_sim, hop2_idx = torch.topk(sim, k2, dim=-1)

    return hop1_sim, hop1_idx, hop2_sim, hop2_idx


class MultiHeadMLP(nn.Module):
    """Project raw features into (C, d) per item."""
    def __init__(self, input_dim, output_shape):
        super(MultiHeadMLP, self).__init__()
        N, C, d = output_shape
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, C * d),
        )
        self.C = C
        self.d = d

    def forward(self, x):
        out = self.mlp(x)  # (N, C*d)
        return out.view(-1, self.C, self.d)  # (N, C, d)


def gather_features_by_index(retrieval_db_features, indices, is_cross_view=False):
    """
    Gather features from retrieval_db_features using indices.

    Args:
        retrieval_db_features: (N, C, D)
        indices:
            - intra-view: (B, C, k1) or (B, C, k1, k2), values in [0, N)
            - cross-view: (B, C, k1) or (B, C, k1, k2), values in [0, N*C)
        is_cross_view: whether indices are in flattened [0, N*C)

    Returns:
        Selected features with an extra trailing D dimension:
            - (B, C, k1, D) or (B, C, k1, k2, D)
    """
    N, C, D = retrieval_db_features.shape

    if indices.dim() == 4:
        B, _, k1, k2 = indices.shape

        if not is_cross_view:
            offsets = torch.arange(C, device=indices.device) * N
            flat_indices = indices + offsets.view(1, C, 1, 1)
        else:
            flat_indices = indices

        flat_features = retrieval_db_features.permute(1, 0, 2).reshape(-1, D)  # (N*C, D)
        flat_indices = flat_indices.unsqueeze(-1).expand(B, C, k1, k2, D)

        selected_features = torch.gather(
            flat_features.unsqueeze(0).expand(B, -1, -1),
            dim=1,
            index=flat_indices.reshape(B, -1, D)
        ).reshape(B, C, k1, k2, D)

    elif indices.dim() == 3:
        B, _, k1 = indices.shape

        if not is_cross_view:
            offsets = torch.arange(C, device=indices.device) * N
            flat_indices = indices + offsets.view(1, C, 1)
        else:
            flat_indices = indices

        flat_features = retrieval_db_features.permute(1, 0, 2).reshape(-1, D)  # (N*C, D)
        flat_indices = flat_indices.unsqueeze(-1).expand(B, C, k1, D)

        selected_features = torch.gather(
            flat_features.unsqueeze(0).expand(B, -1, -1),
            dim=1,
            index=flat_indices.reshape(B, -1, D)
        ).reshape(B, C, k1, D)

    else:
        raise ValueError("indices must be a 3D or 4D tensor.")

    return selected_features


class GraphConvolutionModule(nn.Module):
    """Two-stage message passing: hop-2 -> hop-1 (with self), then hop-1 -> query."""
    def __init__(self, input_dim, hidden_dim, output_dim, is_cross_view=False, dropout_rate=0.2):
        super(GraphConvolutionModule, self).__init__()
        self.reducer_2_1 = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.reducer_1_input = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.is_cross_view = is_cross_view

    def forward(self, hop2_sim, hop2_idx, hop1_sim, hop1_idx, retrieval_db_features):
        hop1_agg = self.propagate_two_hop_to_one_hop(hop2_sim, hop2_idx, hop1_idx, retrieval_db_features)
        hop1_agg = self.dropout(hop1_agg)
        out = self.propagate_one_hop_to_input(hop1_sim, hop1_agg)
        return out

    def propagate_two_hop_to_one_hop(self, hop2_sim, hop2_idx, hop1_idx, retrieval_db_features):
        # Weighted aggregation over hop-2 neighbors
        hop2_features = gather_features_by_index(retrieval_db_features, hop2_idx, self.is_cross_view)
        weights = torch.softmax(hop2_sim, dim=-1)
        hop2_agg = torch.einsum('bckld,bckl->bckd', hop2_features, weights)

        # Concatenate hop-1 self feature with hop-2 aggregated feature
        hop1_self = gather_features_by_index(retrieval_db_features, hop1_idx, self.is_cross_view)
        hop1_combined = torch.cat([hop1_self, hop2_agg], dim=-1)

        B, C, k1, _ = hop1_combined.shape
        hop1_reduced = self.reducer_2_1(hop1_combined.view(B * C * k1, -1)).view(B, C, k1, -1)
        return hop1_reduced

    def propagate_one_hop_to_input(self, hop1_sim, hop1_aggregated_with_self):
        # Weighted aggregation over hop-1 neighbors
        weights = torch.softmax(hop1_sim, dim=-1)
        hop1_agg = torch.einsum('bck,bckd->bcd', weights, hop1_aggregated_with_self)

        B, C, _ = hop1_agg.shape
        out = self.reducer_1_input(hop1_agg.view(B * C, -1)).view(B, C, -1)
        return out


class DimensionalityReducer(nn.Module):
    """Fuse and compress features."""
    def __init__(self, input_dim, output_dim):
        super(DimensionalityReducer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


# -------------------------------
# Minimal sanity check
# -------------------------------
B, C, D, N, k1, k2 = 8, 4, 64, 100, 5, 3
local_query = torch.randn(B, C, D)
local_retrieval_db = torch.randn(N, C, D)

global_query = torch.randn(B, C, D)
global_retrieval_db = torch.randn(N, C, D)

# Intra-view two-hop retrieval
intra_hop1_sim, intra_hop1_idx, intra_hop2_sim, intra_hop2_idx = two_hop_retrieval_separate(
    local_query, local_retrieval_db, k1, k2, 'intra'
)

# Cross-view two-hop retrieval
inter_hop1_sim, inter_hop1_idx, inter_hop2_sim, inter_hop2_idx = two_hop_retrieval_separate(
    global_query, global_retrieval_db, k1, k2, 'inter'
)

# Build retrieval features (example: 7-day sales features -> (N, C, d))
sales_features = torch.randn(N, 7)
mlp_model = MultiHeadMLP(7, (N, C, 64))
retrieval_db_features = mlp_model(sales_features)  # (N, C, d)

# Intra-view message passing
intra_gcn = GraphConvolutionModule(input_dim=D, hidden_dim=D, output_dim=D, is_cross_view=False)
aggregated_local_features = intra_gcn(
    intra_hop2_sim, intra_hop2_idx, intra_hop1_sim, intra_hop1_idx, retrieval_db_features
)

# Cross-view message passing
cross_gcn = GraphConvolutionModule(input_dim=D, hidden_dim=D, output_dim=D, is_cross_view=True)
aggregated_global_features = cross_gcn(
    inter_hop2_sim, inter_hop2_idx, inter_hop1_sim, inter_hop1_idx, retrieval_db_features
)

# Fuse intra- and inter-view features
concatenated_features = torch.cat((aggregated_local_features, aggregated_global_features), dim=-1)  # (B, C, 2D)
reducer_model = DimensionalityReducer(2 * D, D)
reduced_features = reducer_model(concatenated_features.view(B * C, -1)).view(B, C, D)

# Final predictor (example: predict 7-day horizon)
flattened_features = reduced_features.view(B, -1)  # (B, C*D)
sales_predictor = nn.Sequential(
    nn.Linear(C * D, 128),
    nn.ReLU(),
    nn.Linear(128, 7)
)
predicted_sales = sales_predictor(flattened_features)  # (B, 7)
