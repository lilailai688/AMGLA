'''
2025.05.10
lsh
内容：从检索到构图，再到图卷积
'''
'''
2025.05.19
lsh
内容：当前的视图内检索只是输入对历史的检索，没有利用输入和输入间的相关性和过去到输入的相关性
新增二跳节点：（一跳节点是输入对历史的直接相似度检索，二跳节点是与输入相似的一跳节点相似的历史节点（二跳）
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def local_intra_view_retrieval(local_query, local_retrieval_db, k):
    """
    局部视图内检索（支持批量输入），使用独立的 local_query 和 local_retrieval_db

    Args:
        local_query: shape (B, C, D)
        local_retrieval_db: shape (N, C, D)
        k: top-k 值

    Returns:
        topk_similarities: shape (B, C, k)
        topk_indices: shape (B, C, k)
    """
    B, C, D = local_query.shape
    N, _, _ = local_retrieval_db.shape

    # 扩展 retrieval_db 到 (1, N, C, D)，用于广播
    db_expanded = local_retrieval_db.unsqueeze(0)  # (1, N, C, D)

    # query: (B, C, D) -> (B, 1, C, D)
    query_expanded = local_query.unsqueeze(1)

    # 计算所有样本、所有视图下的相似度: (B, C, N)
    similarities = torch.einsum('bcsd,bncd->bcn', [query_expanded, db_expanded])  # (B, C, N)

    # 获取 top-k 的相似度和索引
    topk_values, topk_indices = torch.topk(similarities, k, dim=2, largest=True, sorted=True)

    return topk_values, topk_indices


def build_view_mask(C, N):
    """
    构建一个 (C, N*C) 的掩码矩阵，用于屏蔽与当前视图相同的视图

    Args:
        C: 视图数量
        N: 数据库中产品数量

    Returns:
        mask: shape (C, N*C), 其中 mask[c, i] = True 表示要屏蔽该位置
        以不同视图下的产品进行拼接排列的，
    """
    mask = torch.zeros((C, N * C), dtype=torch.bool)
    for c in range(C):
        for i in range(N):
            pos = i * C + c  # 第 i 个产品，第 c 个视图在展平后的索引
            mask[c, pos] = True  # 屏蔽掉当前视图（相同 view id）
    return mask


def global_inter_view_retrieval_with_mask(global_query, global_retrieval_db, view_mask, k):
    """
    全局跨视图检索，使用预定义的 view_mask 屏蔽相同视图区域（支持批量输入）

    Args:
        global_query: shape (B, C, D)
        global_retrieval_db: shape (N, C, D)
        view_mask: shape (C, N*C)，预定义的掩码，屏蔽相同视图
        k: top-k 值

    Returns:
        topk_similarities: shape (B, C, k)
        topk_indices: shape (B, C, k)
    """
    B, C, D = global_query.shape
    N, _, _ = global_retrieval_db.shape

    # 展平数据库: (N*C, D)
    flat_db = global_retrieval_db.view(N * C, D)

    # 展平查询向量: (B*C, D)
    flat_query = global_query.view(B * C, D)

    # 计算相似度矩阵: (B*C, N*C)
    similarities = torch.mm(flat_query, flat_db.t())

    # Reshape 回 (B, C, N*C)
    similarities = similarities.view(B, C, N * C)

    # 应用掩码（广播到 batch 维度）
    mask = view_mask.unsqueeze(0).expand(B, -1, -1)  # (B, C, N*C)
    masked_similarities = similarities.masked_fill(mask, float('-inf'))

    # 获取全局 top-k
    topk_values, topk_indices = torch.topk(masked_similarities, k, dim=2, largest=True, sorted=True)

    return topk_values, topk_indices



def two_hop_retrieval_separate(query, retrieval_db, k1=5, k2=3, mode='intra'):
    """
    二跳检索，分别返回一跳和二跳结果
    
    Args:
        query: (B, C, D)
        retrieval_db: (N, C, D)
        k1: 一跳top-k
        k2: 二跳top-k
        mode: 'intra'或'inter'
    
    Returns:
        hop1_sim: (B, C, k1) - 一跳相似度
        hop1_idx: (B, C, k1) - 一跳索引
        hop2_sim: (B, C, k1, k2) - 二跳相似度
        hop2_idx: (B, C, k1, k2) - 二跳索引
    """
    B, C, D = query.shape
    N = retrieval_db.shape[0]
    
    # 一跳检索 ------------------------------------------------------------
    if mode == 'intra':
        hop1_sim, hop1_idx = local_intra_view_retrieval(query, retrieval_db, k1) # (B,C,k1)
    else:
        view_mask = build_view_mask(C, N)
        hop1_sim, hop1_idx = global_inter_view_retrieval_with_mask(query, retrieval_db, view_mask, k1) # (B,C,k1)

    # if mode == 'intra':
    
    # 二跳检索 ------------------------------------------------------------
    # 获取一跳节点的所有特征 (B, C, k1, D)  hop1_idx   
    # 方法：使用gather在N维度上索引，同时保持视图对应
    # 二跳特征收集（改进部分）
    retrieval_db_expanded = retrieval_db.permute(1, 0, 2)  # (C, N, D)
    
    if mode == 'intra':
        hop1_features = torch.gather(
            retrieval_db_expanded.unsqueeze(0).expand(B, -1, -1, -1),
            dim=2,
            index=hop1_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        )
    else:
        sample_idx = hop1_idx // C # 确定是哪个商品
        view_idx = hop1_idx % C # 确定是哪个视图
        # 就是行索引和列索引
        # 先取出商品的
        sample_features = torch.gather(
            retrieval_db_expanded.unsqueeze(0).expand(B, -1, -1, -1),
            dim=2,
            index=sample_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        )
        # 再取出视图所对应的 sample_features[b, view_idx[b,c,k], k]
        hop1_features = torch.gather(
            sample_features,
            dim=1,
            index=view_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        )
    
    if mode == 'intra':
        # 视图内二跳检索
        # 计算相似度 (B,C,k1,N)
        sim = torch.einsum('bckd,ncd->bckn', hop1_features, retrieval_db)
        
        # 正确屏蔽一跳节点本身（而不是简单的对角线）
        # hop1_idx: (B,C,k1) 包含原始数据库中的N维索引
        # 创建屏蔽矩阵 (B,C,k1,N)
        mask = torch.zeros((B, C, k1, N), dtype=torch.bool, device=query.device)
        
        # 为每个一跳节点屏蔽其在数据库中的原始位置
        # 使用scatter_高效设置屏蔽位
        mask.scatter_(3, 
                    hop1_idx.unsqueeze(-1),  # (B,C,k1,1)
                    torch.ones_like(hop1_idx, dtype=torch.bool).unsqueeze(-1))

        # 应用屏蔽
        sim = sim.masked_fill(mask, float('-inf'))
        
        # 获取top-k2 (B,C,k1,k2)
        hop2_sim, hop2_idx = torch.topk(sim, k2, dim=-1)
        
    else:
        # 跨视图二跳检索
        view_mask = build_view_mask(C, N)

        
        # 展平一跳特征 (B*C*k1, D)
        flat_hop1 = hop1_features.view(-1, D)
        
        # 计算相似度矩阵 (B*C*k1, N*C)
        flat_db = retrieval_db.view(-1, D)
        sim = torch.matmul(flat_hop1, flat_db.t())  # (B*C*k1, N*C)
        sim = sim.view(B, C, k1, N*C)

        # 双重掩码准备 ---------------------------------------------------
        # 1. 视图掩码 (屏蔽相同视图)
        view_mask_expanded = view_mask.unsqueeze(0).unsqueeze(2)  # (1,C,1,N*C)
        
        # 2. 一跳节点掩码 (屏蔽自身)
        # hop1_idx是(B,C,k1)的全局索引(N*C范围)
        self_mask = torch.zeros((B, C, k1, N*C), dtype=torch.bool, device=sim.device)
        
        # 使用scatter_将一跳节点位置设为True(需要屏蔽)
        self_mask.scatter_(
            dim=3,
            index=hop1_idx.unsqueeze(-1),  # (B,C,k1,1)
            src=torch.ones_like(hop1_idx, dtype=torch.bool).unsqueeze(-1)
        )
  
        # 合并掩码
        combined_mask = view_mask_expanded | self_mask
        
        # 应用视图掩码
        # mask = view_mask.unsqueeze(0).unsqueeze(2)  # (1, C, 1, N*C)
        # sim = sim.masked_fill(mask.expand(B, -1, k1, -1), float('-inf'))

        # 应用双重掩码
        sim = sim.masked_fill(combined_mask, float('-inf'))
        
        # 获取top-k2 (B,C,k1,k2)
        hop2_sim, hop2_idx = torch.topk(sim, k2, dim=-1)
    
    return hop1_sim, hop1_idx, hop2_sim, hop2_idx

# 对邻居节点的未来七天销量特征进行升维
class MultiHeadMLP(nn.Module):
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
        # x: (N, 7)
        out = self.mlp(x)  # (N, C*d)
        return out.view(-1, self.C, self.d)  # (N, C, d)



def gather_features_by_index(retrieval_db_features, indices, is_cross_view=False):
    """
    根据视图内的索引取出查询库中的特征。
    
    Args:
        retrieval_db_features: shape (N, C, D)
        indices: 
            - 如果 is_cross_view=False，shape 可以是 (B, C, k1) 或 (B, C, k1, k2), 其中 values in [0, N)
            - 如果 is_cross_view=True，shape 可以是 (B, C, k1) 或 (B, k1, k2), 其中 values in [0, N*C)
        is_cross_view: bool, 指示是否进行跨视图查询
        
    Returns:
        selected_features: 
            - 如果 indices 是 (B, C, k1)，则返回形状为 (B, C, k1, D)
            - 如果 indices 是 (B, C, k1, k2)，则返回形状为 (B, C, k1, k2, D)
    """
    N, C, D = retrieval_db_features.shape
    
    if indices.dim() == 4:  # 处理二跳索引 (B, C, k1, k2)
        B, _, k1, k2 = indices.shape

        if not is_cross_view:
        
            # Step 1: 构建 flat index: (B, C, k1, k2) -> (B, C, k1, k2)
            offsets = torch.arange(C, device=indices.device) * N  # (C,)
            flat_indices = indices + offsets.view(1, C, 1, 1)     # (B, C, k1, k2)
        else:
            flat_indices = indices

        # Step 2: 扁平化数据库特征 (N, C, D) -> (N*C, D)
        flat_features = retrieval_db_features.permute(1, 0, 2).reshape(-1, D)  # (N*C, D)

        # Step 3: 展平索引以用于 gather
        flat_indices = flat_indices.unsqueeze(-1).expand(B, C, k1, k2, D)  # (B, C, k1, k2, D)

        # Step 4: gather 特征
        selected_features = torch.gather(
            flat_features.unsqueeze(0).expand(B, -1, -1),
            dim=1,
            index=flat_indices.reshape(B, -1, D)
        ).reshape(B, C, k1, k2, D)  # 恢复原始结构

    elif indices.dim() == 3:  # 处理一跳索引 (B, C, k1)
        B, _, k1 = indices.shape
        if not is_cross_view:
            # Step 1: 构建 flat index: (B, C, k1) -> (B, C, k1)
            offsets = torch.arange(C, device=indices.device) * N  # (C,)
            flat_indices = indices + offsets.view(1, C, 1)         # (B, C, k1)
        else:
            flat_indices = indices

        # Step 2: 扁平化数据库特征 (N, C, D) -> (N*C, D)
        flat_features = retrieval_db_features.permute(1, 0, 2).reshape(-1, D)  # (N*C, D)

        # Step 3: 展平索引以用于 gather
        flat_indices = flat_indices.unsqueeze(-1).expand(B, C, k1, D)  # (B, C, k1, D)

        # Step 4: gather 特征
        selected_features = torch.gather(
            flat_features.unsqueeze(0).expand(B, -1, -1),
            dim=1,
            index=flat_indices.reshape(B, -1, D)
        ).reshape(B, C, k1, D)  # 恢复原始结构

    else:
        raise ValueError("Indices tensor must have 3 or 4 dimensions.")
    
    return selected_features



class GraphConvolutionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,  is_cross_view=False, dropout_rate=0.2):
        """
        初始化图卷积模块
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度（用于降维）
            output_dim: 输出特征维度
            dropout_rate: Dropout的概率
        """
        super(GraphConvolutionModule, self).__init__()
        self.reducer_2_1 = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
        )
        self.reducer_1_input = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.is_cross_view = is_cross_view

    def forward(self, hop2_sim, hop2_idx,  hop1_sim, hop1_idx, retrieval_db_features):
        """
        前向传播函数
        
        Args:
            hop2_sim: shape (B, C, k1, k2) - 二跳相似度
            hop2_idx: shape (B, C, k1, k2) - 二跳索引
            hop1_idx: shape (B, C, k1) - 一跳索引
            retrieval_db_features: shape (N, C, d) - 数据库特征
            
            
        Returns:
            input_aggregated_with_self: shape (B, C, output_dim) - 聚合后的输入节点特征
        """
        # Step 1: 二跳 → 一跳（含自身）
        hop1_aggregated_with_self = self.propagate_two_hop_to_one_hop(
            hop2_sim, hop2_idx, hop1_idx, retrieval_db_features
        )
        
        # Dropout after aggregation
        hop1_aggregated_with_self = self.dropout(hop1_aggregated_with_self)
        
        # Step 2: 一跳 → 输入
        input_aggregated = self.propagate_one_hop_to_input(
            hop1_sim, hop1_aggregated_with_self
        )
        
        return input_aggregated
    
    def propagate_two_hop_to_one_hop(self, hop2_sim, hop2_idx, hop1_idx, retrieval_db_features):
        B, C, k1, k2 = hop2_sim.shape
        N, _, d = retrieval_db_features.shape
        
        # 获取二跳节点特征并加权聚合
        hop2_features = gather_features_by_index(retrieval_db_features, hop2_idx, self.is_cross_view)  # (B, C, k1, k2, d)
        weights = torch.softmax(hop2_sim, dim=-1)  # (B, C, k1, k2)
        hop2_agg = torch.einsum('bckld,bckl->bckd', hop2_features, weights)  # (B, C, k1, d)
        
        # 获取一跳节点自身特征
        hop1_self_features = gather_features_by_index(retrieval_db_features, hop1_idx, self.is_cross_view)  # (B, C, k1, d)
        
        # 合并自身 + 二跳聚合
        hop1_combined = torch.cat([hop1_self_features, hop2_agg], dim=-1)  # (B, C, k1, 2*d)
        
        # 使用 reducer 降维
        B_, C_, k1_, _ = hop1_combined.shape
        hop1_combined = self.reducer_2_1(hop1_combined.view(B_ * C_ * k1_, -1))
        hop1_combined = hop1_combined.view(B_, C_, k1_, -1)
        
        return hop1_combined  # (B, C, k1, output_dim)
    
    def propagate_one_hop_to_input(self, hop1_sim, hop1_aggregated_with_self):
        B, C, k1 = hop1_sim.shape
        d = hop1_aggregated_with_self.shape[-1]
        
        # 聚合一跳节点（加权求和）
        weights = torch.softmax(hop1_sim, dim=-1)  # (B, C, k1)
        hop1_agg = torch.einsum('bck,bckd->bcd', weights, hop1_aggregated_with_self)  # (B, C, d)
        
 
        # 使用 reducer 降维
        input_combined = self.reducer_1_input(hop1_agg.view(B * C, -1)).view(B, C, -1)
        
        return input_combined  # (B, C, output_dim)




class DimensionalityReducer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DimensionalityReducer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.mlp(x)


# 测试
B, C, D, N, k1, k2 = 8, 4, 64, 100, 5, 3
# 不同的局部和全局向量
local_query = torch.randn(B, C, D)
local_retrieval_db = torch.randn(N, C, D)

global_query = torch.randn(B, C, D)
global_retrieval_db = torch.randn(N, C, D)

# 测试视图内检索
intra_hop1_sim, intra_hop1_idx, intra_hop2_sim, intra_hop2_idx = two_hop_retrieval_separate(
    local_query, local_retrieval_db, k1, k2, 'intra'
)
print(f"Intra-view hop1: {intra_hop1_sim.shape}, {intra_hop1_idx.shape}")
print(f"Intra-view hop2: {intra_hop2_sim.shape}, {intra_hop2_idx.shape}")




# 测试跨视图检索
inter_hop1_sim, inter_hop1_idx, inter_hop2_sim, inter_hop2_idx = two_hop_retrieval_separate(
    global_query, global_retrieval_db, k1, k2, 'inter'
)
print(f"Inter-view hop1: {inter_hop1_sim.shape}, {inter_hop1_idx.shape}")
print(f"Inter-view hop2: {inter_hop2_sim.shape}, {inter_hop2_idx.shape}")

# 上面的代码已经完成了一跳节点和二跳节点的索引位置查找，即已经完成了一跳和二跳的构图
# 下面就是利用学习到的图结构进行图卷积，先从二跳节点的信息传播到一跳节点，再从一跳节点传播到输入节点


# 构建销量特征库
sales_features = torch.randn(N, 7)

# 使用MultiHeadMLP将其转换为检索库特征
mlp_model = MultiHeadMLP(7, (N, C, 64))  # 假设d=64
retrieval_db_features = mlp_model(sales_features)  # (N, C, d)


# 初始化视图内图卷积模块
inter_graph_conv_module = GraphConvolutionModule(input_dim=D, hidden_dim=D, output_dim=D, is_cross_view=False)

# 执行前向传播
aggregated_local_features = inter_graph_conv_module(intra_hop2_sim,intra_hop2_idx, intra_hop1_sim,intra_hop1_idx, retrieval_db_features)
print("Final aggregated input features:", aggregated_local_features.shape)



# 初始化跨视图图卷积模块
cross_graph_conv_module = GraphConvolutionModule(input_dim=D, hidden_dim=D, output_dim=D,is_cross_view=True)

# 执行前向传播
aggregated_global_features = cross_graph_conv_module(inter_hop2_sim,inter_hop2_idx, inter_hop1_sim,inter_hop1_idx, retrieval_db_features)
print("Final aggregated input features:", aggregated_global_features.shape)



# 对视图内和跨视图的信息进行拼接
concatenated_features = torch.cat((aggregated_local_features, aggregated_global_features), dim=-1)  # (B, C, 2d)
print("Concatenated Features Shape:", concatenated_features.shape)  # 应为 (B, C, 2d)

# 初始化模型
reducer_model = DimensionalityReducer(2*D, D)  # 假设D=d=64

# 使用模型进行维度压缩，视图内和视图外的信息融合
reduced_features = reducer_model(concatenated_features.view(B*C, -1)).view(B, C, D)  # (B, C, d)
print("Reduced Features Shape:", reduced_features.shape)  # 应为 (B, C, d)


# 将不同视图的信息进行融合
# 将特征reshape为(B, C*d)
flattened_features = reduced_features.view(B, -1)  # (B, C*d)

# 定义用于预测销量的MLP
sales_predictor = nn.Sequential(
    nn.Linear(C*D, 128),  # 假设使用中间层大小为128
    nn.ReLU(),
    nn.Linear(128, 7)  # 预测未来7天的销量
)

# 进行销量预测
predicted_sales = sales_predictor(flattened_features)  # (B, 7)
print("Predicted Sales Shape:", predicted_sales.shape)  # 应为 (B, 7)