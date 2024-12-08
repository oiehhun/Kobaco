import torch
from model.light_gcn import LightGCN
from torch_sparse import SparseTensor

class RecommendationContinuousPromptModel(torch.nn.Module):
    def __init__(self, num_users, num_items, edge_index_path):
        super().__init__()
        self.model = LightGCN(num_users=num_users, num_items=num_items)
        edge_index = torch.load(edge_index_path).type(torch.long)
        self.edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_users + num_items, num_users + num_items))
        
    def forward(self, user_id, item_ids):
        device = next(self.model.parameters()).device
        edge_index = self.edge_index.to(device)
        user_id = user_id.to(device)
        item_ids = item_ids.to(device)
        users_emb_final, _, items_emb_final, _ = self.model(edge_index)

        return torch.cat([users_emb_final[user_id].unsqueeze(dim=1), items_emb_final[item_ids]], dim=1)