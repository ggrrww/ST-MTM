import torch
from torch import nn
from .transformer_layers import TransformerLayers

class FFN(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super(FFN, self).__init__()
        self.gen1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gen2 = nn.Linear(d_model, d_hidden)
        self.gen3 = nn.Linear(d_hidden, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        new_x = self.gen1(x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        
        y = self.dropout(self.activation(self.gen2(y)))
        y = self.dropout(self.gen3(y))
        
        return self.norm2(x + y)
        
class Perio_Pretraining(nn.Module):
    def __init__(self, num_nodes, input_len, in_channel, embed_dim, hidden_dim, depth, dropout, mode="pre-train"):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.perio_len = 12
        self.layers = depth
        self.dropout = dropout
        self.input_len = input_len
        self.mode = mode
        
        self.perio_init = nn.Parameter(torch.zeros(self.perio_len, num_nodes))  #(L,N)
        
        self.linear_perio = nn.Linear(1, self.embed_dim)
        # self.linear_TE = nn.Linear(self.TE_len, self.embed_dim)
        
        self.encoder = nn.ModuleList()
        for i in range(self.layers):
            layer = FFN(self.embed_dim, self.hidden_dim, dropout = self.dropout)
            self.encoder.append(layer)
            
        self.out_proj = nn.Linear(self.embed_dim, 1)
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs):
        # history_data: B,L,N,3
        B,T,N,_ = history_data.shape
        source = history_data[..., 0]   #(B,T,N)
        
        t_i_d_data = history_data[..., 1]  #(B,T,N)
        d_i_w_data   = history_data[..., 2]
        pose_data =  ((t_i_d_data + d_i_w_data) * 288).type(torch.LongTensor)  #(B,T,N)
        pose_data = pose_data.to(history_data.device)
        
        perio_emb = self.linear_perio(self.perio_init.unsqueeze(-1)) #(L,N,D)
        L,_, D = perio_emb.shape
        
        perio_emb_expand = perio_emb.unsqueeze(0).expand(B, -1, -1, -1)  # (B, L, N, D)
        pose_emb = perio_emb_expand.gather(
                1,
                pose_data.unsqueeze(-1).expand(-1, -1, -1, D)
        )  # output (B,L,N,D)
        
        if self.mode == "pre-train":
            for layer in self.encoder:
                pose_emb = layer(pose_emb)  #(B,T,N,D)
            
            out = pose_emb
            out = self.out_proj(out).squeeze(-1)  #(B,T,N)
            
            return out, source
        else:
            return pose_emb

        
        
