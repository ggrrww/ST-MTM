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
        
class Perio_Pretraining_Transformer2(nn.Module):
    def __init__(self, num_nodes, input_len, in_channel, embed_dim, hidden_dim, 
                 in_depth, out_depth, dropout, mode="pre-train"):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.perio_len = 288*7
        self.encoder_layers = in_depth
        self.decoder_layers = out_depth
        self.dropout = dropout
        self.input_len = input_len
        self.mode = mode
        
        self.perio_init = nn.Parameter(torch.zeros(self.perio_len, num_nodes))  #(L,N)
        
        self.linear_TE = nn.Linear(self.input_len, self.embed_dim)
        
        self.encoder = nn.ModuleList()
        for i in range(self.encoder_layers):
            layer = FFN(self.embed_dim, self.hidden_dim, dropout = self.dropout)
            self.encoder.append(layer)
            
        self.decoder = nn.ModuleList()
        for i in range(self.decoder_layers):
            layer = FFN(self.embed_dim, self.hidden_dim, dropout = self.dropout)
            self.decoder.append(layer)
            
        self.out_proj = nn.Linear(self.embed_dim, self.input_len)
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs):
        # history_data: B,L,N,3
        B,T,N,_ = history_data.shape
        source = history_data[..., 0]   #(B,T,N)
        
        t_i_d_data = history_data[..., 1]  #(B,T,N)
        d_i_w_data   = history_data[..., 2]
        pose_data =  ((t_i_d_data + d_i_w_data) * 288).type(torch.LongTensor)  #(B,T,N)
        pose_data = pose_data.to(history_data.device)
        
        perio_init_expand = self.perio_init.unsqueeze(0).expand(B, -1, -1) 
        pose_init = perio_init_expand.gather(
                1,
                pose_data
        )  #(B,T,N)
        pose_init = pose_init.transpose(1,2)  #(B,N,T)
        pose_emb = self.linear_TE(pose_init)  #(B,N,D)
        
        if self.mode == "pre-train":
            for layer in self.encoder:
                pose_emb = layer(pose_emb)
            hidden_states = pose_emb
            for layer in self.decoder:
                hidden_states = layer(hidden_states)  #(B,N,D)
            out = hidden_states
            out = self.out_proj(out)  #(B,N,T)
            out = out.transpose(1, 2)  #(B,T,N)
            
            return out, source
        else:
            whole_perio = self.perio_init.transpose(0,1)  #(N,L)
            whole_perio = whole_perio.reshape(N,-1,self.input_len)  #(N,P,L)
            whole_perio_emb = self.linear_TE(whole_perio)  #(N,P,D)
            
            for layer in self.encoder:
                whole_perio_emb = layer(whole_perio_emb)
            hidden_states = whole_perio_emb #(B,N,D)
            return hidden_states
          

        
        
