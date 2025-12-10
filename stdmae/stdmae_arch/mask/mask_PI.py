import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_

from .patch import PatchEmbedding
from .maskgenerator import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers


def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


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
    

class Mask_PI(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, hidden_dim, num_heads, mlp_ratio, dropout,  mask_ratio, encoder_depth, decoder_depth,spatial=False, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.spatial=spatial
        self.selected_feature = 0
        self.dropout = dropout

        # encoder specifics
        # # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)

        # encoder
        self.encoder = nn.ModuleList()
        for i in range(self.encoder_depth):
            layer = FFN(self.embed_dim, self.hidden_dim, dropout = self.dropout)
            self.encoder.append(layer)
            
        # decoder specifics
        # transform layer
        self.enc_2_dec = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # # decoder
        self.decoder = nn.ModuleList()
        for i in range(self.decoder_depth):
            layer = FFN(self.embed_dim, self.hidden_dim, dropout = self.dropout)
            self.decoder.append(layer)

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=True):
        """

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, C, P * L],
                                                which is used in the Pre-training.
                                                P is the number of patches.
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        # patchify and embed input
        if mask:
            patches = self.patch_embedding(long_term_history)  # B, N, d, P
            patches = patches.transpose(-1, -2)  # B, N, P, d
            batch_size, num_nodes, num_time, num_dim  =  patches.shape
            
            Maskg = MaskGenerator(patches.shape[2], self.mask_ratio)
            unmasked_token_index, masked_token_index = Maskg.uniform_rand()
            
            mask_input = self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), patches.shape[-1])
            unmask_input = patches[:, :, unmasked_token_index, :]
            
            for layer in self.encoder:
                mask_input = layer(mask_input)
                
            for layer in self.encoder:
                unmask_input = layer(unmask_input)
            
            mask_hidden = mask_input
            unmask_hidden = unmask_input

        else:
            batch_size, num_nodes, _, _ = long_term_history.shape
            # patchify and embed input
            patches = self.patch_embedding(long_term_history)     # B, N, d, P
            patches = patches.transpose(-1, -2)         # B, N, P, d
            
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches  # B, N, P, d
            for layer in self.encoder:
                encoder_input = layer(encoder_input)
                
            hidden_states_unmasked = encoder_input
            return hidden_states_unmasked, unmasked_token_index, masked_token_index
        # encoding

        return mask_hidden, unmask_hidden, unmasked_token_index, masked_token_index

    def decoding(self, mask_hidden, unmask_hidden, unmasked_token_index, masked_token_index):

        # encoder 2 decoder layer
        hidden_states_unmasked = self.enc_2_dec(unmask_hidden)# B, N, P, d/# B, P, N,  d
        hidden_states_masked = self.enc_2_dec(mask_hidden)
        
        # 获取批次大小和节点数
        B,N = hidden_states_unmasked.shape[0], hidden_states_unmasked.shape[1]
        total_P = len(unmasked_token_index) + len(masked_token_index)
        
        hidden_states_full = torch.zeros(B, N, total_P , self.embed_dim, device=unmask_hidden.device)
        hidden_states_full[:, :, unmasked_token_index, :] = hidden_states_unmasked
        hidden_states_full[:, :, masked_token_index, :] = hidden_states_masked
        
        for layer in self.decoder:
            hidden_states_full = layer(hidden_states_full)

        # prediction (reconstruction)
        reconstruction_full = self.output_layer(hidden_states_full.view(B, N, -1, self.embed_dim))
        reconstruction_full = reconstruction_full.reshape(B, N, -1).unsqueeze(2)

        return reconstruction_full

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        
        # feed forward
        if self.mode == "pre-train":
            # encoding
            mask_hidden, unmask_hidden, unmasked_token_index, masked_token_index = self.encoding(history_data)
            # decoding
            reconstruction_full = self.decoding(mask_hidden, unmask_hidden, unmasked_token_index, masked_token_index)
            # print(reconstruction_full.shape, history_data.shape)
            return reconstruction_full, history_data
        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full

