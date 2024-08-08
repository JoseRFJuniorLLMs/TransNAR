import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv

class TransNAR(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, num_layers, ffn_dim, dropout=0.1):
        super(TransNAR, self).__init__()
        
        # Camada de Embedding
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Camadas Transformer
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Neural Algorithmic Reasoner (NAR)
        self.nar = NAR(embed_dim)
        
        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Decodificador
        self.decoder = nn.Linear(embed_dim, output_dim)
        
        # Camada de normalização final
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index, edge_attr):
        # Embedding e codificação posicional
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Camadas Transformer
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Neural Algorithmic Reasoner
        nar_output = self.nar(x, edge_index, edge_attr)
        
        # Cross-Attention between Transformer and NAR outputs
        cross_attn_output, _ = self.cross_attention(x, nar_output, nar_output)
        
        # Decodificação
        output = self.decoder(cross_attn_output)
        
        # Normalização final
        output = self.final_norm(output)
        
        return output

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Atenção
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feedforward
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x

class NAR(nn.Module):
    def __init__(self, embed_dim):
        super(NAR, self).__init__()
        self.gcn1 = GCNConv(embed_dim, embed_dim * 2)
        self.gcn2 = GCNConv(embed_dim * 2, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        output, _ = self.gru(x.unsqueeze(1))
        return output.squeeze(1)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Inicializa o tensor de codificação posicional
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)

# Exemplo de uso
input_dim = 100
output_dim = 50
embed_dim = 256
num_heads = 8
num_layers = 6
ffn_dim = 1024

model = TransNAR(input_dim, output_dim, embed_dim, num_heads, num_layers, ffn_dim)
input_data = torch.randn(32, 100, input_dim)
edge_index = torch.tensor([[0, 1], [1, 0]])  # Example edge index
edge_attr = torch.randn(edge_index.size(1))  # Example edge attributes
output = model(input_data, edge_index, edge_attr)
print(output.shape)  # Deve imprimir torch.Size([32, 100, 50])
