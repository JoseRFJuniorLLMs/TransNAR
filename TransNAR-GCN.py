import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create the positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class GCNModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNModule, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.gcn(x, edge_index)

class NAR(nn.Module):
    def __init__(self, embed_dim, gcn_out_dim):
        super(NAR, self).__init__()
        self.gcn = GCNModule(embed_dim, gcn_out_dim)
    
    def forward(self, x, edge_index):
        return self.gcn(x, edge_index)

class TransNAR(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, num_layers, ffn_dim, gcn_out_dim, dropout=0.1):
        super(TransNAR, self).__init__()
        
        # Camada de Embedding
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Inicialização dos pesos
        self.initialize_weights()
        
        # Camadas Transformer
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Neural Algorithmic Reasoner (NAR) com GCNConv
        self.nar = NAR(embed_dim, gcn_out_dim)
        
        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Decodificador
        self.decoder = nn.Linear(embed_dim, output_dim)
        
        # Camada de normalização final
        self.final_norm = nn.LayerNorm(output_dim)
        
        # Otimizador
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def initialize_weights(self):
        # Inicialização de Xavier para camadas lineares
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Inicialização normal para camadas de atenção
        # (Query/Key/Value)
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                nn.init.normal_(m.in_proj_weight, std=0.02)
                nn.init.normal_(m.out_proj.weight, std=0.02)

    def forward(self, x, edge_index):
        # Embedding e codificação posicional
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Camadas Transformer
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Neural Algorithmic Reasoner (NAR) com GCNConv
        nar_output = self.nar(x, edge_index)
        
        # Cross-Attention between Transformer and NAR outputs
        cross_attn_output, _ = self.cross_attention(x, nar_output, nar_output)
        
        # Decodificação
        output = self.decoder(cross_attn_output)
        
        # Normalização final
        output = self.final_norm(output)
        
        return output

    def train_model(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                output = self(batch.x, batch.edge_index)
                loss = F.mse_loss(output, batch.y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            self.eval()
            val_loss = 0
            for batch in val_loader:
                output = self(batch.x, batch.edge_index)
                loss = F.mse_loss(output, batch.y)
                val_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")
            
            # Salvar checkpoint do modelo
            torch.save(self.state_dict(), f'transnar_checkpoint_epoch_{epoch+1}.pth')

# Exemplo de uso
input_dim = 100
output_dim = 50
embed_dim = 256
num_heads = 8
num_layers = 6
ffn_dim = 1024
gcn_out_dim = 256  # Dimensão da saída do GCNConv

model = TransNAR(input_dim, output_dim, embed_dim, num_heads, num_layers, ffn_dim, gcn_out_dim)
input_data = torch.randn(32, 100, input_dim)
edge_index = torch.tensor([[0, 1], [1, 0]])  # Example edge index

# Criação de dados de exemplo para treinamento
class ExampleBatch:
    def __init__(self, x, edge_index, y):
        self.x = x
        self.edge_index = edge_index
        self.y = y

train_loader = [ExampleBatch(input_data, edge_index, torch.randn(32, 100, output_dim))]
val_loader = [ExampleBatch(input_data, edge_index, torch.randn(32, 100, output_dim))]

# Treinamento do modelo
model.train_model(train_loader, val_loader, num_epochs=100)
