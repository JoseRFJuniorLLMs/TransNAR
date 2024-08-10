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
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class GCNModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModule, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class NAR(nn.Module):
    def __init__(self, embed_dim, hidden_dim, gcn_out_dim):
        super(NAR, self).__init__()
        self.gcn = GCNModule(embed_dim, hidden_dim, gcn_out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(gcn_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        return self.mlp(x)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + self.dropout(attn_output))

class TransNAR_GCN(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, num_layers, ffn_dim, gcn_hidden_dim, gcn_out_dim, dropout=0.1):
        super(TransNAR_GCN, self).__init__()
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.nar = NAR(embed_dim, gcn_hidden_dim, gcn_out_dim)
        
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        
        self.decoder = nn.Linear(embed_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)
        
        self.initialize_weights()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.normal_(m.in_proj_weight, std=0.02)
                nn.init.normal_(m.out_proj.weight, std=0.02)

    def forward(self, x, edge_index):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        nar_output = self.nar(x, edge_index)
        
        x = self.cross_attention(x, nar_output, nar_output)
        
        output = self.decoder(x)
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
            with torch.no_grad():
                for batch in val_loader:
                    output = self(batch.x, batch.edge_index)
                    loss = F.mse_loss(output, batch.y)
                    val_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
            
            torch.save(self.state_dict(), f'transnar_gcn_checkpoint_epoch_{epoch+1}.pth')

# Exemplo de uso
input_dim = 100
output_dim = 50
embed_dim = 256
num_heads = 8
num_layers = 6
ffn_dim = 1024
gcn_hidden_dim = 128
gcn_out_dim = 256

model = TransNAR_GCN(input_dim, output_dim, embed_dim, num_heads, num_layers, ffn_dim, gcn_hidden_dim, gcn_out_dim)
input_data = torch.randn(32, 100, input_dim)
edge_index = torch.randint(0, 32, (2, 160))  # Exemplo de índice de arestas mais complexo

class ExampleBatch:
    def __init__(self, x, edge_index, y):
        self.x = x
        self.edge_index = edge_index
        self.y = y

train_loader = [ExampleBatch(input_data, edge_index, torch.randn(32, 100, output_dim))]
val_loader = [ExampleBatch(input_data, edge_index, torch.randn(32, 100, output_dim))]

# Treinamento do modelo
model.train_model(train_loader, val_loader, num_epochs=100)