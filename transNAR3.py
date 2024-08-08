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
        
        # Inicialização dos pesos
        self.initialize_weights()
        
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
        
        # Otimizador
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def initialize_weights(self):
        # Inicialização de Xavier para camadas lineares
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Inicialização normal para camadas de atenção
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                nn.init.normal_(m.in_proj_weight, std=0.02)
                nn.init.normal_(m.out_proj.weight, std=0.02)

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

    def train_model(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                output = self(batch.x, batch.edge_index, batch.edge_attr)
                loss = F.mse_loss(output, batch.y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            self.eval()
            val_loss = 0
            for batch in val_loader:
                output = self(batch.x, batch.edge_index, batch.edge_attr)
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

model = TransNAR(input_dim, output_dim, embed_dim, num_heads, num_layers, ffn_dim)
input_data = torch.randn(32, 100, input_dim)
edge_index = torch.tensor([[0, 1], [1, 0]])  # Example edge index
edge_attr = torch.randn(edge_index.size(1))  # Example edge attributes

# Treinamento do modelo
train_loader = ... # Carregador de dados de treinamento
val_loader = ... # Carregador de dados de validação
model.train_model(train_loader, val_loader, num_epochs=100)