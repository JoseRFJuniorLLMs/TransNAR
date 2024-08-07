import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Definição do Dataset
class SimpleDataset(Dataset):
    def __init__(self, num_samples, seq_length, input_dim):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.data = torch.randn(num_samples, seq_length, input_dim)
        self.labels = torch.randint(0, 2, (num_samples, seq_length, 50))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Definição do modelo (usando TransNAR do exemplo anterior)
class TransNAR(nn.Module):
    # ... Definição do modelo como no exemplo anterior ...

# Inicializar o modelo, critério e otimizador
input_dim = 100
output_dim = 50
embed_dim = 256
num_heads = 8
num_layers = 6
ffn_dim = 1024

model = TransNAR(input_dim, output_dim, embed_dim, num_heads, num_layers, ffn_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Criar o DataLoader
num_samples = 1000
seq_length = 100
batch_size = 32
dataset = SimpleDataset(num_samples, seq_length, input_dim)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loop de treinamento
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
