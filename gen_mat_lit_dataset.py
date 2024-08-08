# Criar o dataset de textos matemáticos
math_dataset = TransNARTextDataset('math', num_samples=1000, max_length=512, vocab_size=30522, device=device)
math_dataloader = DataLoader(math_dataset, batch_size=32, shuffle=True)

# Criar o dataset de textos literários
lit_dataset = TransNARTextDataset('literature', num_samples=1000, max_length=512, vocab_size=30522, device=device)
lit_dataloader = DataLoader(lit_dataset, batch_size=32, shuffle=True)

# Treinar o modelo TransNAR
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for (input_ids, attention_masks, labels) in math_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids, attention_masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input_ids.size(0)
    
    epoch_loss = running_loss / len(math_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Math Loss: {epoch_loss:.4f}')

    # Avaliar o modelo no conjunto de dados literário
    model.eval()
    val_loss = 0.0
    for (input_ids, attention_masks, labels) in lit_dataloader:
        with torch.no_grad():
            outputs = model(input_ids, attention_masks)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * input_ids.size(0)
    val_loss /= len(lit_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Literature Validation Loss: {val_loss:.4f}')