# TransNAR - Modelo de Transformer Aprimorado com Neural Algorithmic Reasoner

![TransNAR Architecture](/img.png) 

## Descrição

O **TransNAR** é um modelo avançado de aprendizado de máquina que combina as capacidades de um Transformer com um Neural Algorithmic Reasoner (NAR). Este modelo é projetado para lidar com dados de sequência, aproveitando a atenção multi-cabeça e a codificação posicional dos Transformers, e aprimorando a capacidade de raciocínio com o NAR.

### Componentes Principais

1. **Camada de Embedding**: Converte a entrada de dados em um espaço de alta dimensionalidade.
   
2. **Codificação Posicional**: Adiciona informações sobre a posição dos elementos na sequência para capturar dependências temporais.

3. **Camadas Transformer**: Aplicam mecanismos de atenção multi-cabeça para aprender dependências de longo alcance nas sequências.

4. **Neural Algorithmic Reasoner (NAR)**: Realiza um raciocínio adicional sobre os embeddings produzidos pelas camadas Transformer. Inclui:
   - Camadas de Raciocínio: Processam e ajustam as representações.
   - GRU: Captura dependências temporais e ajusta a representação final.

5. **Decodificador**: Concatena as saídas das camadas Transformer e do NAR e projeta a saída para o espaço desejado.

6. **Camada de Normalização Final**: Normaliza a saída para estabilizar o treinamento e melhorar a performance do modelo.

## Estrutura do Código

- **`TransNAR`**: Classe principal que define a arquitetura do modelo.
- **`TransformerLayer`**: Implementa uma camada do Transformer com atenção multi-cabeça e feed-forward.
- **`NAR`**: Implementa o Neural Algorithmic Reasoner, que inclui camadas de raciocínio e uma GRU.
- **`PositionalEncoding`**: Adiciona informações de posição aos embeddings.

## Uso

Para usar o modelo, siga os passos abaixo:

1. **Instalação**: Certifique-se de ter o PyTorch instalado. Você pode instalá-lo usando o comando:

   ```bash
   pip install torch
   ```

2. **Configuração do Modelo**: Defina os parâmetros do modelo e inicialize-o:

   ```python
   import torch
   from seu_modulo import TransNAR  # Substitua 'seu_modulo' pelo nome do arquivo onde está a classe TransNAR

   # Parâmetros do modelo
   input_dim = 100
   output_dim = 50
   embed_dim = 256
   num_heads = 8
   num_layers = 6
   ffn_dim = 1024

   # Inicializa o modelo
   model = TransNAR(input_dim, output_dim, embed_dim, num_heads, num_layers, ffn_dim)

   # Dados de entrada
   input_data = torch.randn(32, 100, input_dim)  # batch_size = 32, seq_length = 100, input_dim = 100

   # Obtendo a saída
   output = model(input_data)
   print(output.shape)  # Deve imprimir torch.Size([32, 100, 50])
   ```

3. **Treinamento e Avaliação**: Utilize os métodos padrão do PyTorch para treinar e avaliar o modelo. Certifique-se de preparar os dados de entrada e as etiquetas conforme a tarefa que você está resolvendo.

## Contribuições

Contribuições são bem-vindas! Se você tiver melhorias, correções ou novos recursos para adicionar, por favor, abra uma **pull request** ou **issue**.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contato

Para mais informações, entre em contato com [web2ajax@gmail.com](mailto:web2ajax@gmail.com).
