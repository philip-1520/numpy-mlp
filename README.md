# numpy-mpl
Implementação de uma rede neural Multilayer Perceptron (MLP) desenvolvida inteiramente com NumPy, sem utilização de frameworks de Deep Learning como TensorFlow ou PyTorch. O projeto foi estruturado de forma modular para facilitar a compreensão do algoritmo de backpropagation, das funções de ativação e do treinamento por SGD.

## Estrutura
```text
numpy-mpl/

│

├── mpl.py

├── activation_function.py

├── loss_function.py

├── star-classification.py

├── data-stars.csv

└── README.md
```
## Funcionalidades
- Implementação completa de MLP utilizando apenas NumPy
- Inicialização de pesos com He Initialization
- Treinamento utilizando SGD
- Backpropagation implementado manualmente
- Funções de ativação modulares
    - ReLU
    - Sigmoid
    - Softmax
- Funções de perda modulares
    - Sparse Categorical Crossentropy
- Predição para novos dados
- Histórico de treinamento

## Exemplo

Como demonstração foi utilizada uma base de classificação de estrelas a partir do dataset data-stars.csv retirado do repositório: https://github.com/suaide/PGF5393

Cada estrela é descrita por quatro atributos físicos, e a rede deve classificá-la em uma das seguintes categorias:

- Brown Dwarf
- Hypergiant
- Main Sequence
- Red Dwarf
- Supergiant
- White Dwarf

Primeiro deve-se criar as camadas, e então associá-las a uma rede. Segue exemplo:
```python
layer_entry = Layer(4)
layer_hidden = Layer(16, activation_function=ReLU)
layer_output = Layer(6, activation_function=Softmax)

layers = [layer_entry, layer_hidden, layer_output]
nn = MultilayerPerceptron(layers, learning_rate=0.01, loss_function=SparseCategoricalCrossentropy)
epochs = 50

training_history = nn.train(x_train, y_train, epochs)
```

Desempenho da rede no exemplo:

<img width="1200" height="600" alt="Gráfico" src="https://github.com/user-attachments/assets/c1751e0c-8568-4cbb-bda5-78fa3d32e988" />

O treinamento converge rapidamente, reduzindo continuamente a função de perda enquanto aumenta a quantidade de classificações corretas.

A matriz de confusão obtida no conjunto de teste demonstra classificação perfeita para todas as classes do exemplo utilizado. O desempenho apresentado refere-se ao conjunto de dados utilizado como exemplo e pode variar conforme a divisão entre treinamento e teste e os hiperparâmetros empregados.

# MultilayerPerceptron

## Propriedades
- layers -> lista com todos os objetos Layer que pertencem a rede na ordem que devem ser computados
- learning_rate -> taxa de aprendizado utilizada no SGD
- loss_function -> função de perda utilizada no SGD
- size -> quantidade de camadas na rede
  
## Métodos
- input(entry) -> insere o valor entry na entrada da rede neural
- output -> retorna o valor da última camada da rede neural
- summary -> retorna um dicionário mostrando os valores de cada parâmetro de cada camada
- feedforward() -> calcula a partir do valor de entrada até a última camada
- backpropagate(target) -> ajuste os parâmetros da rede neural com SGD a partir do valor de output esperado para o input
- train(x_train, y_train, epochs) -> passa pelos dados conforme o número de épocas, reajustando os parâmetros a cada amostra
- predict(entry_array) -> retorna o valor que a rede neural prevê a partir de cada um dos dados de entrada sem treinar com os dados

# Layer

## Propriedades
- size -> quantidade de neurônios
- activation_function -> função de ativação utilizada para transformar os logits em saída
- x -> valor de entrada/entry
- W -> matriz de pesos m x n, sendo m a quantidade de neurônios da camada anterior e n a quantidade de neurônios da camada atual
- b -> valor dos vieses/biases
- z -> valor dos logits
- y -> valor de saída/output

## Métodos
- compute(): atualiza a saída a partir dos valores de entrada
- learn(grad_signal): atualiza seus parâmetros conforme o sinal da camada superior e retorna o sinal para a próxima camada

# Funções de custo

Toda função de perda tem dois métodos estáticos:
- function(y, target) -> retorna a perda dado a predição y e o valor real target
- derivative(y, target) -> retorna a derivada da perda dado a predição y e o valor real target

# Funções de ativação

Toda função de ativação tem dois métodos estáticos:
- function(z) -> retorna a saída da camada considerando os logits z
- derivative(z) -> retorna a derivada da saída da camada considerando os logits z
