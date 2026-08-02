# numpy-mpl
Criação de classes para implementação de um Multilayer Perceptron a partir da biblioteca Numpy e a aplicação de um exemplo na classificação de estrelas.

Implementa a partir da classe MultilayerPerceptron todas as atividades da rede neural com inicialização He e SGD, facilitando a implementação e manutenção ao encapsular as camadas na classe Layer, e cada função de custo e ativação na sua própria classe.

Como exemplo, foi criada uma rede neural para classificação de estrelas utilizando o dataset data-stars.csv retirado do repositório: https://github.com/suaide/PGF5393

Esse foi o desempenho da rede no exemplo:

<img width="1200" height="600" alt="Gráfico" src="https://github.com/user-attachments/assets/c1751e0c-8568-4cbb-bda5-78fa3d32e988" />

#Exemplo de criação de rede

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

#MultilayerPerceptron

##Propriedades
- layers -> lista com todos os objetos Layer que pertencem a rede na ordem que devem ser computados
- learning_rate -> taxa de aprendizado utilizada no SGD
- loss_function -> função de perda utilizada no SGD
- size -> quantidade de camadas na rede
  
##Métodos
- input(entry) -> insere o valor entry na entrada da rede neural
- output -> retorna o valor da última camada da rede neural
- summary -> retorna um dicionário mostrando os valores de cada parâmetro de cada camada
- feedforward() -> calcula a partir do valor de entrada até a última camada
- backpropagate(target) -> ajuste os parâmetros da rede neural com SGD a partir do valor de output esperado para o input
- train(x_train, y_train, epochs) -> passa pelos dados conforme o número de épocas, reajustando os parâmetros a cada amostra
- predict(entry_array) -> retorna o valor que a rede neural prevê a partir de cada um dos dados de entrada sem treinar com os dados

#Layer

##Propriedades
- size -> quantidade de neurônios
- activation_fuction -> função de ativação utlizada para transformar os logits em saída
- x -> valor de entrada
- W -> matrix de pesos m x n, sendo m a quantidade de neurônios da camada anterior e n a quantidade de neurônios da camada atual
- b -> valor dos pesos
- z -> valor dos logits
- y -> valor de saída

##Métodos
- compute(): atualiza a saída a partir dos valores de entrada
- learn(grad_signal): atualiza seus parâmetros conforme o sinal da camada superior e retorna o sinal para a próxima camada

#Funções de custo

Toda função de perda tem dois métodos estáticos:
- function(y, target) -> retorna a perda dado a predição y e o valor real target
- derivative(y, target) -> retorna a derivada da perda dado a predição y e o valor real target

#Funções de ativação

Toda função de ativação tem dois métodos estáticos:
- function(z) -> retorna a saída da camada considerando os logits z
- derivative(z) -> retorna a derivada da saída da camada considerando os logits z
