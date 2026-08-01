from mpl import MultilayerPerceptron, Layer
from activation_function import ReLU, Softmax
from loss_function import SparseCategoricalCrossentropy

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('data-stars.csv')
features = ['Temperature (K)',	'Luminosity (L/Lo)',	'Radius (R/Ro)',	'Absolute magnitude (Mv)']
target = 'Star category'

x = df[features]
y = df[target]

test_ratio = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio)

sca = StandardScaler()
x_train = sca.fit_transform(x_train)
x_test = sca.transform(x_test)

enc = LabelEncoder()
y_train = enc.fit_transform(y_train)

# Cria cada camada
layer_entry = Layer(4)
layer_hidden = Layer(16, activation_function=ReLU)
layer_output = Layer(6, activation_function=Softmax)

# Gera a rede neural
layers = [layer_entry, layer_hidden, layer_output]
nn = MultilayerPerceptron(layers, learning_rate=0.01, loss_function=SparseCategoricalCrossentropy)
epochs = 50

# Treina a rede neural
training_history = nn.train(x_train, y_train, epochs)

# Avaliação do modelo

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

fig = plt.figure(figsize=(10, 8), constrained_layout=True)
fig.suptitle('Model evaluation', fontsize=20)
gs = fig.add_gridspec(2, 2)

training_plot = fig.add_subplot(gs[0, 0])
training_plot.plot(training_history['loss'].keys(), training_history['loss'].values(), label='Loss', color='red')
training_plot.plot(training_history['correct'].keys(), training_history['correct'].values(), label='Correct', color='blue')
training_plot.set_xlabel('Epoch')
training_plot.legend()
training_plot.grid()

y_pred = nn.predict(x_test)
label_pred = enc.inverse_transform(y_pred)
cm = confusion_matrix(y_test, label_pred)
classes = enc.classes_
matrix_plot = fig.add_subplot(gs[0, 1])
sns.heatmap(cm, ax=matrix_plot, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
matrix_plot.set_ylabel('True')
matrix_plot.set_xlabel('Predicted')

plt.show()
