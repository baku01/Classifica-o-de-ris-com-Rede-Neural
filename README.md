# Classificação de Íris com Rede Neural 🌺

Este notebook implementa uma rede neural para classificar flores de íris usando o conjunto de dados Iris do sklearn. O objetivo é demonstrar passo a passo o processo de carregamento dos dados, pré-processamento, definição e treinamento da rede neural, além da visualização dos resultados.

## Passo a Passo

### 1. Carregamento dos Dados
O conjunto de dados Iris é carregado usando a função `datasets.load_iris()` do sklearn. Em seguida, selecionamos as duas primeiras características (comprimento e largura da sépala) para visualização.

```python
from sklearn import datasets

iris_dataset = datasets.load_iris()
iris_features = [0, 1]
iris_data = iris_dataset.data[:, iris_features]
```

### 2. Visualização dos Dados
Plotamos um gráfico de dispersão das características selecionadas, colorindo as diferentes classes de flores.

```python
import matplotlib.pyplot as plt

plt.scatter(iris_dataset.data[:, 0], iris_dataset.data[:, 1], c=iris_dataset.target)
plt.xlabel(iris_dataset.feature_names[0])
plt.ylabel(iris_dataset.feature_names[1])
plt.show()
```

### 3. Pré-processamento dos Dados
Normalizamos os dados usando `MinMaxScaler` do sklearn para garantir que todas as características estejam na mesma escala.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = scaler.fit_transform(iris_data)
```

### 4. Definição da Rede Neural
Definimos uma rede neural simples usando o PyTorch. A rede consiste em uma camada oculta com ativação de LeakyReLU e uma camada de saída com ativação Softmax.

```python
from torch import nn

input_size = data.shape[1]
hidden_size1 = 50
out_size = len(iris_dataset.target_names)

net = nn.Sequential(
    nn.Linear(input_size, hidden_size1),
    nn.LeakyReLU(),  
    nn.Linear(hidden_size1, out_size),
    nn.Softmax()
)
```

### 5. Treinamento do Modelo
Treinamos a rede neural usando a função de perda de entropia cruzada e otimizador SGD (Gradiente Descendente Estocástico). O treinamento é feito em 1000 épocas.

```python
from torch import optim
import torch

# Definindo funções de otimização
X = torch.FloatTensor(data)
Y = torch.LongTensor(iris_dataset.target)

# Função de perda
criterion = nn.CrossEntropyLoss()

# Otimizador 
optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=0.01)

# Treinamento
for i in range(1000):
    pred = net(X)
    loss = criterion(pred, Y)

    loss.backward()
    optimizer.step()
```

### 6. Visualização da Classificação
A cada época, plotamos a classificação das flores de íris usando a função `plot_sepal`, que mostra as fronteiras de decisão da rede neural.

```python
from torch import Tensor
import numpy

def plot_sepal(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    spacing = min(x_max - x_min, y_max - y_min) / 100

    XX, YY = numpy.meshgrid(numpy.arange(x_min, x_max, spacing),
                            numpy.arange(y_min, y_max, spacing))

    data = numpy.hstack((XX.ravel().reshape(-1, 1), YY.ravel().reshape(-1, 1)))

    db_prob = model(Tensor(data))
    clf = numpy.argmax(db_prob.cpu().detach().numpy(), axis=1)

    Z = clf.reshape(XX.shape)

    plt.contourf(XX, YY, Z, cmap='brg', alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=15, cmap='brg')
    plt.show()
```

### 7. Sumário da Rede Neural
Exibimos um resumo da arquitetura da rede neural após o treinamento.

```python
from torchsummary import summary

summary(net, input_size=(input_size,))
```

## Como Executar
- Execute o notebook linha por linha ou célula por célula para ver os resultados passo a passo.
- Certifique-se de ter uma GPU disponível para treinamento mais rápido, caso contrário, ajuste o código para executar no CPU.
- Ajuste os parâmetros do treinamento conforme necessário, como taxa de aprendizado e número de épocas.