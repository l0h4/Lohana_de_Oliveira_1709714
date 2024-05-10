import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.decomposition import PCA  # Importando PCA para redução de dimensionalidade

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data  # Utilizando todas as características
y = iris.target

# Redução de dimensionalidade com PCA para visualização em 2D
pca = PCA(n_components=2)  # Criando uma instância de PCA com 2 componentes principais
X_pca = pca.fit_transform(X)  # Aplicando PCA para reduzir a dimensionalidade dos dados para 2 componentes principais

h = 0.02  # Tamanho do passo na malha

# Criando mapas de cores para os pontos (treinamento) e a região de decisão (malha)
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])  # Cores para a malha
cmap_bold = ["darkorange", "c", "darkblue"]  # Cores para os pontos de treinamento

# Criando e treinando o classificador de Regressão Logística
clf = LogisticRegression()  # Criando uma instância do classificador de Regressão Logística
clf.fit(X_pca, y)  # Treinando o classificador com os dados de treinamento após a redução de dimensionalidade

# Definindo os limites do plot com base nos dados de entrada após PCA
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1  # Limites do eixo x após PCA
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1  # Limites do eixo y após PCA

# Criando a malha de pontos para o plot
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # Criando uma malha de pontos usando np.meshgrid

# Prevendo a classe para cada ponto na malha usando o classificador treinado
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Prevendo as classes para todos os pontos na malha

# Reformulando os resultados para o formato da malha
Z = Z.reshape(xx.shape)  # Reformulando as classes previstas para o formato da malha

# Plotando a região de decisão
plt.figure(figsize=(8, 6))  # Criando uma nova figura com o tamanho especificado
plt.contourf(xx, yy, Z, cmap=cmap_light)  # Plotando a região de decisão usando cores da malha

# Plotando os pontos de treinamento após PCA
sns.scatterplot(
    x=X_pca[:, 0],  # Coordenadas x dos pontos de treinamento após PCA
    y=X_pca[:, 1],  # Coordenadas y dos pontos de treinamento após PCA
    hue=iris.target_names[y],  # Usando os rótulos de classe como cores
    palette=cmap_bold,  # Especificando a paleta de cores
    alpha=1.0,  # Configurando a transparência dos pontos
    edgecolor="black",  # Cor das bordas dos pontos
)

# Definindo limites do plot
plt.xlim(xx.min(), xx.max())  # Limites do eixo x
plt.ylim(yy.min(), yy.max())  # Limites do eixo y

# Configurando título e rótulos dos eixos
plt.title("3-Class classification (Logistic Regression after PCA)")  # Título do plot
plt.xlabel("Principal Component 1")  # Rótulo do eixo x
plt.ylabel("Principal Component 2")  # Rótulo do eixo y

# Mostrando o plot
plt.show()

