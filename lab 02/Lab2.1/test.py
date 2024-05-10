import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as p1
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()
X_test = iris.data
y_test = iris.target

# Carregando o modelo previamente treinado
loaded_model = p1.load(open('number.pkl', 'rb'))

# Carregando os dados de treinamento
X_train = loaded_model['X_train']
y_train = loaded_model['y_train']

# Redução de dimensionalidade com PCA para visualização em 2D
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Aplicando a transformação de PCA aos dados de teste usando o mesmo objeto PCA que foi ajustado aos dados de treinamento
X_test_pca = pca.transform(X_test)

# Criando e treinando o classificador de Regressão Logística
clf = LogisticRegression()  # Criando uma instância do classificador de Regressão Logística
clf.fit(X_train, y_train)  # Treinando o classificador com os dados de treinamento após a redução de dimensionalidade

# Prevendo as probabilidades de pertencimento a cada classe para cada ponto de teste
probs = clf.predict_proba(X_test_pca)

# Criando mapas de cores para os pontos e a região de decisão
cmap_light = ListedColormap(["cyan", "orange", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

# Definindo os limites do gráfico
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Prevendo a classe para cada ponto na malha usando o classificador treinado
Z = np.argmax(probs, axis=1)

# Plotando a região de decisão
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=cmap_light)

# Plotando os pontos de teste após PCA
sns.scatterplot(
    x=X_test_pca[:, 0],
    y=X_test_pca[:, 1],
    hue=y_test,
    palette=cmap_bold,
    alpha=0.7,
    edgecolor="black",
    marker='o',
    s=80
)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title("3-Class classification (Logistic Regression after PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()
