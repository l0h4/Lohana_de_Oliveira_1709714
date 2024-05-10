import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data  # Utilizando todas as características
y = iris.target

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% para treinamento, 20% para teste

# Redução de dimensionalidade com PCA para visualização em 2D
pca = PCA(n_components=2)  # Criando uma instância de PCA com 2 componentes principais
X_train_pca = pca.fit_transform(X_train)  # Aplicando PCA aos dados de treinamento
X_test_pca = pca.transform(X_test)  # Aplicando PCA aos dados de teste

# Criando e treinando o classificador de Regressão Logística
clf = LogisticRegression()  # Criando uma instância do classificador de Regressão Logística
clf.fit(X_train_pca, y_train)  # Treinando o classificador com os dados de treinamento após a redução de dimensionalidade

# Fazendo a predição com os dados de teste
y_pred = clf.predict(X_test_pca)

# Exibindo as predições
print("Predições:")
print(y_pred)
