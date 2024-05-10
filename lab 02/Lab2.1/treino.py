import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle as p1

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data  # Utilizando todas as características
y = iris.target

# Dividindo os dados em conjuntos de treinamento e teste
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% para treinamento, 20% para teste

# Redução de dimensionalidade com PCA para visualização em 2D
pca = PCA(n_components=2)  # Criando uma instância de PCA com 2 componentes principais
X_train_pca = pca.fit_transform(X_train)  # Aplicando PCA aos dados de treinamento

# Criando e treinando o classificador de Regressão Logística
clf = LogisticRegression()  # Criando uma instância do classificador de Regressão Logística
clf.fit(X_train_pca, y_train)  # Treinando o classificador com os dados de treinamento após a redução de dimensionalidade

# Abrindo o arquivo para escrita em modo binário
preditor_Pickle = open('number.pkl', 'wb')
p1.dump(clf, preditor_Pickle)

# Confirmação de que o modelo foi salvo com sucesso
print("O modelo de Regressão Logística foi salvo com sucesso usando pickle!")

# Confirmação de que o modelo foi treinado
print("O modelo de Regressão Logística foi treinado com sucesso!")
