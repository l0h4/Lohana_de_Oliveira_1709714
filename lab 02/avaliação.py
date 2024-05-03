from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Optdigits
optdigits = load_digits()

# Extrair as matrizes de características (X) e os vetores de classes (y)
X = optdigits.data
y = optdigits.target

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir o classificador KNN
n_neighbors = 15
clf = KNeighborsClassifier(n_neighbors)

# Treinar o classificador KNN com os dados de treino
clf.fit(X_train, y_train)

# Fazer previsões nos dados de treino
y_pred_train = clf.predict(X_train)

# Calcular a precisão do classificador nos dados de treino
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Accuracy on training data:", accuracy_train)