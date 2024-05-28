import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Carregar o dataset
data, labels = load_digits(return_X_y=True)

# Aplicar K-means nos dados originais
n_clusters = 10  # número de clusters, que corresponde ao número de dígitos (0-9)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data)
clusters = kmeans.predict(data)

# Redução de dimensão com PCA
pca = PCA(2)  # Reduzindo para 2 componentes
reduced_data = pca.fit_transform(data)

# Aplicar K-means nos dados reduzidos
kmeans_reduced = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_reduced.fit(reduced_data)
reduced_clusters = kmeans_reduced.predict(reduced_data)
centroids = kmeans_reduced.cluster_centers_

# Plotar gráfico de dispersão
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=reduced_clusters, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')  # Centróides
plt.colorbar(scatter, ticks=range(n_clusters), label='Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-means Clustering com Redução de Dimensão PCA')
plt.show()

# Visualizar alguns dígitos e seus respectivos clusters
fig, axs = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(data[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Cluster: {clusters[i]}')
    ax.axis('off')

plt.tight_layout()
plt.show()
