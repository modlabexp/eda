#Exp 5:
!pip install scikit-learn matplotlib numpy -q

import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 2. Select 5000 random images
np.random.seed(42)
idx = np.random.choice(len(X), 5000, replace=False)
X, y = X[idx], y[idx]

# 3. Standardize data
X = StandardScaler().fit_transform(X)

# 4. PCA: Reduce 784 dimensions to 2
start = time.time()
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)

print("PCA variance:", pca.explained_variance_ratio_)
print("PCA time:", round(time.time() - start, 2), "seconds")

# 5. t-SNE: Reduce to 2 dimensions
start = time.time()
tsne = TSNE(n_components=2, random_state=42,
            perplexity=30)
Xt = tsne.fit_transform(X)

print("t-SNE time:", round(time.time() - start, 2), "seconds")

# 6. Scatter plots
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

for a, Z, title in [
    (ax[0], Xp, 'MNIST PCA'),
    (ax[1], Xt, 'MNIST t-SNE')
]:
    sc = a.scatter(Z[:, 0], Z[:, 1], c=y,
                   cmap='tab10', s=8, alpha=0.6)
    a.set_title(title)
    a.set_xlabel('Dimension 1')
    a.set_ylabel('Dimension 2')

fig.colorbar(sc, ax=ax, label='Digit')
plt.tight_layout()
plt.show()

# 7. Cumulative explained variance
pca_all = PCA().fit(X)
variance = np.cumsum(pca_all.explained_variance_ratio_)

plt.plot(variance)
plt.axhline(0.95, linestyle='--', color='red')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()
