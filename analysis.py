"""ML-3: Unsupervised Learning - K-Means + PCA on the Wine dataset (labels hidden)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

SEED = 42

# 1. Load data; drop labels (kept aside ONLY for a post-hoc sanity check)
wine = load_wine(as_frame=True)
X = wine.data.copy()
hidden_labels = wine.target
print("Shape:", X.shape, "| features:", list(X.columns))

# 2. Scale
X_scaled = StandardScaler().fit_transform(X)

# 3. Elbow + silhouette for k = 2..10
ks = range(2, 11)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k}: inertia={km.inertia_:.1f}  silhouette={sils[-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertias, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(list(ks), sils, "o-", color="tab:green"); ax[1].set(title="Silhouette score", xlabel="k", ylabel="Mean silhouette")
best_k = list(ks)[int(np.argmax(sils))]
ax[1].axvline(best_k, ls="--", color="gray")
plt.tight_layout(); plt.savefig("figures/elbow_silhouette.png", dpi=150); plt.close()
print("Best k by silhouette:", best_k)

# 4. Final K-Means
K = best_k
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X_scaled)
X["cluster"] = km.labels_

# 5. PCA
pca = PCA(n_components=2, random_state=SEED)
Z = pca.fit_transform(X_scaled)
evr = pca.explained_variance_ratio_
print("Explained variance ratio:", evr.round(4), "| total:", evr.sum().round(4))
full = PCA().fit(X_scaled).explained_variance_ratio_
print("Cumulative (all PCs):", np.cumsum(full).round(3))

# 6. Plot
plt.figure(figsize=(7, 5.5))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=km.labels_, cmap="viridis", s=45, edgecolor="k", linewidth=0.3)
cent = pca.transform(km.cluster_centers_)
plt.scatter(cent[:, 0], cent[:, 1], c="red", marker="X", s=200, edgecolor="k", label="Centroids")
plt.xlabel(f"PC1 ({evr[0]:.1%} variance)"); plt.ylabel(f"PC2 ({evr[1]:.1%} variance)")
plt.title(f"Wine: K-Means (k={K}) on PCA projection ({evr.sum():.1%} variance captured)")
plt.legend(*sc.legend_elements(), title="Cluster", loc="best"); plt.tight_layout()
plt.savefig("figures/pca_clusters.png", dpi=150); plt.close()

# 7. Interpretation helpers
print("\nCluster sizes:\n", X["cluster"].value_counts().sort_index())
profile = X.groupby("cluster").mean().round(2)
print("\nCluster means (original units):\n", profile.T)
z_profile = pd.DataFrame(X_scaled, columns=wine.data.columns).groupby(km.labels_).mean().round(2)
print("\nCluster means (z-scores):\n", z_profile.T)
pc_load = pd.DataFrame(pca.components_.T, index=wine.data.columns, columns=["PC1", "PC2"]).round(2)
print("\nPCA loadings:\n", pc_load.sort_values("PC1"))
print("\nExamples per cluster:")
for c in range(K):
    print(f"\n--- Cluster {c} ---"); print(X[X.cluster == c].head(3).round(2).to_string())
print("\nCrosstab (post-hoc check vs hidden cultivar):\n", pd.crosstab(km.labels_, hidden_labels))
print("Adjusted Rand Index:", round(adjusted_rand_score(hidden_labels, km.labels_), 3))
profile.T.to_csv("cluster_profiles.csv")
