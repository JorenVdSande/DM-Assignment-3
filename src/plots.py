from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.base import clone
from scipy.sparse import issparse


def plot_elbow_method(X, max_k):
    sse = []
    k_range = range(2, max_k + 1)

    print("Calculating SSE for Elbow Method...")
    for k in k_range:
        # We use KMeans directly here to access the .inertia_ attribute
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # plot results
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sse, marker='o', linestyle='--')
    plt.title('Elbow Method: SSE vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE (Inertia)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()


def plot_silhouette_analysis(X, model_instance, max_k=10):
    """
    Takes a 'template' model and updates n_clusters for each iteration.
    Plots the silhouette score for 
    """
    silhouettes = []
    k_range = range(2, max_k + 1)
    
    is_dense_required = isinstance(model_instance, (AgglomerativeClustering, SpectralClustering))
    # Only call .toarray() if the model needs it AND X is a sparse matrix
    X_input = X.toarray() if (is_dense_required and issparse(X)) else X
    
    for k in k_range:
        # 'set_params' allows you to update n_clusters on a copy of your model
        model = clone(model_instance).set_params(n_clusters=k)
        
        labels = model.fit_predict(X_input)
        score = silhouette_score(X_input, labels)
        silhouettes.append(score)
        print(f" - k={k}: {score:.4f}")
    
    # plot results
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouettes, marker='s', color='darkorange', linestyle='-', linewidth=2)
    
    plt.title('Silhouette Score: Evaluating Cluster Separation', fontsize=12)
    plt.xlabel('Number of Clusters (k)', fontsize=10)
    plt.ylabel('Average Silhouette Score', fontsize=10)
    plt.xticks(k_range)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Highlight the best k
    best_k = k_range[silhouettes.index(max(silhouettes))]
    plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
    plt.legend()
    
    plt.show()
    return best_k