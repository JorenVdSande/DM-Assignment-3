import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from scipy.sparse import issparse


def run_clustering(X, model):
    """
    Interchangeable clustering function.
    """
    needs_dense = isinstance(model, (AgglomerativeClustering, SpectralClustering))
    
    if needs_dense and issparse(X):
        X_input = X.toarray()
    else:
        X_input = X
    
    labels = model.fit_predict(X_input)
    return labels


def inspect_clusters(df, n_clusters, top_n: int = 20):
    """
    Manually calculates top words for any clustering method
    by grouping the dataframe by the cluster label.
    """
    print(f"\n--- Top {top_n} Terms per Cluster ---")
    
    for i in range(n_clusters):
        cluster_text = df[df['label'] == i]['token_text']
        all_words = " ".join(cluster_text).split()
        word_counts = pd.Series(all_words).value_counts() # Use Pandas to count frequencies
        top_words = word_counts.head(top_n).index.tolist() # Get the top N words
        
        print(f"Cluster {i} ({len(cluster_text)} articles):")
        print(f"{', '.join(top_words)}\n")

