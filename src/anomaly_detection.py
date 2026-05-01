from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
import numpy as np


def detect_anomalies_forest(X, n_anomalies=50):
    """
    Tree-based anomaly detection using Isolation Forest.
    Identifies anomalies by isolating observations through random partitioning;
    outliers are easier to isolate and require fewer splits.
    """
    # contamination is the expected proportion of outliers
    contamination = n_anomalies/2164  # 2164 = #documents
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)

    # Predict: -1 for outliers, 1 for normal data
    outlier_labels = model.fit_predict(X)

    # Alternatively, get the raw scores (lower is more anomalous)
    scores = model.decision_function(X)

    return outlier_labels, scores


def detect_anomalies_lof(X, n_anomalies=50):
    """
    Density-based anomaly detection using Local Outlier Factor (LOF).
    Identifies anomalies by comparing the local density of a point to its
    neighbors; points with significantly lower density are flagged.
    """
    # LOF works by comparing local densities
    contamination = n_anomalies/2164
    model = LocalOutlierFactor(n_neighbors=100, contamination=contamination)
    
    outlier_labels = model.fit_predict(X)
    scores = model.negative_outlier_factor_  # Lower (more negative) is more anomalous
    
    return outlier_labels, scores


def detect_anomalies_knn(X, k=5):
    """
    Distance-based anomaly detection using k-NN.
    Anomalies are points with the largest average distance to their k-neighbors.
    """
    # Use 'cosine' metric for text/SVD data as it handles orientation better than Euclidean
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(X)
    
    # distances is a matrix of [n_samples, k]
    distances, indices = knn.kneighbors(X)
    
    # We take the mean distance to the k neighbors.
    # Larger distance = more anomalous.
    avg_distances = distances.mean(axis=1)
    
    # times (-1) to flip the scores to make them consistent with the
    # "lower is more anomalous" in de previous algorithms
    return - avg_distances