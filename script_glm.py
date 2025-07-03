import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from ucimlrepo import fetch_ucirepo 

def load_dataset(): # How to import the dataset with UCI repository
    # Import dataset
    superconductivty_data = fetch_ucirepo(id=464) 
  
    # Data (dataframe)
    X = superconductivty_data.data.features 
    y = superconductivty_data.data.targets 

    return X, y

class PerClusterClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clusterer, base_estimator):
        self.clusterer = clusterer
        self.base_estimator = base_estimator # Base class for all the estimators

    def fit(self, X, y):
        # Fit the clustering algorithm
        self.clusterer_ = clone(self.clusterer) # Construct a new unfitted estimator with the same parameters for each entrance 
        cluster_labels = self.clusterer_.fit_predict(X) # Use the fit/predict method
        self.cluster_labels_ = cluster_labels
        self.models_ = {}
        
        # For each cluster, fit a supervised model
        for cluster in np.unique(cluster_labels):
            idx = np.where(cluster_labels == cluster)[0]
            X_cluster = X[idx]
            y_cluster = y[idx]
            model = clone(self.base_estimator)
            model.fit(X_cluster, y_cluster)
            self.models_[cluster] = model
        return self

    def predict(self, X):
        cluster_labels = self.clusterer_.predict(X)
        y_pred = np.empty(X.shape[0], dtype=object)
        for cluster in np.unique(cluster_labels):
            idx = np.where(cluster_labels == cluster)[0]
            if cluster in self.models_:
                y_pred[idx] = self.models_[cluster].predict(X[idx])
            else:
                # Handle unseen clusters (optional: fallback strategy)
                y_pred[idx] = None
        return y_pred

        