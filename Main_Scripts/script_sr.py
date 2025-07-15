import statistics as st

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

class Clustering_SR(BaseEstimator, RegressorMixin):
    
    def __init__(self, clusterer, n_iterations, maxsize, maxdepth, binary_operators, unary_operators, select_k_features):
        self.n_iterations = n_iterations
        self.clusterer = clusterer
        self.n_iterations = n_iterations
        self.maxsize = maxsize
        self.maxdepth = maxdepth
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.select_k_features = select_k_features
        
        
    def fit(self, X, y):
        """Adjusting a model for each cluster"""

        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        self.X = X
        self.y = y

        # Fit the clustering algorithm
        self.clusterer_ = clone(self.clusterer)

        cluster_labels = self.clusterer_.fit_predict(X)

        self.cluster_labels_ = cluster_labels

        unique_clusters = np.unique(cluster_labels)
        valid_clusters = [c for c in unique_clusters if c != -1]

        self.models_ = {}

        # For each cluster, fit a supervised model
        self.data_by_cluster_ = {}
        
        self.default_pysr_params = dict(
                populations = 20,
                model_selection="best",
                batching=True,
                batch_size=64,
                progress = True,
            )

        for cluster in valid_clusters:  # identify the classes
            idx = np.where(cluster_labels == cluster)[0]

            X_cluster = X.iloc[idx]
            y_cluster = y.iloc[idx]

            self.data_by_cluster_[cluster] = {
                "X": X_cluster,
                "y": y_cluster,
                "indices": idx,
            }
                
            self.constraints = {}
            if "/" in self.binary_operators:
                self.constraints["/"] = (-1, 9)
            
            if "square" in self.unary_operators:
                self.constraints["square"] = 9
            
            if "cube" in self.unary_operators:
                self.constraints["cube"] = 9
                
            if "exp" in self.unary_operators:
                self.constraints["exp"] = 9
        
            model = PySRRegressor(
                maxsize = self.maxsize,
                maxdepth=self.maxdepth,
                binary_operators=self.binary_operators,
                unary_operators=self.unary_operators,
                niterations=self.n_iterations,
                select_k_features=self.select_k_features,
                **self.default_pysr_params,
            )
            
            model.fit(X_cluster, y_cluster)
            
            self.models_[cluster] = model

        return self

    def predict(self, X):
        """Making predictions considering each cluster"""

        X = self.scaler_X_.transform(X)
        cluster_labels = self.clusterer_.predict(X)
        y_pred = np.zeros(X.shape[0])

        for cluster in np.unique(cluster_labels):
            idx = np.where(cluster_labels == cluster)[0]
            if cluster in self.models_:
                y_pred[idx] = self.models_[cluster].predict(X[idx])
            else:
                y_pred[idx] = np.nan

        return y_pred


def cross_validation(X, y, clusterer, n_iterations, maxsize, maxdepth, binary_operators, unary_operators, select_k_features, random_seed=1203, n_splits=5):
    """Cross validation with clustering"""

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    all_rmse = {}

    for split, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = Clustering_SR(clusterer=clone(clusterer), 
                              n_iterations=n_iterations, 
                              maxsize=maxsize,
                              maxdepth=maxdepth, 
                              binary_operators=binary_operators, 
                              unary_operators=unary_operators, 
                              select_k_features=select_k_features)
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)

        all_rmse[split] = rmse
        
    fold_values = list(all_rmse.values())
    all_rmse["Mean"] = st.mean(fold_values)
    all_rmse["Median"] = st.median(fold_values)

    return all_rmse
