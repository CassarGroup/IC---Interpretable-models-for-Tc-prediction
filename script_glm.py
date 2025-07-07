import statistics as st

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

random_seed = 1203


def load_dataset():
    """Import the dataset from UCI repository"""

    # Import dataset
    superconductivty_data = fetch_ucirepo(id=464)

    # Data (dataframe)
    X = superconductivty_data.data.features
    y = superconductivty_data.data.targets

    return X, y


class Clustering_GLM(BaseEstimator, RegressorMixin):
    def __init__(self, clusterer, distribution):
        self.clusterer = clusterer
        self.distribution = distribution

    def fit(self, X, y):
        """Ajustando um modelo para cada cluster"""

        self.scaler_X_ = StandardScaler()
        X = self.scaler_X_.fit_transform(X)

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

        for cluster in valid_clusters:  # identify the classes
            idx = np.where(cluster_labels == cluster)[0]
            X_cluster = X[idx]
            y_cluster = y[idx]
            y_cluster = np.clip(y_cluster, 1e-6, np.inf)

            self.data_by_cluster_[cluster] = {
                "X": X_cluster,
                "y": y_cluster,
                "indices": idx,
            }

            model_glm = sm.GLM(y_cluster, X_cluster, family=self.distribution)
            glm = model_glm.fit()

            self.models_[cluster] = glm

        return self

    def predict(self, X):
        """Realiza previsão considerando cada cluster"""

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

    def rmse(self, X, y):
        # TODO: alterar para um cálculo vetorial com sklearn

        y_pred = list(self.predict(X))
        y_true = list(y)
        rmse = 0
        for t, p in zip(y_true, y_pred):
            rmse += (t - p) ** 2
        rmse = (rmse / len(y)) ** (1 / 2)
        return rmse

    def cross_validation(self, X, y, n_splits=5):
        """Realiza validação cruzada com clusterização"""

        scaler_X_ = StandardScaler()
        X = scaler_X_.fit_transform(X)

        all_rmse = {}

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        cluster_rmse = []  # RMSE only for the cluster
        split = 0

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = Clustering_GLM(
                clusterer=clone(self.clusterer), distribution=self.distribution
            )
            model.fit(X_train, y_train)

            # Make the prediction
            y_pred = model.predict(X_test)

            # Calculate RMSE
            rmse = root_mean_squared_error(y_test, y_pred)
            cluster_rmse.append(rmse)

            all_rmse[split] = np.mean(cluster_rmse)
            split += 1

        fold_values = list(all_rmse.values())
        all_rmse["Mean"] = st.mean(fold_values)
        all_rmse["Median"] = st.median(fold_values)

        return all_rmse
