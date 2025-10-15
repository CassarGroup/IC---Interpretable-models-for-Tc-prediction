import statistics as st
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 1203
TEST_SIZE = 0.1

class Clustering_GLM(BaseEstimator, RegressorMixin):
    def __init__(self, clusterer, distribution):
        self.clusterer = clusterer
        self.distribution = distribution

    def fit(self, X, y, X_test):
        """Ajustando um modelo para cada cluster"""

        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        X_test_scaled = self.scaler_X_test_.transform(X_test)
        
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X.index)

        self.X = X
        self.y = y
        self.X_test = X_test

        # Fit the clustering algorithm
        self.clusterer_ = clone(self.clusterer)

        cluster_labels = self.clusterer_.fit_predict(X)
        cluster_labels_test = self.clusterer_.predict(X_test)

        self.cluster_labels_ = cluster_labels

        unique_clusters = np.unique(cluster_labels)
        valid_clusters = [c for c in unique_clusters if c != -1]

        self.models_ = {}

        # For each cluster, fit a supervised model
        self.data_by_cluster_ = {}

        for cluster in valid_clusters:  # identify the classes
            idx = np.where(cluster_labels == cluster)[0]
            idx_test = np.where(cluster_labels_test == cluster)[0]

            X_cluster = X.iloc[idx]
            y_cluster = y.iloc[idx]
            X_cluster_test = X_test.iloc[idx]

            self.data_by_cluster_[cluster] = {
                "X": X_cluster,
                "y": y_cluster,
                "indices": idx,
            }

            self.data_by_cluster_test_[cluster] = {
                "X": X_cluster_test,
                "indices": idx_test
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

    
    def feature_importance(self, cluster, k=None, plot=True):
        """
        Retorna as features mais importantes para um cluster específico.
        """
        
        df_glm = pd.DataFrame({
            'Feature': self.models_[cluster].params.index,
            'Feature Importance': self.models_[cluster].params.values
        })

        df_glm = df_glm.sort_values(by='Feature Importance', ascending=False)
        
        if plot:
            plt.figure(figsize=(10, 15))
            plt.barh(df_glm['Feature'], df_glm['Feature Importance'], color="green")
            plt.xlabel('Weights')
            plt.title(f'Feature Importance (GLM) - Cluster {cluster}')
            plt.gca().invert_yaxis() 
            plt.show()

        if k is not None:
            df_glm = pd.DataFrame({
            'Feature': self.models_[cluster].params.index,
            'Feature Importance': abs(self.models_[cluster].params).values
        })
            df_glm = df_glm.sort_values(by='Feature Importance', ascending=False)
            df_glm = df_glm.head(k)
        
        return df_glm
        

    def shap(self, cluster, instance=None):

        glm = self.models_[cluster]
        X_cluster = self.data_by_cluster_test_[cluster]["X"]

        X_train, X_test = train_test_split(X_cluster, test_size = 0.2, random_state=1203)

        explainer = shap.Explainer(glm.predict, X_test)
        shap_values = explainer(X_cluster)


        plt.title(f"Shap values for the Cluster {cluster} model")
        shap.plots.bar(shap_values)

        if instance is not None:
            plt.title(f"Feature Importance: cluster {cluster}, instance {instance}")
            shap.plots.waterfall(shap_values[instance]) 
            plt.show()
        else:
            plt.title(f"Feature Importance: cluster {cluster}")
            shap.summary_plot(shap_values, X_test, color="coolwarm")
            plt.show()

        return shap_values

def cross_validation(X, y, clusterer, distribution, random_seed=1203, n_splits=5):
    """Realiza validação cruzada com clusterização"""

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    all_rmse = {}

    for split, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = Clustering_GLM(clusterer=clone(clusterer), distribution=distribution)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)

        all_rmse[split] = rmse
        
    fold_values = list(all_rmse.values())
    all_rmse["Mean"] = st.mean(fold_values)
    all_rmse["Median"] = st.median(fold_values)

    return all_rmse


