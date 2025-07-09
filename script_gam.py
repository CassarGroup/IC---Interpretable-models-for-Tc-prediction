import numpy as np
import statistics as st
import pygam
from pygam import s
from sklearn.base import BaseEstimator, RegressorMixin, clone
from ucimlrepo import fetch_ucirepo
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_SEED = 1203
TEST_SIZE = 0.9

class Clustering_GAM(BaseEstimator, RegressorMixin):
    def __init__(self, clusterer, distribution, link, n_splines, lam):
        self.clusterer = clusterer
        self.distribution = distribution
        self.link = link
        self.n_splines = n_splines
        self.lam = lam

    def fit(self, X, y):

        self.scaler_X_ = StandardScaler()
        X = self.scaler_X_.fit_transform(X)
        
        self.X = X
        self.y = y

        # Fit the clustering algorithm
        self.clusterer_ = clone(
            self.clusterer
        )  # Construct a new unfitted estimator with the same parameters for each entrance
        
        cluster_labels = self.clusterer_.fit_predict(
            X
        )  # Use the fit/predict method for data clustering
        self.cluster_labels_ = cluster_labels

        unique_clusters = np.unique(cluster_labels)
        valid_clusters = [c for c in unique_clusters if c != -1]

        self.models_ = {}
        # For each cluster, fit a supervised model
        self.data_by_cluster_ = {}
        
        self.terms = s(0)
        
        for i in range(1, X.shape[1]):
            self.terms += s(i)

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
            
            model_gam = pygam.pygam.GAM(self.terms, 
                                        distribution=self.distribution, 
                                        link=self.link, 
                                        lam=self.lam, 
                                        n_splines=self.n_splines)

            glm = model_gam.fit(X_cluster,y_cluster)

            self.models_[cluster] = glm

        return self

    def predict(self, X):
        # Assign clusters to new data
        
        X = self.scaler_X_.transform(X)
        cluster_labels = self.clusterer_.predict(X)
        y_pred = np.empty(X.shape[0])
        
        for cluster in np.unique(cluster_labels):
            idx = np.where(cluster_labels == cluster)[0]
            if cluster in self.models_:
                y_pred[idx] = self.models_[cluster].predict(X[idx])
            else:
                y_pred[idx] = np.nan
                
        return y_pred


def cross_validation(X, y, clusterer, distribution_name, link_name, lam, n_splines, n_splits=5):
    
    all_rmse = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    split = 0
    for train_idx, test_idx in kf.split(X):
        
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = Clustering_GAM(clusterer=clone(clusterer), 
                          distribution=distribution_name, 
                          link = link_name,
                          lam = lam,
                          n_splines= n_splines)
            
            model.fit(X_train, y_train)

            # Make the prediction
            y_pred = model.predict(X_test)

            # Calculate RMSE
            rmse = root_mean_squared_error(y_test, y_pred)

            all_rmse[split] = rmse
            split += 1

    fold_values = list(all_rmse.values())
    all_rmse["Mean"] = st.mean(fold_values)
    all_rmse["Median"] = st.median(fold_values)

    return all_rmse

def train_validation(X, y, clusterer, distribution_name, link_name, lam, n_splines):
    
    X_validation, X_train, y_validation, y_train = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
        
    model = Clustering_GAM(clusterer=clone(clusterer), 
                          distribution=distribution_name, 
                          link = link_name,
                          lam = lam,
                          n_splines= n_splines)
            
    model.fit(X_train, y_train)

    y_pred = model.predict(X_validation)
    rmse = root_mean_squared_error(y_validation, y_pred)

    return rmse
    
