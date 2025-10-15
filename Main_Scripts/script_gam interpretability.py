import numpy as np
import pandas as pd
import statistics as st
import shap
import matplotlib.pyplot as plt
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
TEST_SIZE = 0.1

class Clustering_GAM(BaseEstimator, RegressorMixin):
    def __init__(self, clusterer, distribution, link, n_splines, lam):
        self.clusterer = clusterer
        self.distribution = distribution
        self.link = link
        self.n_splines = n_splines
        self.lam = lam

    def fit(self, X, y):

        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)

        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        self.X = X
        self.y = y

        # Fit the clustering algorithm
        self.clusterer_ = clone(
            self.clusterer
        )  
        # Construct a new unfitted estimator with the same parameters for each entrance
        
        cluster_labels = self.clusterer_.fit_predict(X)  
        
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
            X_cluster = X.iloc[idx]
            y_cluster = y.iloc[idx]
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
        y_pred = np.zeros(X.shape[0])
        
        for cluster in np.unique(cluster_labels):
            idx = np.where(cluster_labels == cluster)[0]
            if cluster in self.models_:
                y_pred[idx] = self.models_[cluster].predict(X[idx])
            else:
                y_pred[idx] = np.nan
                
        return y_pred
    
    def shap(self, cluster, instance=None):

        glm = self.models_[cluster]
        X_cluster = self.data_by_cluster_[cluster]["X"]

        explainer = shap.Explainer(glm.predict, X_cluster)
        shap_values = explainer(X_cluster)

        shap.plots.bar(shap_values)

        if instance is not None:
            shap.plots.waterfall(shap_values[instance]) 
        else:
            shap.summary_plot(shap_values, X_cluster, color="coolwarm")

        return shap_values
    
    def partial_dependence_plot(self, cluster):

        gam = self.models_[cluster]
        X_cluster = self.data_by_cluster_[cluster]["X"]
        
        num_vars = 81  

        fig, axs = plt.subplots(9, 9, figsize=(40, 40))  
        axs = axs.flatten() 

        for i in range(num_vars):
            XX = gam.generate_X_grid(term=i)  
            
            y_partial = gam.partial_dependence(term=i, X=XX)
            y_conf = gam.partial_dependence(term=i, X=XX, width=.95)[1]  

            axs[i].plot(XX[:, i], y_partial, label="Estimated effect")
            axs[i].plot(XX[:, i], y_conf, c='r', ls='--', label="Confidence Interval 95%")
            
            axs[i].set_title(f"Variable {X_cluster.columns[i]}")
            axs[i].set_xlabel("Variable value")
            axs[i].set_ylabel("Effect on target")
            axs[i].legend()

        plt.tight_layout()
        plt.show()
    
    def partial_dependence_shap(self, cluster, k=5):
        glm = self.models_[cluster]
        X_cluster = self.data_by_cluster_[cluster]["X"]

        explainer = shap.Explainer(glm.predict, X_cluster)
        shap_values = explainer(X_cluster)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_names = X_cluster.columns

        feature_importance_shap = pd.DataFrame({
            "feature": X_cluster.columns,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        for i in feature_importance_shap.head(k).itertuples():
            shap.dependence_plot(i.feature, shap_values.values, X_cluster, show=True)

def cross_validation(X, y, clusterer, distribution_name, link_name, lam, n_splines, n_splits=5):
    
    all_rmse = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    split = 0
    for train_idx, test_idx in kf.split(X):
        
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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

    TEST_SIZE = 0.1
    RANDOM_SEED = 1203
    
    X_train, X_validation, y_train, y_validation = train_test_split(
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
    
