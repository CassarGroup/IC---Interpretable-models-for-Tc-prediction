import numpy as np
import statistics as st
from sklearn.base import BaseEstimator, RegressorMixin, clone
from ucimlrepo import fetch_ucirepo 
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

random_seed = 1203

def load_dataset(): # Import the dataset from UCI repository
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
        self.X = X
        self.y = y
        # Fit the clustering algorithm
        self.clusterer_ = clone(self.clusterer) # Construct a new unfitted estimator with the same parameters for each entrance 
        cluster_labels = self.clusterer_.fit_predict(X) # Use the fit/predict method for data clustering 
        self.cluster_labels_ = cluster_labels
        self.models_ = {}
        # For each cluster, fit a supervised model
        self.data_by_cluster_ = {}  

        for cluster in np.unique(cluster_labels): # identify the classes
            idx = np.where(cluster_labels == cluster)[0]
            X_cluster = X[idx]
            y_cluster = y[idx]
            
            self.data_by_cluster_[cluster] = {
                    "X": X_cluster,
                    "y": y_cluster,
                    "indices": idx
                }
            
            # link_function = sm.families.links.Log()

            # distribution = sm.families.Gamma(link=link_function)
            #X_cluster = sm.add_constant(X_cluster, has_constant='add')

            model_glm = sm.GLM(y_cluster, X_cluster, family=self.distribution)

            glm = model_glm.fit()

            self.models_[cluster] = glm
        return self

    def predict(self, X):
        # Assign clusters to new data
        #X = sm.add_constant(X, has_constant='add')
        cluster_labels = self.clusterer_.predict(X)
        y_pred = np.empty(X.shape[0])
        for cluster in np.unique(cluster_labels):
            idx = np.where(cluster_labels == cluster)[0]
            if cluster in self.models_:
                y_pred[idx] = self.models_[cluster].predict(X[idx])
            else:
                y_pred[idx] = np.nan
        return y_pred
    
    def rmse(self, X, y):
        y_pred = list(self.predict(X))
        y_true = list(y)
        rmse = 0
        for t, p in zip(y_true, y_pred):
            rmse += (t-p) ** 2
        rmse = (rmse/ len(y)) ** (1/2)
        return rmse
    
    def cross_validation(self, n_splits=5):
        cluster_labels = self.clusterer_.predict(self.X) # predict only the label of the entrance
        all_rmse = {}

        for cluster in np.unique(cluster_labels):
            # Cross validation for each cluster
            idx = np.where(cluster_labels == cluster)[0]
            X_cluster = self.X[idx]
            #X_cluster = sm.add_constant(X_cluster, has_constant='add')
            y_cluster = self.y[idx]

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            cluster_rmse = [] # RMSE only for the cluster

            for train_idx, test_idx in kf.split(X_cluster):
                X_train, X_test = X_cluster[train_idx], X_cluster[test_idx]
                y_train, y_test = y_cluster[train_idx], y_cluster[test_idx]
                
                # Training a new GLM model for each cluster
                model_glm = sm.GLM(y_train, X_train, family=self.distribution)
                glm = model_glm.fit()

                # Make the prediction
                y_pred = glm.predict(X_test)

                # Calculate RMSE
                rmse = root_mean_squared_error(y_test, y_pred)
                cluster_rmse.append(rmse)

            all_rmse[cluster] = np.mean(cluster_rmse)
        cluster_values = list(all_rmse.values())
        all_rmse["Mean"] = st.mean(cluster_values)
        all_rmse["Median"] = st.median(cluster_values)
        
        return all_rmse

        