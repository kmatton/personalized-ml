""""
Class to perform clustering analysis.
"""
import numpy as np
from sklearn.cluster import KMeans


class ClusterModel:
    """
    Generic base class
    """
    def __init__(self, num_clusters, cluster_feature):
        self.num_clusters = num_clusters
        self.cluster_feature = cluster_feature
        self.model = None

    def learn_clusters(self, dataset):
        print('training cluster model')
        X = np.array(dataset[self.cluster_feature])
        self._learn_clusters(X)

    def _learn_clusters(self, X):
        raise NotImplementedError

    def predict_clusters(self, dataset):
        if self.model is None:
            print("ERROR: need to train model before using for predictions")
            print("Exiting...")
            exit(1)
        X = np.array(dataset[self.cluster_feature])
        preds = self.model.predict(X)
        return preds

    def save_model(self):
        raise NotImplementedError


class KMeansModel(ClusterModel):
    """
    Apply clustering using kmeans algorithms
    """
    def __init__(self, dataset, num_clusters, random_state=0):
        self.random_state = random_state
        super().__init__(dataset, num_clusters)

    def _learn_clusters(self, X):
        self.model = KMeans(n_clusters=self.num_clusters, random_state=self.random_state).fit(X)
