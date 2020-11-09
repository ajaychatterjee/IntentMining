import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances
from collections import Counter
from sklearn.cluster import DBSCAN
from sentenceEmbedding import SentenceEmbedding
import re


class ITER_DBSCAN(DBSCAN):
    """
    ITER-DBSCAN Implementation - Iteratively adapt dbscan parameters for unbalanced data (text) clustering
    The change of core parameters of DBSCAN i.e. distance and minimum samples parameters are changed smoothly to
    find high to low density clusters. At each iteration distance parameter is increased by 0.01 and minimum samples
    are decreased by 1. The algorithm uses cosine distance for cluster creation
    :params
    :initial_distance: initial distance for initial cluster creation (default: 0.10)
    :initial_minimum_samples: initial minimum sample count for initial cluster creation (default: 20)
    :delta_distance: change in distance parameter at each iteration(default: 0.01)
    :delta_minimum_samples: change in minimum sample parameter (of DBSCAN) at each iteration(default: 0.01)
    :max_iteration : maximum number of iteration the DBSCAN algorithm will run for cluster creation(default: 5)
    :threshold: threshold parameter controls the size of the cluster, any cluster contains more than threshold parameter
                will be discarded. (default: 300)
    :features: default values is None, the algorithm expects a list of short texts. In case the representation is
                pre-computed for text or data sources (pass featyres values as "precomputed").
    """

    def __init__(self, initial_distance=0.10, initial_minimum_samples=20, delta_distance=0.01, delta_minimum_samples=1,
                 max_iteration=5, threshold=300, features=None
                 ):

        self.initial_distance = initial_distance
        self.initial_minimum_samples = initial_minimum_samples
        self.delta_distance = delta_distance
        self.delta_minimum_samples = delta_minimum_samples
        self.max_iteration = max_iteration
        self.threshold = threshold
        self.features = features
        self.labels_ = None

    def preprocess_data(self, features):
        self.features = features
        clean_text_array = []
        for feature in features:
            text = re.sub(r'[^\w\s]', ' ', feature)
            text = text.lower()
            tokens = text.split(' ')
            tokens = [token.strip() for token in tokens if len(token.strip()) > 0]
            text = ' '.join(tokens)
            text = text.strip()
            clean_text_array.append(text)

        return clean_text_array

    def compute(self, data):
        if not (type(data) is np.ndarray or type(data) is list):
            raise Exception("Please pass a list of string or a list of feature vectors.")

        if type(data[0]) is str:
            #data = self.preprocess_data(data)
            if self.features is not 'precomputed':
                embedding_model = SentenceEmbedding()
                data = embedding_model.getEmbeddings(data)

        df = pd.DataFrame(index=range(len(data)), columns=['features', 'labels'])
        df['features'] = data
        df['labels'] = [-1] * len(data)
        cluster_id = 0
        for i in range(self.max_iteration):
            features = np.array(df.loc[df.labels == -1]['features'].values.tolist())
            distance_matrix = pairwise_distances(features, metric='cosine')

            if 5 > len(features): break
            cluster_labels = DBSCAN(eps=self.initial_distance, min_samples=self.initial_minimum_samples,
                                    metric='precomputed')
            labels = cluster_labels.fit_predict(distance_matrix)
            cluster_labels = [str(c) for c in labels]
            label_freq = Counter(cluster_labels)
            label_set = cluster_labels
            new_label_set = [-1 if label_freq[l] > self.threshold or l == '-1' else int(l) + cluster_id for l in
                             label_set]

            cluster_values = [k for k in new_label_set if k != -1]

            if len(cluster_values) > 0:
                min_cluster_id = min(cluster_id, min(cluster_values))
            else:
                min_cluster_id = cluster_id
            max_cluster_id = min_cluster_id + len(list(set(cluster_values)))

            unique_cluster_ids = list(set(cluster_values))
            id_mapper = dict()
            for i in range(len(unique_cluster_ids)):
                id_mapper[unique_cluster_ids[i]] = min_cluster_id
                min_cluster_id += 1

            new_label_set = [-1 if l == -1 else id_mapper[l] for l in new_label_set]
            cluster_id = max_cluster_id
            df.loc[df['labels'] == -1, 'labels'] = new_label_set
            self.initial_distance += self.delta_distance
            self.initial_minimum_samples -= self.delta_minimum_samples

            if self.initial_minimum_samples == 2:
                break

        self.labels_ = df['labels'].values.tolist()

    def fit_predict(self, X):
        self.compute(X)
        return self.labels_

    def fit(self, X):
        self.compute(X)
