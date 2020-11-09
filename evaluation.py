import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from ITER_DBSCAN import ITER_DBSCAN
from sentenceEmbedding import getEmbeddings
from sklearn.cluster import DBSCAN
import hdbscan
from tqdm import tqdm
from sklearn import metrics
from time import time
import numpy as np
import itertools
import traceback
import warnings
warnings.filterwarnings('ignore')


class EvaluateDataset(object):
    def __init__(self, filename, filetype, text_column, target_column):
        self.filename = filename
        self.text_column = text_column
        self.target_column = target_column
        self.filetype = filetype
        self.df = None

    def load_data(self):
        """
        Load data into a data frame (using filename)
        :return:
        """
        if self.filetype not in ['csv', 'xlsx']:
            raise Exception("Only supports csv and excel file.")
        try:
            if self.filetype == 'csv':
                self.df = pd.read_csv(self.filename)
            else:
                self.df = pd.read_excel(self.filename)
        except:
            raise Exception("Failed to load file!!")

    def extract_feature(self):
        """
        extract feature representation of short text using Universenal sentence encoder
        :return:
        """
        data = self.df[self.text_column].values.tolist()
        feature = getEmbeddings(data)
        self.df['features'] = feature

    def run_iter_dbscan(self, df, dist, max_iter, min_sample):
        """
        run iter-dbscan algorithm - computes cluster labels for short text
        :param df: loaded dataframe
        :param dist: iter-dbscan parameter initial distance
        :param max_iter: iter-dbscan parameter maximum iteration
        :param min_sample: iter-dbscan parameter minimum samples
        :return:
        """
        clustering_model = ITER_DBSCAN(initial_distance=dist, initial_minimum_samples=min_sample,
                                       max_iteration=max_iter,
                                       features='precomputed')
        cluster_labels = clustering_model.fit_predict(df['features'].values.tolist())
        cluster_labels = ['None' if c == -1 else c for c in cluster_labels]
        self.df['cluster_ids'] = cluster_labels

    def run_dbscan(self, df, min_distance, minimum_samples):
        """
        run dbscan algorithm algorithm - computes cluster labels for short text
        :param df: loaded dataframe
        :param min_distance: dbscan parameter minimum distance
        :param minimum_samples: dbscan parameter minimum samples
        :return:
        """
        clusterer = DBSCAN(eps=min_distance, min_samples=minimum_samples, metric='cosine')
        cluster_labels = clusterer.fit_predict(np.array(df['features'].values.tolist()))
        cluster_labels = ['None' if c == -1 else c for c in cluster_labels]
        self.df['cluster_ids'] = cluster_labels

    def run_hdbscan(self, df, min_cluster_size, min_samples):
        """
        run hdbscan algorithm algorithm - computes cluster labels for short text
        :param df: loaded dataframe
        :param min_cluster_size: hdbscan parameter minimum cluster size
        :param min_samples: dbscan parameter minimum samples
        :return:
        """
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, algorithm='generic', metric='cosine',
                                    min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(np.array(df['features'].values.tolist()))
        cluster_labels = ['None' if c == -1 else c for c in cluster_labels]
        self.df['cluster_ids'] = cluster_labels

    def generate_labels(self):
        """calculate cluster purity
        """
        self.df['representative_label'] = ['None'] * len(self.df)
        total = 0
        noise_count = 0
        purities = []
        for cluster_id in self.df.cluster_ids.unique():
            if cluster_id == 'None': continue
            from collections import Counter
            tmp_df = self.df.loc[self.df.cluster_ids == cluster_id][self.target_column].values.tolist()
            counts = Counter(tmp_df)
            intent = None
            cur_value = 0
            for key, value in counts.items():
                if value > cur_value:
                    cur_value = value
                    intent = key
            if len(tmp_df) == 0: continue
            purity = round(cur_value / len(tmp_df), 2)
            purities.append(purity)
            total += purity
            if purity >= 0.5:
                self.df.loc[self.df.cluster_ids == cluster_id, 'representative_label'] = intent
            else:
                noise_count += 1

        return noise_count

    """Propagating labels to the nearby points
    """

    def label_propagation(self):
        """propagate labels to unlabelled samples
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import LogisticRegression
        X = np.array(self.df.loc[self.df.representative_label != 'None']['features'].values.tolist())
        labels = self.df.loc[self.df.representative_label != 'None']['representative_label'].values.tolist()
        le = LabelEncoder()
        y = le.fit_transform(labels)
        # print('Feature matrix shape: ', X.shape)
        # print('Target matrix shape: ', y.shape)
        clf = LogisticRegression(class_weight='balanced', C=0.8, solver='newton-cg')
        clf.fit(X, y)
        feat = np.array(self.df['features'].values.tolist())
        y_pred = clf.predict(feat)
        labels = [le.classes_[i] for i in y_pred]
        self.df['predictedIntent'] = labels

    def evaulate_iter_dbscan(self, all_parameters):
        return self.run_iter(all_parameters, 'ITER_DBSCAN')




    def run_iter(self, all_parameters, algorithm):
        self.load_data()
        self.extract_feature()
        param_results = []
        for i in tqdm(range(len(all_parameters))):
            try:
                start_time = time()
                if algorithm == 'ITER_DBSCAN':
                    self.run_iter_dbscan(self.df, dist=all_parameters[i]['distance'],
                                         max_iter=all_parameters[i]['max_iteration'],
                                         min_sample=all_parameters[i]['minimum_samples'])
                elif algorithm == 'DBSCAN':
                    self.run_dbscan(self.df, min_distance=all_parameters[i]['min_distance'],
                                    minimum_samples=all_parameters[i]['minimum_samples'])
                elif algorithm == 'HDBSCAN':
                    self.run_hdbscan(self.df, min_cluster_size=all_parameters[i]['min_cluster_size'],
                                     min_samples=all_parameters[i]['min_samples'])
                else:
                    continue

                time_taken = time() - start_time
                noise_count = self.generate_labels()

                if 2 > len(self.df[self.df['representative_label'] != 'None']['representative_label'].unique()):
                    continue

                self.label_propagation()
                per_labelled = round(len(self.df.loc[self.df.cluster_ids != 'None']) / len(self.df) * 100, 2)
                num_clusters = len(list(set(self.df.loc[self.df.cluster_ids != 'None']['cluster_ids'].values.tolist())))
                true_intent = self.df[self.target_column].values.tolist()
                predicted_intent = self.df['predictedIntent'].values.tolist()
                h_score = round(metrics.homogeneity_score(true_intent, predicted_intent), 2)
                c_score = round(metrics.completeness_score(true_intent, predicted_intent), 2)
                nmf = round(metrics.normalized_mutual_info_score(true_intent, predicted_intent), 2)
                amf = round(metrics.adjusted_mutual_info_score(true_intent, predicted_intent), 2)
                ars = round(metrics.adjusted_rand_score(true_intent, predicted_intent), 2)

                le = LabelEncoder()

                le = le.fit(self.df[self.target_column].values.tolist())

                true = le.transform(self.df[self.target_column].values.tolist())
                pred = le.transform(self.df['predictedIntent'].values.tolist())
                accuracy = metrics.accuracy_score(true, pred)
                precision = metrics.precision_score(true, pred, average='weighted')
                recall = metrics.recall_score(true, pred, average='weighted')
                f1 = metrics.f1_score(true, pred, average='weighted')

                param = all_parameters[i]
                param['time'] = round(time_taken, 2)
                param['percentage_labelled'] = per_labelled
                param['clusters'] = num_clusters
                param['noisy_clusters'] = noise_count
                param['homogeneity_score'] = h_score
                param['completeness_score'] = c_score
                param['normalized_mutual_info_score'] = nmf
                param['adjusted_mutual_info_score'] = amf
                param['adjusted_rand_score'] = ars
                param['accuracy'] = round(accuracy, 3) * 100.0
                param['precision'] = round(precision, 3) * 100.0
                param['recall'] = round(recall, 3) * 100.0
                param['f1'] = round(f1, 3) * 100.0
                param['accuracy'] = accuracy

                param['intents'] = len(
                    self.df.loc[self.df['representative_label'] != 'None']['representative_label'].value_counts())
                param_results.append(param)
            except Exception as e:
                print(str(e))
                traceback.print_exc()
                pass
        return param_results
