# Overview (Intent Mining from past conversations for Conversational Agent)
ITER-DBSCAN implementation for unbalanced data clustering. The algorithm is 
is tested on short text dataset (conversational intent mining from utterances) 
and achieve state-of-the art result. The work in accepted in COLING-2020.
All the dataset and results are shared for future evaluation. 

paper Link: https://www.aclweb.org/anthology/2020.coling-main.366/

Please note, we have only shared the base ITER-DBSCAN implementation. The
parallelized implementation of ITER-DBSCAN is not shared. 

All the raw and processed dataset is shared for future research in **Data** and 
**ProcessedData** folder. 

The result of ITER-DBSCAN and parallelized ITER-DBSCAN evaluation on the dataset
is shared in **NewResults** and **publishedResults** folder.

# Code (API Reference)
API Reference :
ITER-DBSCAN Implementation - Iteratively adapt dbscan parameters for unbalanced data (text) clustering
    The change of core parameters of DBSCAN i.e. distance and minimum samples parameters are changed smoothly to
    find high to low density clusters. At each iteration distance parameter is increased by 0.01 and minimum samples
    are decreased by 1. The algorithm uses cosine distance for cluster creation.

**ITER-DBSCAN(initial_distance, initial_minimum_samples, delta_distance, delta_minimum_samples, max_iteration, threshold, features)**
Parameters:
- initial_distance: initial distance for initial cluster creation (default: 0.10)
- initial_minimum_samples: initial minimum sample count for initial cluster creation (default: 20)
- delta_distance: change in distance parameter at each iteration(default: 0.01)
- delta_minimum_samples: change in minimum sample parameter (of DBSCAN) at each iteration(default: 0.01)
- max_iteration : maximum number of iteration the DBSCAN algorithm will run for cluster creation(default: 5)
- threshold: threshold parameter controls the size of the cluster, any cluster contains more than threshold parameter will be discarded. (default: 300)
- features: default values is None, the algorithm expects a list of short texts. In case the representation is pre-computed for text or data sources (pass features values as "precomputed"). default: None

In our experiments, delta_distance and delta_minimum_samples changed constantly by
a factor of 0.01 and 1 respectively.

# API Usage
  Download ITER-DBSCAN package from Pypi repository.
  `pip install ShortTextClustering`

# Sample Code
# Load Packages


```python
import pandas as pd
from ShortTextClustering.ITER_DBSCAN import ITER_DBSCAN
from ShortTextClustering.evaluation import EvaluateDataset
```

# Load Dataset


```python
df = pd.read_excel("WebApplicationsCorpus.xlsx")
```


```python
df.head(5)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>intent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alternative to Facebook</td>
      <td>Find Alternative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>How do I delete my Facebook account?</td>
      <td>Delete Account</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Are there any good Pandora alternatives with g...</td>
      <td>Find Alternative</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Is it possible to export my data from Trello t...</td>
      <td>Export Data</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Is there an online alternative to iGoogle</td>
      <td>Find Alternative</td>
    </tr>
  </tbody>
</table>
</div>



# Distribution of intents


```python
df.intent.value_counts()
```




    Find Alternative    23
    Filter Spam         20
    Delete Account      17
    Sync Accounts        9
    Change Password      8
    None                 6
    Export Data          5
    Name: intent, dtype: int64



# Remove Intent type "None"


```python
print('Before: ', len(df))
df = df.loc[df.intent != 'None']
print('After: ', len(df))
df = df.reset_index()
del df['index']
```

    Before:  88
    After:  82
    


```python
df.intent.value_counts()
```




    Find Alternative    23
    Filter Spam         20
    Delete Account      17
    Sync Accounts        9
    Change Password      8
    Export Data          5
    Name: intent, dtype: int64



# Generate cluster labels for short text dataset


```python
dataset = df.data.values.tolist()
```


```python
%%time
model = ITER_DBSCAN(initial_distance=0.3, initial_minimum_samples=16, delta_distance=0.01, delta_minimum_samples=1, max_iteration=15)
```

    Wall time: 0 ns
    


```python
%%time
labels = model.fit_predict(dataset)
```

    Wall time: 48 ms
    


```python
df['cluster_ids'] = labels
```

# Cluster distribution
Noisy points are marked as -1


```python
df.cluster_ids.value_counts()
```




    -1    33
     0    13
     1    12
     3     5
     2     5
     6     4
     4     4
     7     3
     5     3
    Name: cluster_ids, dtype: int64



# Clustered Data result


```python
df.loc[df.cluster_ids == 0]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>intent</th>
      <th>cluster_ids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>How do I delete my Facebook account?</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>How can I delete my 160by2 account?</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>How can I permanently delete my Yahoo mail acc...</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>How to delete my imgur account?</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>How to delete a Sify Mail account</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>How to permanently delete a 37signals ID</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>How can I delete my Hunch account?</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>How can I delete my Twitter account?</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>76</th>
      <td>How do I delete my LinkedIn account?</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>How do I delete my Gmail account?</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>How do I delete my Experts Exchange account?</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>How do I delete my Ohloh profile?</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>How can I permanently delete my MySpace account?</td>
      <td>Delete Account</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Evaluate ITER-DBSCAN performance on a dataset with different parameters


```python
evaluate_dataset = EvaluateDataset(filename='WebApplicationsCorpus.xlsx', filetype='xlsx', text_column='data', 
                                   target_column='intent')
```


```python
parameters = [
             {
               "distance":0.3, 
               "minimum_samples":16, 
               "delta_distance":0.01, 
               "delta_minimum_samples":1, 
               "max_iteration":15
             },
             {
               "distance":0.25, 
               "minimum_samples":14, 
               "delta_distance":0.01, 
               "delta_minimum_samples":1, 
               "max_iteration":12
             }, 
             {
               "distance":0.28, 
               "minimum_samples":12, 
               "delta_distance":0.01, 
               "delta_minimum_samples":1, 
               "max_iteration":12
             }
             ]
```

# Generate different metrics of parameter evaluation with ITER-DBSCAN


```python
%%time
results = evaluate_dataset.evaulate_iter_dbscan(parameters)
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 14.10it/s]

    Wall time: 229 ms
    

    
    


```python
result_df = pd.DataFrame.from_dict(results)
```


```python
result_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>distance</th>
      <th>minimum_samples</th>
      <th>delta_distance</th>
      <th>delta_minimum_samples</th>
      <th>max_iteration</th>
      <th>time</th>
      <th>percentage_labelled</th>
      <th>clusters</th>
      <th>noisy_clusters</th>
      <th>homogeneity_score</th>
      <th>completeness_score</th>
      <th>normalized_mutual_info_score</th>
      <th>adjusted_mutual_info_score</th>
      <th>adjusted_rand_score</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>intents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.30</td>
      <td>16</td>
      <td>0.01</td>
      <td>1</td>
      <td>15</td>
      <td>0.06</td>
      <td>56.82</td>
      <td>8</td>
      <td>0</td>
      <td>0.76</td>
      <td>0.88</td>
      <td>0.81</td>
      <td>0.79</td>
      <td>0.81</td>
      <td>0.852273</td>
      <td>75.0</td>
      <td>85.2</td>
      <td>79.7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.25</td>
      <td>14</td>
      <td>0.01</td>
      <td>1</td>
      <td>12</td>
      <td>0.03</td>
      <td>42.05</td>
      <td>6</td>
      <td>0</td>
      <td>0.70</td>
      <td>0.82</td>
      <td>0.76</td>
      <td>0.73</td>
      <td>0.74</td>
      <td>0.818182</td>
      <td>72.4</td>
      <td>81.8</td>
      <td>76.6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.28</td>
      <td>12</td>
      <td>0.01</td>
      <td>1</td>
      <td>12</td>
      <td>0.04</td>
      <td>46.59</td>
      <td>7</td>
      <td>0</td>
      <td>0.73</td>
      <td>0.85</td>
      <td>0.79</td>
      <td>0.77</td>
      <td>0.78</td>
      <td>0.840909</td>
      <td>74.1</td>
      <td>84.1</td>
      <td>78.7</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


# Citation
If you are using this code in your work, please cite this paper:

`@inproceedings{chatterjee-sengupta-2020-intent,
    title = "Intent Mining from past conversations for Conversational Agent",
    author = "Chatterjee, Ajay  and
      Sengupta, Shubhashis",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.366",
    pages = "4140--4152",
    abstract = "Conversational systems are of primary interest in the AI community. Organizations are increasingly using chatbot to provide round-the-clock support and to increase customer engagement. Many commercial bot building frameworks follow a standard approach that requires one to build and train an intent model to recognize user input. These frameworks require a collection of user utterances and corresponding intent to train an intent model. Collecting a substantial coverage of training data is a bottleneck in the bot building process. In cases where past conversation data is available, the cost of labeling hundreds of utterances with intent labels is time-consuming and laborious. In this paper, we present an intent discovery framework that can mine a vast amount of conversational logs and to generate labeled data sets for training intent models. We have introduced an extension to the DBSCAN algorithm and presented a density-based clustering algorithm ITER-DBSCAN for unbalanced data clustering. Empirical evaluation on one conversation dataset, six different intent dataset, and one short text clustering dataset show the effectiveness of our hypothesis.",
}
`
