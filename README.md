ITER-DBSCAN implementation for unbalanced data clustering. The algorithm is 
is tested on short text dataset (conversational intent mining from utterances) 
and achieve state-of-the art result. The work in accepted in COLING-2020.
All the dataset and results are shared for future evaluation. 

Please note, we have only shared the base ITER-DBSCAN implementation. The
parallelized implementation of ITER-DBSCAN is not shared. 

All the raw and processed dataset is shared for future research in **Data** and 
**ProcessedData** folder. 

The result of ITER-DBSCAN and parallelized ITER-DBSCAN evaluation on the dataset
is shared in **NewResults** and **publishedResults** folder.  

The based implementation of the ITER-DBSCAN is available in ITER_DBSCAN.py
script. Later, a pypi installable will be shared. 

API Reference :

ITER-DBSCAN Implementation - Iteratively adapt dbscan parameters for unbalanced data (text) clustering
    The change of core parameters of DBSCAN i.e. distance and minimum samples parameters are changed smoothly to
    find high to low density clusters. At each iteration distance parameter is increased by 0.01 and minimum samples
    are decreased by 1. The algorithm uses cosine distance for cluster creation

ITER-DBSCAN(initial_distance, initial_minimum_samples, delta_distance, delta_minimum_samples,
                 max_iteration, threshold, features)
     
Parameters:
initial_distance: initial distance for initial cluster creation (default: 0.10)

initial_minimum_samples: initial minimum sample count for initial cluster creation (default: 20)

delta_distance: change in distance parameter at each iteration(default: 0.01)

delta_minimum_samples: change in minimum sample parameter (of DBSCAN) at each iteration(default: 0.01)

max_iteration : maximum number of iteration the DBSCAN algorithm will run for cluster creation(default: 5)

threshold: threshold parameter controls the size of the cluster, any cluster contains more than threshold parameter
            will be discarded. (default: 300)
            
features: default values is None, the algorithm expects a list of short texts. In case the representation is
            pre-computed for text or data sources (pass features values as "precomputed").
            
                
In our experiments, delta_distance and delta_minimum_samples changed constantly by
a factor of 0.01 and 1 respectively. 

**Citation**
If you are using this code in your work, please cite this paper:

@misc{chatterjee2020intent,
      title={Intent Mining from past conversations for Conversational Agent}, 
      author={Ajay Chatterjee and Shubhashis Sengupta},
      year={2020},
      eprint={2005.11014},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
