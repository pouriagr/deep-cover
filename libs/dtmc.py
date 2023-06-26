import numpy as np
import pandas as pd
from libs.sparse_matrix import SparseMatrix

class DTMC:
  def __init__(self, traces, traces_dist_to_centroids, states_count):
    traces_df = pd.DataFrame(data=traces.transpose(),columns=np.arange(len(traces)).astype(str))
    if traces_dist_to_centroids is None:
      traces_dist_to_centroids_df = None
    else:
      traces_dist_to_centroids_df = pd.DataFrame(data=traces_dist_to_centroids.transpose(),columns=np.arange(len(traces_dist_to_centroids)).astype(str))
    transition_matrix = SparseMatrix((states_count, states_count))

    def count_transitions(x):
      transition_matrix.set_value(
          np.array(x)[0], 
          np.array(x)[1], 
          transition_matrix.get_value(np.array(x)[0], np.array(x)[1])+1
      )
      return 0

    for c in traces_df.columns:
      _ = traces_df[c].rolling(window=2, center=False).apply(lambda x: count_transitions(x))

    self.transition_matrix = transition_matrix
    self.traces_df = traces_df
    self.traces_dist_to_centroids_df = traces_dist_to_centroids_df