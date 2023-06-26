import pandas as pd
import numpy as np

class CoverageCriterias:
  def __init__(
    self, 
    dtmc_model, 
    last_states_df, 
    last_states_true_based_df, 
    all_states_counts, 
    dataset_classes_count, 
    mean_dist_to_centroids, 
    max_dist_to_centroids
  ):
    
    self.last_states_true_based_df = last_states_true_based_df
    self.last_states_df = last_states_df
    self.all_states_counts = all_states_counts
    self.dataset_classes_count = dataset_classes_count
    self.mean_dist_to_centroids = mean_dist_to_centroids
    self.max_dist_to_centroids = max_dist_to_centroids
    
    self.last_states_in_training = np.unique(self.last_states_df['last_state'].astype(str))
    self.last_states_and_true_label_in_training = (
      self.last_states_true_based_df['last_state'].astype(str)\
      + '.' + self.last_states_true_based_df['true_label'].astype(str)
    ).values
    
    self.train_states_np, self.train_states_count_np = np.unique(dtmc_model.traces_df, return_counts=True)

    train_traces_df = dtmc_model.traces_df.copy()
    train_transitions_df = train_traces_df.transpose().reset_index(drop=True)[0:-1]\
                                          .reset_index(drop=True).astype(str) + '_' +\
                                          train_traces_df.transpose()\
                                          .reset_index(drop=True)[1:]\
                                          .reset_index(drop=True).astype(str)
    self.train_transitions_np, self.train_transitions_count_np = np.unique(train_transitions_df, return_counts=True)

  def accuracy(self, traces_features_df):
    return 1 - traces_features_df['trace_label_is_wrong'].sum()/len(traces_features_df)

  def accuracy_on_coverage(self, traces_features_df, traces_covering_type_np):
    traces_features_df['traces_covering_type'] = traces_covering_type_np.astype(float)
    traces_features_df['accuracy_weight'] = 1
    agg_df = traces_features_df.query('traces_covering_type>-1')[[
      'traces_covering_type', 'trace_label_is_wrong', 'accuracy_weight'
    ]].groupby('traces_covering_type').sum().reset_index()

    accuracy_on_each_block = 1-(agg_df['trace_label_is_wrong']/agg_df['accuracy_weight'])
    acc = (accuracy_on_each_block * agg_df['accuracy_weight']).sum()/agg_df['accuracy_weight'].sum()
    return acc
  
  def new_last_state_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    seen_last_states_in_this_test = np.unique(traces_df[traces_df.shape[1]-1].values.astype(str))
    coverage_coe = np.logical_not(np.isin(seen_last_states_in_this_test, self.last_states_in_training)).sum() /\
                   (self.all_states_counts - len(self.last_states_in_training))
    
    traces_covering_type_np = np.logical_not(np.isin(traces_df[traces_df.shape[1]-1].values.astype(str), self.last_states_in_training))
    traces_covering_type_np = traces_df[traces_df.shape[1]-1].values.astype(int) * traces_covering_type_np - 1
    
    return coverage_coe, traces_covering_type_np
  
  def all_labels_and_last_states_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    seen_last_states_in_this_test = np.unique(
      traces_df[traces_df.shape[1]-1].astype(str) + '.' + traces_features_df['true_label'].astype(str)
    )
    
    coverage_coe = len(seen_last_states_in_this_test) / (self.all_states_counts * self.dataset_classes_count)
    traces_covering_type_np = (traces_df[traces_df.shape[1]-1].values.astype(float) + 
                               traces_features_df['true_label'].apply(lambda x: float('0.'+str(x))).values)
    
    return coverage_coe, traces_covering_type_np
  
  def weighted_all_labels_and_last_states_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    seen_last_states_in_this_test = np.unique(
      traces_df[traces_df.shape[1]-1].astype(str) + '.' + traces_features_df['true_label'].astype(str)
    )
    weights = 1/(self.last_states_true_based_df['count'])
    
    mutual_last_states_and_labels = np.isin(self.last_states_and_true_label_in_training, 
                                            seen_last_states_in_this_test)
    not_mutual_last_states_and_labels_count = len(seen_last_states_in_this_test) - np.sum(mutual_last_states_and_labels)

    coverage_coe =  ((mutual_last_states_and_labels * weights).sum() + not_mutual_last_states_and_labels_count)/\
                    (weights.sum() + not_mutual_last_states_and_labels_count)
    
    traces_covering_type_np = (traces_df[traces_df.shape[1]-1].values.astype(float) + 
                               traces_features_df['true_label'].apply(lambda x: float('0.'+str(x))).values)
    
    return coverage_coe, traces_covering_type_np
  
  def n_step_last_state_boundry_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    last_states_dist = traces_dist_to_centroids_df[traces_dist_to_centroids_df.shape[1]-1]
    covered_layers = (last_states_dist/self.mean_dist_to_centroids).astype(str)

    coverage_value = np.unique(
      covered_layers[last_states_dist > self.max_dist_to_centroids] + '.' +\
      traces_df[traces_df.shape[1]-1].values[last_states_dist > self.max_dist_to_centroids].astype(str)
    ).__len__()

    traces_covering_type_np = (covered_layers + '.' +\
                               traces_df[traces_df.shape[1]-1].values.astype(str))
    _, traces_covering_type_np = np.unique(traces_covering_type_np, return_inverse=True)
    traces_covering_type_np[last_states_dist < self.max_dist_to_centroids] = -1

    return coverage_value, traces_covering_type_np


  def basic_last_state_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    temp_last_states = np.unique(traces_df[traces_df.shape[1]-1].astype(str))
    coverage_coe = np.isin(temp_last_states, self.last_states_in_training).sum() / len(self.last_states_in_training)

    traces_covering_type_np = traces_df[traces_df.shape[1]-1].values.astype(int)
    traces_covering_type_np[np.logical_not(np.isin(traces_covering_type_np.astype(str), self.last_states_in_training))] = -1

    return coverage_coe, traces_covering_type_np

  
  def basic_labels_and_last_state_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    seen_last_states_and_true_label_in_this_test = np.unique(
      traces_df[traces_df.shape[1]-1].astype(str) + '.' + traces_features_df['true_label'].astype(str)
    )
    coverage_coe = np.isin(seen_last_states_and_true_label_in_this_test, self.last_states_and_true_label_in_training).sum() /\
                  len(self.last_states_and_true_label_in_training)
    
    traces_covering_type_np = (traces_df[traces_df.shape[1]-1].astype(str) + '.' + traces_features_df['true_label'].astype(str))
    traces_covering_type_np[np.logical_not(np.isin(traces_covering_type_np, self.last_states_and_true_label_in_training))] = '-1'
    traces_covering_type_np = traces_covering_type_np.values.astype(float)

    return coverage_coe, traces_covering_type_np

  
  def weighted_basic_labels_and_last_state_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    seen_last_states_and_true_label_in_this_test = np.unique(
      traces_df[traces_df.shape[1]-1].astype(str) + '.' + traces_features_df['true_label'].astype(str)
    )

    mutual = np.isin(self.last_states_and_true_label_in_training, seen_last_states_and_true_label_in_this_test)
    weights = self.last_states_true_based_df['count'].values

    coverage_coe = (mutual*weights).sum() / (weights).sum()

    traces_covering_type_np = (traces_df[traces_df.shape[1]-1].astype(str) + '.' + traces_features_df['true_label'].astype(str))
    traces_covering_type_np[np.logical_not(np.isin(traces_covering_type_np, self.last_states_and_true_label_in_training))] = '-1'
    traces_covering_type_np = traces_covering_type_np.values.astype(float)

    return coverage_coe, traces_covering_type_np

  
  def basic_states_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    traces_states = np.unique(traces_df)
    coverage_coe = np.isin(traces_states, self.train_states_np).sum() / len(self.train_states_np)

    traces_df['all_seen_states'] = traces_df.apply(lambda x: np.unique(x.values), axis=1)
    traces_df['no_new_state'] = traces_df['all_seen_states'].apply(lambda x: np.logical_not(np.isin(x,self.train_states_np)).sum()==0)
    _, traces_covering_type_np = np.unique(traces_df['all_seen_states'].astype(str).values, return_inverse=True)
    traces_covering_type_np = (traces_covering_type_np + 1) * traces_df['no_new_state'].values - 1

    return coverage_coe, traces_covering_type_np
    
  
  def weighted_states_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    traces_states = np.unique(traces_df)

    mutual_states = np.isin(self.train_states_np, traces_states)
    weights = 1/(self.train_states_count_np)

    not_mutual_states_count = len(traces_states) - np.sum(mutual_states)

    coverage_coe = (np.sum(mutual_states * weights)+not_mutual_states_count) / (np.sum(weights)+not_mutual_states_count)

    traces_df['all_seen_states'] = traces_df.apply(lambda x: str(np.unique(x.values)), axis=1)
    _, traces_covering_type_np = np.unique(traces_df['all_seen_states'].values, return_inverse=True) 
    return coverage_coe, traces_covering_type_np

  
  def n_step_states_boundry_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    max_states_dist = traces_dist_to_centroids_df.max(axis=1).values
    max_states_dist_related_states_index = np.argmax(traces_dist_to_centroids_df.values, axis=1)
    max_states_dist_related_states = np.diag(traces_df.values[:, max_states_dist_related_states_index])
    covered_layers = (max_states_dist/self.mean_dist_to_centroids).astype(str)

    coverage_value = np.unique(
      pd.Series(covered_layers[max_states_dist > self.max_dist_to_centroids]) + '.' +\
      pd.Series(max_states_dist_related_states[max_states_dist > self.max_dist_to_centroids].astype(str))
    ).__len__()

    _, traces_covering_type_np = np.unique(pd.Series(covered_layers) + '.' + pd.Series(max_states_dist_related_states.astype(str)), return_inverse=True)
    traces_covering_type_np[max_states_dist < self.max_dist_to_centroids] = -1

    return coverage_value, traces_covering_type_np

  
  def basic_trans_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    temp_transes = np.unique(transitions_df)
    coverage_coe = np.isin(temp_transes, self.train_transitions_np).sum() / len(self.train_transitions_np)

    transitions_df = transitions_df.transpose()
    transitions_df['all_seen_traces'] = transitions_df.apply(lambda x: np.unique(x.values), axis=1)
    transitions_df['no_new_transition'] = transitions_df['all_seen_traces'].apply(lambda x: np.logical_not(np.isin(x,self.train_transitions_np)).sum()==0)
    _, traces_covering_type_np = np.unique(transitions_df['all_seen_traces'].astype(str).values, return_inverse=True)
    traces_covering_type_np = (traces_covering_type_np + 1) * transitions_df['no_new_transition'].values - 1

    return coverage_coe, traces_covering_type_np

  def weighted_trans_coverage(self, traces_features_df, traces_df, traces_dist_to_centroids_df, transitions_df):
    temp_transes = np.unique(transitions_df)
    mutual_transes = np.isin(self.train_transitions_np, temp_transes)
    not_mutual_transes = np.logical_not(np.isin(temp_transes, self.train_transitions_np))
    weights = 1/(self.train_transitions_count_np)
    coverage_coe = (np.sum(mutual_transes * weights)+np.sum(not_mutual_transes)) / (np.sum(weights)+np.sum(not_mutual_transes))

    transitions_df = transitions_df.transpose()
    transitions_df['all_seen_traces'] = transitions_df.apply(lambda x: str(np.unique(x.values)), axis=1)
    _, traces_covering_type_np = np.unique(transitions_df['all_seen_traces'].values, return_inverse=True)
    return coverage_coe, traces_covering_type_np