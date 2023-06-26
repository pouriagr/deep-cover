import numpy as np
import pandas as pd

class FeatureExtractor:
  def __init__(self):
    pass

  def __count_new_transitions(self, dtmc_matrix, one_trace_array):
    zeros_count = 0
    state1 = one_trace_array[0]
    for state in one_trace_array[1:]:
      state2 = state
      if dtmc_matrix.get_value(state1,state2) == 0:
        zeros_count += 1
      state1=state2

    return zeros_count

  def __count_new_states(self, dtmc_states_label, one_trace_array):
    return np.sum(np.logical_not(np.isin(one_trace_array, dtmc_states_label)))
  
  def __cal_last_state_share_rate_and_count(self, last_states_df, one_trace_array, pred_label):
    trace_last_states_label = one_trace_array[-1]
    filtered_df = last_states_df.query(f'(last_state=={trace_last_states_label})&(pred_label=={pred_label})')
    if len(filtered_df) == 0:
      return 0, 0
    
    return round(filtered_df['prob'].values[0],2),  filtered_df['count'].values[0]

  def __cal_trace_probability(self, dtmc_matrix, one_trace_array):
    probability_ls = []
    state1 = one_trace_array[0]
    for state in one_trace_array[1:]:
      state2 = state

      probability_ls.append(
          dtmc_matrix.get_transition_probability(state1,state2)
      )
      state1=state2

    probability_ls = np.array(probability_ls)
    return probability_ls.sum(), probability_ls.mean(), np.prod(probability_ls)
  
  def extract_features(self, dtmc_model, last_states_df, true_test, dtmc_model_test, pred_test):
    trace_has_new_transition = []
    trace_has_new_state = []
    trace_share_rate_at_last_state = []
    trace_count_at_last_state = []
    trace_probability_prod_ls = []
    trace_probability_sum_ls = []
    trace_probability_mean_ls = []
    trace_label_is_wrong = []

    dtmc_states_label = np.unique(dtmc_model.traces_df.values.reshape(-1))
    for index in range(len(true_test)):
      current_trace_array = dtmc_model_test.traces_df[str(index)].values
      trace_has_new_transition.append(self.__count_new_transitions(dtmc_model.transition_matrix, current_trace_array))
      trace_has_new_state.append(self.__count_new_states(dtmc_states_label, current_trace_array))
      
      last_state_share_rate, last_state_count =  self.__cal_last_state_share_rate_and_count(last_states_df, current_trace_array, pred_test[index])
      trace_share_rate_at_last_state.append(last_state_share_rate)
      trace_count_at_last_state.append(last_state_count)

      trace_probability_sum, trace_probability_mean, trace_probability_prod = self.__cal_trace_probability(dtmc_model.transition_matrix, current_trace_array)
      trace_probability_sum_ls.append(trace_probability_sum)
      trace_probability_mean_ls.append(trace_probability_mean)
      trace_probability_prod_ls.append(trace_probability_prod)

      trace_label_is_wrong.append(int(not(pred_test[index] == true_test[index])))

    results_df = pd.DataFrame()
    results_df['trace_has_new_transition'] = trace_has_new_transition
    results_df['trace_has_new_state'] = trace_has_new_state
    results_df['trace_share_rate_at_last_state'] = trace_share_rate_at_last_state
    results_df['trace_count_at_last_state'] = trace_count_at_last_state
    results_df['trace_label_is_wrong'] = trace_label_is_wrong
    results_df['trace_probability_sum'] = trace_probability_sum_ls
    results_df['trace_probability_mean'] = trace_probability_mean_ls
    results_df['trace_probability_prod'] = trace_probability_prod_ls
    results_df['pred_label'] = pred_test
    results_df['true_label'] = true_test

    return results_df