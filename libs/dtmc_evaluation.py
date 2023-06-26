import pandas as pd
import numpy as np

class DtmcEvaluator:
  def __init__(self):
    pass
  
  def calculate_true_label_based_last_state_df(self, traces_df, true_label):
    last_states = traces_df[-1:].values[0]

    last_states_df = pd.DataFrame()
    last_states_df['last_state'] = last_states
    last_states_df['true_label'] = np.argmax(true_label, axis=1)
    last_states_df['count'] = 1

    last_states_df = last_states_df[['last_state', 'true_label', 'count']].groupby(['last_state', 'true_label']).sum().reset_index()

    last_states_df = last_states_df.merge(
        last_states_df[['last_state', 'count']].groupby('last_state').sum().reset_index().rename(columns={'count':'group_count'}),
        on='last_state'
    )
    last_states_df['prob'] = last_states_df['count']/last_states_df['group_count']
    return last_states_df

  def calculate_dtmc_purity(self, traces_df, pred_label):
    last_states = traces_df[-1:].values[0]

    last_states_df = pd.DataFrame()
    last_states_df['last_state'] = last_states
    last_states_df['pred_label'] = np.argmax(pred_label,axis=1)
    last_states_df['count'] = 1

    last_states_df = last_states_df[['last_state', 'pred_label', 'count']].groupby(['last_state', 'pred_label']).sum().reset_index()

    last_states_df = last_states_df.merge(
        last_states_df[['last_state', 'count']].groupby('last_state').sum().reset_index().rename(columns={'count':'group_count'}),
        on='last_state'
    )
    last_states_df['prob'] = last_states_df['count']/last_states_df['group_count']

    def find_max_share(x):
      i = np.argmax(x['prob'])
      return [x['prob'].values[i]*x['group_count'].values[i],x['group_count'].values[i]]

    purity_df = np.array(last_states_df[['last_state', 'group_count', 'prob']].groupby(['last_state']).apply(lambda x: find_max_share(x)).to_list())
    last_states_count_mean = last_states_df[['last_state', 'group_count']].drop_duplicates(['last_state'])['group_count'].mean()
    return np.sum(purity_df[:,0])/np.sum(purity_df[:,1]), last_states_count_mean, last_states_df