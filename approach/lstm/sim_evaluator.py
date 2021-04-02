"""
Created on Fri Jan 10 11:40:46 2020

@author: Manuel Camargo

Modified by Martin Kaeppel
"""

import numpy as np


class Evaluator():

    def measure(self, metric, data, feature=None):
        import pandas as pd
        data_df = pd.DataFrame(data)
        data_df.to_csv('I:/data_in_sim_evaluator', sep=";")
        data_df_n = data_df.loc[data_df['implementation'] == 'Random Choice']
        print(data_df_n)
        exit(-1)
        #balanced_accuracy_score(y_test_max, y_predicted_max)
        evaluator = self._get_metric_evaluator(metric)
        return evaluator(data, feature)

    def _get_metric_evaluator(self, metric):
        if metric == 'accuracy':
            return self._accuracy_evaluation
        elif metric == 'mae_suffix':
            return self._mae_remaining_evaluation
        else:
            raise ValueError(metric)

    def _accuracy_evaluation(self, data, feature):
        data = data.copy()
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'implementation']]
        eval_acc = (lambda x:
                    1 if x[feature + '_expect'] == x[feature + '_pred'] else 0)
        data[feature + '_acc'] = data.apply(eval_acc, axis=1)
        # agregate true positives
        data = (data.groupby(['implementation', 'run_num'])[feature + '_acc']
                .agg(['sum', 'count'])
                .reset_index())
        # calculate accuracy
        data['accuracy'] = np.divide(data['sum'], data['count'])
        return data

    def _mae_remaining_evaluation(self, data, feature):
        data = data.copy()
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'implementation']]
        ae = (lambda x: np.abs(np.sum(x[feature + '_expect']) -
                               np.sum(x[feature + '_pred'])))
        data['ae'] = data.apply(ae, axis=1)
        data = (data.groupby(['implementation', 'run_num'])['ae']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae'}))
        return data


