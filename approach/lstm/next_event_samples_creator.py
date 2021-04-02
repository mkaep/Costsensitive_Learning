# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:28:18 2020

@author: Manuel Camargo

Modified by Martin Kaeppel
"""
import itertools

import pandas as pd


class NextEventSamplesCreator():
    """
    This is the man class encharged of the model training
    """

    def __init__(self):
        self.log = pd.DataFrame
        self.ac_index = dict()
        self.rl_index = dict()

    def create_samples(self, params, log, ac_index, rl_index):
        self.log = log
        self.ac_index = ac_index
        self.rl_index = rl_index
        sampler = self._get_model_specific_sampler(params['model_type'])
        return sampler(params)

    def _get_model_specific_sampler(self, model_type):
        if model_type == 'shared_cat':
            return self._sample_next_event_shared_cat
        else:
            raise ValueError(model_type)

    def _sample_next_event_shared_cat(self, parms):
        """
        Extraction of prefixes and expected suffixes from event log.
        Args:
            df_test (dataframe): testing dataframe in pandas format.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            pref_size (int): size of the prefixes to extract.
        Returns:
            list: list of prefixes and expected sufixes.
        """
        columns = ['ac_index', 'rl_index', 'dur_norm']
        self.log = self.reformat_events(columns)
        examples = {'prefixes': dict(), 'next_evt': dict()}
        # n-gram definition
        equi = {'ac_index': 'activities',
                'rl_index': 'roles',
                'dur_norm': 'times'}
        for i, _ in enumerate(self.log):
            for x in columns:
                serie = [self.log[i][x][:idx]
                         for idx in range(1, len(self.log[i][x]))]
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                y_serie = y_serie[1:]
                examples['prefixes'][equi[x]] = (
                    examples['prefixes'][equi[x]] + serie
                    if i > 0 else serie)
                examples['next_evt'][equi[x]] = (
                    examples['next_evt'][equi[x]] + y_serie
                    if i > 0 else y_serie)
        return examples


    def reformat_events(self, columns):
        """Creates series of activities, roles and relative times per trace.
        Args:
            log_df: dataframe.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
        Returns:
            list: lists of activities, roles and relative times.
        """
        temp_data = list()
        log_df = self.log.to_dict('records')
        key = 'end_timestamp'
        log_df = sorted(log_df, key=lambda x: (x['caseid'], key))
        for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for x in columns:
                serie = [y[x] for y in trace]
                if x == 'ac_index':
                    serie.insert(0, self.ac_index[('Start')])
                    serie.append(self.ac_index[('End')])
                elif x == 'rl_index':
                    serie.insert(0, self.rl_index[('Start')])
                    serie.append(self.rl_index[('End')])
                else:
                    serie.insert(0, 0)
                    serie.append(0)
                temp_dict = {**{x: serie}, **temp_dict}
            temp_dict = {**{'caseid': key}, **temp_dict}
            temp_data.append(temp_dict)
        return temp_data
