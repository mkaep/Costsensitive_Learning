# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:13:15 2020

@author: Manuel Camargo
@author: Martin Kaeppel modified on Jan 31 09:36
"""
import itertools
import numpy as np
import pandas as pd

from nltk.util import ngrams
import keras.utils as ku

from approach.lstm import nn_support as nsup


class SequencesCreator():

    def __init__(self, log, ac_index, rl_index):
        """constructor"""
        self.log = log
        print(log)
        self.ac_index = ac_index
        self.rl_index = rl_index

    def vectorize(self, model_type):
        loader = self._get_vectorizer(model_type)
        return loader()

    def _get_vectorizer(self, model_type):
        if model_type == 'shared_cat':
            return self._vectorize_shared_cat
        else:
            raise ValueError(model_type)

    # Methode modifozoert damit mit eigenen Testdaten geearbeitet wird zudem ist der Feature Manager nicht notwendig
    def _vectorize_shared_cat(self):
        """
        Example function with types documented in the docstring.
        parms:
            log_df (dataframe): event log data.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        equi = {'ac_index': 'activities',
                'rl_index': 'roles',
                'dur_norm': 'times'}

        columns = list(equi.keys())

        vec = {'prefixes': dict(),
               'next_evt': dict(),
               'max_dur': np.max(self.log.dur)}

        self.log = self.reformat_events(columns)

        # n-gram definition
        for i, _ in enumerate(self.log):
            for x in columns:
                serie = list(ngrams(self.log[i][x], 5,
                                     pad_left=True, left_pad_symbol=0))
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                y_serie = y_serie[1:]
                vec['prefixes'][equi[x]] = vec['prefixes'][equi[x]] + serie if i > 0 else serie
                vec['next_evt'][equi[x]] = vec['next_evt'][equi[x]] + y_serie if i > 0 else y_serie
             

        # Transform task, dur and role prefixes in vectors
        for value in equi.values():
            vec['prefixes'][value] = np.array(vec['prefixes'][value])
            vec['next_evt'][value] = np.array(vec['next_evt'][value])
        # Reshape dur (prefixes, n-gram size, 1) i.e. time distribute
        vec['prefixes']['times'] = vec['prefixes']['times'].reshape(
                (vec['prefixes']['times'].shape[0],
                 vec['prefixes']['times'].shape[1], 1))
        # one-hot encode target values
        vec['next_evt']['activities'] = ku.to_categorical(
            vec['next_evt']['activities'], num_classes=len(self.ac_index))
        vec['next_evt']['roles'] = ku.to_categorical(
            vec['next_evt']['roles'], num_classes=len(self.rl_index))

        return vec

    # =============================================================================
    # Reformat events, modified by removing paramater one timestamp since we work only with end timestamü
    # =============================================================================
    def reformat_events(self, columns):
        """Creates series of activities, roles and relative times per trace.
        parms:
            self.log: dataframe.
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
