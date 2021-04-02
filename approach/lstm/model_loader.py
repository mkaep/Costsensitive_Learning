# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:53:54 2020

@author: Manuel Camargo

Modified by Martin Kaeppel
"""
from approach.lstm.models import model_shared_cat as mshcat
from approach.lstm.models import model_shared_cat_intercase as mshcati
from approach.lstm.models import model_seq2seq as seq
from approach.lstm.models import model_seq2seq_intercase as seqi


class ModelLoader():

    def __init__(self, parms):
        self.parms = parms

    def train(self, model_type, examples, ac_weights, rl_weights, output_folder, class_weights=None):
        loader = self._get_trainer(model_type)
        loader(examples, ac_weights, rl_weights, output_folder, self.parms, class_weights=class_weights)

    def _get_trainer(self, model_type):
        if model_type == 'shared_cat':
            return mshcat._training_model
        elif model_type == 'shared_cat_inter':
            return mshcati._training_model
        elif model_type == 'seq2seq':
            return seq._training_model
        elif model_type == 'seq2seq_inter':
            return seqi._training_model
        else:
            raise ValueError(model_type)